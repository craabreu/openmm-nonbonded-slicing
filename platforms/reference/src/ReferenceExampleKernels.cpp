/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "ReferenceExampleKernels.h"
#include "NativeNonbondedForce.h"
#include "internal/NativeNonbondedForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/reference/ReferenceBondForce.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include <cstring>

#include "ReferenceLJCoulombIxn.h"
#include "ReferenceLJCoulomb14.h"

using namespace ExamplePlugin;
using namespace OpenMM;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}

static RealVec* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (RealVec*) data->periodicBoxVectors;
}

ReferenceCalcNativeNonbondedForceKernel::~ReferenceCalcNativeNonbondedForceKernel() {
    if (neighborList != NULL)
        delete neighborList;
}

void ReferenceCalcNativeNonbondedForceKernel::initialize(const System& system, const NativeNonbondedForce& force) {

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionsWithOffsets.insert(exception);
    }
    numParticles = force.getNumParticles();
    exclusions.resize(numParticles);
    vector<int> nb14s;
    map<int, int> nb14Index;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        exclusions[particle1].insert(particle2);
        exclusions[particle2].insert(particle1);
        if (chargeProd != 0.0 || epsilon != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end()) {
            nb14Index[i] = nb14s.size();
            nb14s.push_back(i);
        }
    }

    // Build the arrays.

    num14 = nb14s.size();
    bonded14IndexArray.resize(num14, vector<int>(2));
    bonded14ParamArray.resize(num14, vector<double>(3));
    particleParamArray.resize(numParticles, vector<double>(3));
    baseParticleParams.resize(numParticles);
    baseExceptionParams.resize(num14);
    for (int i = 0; i < numParticles; ++i)
       force.getParticleParameters(i, baseParticleParams[i][0], baseParticleParams[i][1], baseParticleParams[i][2]);
    for (int i = 0; i < num14; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(nb14s[i], particle1, particle2, baseExceptionParams[i][0], baseExceptionParams[i][1], baseExceptionParams[i][2]);
        bonded14IndexArray[i][0] = particle1;
        bonded14IndexArray[i][1] = particle2;
    }
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        particleParamOffsets[make_pair(param, particle)] = {charge, sigma, epsilon};
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionParamOffsets[make_pair(param, nb14Index[exception])] = {charge, sigma, epsilon};
    }
    nonbondedMethod = CalcNativeNonbondedForceKernel::NonbondedMethod(force.getNonbondedMethod());
    nonbondedCutoff = force.getCutoffDistance();
    if (nonbondedMethod == NoCutoff) {
        neighborList = NULL;
        useSwitchingFunction = false;
    }
    else {
        neighborList = new NeighborList();
        useSwitchingFunction = force.getUseSwitchingFunction();
        switchingDistance = force.getSwitchingDistance();
    }
    if (nonbondedMethod == Ewald) {
        double alpha;
        NativeNonbondedForceImpl::calcEwaldParameters(system, force, alpha, kmax[0], kmax[1], kmax[2]);
        ewaldAlpha = alpha;
    }
    else if (nonbondedMethod == PME) {
        double alpha;
        NativeNonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSize[0], gridSize[1], gridSize[2], false);
        ewaldAlpha = alpha;
    }
    else if (nonbondedMethod == LJPME) {
        double alpha;
        NativeNonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSize[0], gridSize[1], gridSize[2], false);
        ewaldAlpha = alpha;
        NativeNonbondedForceImpl::calcPMEParameters(system, force, alpha, dispersionGridSize[0], dispersionGridSize[1], dispersionGridSize[2], true);
        ewaldDispersionAlpha = alpha;
        useSwitchingFunction = false;
    }
    if (nonbondedMethod == NoCutoff || nonbondedMethod == CutoffNonPeriodic)
        exceptionsArePeriodic = false;
    else
        exceptionsArePeriodic = force.getExceptionsUsePeriodicBoundaryConditions();
    rfDielectric = force.getReactionFieldDielectric();
    if (force.getUseDispersionCorrection())
        dispersionCoefficient = NativeNonbondedForceImpl::calcDispersionCorrection(system, force);
    else
        dispersionCoefficient = 0.0;
}

double ReferenceCalcNativeNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    computeParameters(context);
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    double energy = 0;
    ReferenceLJCoulombIxn clj;
    bool periodic = (nonbondedMethod == CutoffPeriodic);
    bool ewald  = (nonbondedMethod == Ewald);
    bool pme  = (nonbondedMethod == PME);
    bool ljpme = (nonbondedMethod == LJPME);
    if (nonbondedMethod != NoCutoff) {
        computeNeighborListVoxelHash(*neighborList, numParticles, posData, exclusions, extractBoxVectors(context), periodic || ewald || pme || ljpme, nonbondedCutoff, 0.0);
        clj.setUseCutoff(nonbondedCutoff, *neighborList, rfDielectric);
    }
    if (periodic || ewald || pme || ljpme) {
        Vec3* boxVectors = extractBoxVectors(context);
        double minAllowedSize = 1.999999*nonbondedCutoff;
        if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize)
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
        clj.setPeriodic(boxVectors);
        clj.setPeriodicExceptions(exceptionsArePeriodic);
    }
    if (ewald)
        clj.setUseEwald(ewaldAlpha, kmax[0], kmax[1], kmax[2]);
    if (pme)
        clj.setUsePME(ewaldAlpha, gridSize);
    if (ljpme){
        clj.setUsePME(ewaldAlpha, gridSize);
        clj.setUseLJPME(ewaldDispersionAlpha, dispersionGridSize);
    }
    if (useSwitchingFunction)
        clj.setUseSwitchingFunction(switchingDistance);
    clj.calculatePairIxn(numParticles, posData, particleParamArray, exclusions, forceData, includeEnergy ? &energy : NULL, includeDirect, includeReciprocal);
    if (includeDirect) {
        ReferenceBondForce refBondForce;
        ReferenceLJCoulomb14 nonbonded14;
        if (exceptionsArePeriodic) {
            Vec3* boxVectors = extractBoxVectors(context);
            nonbonded14.setPeriodic(boxVectors);
        }
        refBondForce.calculateForce(num14, bonded14IndexArray, posData, bonded14ParamArray, forceData, includeEnergy ? &energy : NULL, nonbonded14);
        if (periodic || ewald || pme) {
            Vec3* boxVectors = extractBoxVectors(context);
            energy += dispersionCoefficient/(boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2]);
        }
    }
    return energy;
}

void ReferenceCalcNativeNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NativeNonbondedForce& force) {
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionsWithOffsets.insert(exception);
    }
    vector<int> nb14s;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if (chargeProd != 0.0 || epsilon != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end())
            nb14s.push_back(i);
    }
    if (nb14s.size() != num14)
        throw OpenMMException("updateParametersInContext: The number of non-excluded exceptions has changed");

    // Record the values.

    for (int i = 0; i < numParticles; ++i)
        force.getParticleParameters(i, baseParticleParams[i][0], baseParticleParams[i][1], baseParticleParams[i][2]);
    for (int i = 0; i < num14; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(nb14s[i], particle1, particle2, baseExceptionParams[i][0], baseExceptionParams[i][1], baseExceptionParams[i][2]);
        bonded14IndexArray[i][0] = particle1;
        bonded14IndexArray[i][1] = particle2;
    }
    
    // Recompute the coefficient for the dispersion correction.

    NativeNonbondedForce::NonbondedMethod method = force.getNonbondedMethod();
    if (force.getUseDispersionCorrection() && (method == NativeNonbondedForce::CutoffPeriodic || method == NativeNonbondedForce::Ewald || method == NativeNonbondedForce::PME))
        dispersionCoefficient = NativeNonbondedForceImpl::calcDispersionCorrection(context.getSystem(), force);
}

void ReferenceCalcNativeNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != PME && nonbondedMethod != LJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME or LJPME");
    alpha = ewaldAlpha;
    nx = gridSize[0];
    ny = gridSize[1];
    nz = gridSize[2];
}

void ReferenceCalcNativeNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != LJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using LJPME");
    alpha = ewaldDispersionAlpha;
    nx = dispersionGridSize[0];
    ny = dispersionGridSize[1];
    nz = dispersionGridSize[2];
}

void ReferenceCalcNativeNonbondedForceKernel::computeParameters(ContextImpl& context) {
    // Compute particle parameters.

    vector<double> charges(numParticles), sigmas(numParticles), epsilons(numParticles);
    for (int i = 0; i < numParticles; i++) {
        charges[i] = baseParticleParams[i][0];
        sigmas[i] = baseParticleParams[i][1];
        epsilons[i] = baseParticleParams[i][2];
    }
    for (auto& offset : particleParamOffsets) {
        double value = context.getParameter(offset.first.first);
        int index = offset.first.second;
        charges[index] += value*offset.second[0];
        sigmas[index] += value*offset.second[1];
        epsilons[index] += value*offset.second[2];
    }
    for (int i = 0; i < numParticles; i++) {
        particleParamArray[i][0] = 0.5*sigmas[i];
        particleParamArray[i][1] = 2.0*sqrt(epsilons[i]);
        particleParamArray[i][2] = charges[i];
    }

    // Compute exception parameters.

    charges.resize(num14);
    sigmas.resize(num14);
    epsilons.resize(num14);
    for (int i = 0; i < num14; i++) {
        charges[i] = baseExceptionParams[i][0];
        sigmas[i] = baseExceptionParams[i][1];
        epsilons[i] = baseExceptionParams[i][2];
    }
    for (auto& offset : exceptionParamOffsets) {
        double value = context.getParameter(offset.first.first);
        int index = offset.first.second;
        charges[index] += value*offset.second[0];
        sigmas[index] += value*offset.second[1];
        epsilons[index] += value*offset.second[2];
    }
    for (int i = 0; i < num14; i++) {
        bonded14ParamArray[i][0] = sigmas[i];
        bonded14ParamArray[i][1] = 4.0*epsilons[i];
        bonded14ParamArray[i][2] = charges[i];
    }
}