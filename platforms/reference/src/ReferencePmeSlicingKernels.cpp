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

#include "ReferencePmeSlicingKernels.h"
#include "SlicedPmeForce.h"
#include "internal/SlicedPmeForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/reference/ReferenceBondForce.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include <cstring>

#include "openmm/reference/ReferenceLJCoulombIxn.h"
#include "openmm/reference/ReferenceLJCoulomb14.h"

using namespace PmeSlicing;
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

ReferenceCalcSlicedPmeForceKernel::~ReferenceCalcSlicedPmeForceKernel() {
    if (neighborList != NULL)
        delete neighborList;
}

void ReferenceCalcSlicedPmeForceKernel::initialize(const System& system, const SlicedPmeForce& force) {

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge;
        force.getExceptionParameterOffset(i, param, exception, charge);
        exceptionsWithOffsets.insert(exception);
    }
    numParticles = force.getNumParticles();
    exclusions.resize(numParticles);
    vector<int> nb14s;
    map<int, int> nb14Index;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd;
        force.getExceptionParameters(i, particle1, particle2, chargeProd);
        exclusions[particle1].insert(particle2);
        exclusions[particle2].insert(particle1);
        if (chargeProd != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end()) {
            nb14Index[i] = nb14s.size();
            nb14s.push_back(i);
        }
    }

    // Build the arrays.

    num14 = nb14s.size();
    bonded14IndexArray.resize(num14, vector<int>(2));
    bonded14ParamArray.resize(num14, vector<double>(3));
    particleParamArray.resize(numParticles, vector<double>(3));
    particleCharges.resize(numParticles);
    exceptionCharges.resize(num14);
    for (int i = 0; i < numParticles; ++i)
       particleCharges[i] = force.getParticleCharge(i);
    for (int i = 0; i < num14; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(nb14s[i], particle1, particle2, exceptionCharges[i]);
        bonded14IndexArray[i][0] = particle1;
        bonded14IndexArray[i][1] = particle2;
    }
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge;
        force.getParticleParameterOffset(i, param, particle, charge);
        particleParamOffsets[make_pair(param, particle)] = charge;
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge;
        force.getExceptionParameterOffset(i, param, exception, charge);
        exceptionParamOffsets[make_pair(param, nb14Index[exception])] = charge;
    }
    nonbondedCutoff = force.getCutoffDistance();
    neighborList = new NeighborList();
    double alpha;
    SlicedPmeForceImpl::calcPMEParameters(system, force, alpha, gridSize[0], gridSize[1], gridSize[2], false);
    ewaldAlpha = alpha;
    exceptionsArePeriodic = force.getExceptionsUsePeriodicBoundaryConditions();
}

double ReferenceCalcSlicedPmeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    computeParameters(context);
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    double energy = 0;
    ReferenceLJCoulombIxn clj;
    computeNeighborListVoxelHash(*neighborList, numParticles, posData, exclusions, extractBoxVectors(context), true, nonbondedCutoff, 0.0);
    clj.setUseCutoff(nonbondedCutoff, *neighborList, 1.0);
    Vec3* boxVectors = extractBoxVectors(context);
    double minAllowedSize = 1.999999*nonbondedCutoff;
    if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize)
        throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
    clj.setPeriodic(boxVectors);
    clj.setPeriodicExceptions(exceptionsArePeriodic);
    clj.setUsePME(ewaldAlpha, gridSize);
    clj.calculatePairIxn(numParticles, posData, particleParamArray, exclusions, forceData, includeEnergy ? &energy : NULL, includeDirect, includeReciprocal);
    if (includeDirect) {
        ReferenceBondForce refBondForce;
        ReferenceLJCoulomb14 nonbonded14;
        if (exceptionsArePeriodic) {
            Vec3* boxVectors = extractBoxVectors(context);
            nonbonded14.setPeriodic(boxVectors);
        }
        refBondForce.calculateForce(num14, bonded14IndexArray, posData, bonded14ParamArray, forceData, includeEnergy ? &energy : NULL, nonbonded14);
    }
    return energy;
}

void ReferenceCalcSlicedPmeForceKernel::copyParametersToContext(ContextImpl& context, const SlicedPmeForce& force) {
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge;
        force.getExceptionParameterOffset(i, param, exception, charge);
        exceptionsWithOffsets.insert(exception);
    }
    vector<int> nb14s;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd;
        force.getExceptionParameters(i, particle1, particle2, chargeProd);
        if (chargeProd != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end())
            nb14s.push_back(i);
    }
    if (nb14s.size() != num14)
        throw OpenMMException("updateParametersInContext: The number of non-excluded exceptions has changed");

    // Record the values.

    for (int i = 0; i < numParticles; ++i)
        particleCharges[i] = force.getParticleCharge(i);
    for (int i = 0; i < num14; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(nb14s[i], particle1, particle2, exceptionCharges[i]);
        bonded14IndexArray[i][0] = particle1;
        bonded14IndexArray[i][1] = particle2;
    }
}

void ReferenceCalcSlicedPmeForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = ewaldAlpha;
    nx = gridSize[0];
    ny = gridSize[1];
    nz = gridSize[2];
}

void ReferenceCalcSlicedPmeForceKernel::computeParameters(ContextImpl& context) {
    // Compute particle parameters.

    vector<double> charges(numParticles);
    for (int i = 0; i < numParticles; i++)
        charges[i] = particleCharges[i];
    for (auto& offset : particleParamOffsets) {
        double value = context.getParameter(offset.first.first);
        int index = offset.first.second;
        charges[index] += value*offset.second;
    }
    for (int i = 0; i < numParticles; i++) {
        particleParamArray[i][0] = 0.5;
        particleParamArray[i][1] = 0.0;
        particleParamArray[i][2] = charges[i];
    }

    // Compute exception parameters.

    charges.resize(num14);
    for (int i = 0; i < num14; i++)
        charges[i] = exceptionCharges[i];
    for (auto& offset : exceptionParamOffsets) {
        double value = context.getParameter(offset.first.first);
        int index = offset.first.second;
        charges[index] += value*offset.second;
    }
    for (int i = 0; i < num14; i++) {
        bonded14ParamArray[i][0] = 1.0;
        bonded14ParamArray[i][1] = 0.0;
        bonded14ParamArray[i][2] = charges[i];
    }
}