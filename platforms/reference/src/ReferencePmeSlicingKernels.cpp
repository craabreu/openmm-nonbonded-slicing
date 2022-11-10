/* -------------------------------------------------------------------------- *
 *                             OpenMM PME Slicing                             *
 *                             ==================                             *
 *                                                                            *
 * An OpenMM plugin for slicing Particle Mesh Ewald calculations on the basis *
 * of atom pairs and applying a different switching parameter to each slice.  *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-pme-slicing                             *
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
#include <numeric>
#include <iostream>

#include "internal/ReferenceCoulombIxn.h"
#include "internal/ReferenceCoulomb14.h"

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
    numParticles = force.getNumParticles();
    numSubsets = force.getNumSubsets();
    numSlices = numSubsets*(numSubsets+1)/2;
    subsets.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        subsets[i] = force.getParticleSubset(i);
    sliceLambda.resize(numSlices);
    sliceSwitchingParameter.resize(numSlices, "");
    for (int index = 0; index < force.getNumSwitchingParameters(); index++) {
        string parameter;
        int i, j;
        force.getSwitchingParameter(index, parameter, i, j);
        sliceSwitchingParameter[j*(j+1)/2+i] = parameter;
    }

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int index = 0; index < force.getNumExceptionChargeOffsets(); index++) {
        string param;
        int exception;
        double charge;
        force.getExceptionChargeOffset(index, param, exception, charge);
        exceptionsWithOffsets.insert(exception);
    }
    exclusions.resize(numParticles);
    nb14s.resize(numSlices);
    for (int index = 0; index < force.getNumExceptions(); index++) {
        int particle1, particle2;
        double chargeProd;
        force.getExceptionParameters(index, particle1, particle2, chargeProd);
        exclusions[particle1].insert(particle2);
        exclusions[particle2].insert(particle1);
        if (chargeProd != 0.0 || exceptionsWithOffsets.find(index) != exceptionsWithOffsets.end()) {
            int s1 = subsets[particle1];
            int s2 = subsets[particle2];
            int slice = s1 > s2 ? s1*(s1+1)/2+s2 : s2*(s2+1)/2+s1;
            nb14s[slice].push_back(index);
        }
    }

    // Build the arrays.

    particleParamArray.resize(numParticles);
    particleCharges.resize(numParticles);
    for (int i = 0; i < numParticles; ++i)
       particleCharges[i] = force.getParticleCharge(i);

    for (int index = 0; index < force.getNumParticleChargeOffsets(); index++) {
        string param;
        int particle;
        double charge;
        force.getParticleChargeOffset(index, param, particle, charge);
        particleParamOffsets[make_pair(param, particle)] = charge;
    }

    num14.resize(numSlices);
    for (int slice = 0; slice < numSlices; slice++)
        num14[slice] = nb14s[slice].size();
    total14 = accumulate(num14.begin(), num14.end(), 0);

    bonded14IndexArray.resize(numSlices);
    bonded14ParamArray.resize(numSlices);
    exceptionChargeProds.resize(numSlices);
    for (int slice = 0; slice < numSlices; slice++) {
        int n = num14[slice];
        bonded14IndexArray[slice].resize(n, vector<int>(2));
        bonded14ParamArray[slice].resize(n, vector<double>(2));
        exceptionChargeProds[slice].resize(n);
        for (int i = 0; i < n; ++i)
            force.getExceptionParameters(
                nb14s[slice][i],
                bonded14IndexArray[slice][i][0],
                bonded14IndexArray[slice][i][1],
                exceptionChargeProds[slice][i]
            );
    }

    for (int index = 0; index < force.getNumExceptionChargeOffsets(); index++) {
        string param;
        int exception;
        double charge;
        force.getExceptionChargeOffset(index, param, exception, charge);
        exceptionParamOffsets[exception].push_back(make_pair(param,charge));
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
    vector<double> sliceEnergies(numSlices, 0.0);

    computeNeighborListVoxelHash(*neighborList, numParticles, posData, exclusions, extractBoxVectors(context), true, nonbondedCutoff, 0.0);

    ReferenceCoulombIxn coulomb;
    coulomb.setCutoff(nonbondedCutoff, *neighborList);
    Vec3* boxVectors = extractBoxVectors(context);
    double minAllowedSize = 1.999999*nonbondedCutoff;
    if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize)
        throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
    coulomb.setPeriodic(boxVectors);
    coulomb.setPeriodicExceptions(exceptionsArePeriodic);
    coulomb.setPME(ewaldAlpha, gridSize);
    coulomb.calculateEwaldIxn(numParticles, posData,
                              numSubsets, subsets, sliceLambda,
                              particleParamArray, exclusions,
                              forceData, sliceEnergies,
                              includeDirect, includeReciprocal);

    if (includeDirect) {
        ReferenceBondForce refBondForce;
        ReferenceCoulomb14 nonbonded14;
        if (exceptionsArePeriodic) {
            Vec3* boxVectors = extractBoxVectors(context);
            nonbonded14.setPeriodic(boxVectors);
        }
        for (int slice = 0; slice < numSlices; slice++)
            refBondForce.calculateForce(num14[slice], bonded14IndexArray[slice], posData, bonded14ParamArray[slice],
                                        forceData, includeEnergy ? &sliceEnergies[slice] : NULL, nonbonded14);
    }

    double energy = 0;
    for (int slice = 0; slice < numSlices; slice++)
        energy += sliceLambda[slice]*sliceEnergies[slice];
    return energy;
}

void ReferenceCalcSlicedPmeForceKernel::copyParametersToContext(ContextImpl& context, const SlicedPmeForce& force) {
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Get particle subsets and switching parameters.

    for (int i = 0; i < numParticles; i++)
        subsets[i] = force.getParticleSubset(i);

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int index = 0; index < force.getNumExceptionChargeOffsets(); index++) {
        string param;
        int exception;
        double charge;
        force.getExceptionChargeOffset(index, param, exception, charge);
        exceptionsWithOffsets.insert(exception);
    }
    for (int slice = 0; slice < numSlices; slice++)
        nb14s[slice].clear();
    for (int index = 0; index < force.getNumExceptions(); index++) {
        int particle1, particle2;
        double chargeProd;
        force.getExceptionParameters(index, particle1, particle2, chargeProd);
        if (chargeProd != 0.0 || exceptionsWithOffsets.find(index) != exceptionsWithOffsets.end()) {
            int s1 = subsets[particle1];
            int s2 = subsets[particle2];
            int slice = s1 > s2 ? s1*(s1+1)/2+s2 : s2*(s2+1)/2+s1;
            nb14s[slice].push_back(index);
        }
    }
    for (int slice = 0; slice < numSlices; slice++)
        num14[slice] = nb14s[slice].size();
    if (accumulate(num14.begin(), num14.end(), 0) != total14)
        throw OpenMMException("updateParametersInContext: The number of non-excluded exceptions has changed");

    // Record the values.

    for (int i = 0; i < numParticles; ++i)
        particleCharges[i] = force.getParticleCharge(i);

    for (int slice = 0; slice < numSlices; slice++) {
        int n = num14[slice];
        bonded14IndexArray[slice].resize(n, vector<int>(2));
        bonded14ParamArray[slice].resize(n, vector<double>(2));
        exceptionChargeProds[slice].resize(n);
        for (int i = 0; i < n; ++i)
            force.getExceptionParameters(
                nb14s[slice][i],
                bonded14IndexArray[slice][i][0],
                bonded14IndexArray[slice][i][1],
                exceptionChargeProds[slice][i]
            );
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

    for (int i = 0; i < numParticles; i++)
        particleParamArray[i] = particleCharges[i];
    for (auto& offset : particleParamOffsets) {
        double value = context.getParameter(offset.first.first);
        int index = offset.first.second;
        particleParamArray[index] += value*offset.second;
    }

    // Compute switching parameter values.

    for (int slice = 0; slice < numSlices; slice++) {
        string parameter = sliceSwitchingParameter[slice];
        sliceLambda[slice] = parameter == "" ? 1.0 : context.getParameter(parameter);
    }

    // Compute exception parameters.

    for (int slice = 0; slice < numSlices; slice++)
        for (int i = 0; i < num14[slice]; i++) {
            double chargeProd = exceptionChargeProds[slice][i];
            for (auto offset : exceptionParamOffsets[nb14s[slice][i]])
                chargeProd += context.getParameter(offset.first)*offset.second;
            bonded14ParamArray[slice][i][0] = chargeProd;
            bonded14ParamArray[slice][i][1] = sliceLambda[slice];
        }
}