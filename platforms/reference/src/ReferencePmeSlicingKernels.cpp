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
#include "SlicedNonbondedForce.h"
#include "internal/SlicedNonbondedForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/reference/ReferenceBondForce.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include <cstring>
#include <numeric>
#include <vector>
#include <algorithm>
#include <iostream>

#include "internal/ReferenceCoulombIxn.h"
#include "internal/ReferenceCoulomb14.h"
#include "internal/ReferenceSlicedLJCoulombIxn.h"
#include "internal/ReferenceSlicedLJCoulomb14.h"

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

static map<string, double>& extractEnergyParameterDerivatives(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->energyParameterDerivatives;
}

ReferenceCalcSlicedPmeForceKernel::~ReferenceCalcSlicedPmeForceKernel() {
    if (neighborList != NULL)
        delete neighborList;
}

void ReferenceCalcSlicedPmeForceKernel::initialize(const System& system, const SlicedPmeForce& force) {
    numParticles = force.getNumParticles();
    numSubsets = force.getNumSubsets();
    numSlices = numSubsets*(numSubsets+1)/2;
    sliceLambda.resize(numSlices);

    subsets.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        subsets[i] = force.getParticleSubset(i);

    vector<string> derivs;
    for (int index = 0; index < force.getNumSwitchingParameterDerivatives(); index++) {
        string parameter = force.getSwitchingParameterDerivativeName(index);
        derivs.push_back(parameter);
    }

    sliceSwitchParamIndices.resize(numSlices, -1);
    sliceSwitchParamDerivative.resize(numSlices, -1);
    for (int index = 0; index < force.getNumSwitchingParameters(); index++) {
        string parameter;
        int i, j;
        force.getSwitchingParameter(index, parameter, i, j);
        switchParamName.push_back(parameter);
        int slice = i > j ? i*(i+1)/2+j : j*(j+1)/2+i;
        sliceSwitchParamIndices[slice] = index;
        auto begin = derivs.begin();
        auto end = derivs.end();
        auto position = find(begin, end, parameter);
        if (position != end)
            sliceSwitchParamDerivative[slice] = position-begin;
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
                                        forceData, &sliceEnergies[slice], nonbonded14);
    }

    map<string, double>& energyParamDerivs = extractEnergyParameterDerivatives(context);
    for (int slice = 0; slice < numSlices; slice++) {
        int index = sliceSwitchParamDerivative[slice];
        if (index != -1)
            energyParamDerivs[switchParamName[index]] += sliceEnergies[slice];
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
        int index = sliceSwitchParamIndices[slice];
        sliceLambda[slice] = index == -1 ? 1.0 : context.getParameter(switchParamName[index]);
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

ReferenceCalcSlicedNonbondedForceKernel::~ReferenceCalcSlicedNonbondedForceKernel() {
    if (neighborList != NULL)
        delete neighborList;
}

void ReferenceCalcSlicedNonbondedForceKernel::initialize(const System& system, const SlicedNonbondedForce& force) {
    numParticles = force.getNumParticles();
    numSubsets = force.getNumSubsets();
    numSlices = numSubsets*(numSubsets+1)/2;
    sliceLambdas.resize(numSlices, (vector<double>){1, 1});
    sliceScalingParams.resize(numSlices, (vector<int>){-1, -1});
    sliceScalingParamDerivs.resize(numSlices, (vector<int>){-1, -1});

    subsets.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        subsets[i] = force.getParticleSubset(i);

    int numDerivs = force.getNumScalingParameterDerivatives();
    vector<string> derivs(numDerivs);
    for (int i = 0; i < numDerivs; i++)
        derivs[i] = force.getScalingParameterDerivativeName(i);

    int numScalingParams = force.getNumScalingParameters();
    scalingParams.resize(numScalingParams);
    for (int index = 0; index < numScalingParams; index++) {
        int i, j;
        bool includeLJ, includeCoulomb;
        force.getScalingParameter(index, scalingParams[index], i, j, includeLJ, includeCoulomb);
        int slice = i > j ? i*(i+1)/2+j : j*(j+1)/2+i;
        sliceScalingParams[slice] = {includeLJ ? index : -1, includeCoulomb ? index : -1};
        int pos = find(derivs.begin(), derivs.end(), scalingParams[index]) - derivs.begin();
        if (pos < numDerivs)
            sliceScalingParamDerivs[slice] = {includeLJ ? pos : -1, includeCoulomb ? pos : -1};
    }

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionsWithOffsets.insert(exception);
    }
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
    bonded14SliceArray.resize(num14);
    particleParamArray.resize(numParticles, vector<double>(4));
    baseParticleParams.resize(numParticles);
    baseExceptionParams.resize(num14);
    for (int i = 0; i < numParticles; ++i)
       force.getParticleParameters(i, baseParticleParams[i][0], baseParticleParams[i][1], baseParticleParams[i][2]);
    for (int i = 0; i < num14; ++i) {
        int particle1, particle2;
        force.getExceptionParameters(nb14s[i], particle1, particle2, baseExceptionParams[i][0], baseExceptionParams[i][1], baseExceptionParams[i][2]);
        bonded14IndexArray[i][0] = particle1;
        bonded14IndexArray[i][1] = particle2;
        int s1 = subsets[particle1];
        int s2 = subsets[particle2];
        bonded14SliceArray[i] = s1 > s2 ? s1*(s1+1)/2+s2 : s2*(s2+1)/2+s1;
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
    nonbondedMethod = CalcSlicedNonbondedForceKernel::NonbondedMethod(force.getNonbondedMethod());
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
        SlicedNonbondedForceImpl::calcEwaldParameters(system, force, alpha, kmax[0], kmax[1], kmax[2]);
        ewaldAlpha = alpha;
    }
    else if (nonbondedMethod == PME) {
        double alpha;
        SlicedNonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSize[0], gridSize[1], gridSize[2], false);
        ewaldAlpha = alpha;
    }
    else if (nonbondedMethod == LJPME) {
        double alpha;
        SlicedNonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSize[0], gridSize[1], gridSize[2], false);
        ewaldAlpha = alpha;
        SlicedNonbondedForceImpl::calcPMEParameters(system, force, alpha, dispersionGridSize[0], dispersionGridSize[1], dispersionGridSize[2], true);
        ewaldDispersionAlpha = alpha;
        useSwitchingFunction = false;
    }
    if (nonbondedMethod == NoCutoff || nonbondedMethod == CutoffNonPeriodic)
        exceptionsArePeriodic = false;
    else
        exceptionsArePeriodic = force.getExceptionsUsePeriodicBoundaryConditions();
    rfDielectric = force.getReactionFieldDielectric();
    if (force.getUseDispersionCorrection())
        dispersionCoefficient = SlicedNonbondedForceImpl::calcDispersionCorrection(system, force);
    else
        dispersionCoefficient = 0.0;
}

double ReferenceCalcSlicedNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    computeParameters(context);
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    ReferenceSlicedLJCoulombIxn clj;
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
    vector<vector<double>> sliceEnergies(numSlices, (vector<double>){0.0, 0.0});
    if (useSwitchingFunction)
        clj.setUseSwitchingFunction(switchingDistance);
    clj.calculatePairIxn(numParticles, posData, numSubsets, subsets, particleParamArray, sliceLambdas, exclusions, forceData, sliceEnergies, includeDirect, includeReciprocal);

    if (includeDirect) {
        ReferenceSlicedLJCoulomb14 nonbonded14;
        if (exceptionsArePeriodic) {
            Vec3* boxVectors = extractBoxVectors(context);
            nonbonded14.setPeriodic(boxVectors);
        }
        for (int k = 0; k < num14; k++) {
            int slice = bonded14SliceArray[k];
            nonbonded14.calculateBondIxn(bonded14IndexArray[k], posData, bonded14ParamArray[k],
                                         forceData, sliceLambdas[slice], sliceEnergies[slice]);
        }
        if (periodic || ewald || pme) {
            Vec3* boxVectors = extractBoxVectors(context);
            sliceEnergies[0][0] += dispersionCoefficient/(boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2]);
        }
    }

    double energy = 0;
    if (includeEnergy)
        for (int slice = 0; slice < numSlices; slice++)
            for (int term = 0; term < 2; term++)
                energy += sliceLambdas[slice][term]*sliceEnergies[slice][term];

    map<string, double>& energyParamDerivs = extractEnergyParameterDerivatives(context);
    for (int slice = 0; slice < numSlices; slice++)
        for (int term = 0; term < 2; term++) {
            int index = sliceScalingParamDerivs[slice][term];
            if (index != -1)
                energyParamDerivs[scalingParams[index]] += sliceEnergies[slice][term];
        }

    return energy;
}

void ReferenceCalcSlicedNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const SlicedNonbondedForce& force) {
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Get particle subsets.

    for (int i = 0; i < numParticles; i++)
        subsets[i] = force.getParticleSubset(i);

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
        int s1 = subsets[particle1];
        int s2 = subsets[particle2];
        bonded14SliceArray[i] = s1 > s2 ? s1*(s1+1)/2+s2 : s2*(s2+1)/2+s1;
    }

    // Recompute the coefficient for the dispersion correction.

    SlicedNonbondedForce::NonbondedMethod method = force.getNonbondedMethod();
    if (force.getUseDispersionCorrection() && (method == SlicedNonbondedForce::CutoffPeriodic || method == SlicedNonbondedForce::Ewald || method == SlicedNonbondedForce::PME))
        dispersionCoefficient = SlicedNonbondedForceImpl::calcDispersionCorrection(context.getSystem(), force);
}

void ReferenceCalcSlicedNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != PME && nonbondedMethod != LJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME or LJPME");
    alpha = ewaldAlpha;
    nx = gridSize[0];
    ny = gridSize[1];
    nz = gridSize[2];
}

void ReferenceCalcSlicedNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != LJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using LJPME");
    alpha = ewaldDispersionAlpha;
    nx = dispersionGridSize[0];
    ny = dispersionGridSize[1];
    nz = dispersionGridSize[2];
}

void ReferenceCalcSlicedNonbondedForceKernel::computeParameters(ContextImpl& context) {

    // Compute scaling parameter values.

    for (int slice = 0; slice < numSlices; slice++) {
        vector<int> indices = sliceScalingParams[slice];
        int index = max(indices[0], indices[1]);
        if (index != -1) {
            double paramValue = context.getParameter(scalingParams[index]);
            for (int i = 0; i < 2; i++)
                sliceLambdas[slice][i] = indices[i] == -1 ? 1.0 : paramValue;
        }
    }

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