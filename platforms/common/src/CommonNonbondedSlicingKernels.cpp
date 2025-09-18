/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2025 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "CommonNonbondedSlicingKernels.h"
#include "CommonNonbondedSlicingKernelSources.h"
#include "SlicedNonbondedForce.h"
#include "internal/SlicedNonbondedForceImpl.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/Context.h"
#include "openmm/common/BondedUtilities.h"
#include "openmm/common/CommonKernelUtilities.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/NonbondedUtilities.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include <assert.h>
#include <cmath>

#include <iostream>


using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

class CommonCalcSlicedNonbondedForceKernel::ForceInfo : public ComputeForceInfo {
public:
    ForceInfo(const SlicedNonbondedForce& force) : force(force) {
        particleOffset.resize(force.getNumParticles(), -1);
        exceptionOffset.resize(force.getNumExceptions(), -1);
        for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
            string parameter;
            int particleIndex;
            double chargeScale, sigmaScale, epsilonScale;
            force.getParticleParameterOffset(i, parameter, particleIndex, chargeScale, sigmaScale, epsilonScale);
            particleOffset[particleIndex] = i;
        }
        for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
            string parameter;
            int exceptionIndex;
            double chargeProdScale, sigmaScale, epsilonScale;
            force.getExceptionParameterOffset(i, parameter, exceptionIndex, chargeProdScale, sigmaScale, epsilonScale);
            exceptionOffset[exceptionIndex] = i;
        }
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        if (particleOffset[particle1] != -1 || particleOffset[particle2] != -1) {
            if (particleOffset[particle1] == -1 || particleOffset[particle2] == -1)
                return false;
            string parameter1, parameter2;
            int particleIndex1, particleIndex2;
            double chargeScale1, sigmaScale1, epsilonScale1, chargeScale2, sigmaScale2, epsilonScale2;
            force.getParticleParameterOffset(particleOffset[particle1], parameter1, particleIndex1, chargeScale1, sigmaScale1, epsilonScale1);
            force.getParticleParameterOffset(particleOffset[particle2], parameter2, particleIndex2, chargeScale2, sigmaScale2, epsilonScale2);
            if (parameter1 != parameter2 || chargeScale1 != chargeScale2 || sigmaScale1 != sigmaScale2 || epsilonScale1 != epsilonScale2)
                return false;
        }
        double charge1, charge2, sigma1, sigma2, epsilon1, epsilon2;
        force.getParticleParameters(particle1, charge1, sigma1, epsilon1);
        force.getParticleParameters(particle2, charge2, sigma2, epsilon2);
        int subset1 = force.getParticleSubset(particle1);
        int subset2 = force.getParticleSubset(particle2);
        return (charge1 == charge2 && sigma1 == sigma2 && epsilon1 == epsilon2 && subset1 == subset2);
    }
    int getNumParticleGroups() {
        return force.getNumExceptions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(index, particle1, particle2, chargeProd, sigma, epsilon);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        if (exceptionOffset[group1] != -1 || exceptionOffset[group2] != -1) {
            if (exceptionOffset[group1] == -1 || exceptionOffset[group2] == -1)
                return false;
            string parameter1, parameter2;
            int exceptionIndex1, exceptionIndex2;
            double chargeProdScale1, sigmaScale1, epsilonScale1, chargeProdScale2, sigmaScale2, epsilonScale2;
            force.getExceptionParameterOffset(exceptionOffset[group1], parameter1, exceptionIndex1, chargeProdScale1, sigmaScale1, epsilonScale1);
            force.getExceptionParameterOffset(exceptionOffset[group2], parameter2, exceptionIndex2, chargeProdScale2, sigmaScale2, epsilonScale2);
            if (parameter1 != parameter2 || chargeProdScale1 != chargeProdScale2 || sigmaScale1 != sigmaScale2 || epsilonScale1 != epsilonScale2)
                return false;
        }
        int particle1, particle2, subset1, subset2, slice1, slice2;
        double chargeProd1, chargeProd2, sigma1, sigma2, epsilon1, epsilon2;
        force.getExceptionParameters(group1, particle1, particle2, chargeProd1, sigma1, epsilon1);
        subset1 = force.getParticleSubset(particle1);
        subset2 = force.getParticleSubset(particle2);
        slice1 = sliceIndex(subset1, subset2);
        force.getExceptionParameters(group2, particle1, particle2, chargeProd2, sigma2, epsilon2);
        subset1 = force.getParticleSubset(particle1);
        subset2 = force.getParticleSubset(particle2);
        slice2 = sliceIndex(subset1, subset2);
        return (chargeProd1 == chargeProd2 && sigma1 == sigma2 && epsilon1 == epsilon2 && slice1 == slice2);
    }
private:
    const SlicedNonbondedForce& force;
    vector<int> particleOffset, exceptionOffset;
};

class CommonCalcSlicedNonbondedForceKernel::PmeIO : public CalcPmeReciprocalForceKernel::IO {
public:
    PmeIO(ComputeContext& cc, ComputeKernel addForcesKernel) : cc(cc), addForcesKernel(addForcesKernel) {
        forceTemp.initialize<mm_float4>(cc, cc.getNumAtoms(), "PmeForce");
        addForcesKernel->addArg(forceTemp);
        addForcesKernel->addArg();
    }
    float* getPosq() {
        ContextSelector selector(cc);
        cc.getPosq().download(posq);
        return (float*) &posq[0];
    }
    void setForce(float* force) {
        forceTemp.upload(force);
        addForcesKernel->setArg(1, cc.getLongForceBuffer());
        addForcesKernel->execute(cc.getNumAtoms());
    }
private:
    ComputeContext& cc;
    vector<mm_float4> posq;
    ComputeArray forceTemp;
    ComputeKernel addForcesKernel;
};

// class CommonCalcSlicedNonbondedForceKernel::PmePreComputation : public ComputeContext::ForcePreComputation {
// public:
//     PmePreComputation(ComputeContext& cc, Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : cc(cc), pme(pme), io(io) {
//     }
//     void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
//         Vec3 boxVectors[3];
//         cc.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
//         pme.getAs<CalcPmeReciprocalForceKernel>().beginComputation(io, boxVectors, includeEnergy);
//     }
// private:
//     ComputeContext& cc;
//     Kernel pme;
//     CalcPmeReciprocalForceKernel::IO& io;
// };

// class CommonCalcSlicedNonbondedForceKernel::PmePostComputation : public ComputeContext::ForcePostComputation {
// public:
//     PmePostComputation(Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : pme(pme), io(io) {
//     }
//     double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
//         return pme.getAs<CalcPmeReciprocalForceKernel>().finishComputation(io);
//     }
// private:
//     Kernel pme;
//     CalcPmeReciprocalForceKernel::IO& io;
// };

class CommonCalcSlicedNonbondedForceKernel::SyncQueuePreComputation : public ComputeContext::ForcePreComputation {
public:
    SyncQueuePreComputation(ComputeContext& cc, ComputeQueue queue, ComputeEvent event, int forceGroup) : cc(cc), queue(queue), event(event), forceGroup(forceGroup) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            event->enqueue();
            event->queueWait(queue);
        }
    }
private:
    ComputeContext& cc;
    ComputeQueue queue;
    ComputeEvent event;
    int forceGroup;
};

class CommonCalcSlicedNonbondedForceKernel::SyncQueuePostComputation : public ComputeContext::ForcePostComputation {
public:
    SyncQueuePostComputation(ComputeContext& cc, ComputeEvent event, ComputeArray& pmeEnergyBuffer, int forceGroup) : cc(cc), event(event),
            pmeEnergyBuffer(pmeEnergyBuffer), forceGroup(forceGroup) {
        bufferSize = pmeEnergyBuffer.getSize();
        hasDerivatives = false;
    }
    void setKernel(ComputeKernel kernel) {
        addEnergyKernel = kernel;
        addEnergyKernel->addArg(pmeEnergyBuffer);
        addEnergyKernel->addArg(cc.getEnergyBuffer());
        addEnergyKernel->addArg(bufferSize);
    }
    void addDerivatives(ComputeArray& pmeEnergyParamDerivBuffer, int numParamDerivs) {
        addEnergyKernel->addArg(pmeEnergyParamDerivBuffer);
        addEnergyKernel->addArg(cc.getEnergyParamDerivBuffer());
        addEnergyKernel->addArg(numParamDerivs);
        hasDerivatives = true;
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            event->wait();
            if (includeEnergy || hasDerivatives)
                addEnergyKernel->execute(pmeEnergyBuffer.getSize());
        }
        return 0.0;
    }
private:
    ComputeContext& cc;
    ComputeEvent event;
    ComputeKernel addEnergyKernel;
    ComputeArray& pmeEnergyBuffer;
    int forceGroup;
    int bufferSize;
    bool hasDerivatives;
};


class CommonCalcSlicedNonbondedForceKernel::DispersionCorrectionPostComputation : public ComputeContext::ForcePostComputation {
public:
    DispersionCorrectionPostComputation(ComputeContext& cc, vector<double>& coefficients, vector<mm_double2>& sliceLambdas, vector<ScalingParameterInfo>& sliceScalingParams, int forceGroup) :
                                        cc(cc), coefficients(coefficients), sliceLambdas(sliceLambdas), sliceScalingParams(sliceScalingParams), forceGroup(forceGroup) {
        numSlices = coefficients.size();
        hasDerivatives = false;
        for (auto info : sliceScalingParams)
            hasDerivatives = hasDerivatives || info.hasDerivativeLJ;
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        double energy = 0.0;
        if ((includeEnergy || hasDerivatives) && (groups&(1<<forceGroup)) != 0) {
            Vec3 a, b, c;
            cc.getPeriodicBoxVectors(a, b, c);
            double volume = a[0]*b[1]*c[2];
            if (includeEnergy)
                for (int slice = 0; slice < numSlices; slice++)
                    energy += sliceLambdas[slice].y*coefficients[slice]/volume;
            if (hasDerivatives) {
                map<string, double>& energyParamDerivs = cc.getEnergyParamDerivWorkspace();
                for (int slice = 0; slice < numSlices; slice++) {
                    ScalingParameterInfo info = sliceScalingParams[slice];
                    if (info.hasDerivativeLJ)
                        energyParamDerivs[info.nameLJ] += coefficients[slice]/volume;
                }
            }
        }
        return energy;
    }
private:
    ComputeContext& cc;
    vector<double>& coefficients;
    vector<mm_double2>& sliceLambdas;
    vector<ScalingParameterInfo>& sliceScalingParams;
    int forceGroup;
    int numSlices;
    bool hasDerivatives;
};

CommonCalcSlicedNonbondedForceKernel::~CommonCalcSlicedNonbondedForceKernel() {
    ContextSelector selector(cc);
    if (pmeio != NULL)
        delete pmeio;
}

string CommonCalcSlicedNonbondedForceKernel::getDerivativeExpression(string param, bool conditionCoulomb, bool conditionLJ) {
    stringstream exprCoulomb, exprLJ, exprBoth;
    int countCoulomb = 0, countLJ = 0, countBoth = 0;
    for (int slice = 0; slice < numSlices; slice++) {
        ScalingParameterInfo info = sliceScalingParams[slice];
        bool coulomb = conditionCoulomb && info.nameCoulomb == param;
        bool lj = conditionLJ && info.nameLJ == param;
        if (coulomb && lj)
            exprBoth<<(countBoth++ ? " || " : "")<<"slice=="<<slice;
        else if (coulomb)
            exprCoulomb<<(countCoulomb++ ? " || " : "")<<"slice=="<<slice;
        else if (lj)
            exprLJ<<(countLJ++ ? " || " : "")<<"slice=="<<slice;
    }

    stringstream derivative;
    if (countBoth)
        derivative<<"("<<exprBoth.str()<<")*(clEnergy + ljEnergy)";
    if (countCoulomb)
        derivative<<(countBoth ? " + " : "")<<"("<<exprCoulomb.str()<<")*clEnergy";
    if (countLJ)
        derivative<<(countBoth+countCoulomb ? " + " : "")<<"("<<exprLJ.str()<<")*ljEnergy";

    return derivative.str();
}

void CommonCalcSlicedNonbondedForceKernel::commonInitialize(const System& system, const SlicedNonbondedForce& force, FFT3DFactory& fftFactory, bool usePmeQueue,
        bool deviceIsCpu, bool useFixedPointChargeSpreading, bool useCpuPme) {
    this->usePmeQueue = false;
    this->deviceIsCpu = deviceIsCpu;
    this->useFixedPointChargeSpreading = useFixedPointChargeSpreading;
    this->useCpuPme = useCpuPme;
    ContextSelector selector(cc);
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "slicedNonbonded"+cc.intToString(forceIndex)+"_";

    int numParticles = force.getNumParticles();
    numSubsets = force.getNumSubsets();
    numSlices = force.getNumSlices();
    sliceLambdasVec.resize(numSlices, mm_double2(1, 1));
    subsetSelfEnergy.resize(numSubsets, mm_double2(0, 0));
    sliceBackgroundEnergyVolume.resize(numSlices, 0.0);
    sliceScalingParams.resize(numSlices, ScalingParameterInfo());

    subsetsVec.resize(cc.getPaddedNumAtoms(), 0);
    for (int i = 0; i < numParticles; i++)
        subsetsVec[i] = force.getParticleSubset(i);
    subsets.initialize<int>(cc, cc.getPaddedNumAtoms(), "subsets");
    subsets.upload(subsetsVec);

    int numDerivs = force.getNumScalingParameterDerivatives();
    hasDerivatives = numDerivs > 0;
    for (int i = 0; i < numDerivs; i++)
        requestedDerivatives.insert(force.getScalingParameterDerivativeName(i));

    for (int index = 0; index < force.getNumScalingParameters(); index++) {
        string name;
        int subset1, subset2;
        bool includeCoulomb, includeLJ;
        force.getScalingParameter(index, name, subset1, subset2, includeCoulomb, includeLJ);
        bool hasDerivative = requestedDerivatives.find(name) != requestedDerivatives.end();
        sliceScalingParams[sliceIndex(subset1, subset2)].addInfo(name, includeCoulomb, includeLJ, hasDerivative);
    }

    size_t sizeOfReal = cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float);
    sliceLambdas.initialize(cc, numSlices, 2*sizeOfReal, "sliceLambdas");
    if (cc.getUseDoublePrecision())
        sliceLambdas.upload(sliceLambdasVec);
    else {
        vector<mm_float2> sliceLambdasVecFloat(numSlices);
        for (size_t i = 0; i < numSlices; i++)
            sliceLambdasVecFloat[i] = mm_float2(sliceLambdasVec[i].x, sliceLambdasVec[i].y);
        sliceLambdas.upload(sliceLambdasVecFloat);
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
    vector<pair<int, int> > exclusions;
    vector<int> exceptions;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        exclusions.push_back(pair<int, int>(particle1, particle2));
        if (chargeProd != 0.0 || epsilon != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end()) {
            exceptionIndex[i] = exceptions.size();
            exceptions.push_back(i);
        }
    }

    // Initialize nonbonded interactions.

    vector<mm_float4> baseParticleParamVec(cc.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
    vector<vector<int> > exclusionList(numParticles);
    hasCoulomb = false;
    hasLJ = false;
    for (int i = 0; i < numParticles; i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        baseParticleParamVec[i] = mm_float4(charge, sigma, epsilon, 0);
        exclusionList[i].push_back(i);
        if (charge != 0.0)
            hasCoulomb = true;
        if (epsilon != 0.0)
            hasLJ = true;
    }
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        if (charge != 0.0)
            hasCoulomb = true;
        if (epsilon != 0.0)
            hasLJ = true;
    }
    for (auto exclusion : exclusions) {
        exclusionList[exclusion.first].push_back(exclusion.second);
        exclusionList[exclusion.second].push_back(exclusion.first);
    }
    nonbondedMethod = CalcSlicedNonbondedForceKernel::NonbondedMethod(force.getNonbondedMethod());
    bool useCutoff = (nonbondedMethod != NoCutoff);
    bool usePeriodic = (nonbondedMethod != NoCutoff && nonbondedMethod != CutoffNonPeriodic);
    doLJPME = (nonbondedMethod == LJPME && hasLJ);
    hasReciprocal = (nonbondedMethod == Ewald) || ((nonbondedMethod == PME || nonbondedMethod == LJPME) && hasCoulomb) || doLJPME;
    usePosqCharges = hasCoulomb ? cc.requestPosqCharges() : false;
    map<string, string> defines;
    defines["HAS_COULOMB"] = (hasCoulomb ? "1" : "0");
    defines["HAS_LENNARD_JONES"] = (hasLJ ? "1" : "0");
    defines["USE_LJ_SWITCH"] = (useCutoff && force.getUseSwitchingFunction() ? "1" : "0");
    if (useCutoff) {
        // Compute the reaction field constants.

        double reactionFieldK = pow(force.getCutoffDistance(), -3.0)*(force.getReactionFieldDielectric()-1.0)/(2.0*force.getReactionFieldDielectric()+1.0);
        double reactionFieldC = (1.0 / force.getCutoffDistance())*(3.0*force.getReactionFieldDielectric())/(2.0*force.getReactionFieldDielectric()+1.0);
        defines["REACTION_FIELD_K"] = cc.doubleToString(reactionFieldK);
        defines["REACTION_FIELD_C"] = cc.doubleToString(reactionFieldC);

        // Compute the switching coefficients.

        if (force.getUseSwitchingFunction()) {
            defines["LJ_SWITCH_CUTOFF"] = cc.doubleToString(force.getSwitchingDistance());
            defines["LJ_SWITCH_C3"] = cc.doubleToString(10/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 3.0));
            defines["LJ_SWITCH_C4"] = cc.doubleToString(15/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 4.0));
            defines["LJ_SWITCH_C5"] = cc.doubleToString(6/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 5.0));
        }
    }
    // if (force.getUseDispersionCorrection() && cc.getContextIndex() == 0 && !doLJPME)
    //     dispersionCoefficient = SlicedNonbondedForceImpl::calcDispersionCorrection(system, force);
    // else
    //     dispersionCoefficient = 0.0;
    if (force.getUseDispersionCorrection() && cc.getContextIndex() == 0 && hasLJ && useCutoff && usePeriodic && !doLJPME)
        dispersionCoefficients = SlicedNonbondedForceImpl::calcDispersionCorrections(system, force);
    alpha = 0;
    ewaldSelfEnergy = 0.0;
    backgroundEnergyVolume = 0.0;
    // totalCharge = 0.0;
    vector<double> subsetCharges(numSubsets, 0.0);
    map<string, string> paramsDefines;
    paramsDefines["NUM_SUBSETS"] = cc.intToString(numSubsets);
    paramsDefines["ONE_4PI_EPS0"] = cc.doubleToString(ONE_4PI_EPS0);
    paramsDefines["EPSILON0"] = cc.doubleToString(EPSILON0);
    paramsDefines["WORK_GROUP_SIZE"] = cc.intToString(cc.ThreadBlockSize);
    hasOffsets = (force.getNumParticleParameterOffsets() > 0 || force.getNumExceptionParameterOffsets() > 0);
    if (hasOffsets)
        paramsDefines["HAS_OFFSETS"] = "1";
    if (force.getNumParticleParameterOffsets() > 0)
        paramsDefines["HAS_PARTICLE_OFFSETS"] = "1";
    if (force.getNumExceptionParameterOffsets() > 0)
        paramsDefines["HAS_EXCEPTION_OFFSETS"] = "1";
    if (usePosqCharges)
        paramsDefines["USE_POSQ_CHARGES"] = "1";
    if (doLJPME)
        paramsDefines["INCLUDE_LJPME_EXCEPTIONS"] = "1";
    if (nonbondedMethod == Ewald) {
        // Compute the Ewald parameters.

        int kmaxx, kmaxy, kmaxz;
        SlicedNonbondedForceImpl::calcEwaldParameters(system, force, alpha, kmaxx, kmaxy, kmaxz);
        defines["EWALD_ALPHA"] = cc.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cc.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        if (cc.getContextIndex() == 0) {
            paramsDefines["INCLUDE_EWALD"] = "1";
            paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cc.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
            for (int i = 0; i < numParticles; i++) {
                int subset = subsetsVec[i];
                double charge = baseParticleParamVec[i].x;
                subsetSelfEnergy[subset].x -= charge*charge*ONE_4PI_EPS0*alpha/sqrt(M_PI);
                subsetCharges[subset] += charge;
                // totalCharge += charge;
            }
            for (int i = 0; i < numSubsets; i++) {
                int slice = sliceIndex(i, i);
                ewaldSelfEnergy += sliceLambdasVec[slice].x*subsetSelfEnergy[i].x;
                double factor = -subsetCharges[i]/(8*EPSILON0*alpha*alpha);
                sliceBackgroundEnergyVolume[slice] += subsetCharges[i]*factor;
                for (int j = i + 1; j < numSubsets; j++)
                    sliceBackgroundEnergyVolume[sliceIndex(i, j)] += 2.0*subsetCharges[j]*factor;
            }

            // Prepare the reciprocal space kernels.

            ewaldDefines["NUM_ATOMS"] = cc.intToString(numParticles);
            ewaldDefines["NUM_SUBSETS"] = cc.intToString(numSubsets);
            ewaldDefines["NUM_SLICES"] = cc.intToString(numSlices);
            ewaldDefines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
            ewaldDefines["KMAX_X"] = cc.intToString(kmaxx);
            ewaldDefines["KMAX_Y"] = cc.intToString(kmaxy);
            ewaldDefines["KMAX_Z"] = cc.intToString(kmaxz);
            ewaldDefines["EXP_COEFFICIENT"] = cc.doubleToString(-1.0/(4.0*alpha*alpha));
            ewaldDefines["ONE_4PI_EPS0"] = cc.doubleToString(ONE_4PI_EPS0);
            ewaldDefines["M_PI"] = cc.doubleToString(M_PI);
            ewaldDefines["HAS_DERIVATIVES"] = hasDerivatives ? "1" : "0";
            int elementSize = (cc.getUseDoublePrecision() ? sizeof(mm_double2) : sizeof(mm_float2));
            cosSinSums.initialize(cc, numSubsets*(2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1), elementSize, "cosSinSums");
        }
    }
    else if (hasReciprocal) {
        // Compute the PME parameters.

        SlicedNonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSizeX, gridSizeY, gridSizeZ, false);
        gridSizeX = cc.findLegalFFTDimension(gridSizeX);
        gridSizeY = cc.findLegalFFTDimension(gridSizeY);
        gridSizeZ = cc.findLegalFFTDimension(gridSizeZ);
        if (doLJPME) {
            SlicedNonbondedForceImpl::calcPMEParameters(system, force, dispersionAlpha, dispersionGridSizeX,
                                                  dispersionGridSizeY, dispersionGridSizeZ, true);
            dispersionGridSizeX = cc.findLegalFFTDimension(dispersionGridSizeX);
            dispersionGridSizeY = cc.findLegalFFTDimension(dispersionGridSizeY);
            dispersionGridSizeZ = cc.findLegalFFTDimension(dispersionGridSizeZ);
        }
        defines["EWALD_ALPHA"] = cc.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cc.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        defines["DO_LJPME"] = doLJPME ? "1" : "0";
        if (doLJPME) {
            defines["EWALD_DISPERSION_ALPHA"] = cc.doubleToString(dispersionAlpha);
            double invRCut6 = pow(force.getCutoffDistance(), -6);
            double dalphaR = dispersionAlpha * force.getCutoffDistance();
            double dar2 = dalphaR*dalphaR;
            double dar4 = dar2*dar2;
            double multShift6 = -invRCut6*(1.0 - exp(-dar2) * (1.0 + dar2 + 0.5*dar4));
            defines["INVCUT6"] = cc.doubleToString(invRCut6);
            defines["MULTSHIFT6"] = cc.doubleToString(multShift6);
        }
        if (cc.getContextIndex() == 0) {
            paramsDefines["INCLUDE_EWALD"] = "1";
            paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cc.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
            for (int i = 0; i < numParticles; i++) {
                int subset = subsetsVec[i];
                double charge = baseParticleParamVec[i].x;
                subsetSelfEnergy[subset].x -= charge*charge*ONE_4PI_EPS0*alpha/sqrt(M_PI);
                if (doLJPME)
                    subsetSelfEnergy[subset].y += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
                subsetCharges[subset] += charge;
                // totalCharge += charge;
            }
            for (int i = 0; i < numSubsets; i++) {
                int slice = sliceIndex(i, i);
                ewaldSelfEnergy += sliceLambdasVec[slice].x*subsetSelfEnergy[i].x;
                if (doLJPME)
                    ewaldSelfEnergy += sliceLambdasVec[slice].y*subsetSelfEnergy[i].y;
                double factor = -subsetCharges[i]/(8*EPSILON0*alpha*alpha);
                sliceBackgroundEnergyVolume[slice] += subsetCharges[i]*factor;
                for (int j = i + 1; j < numSubsets; j++)
                    sliceBackgroundEnergyVolume[sliceIndex(i, j)] += 2.0*subsetCharges[j]*factor;
            }
            // for (int i = 0; i < numParticles; i++) {
            //     ewaldSelfEnergy -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
            //     totalCharge += baseParticleParamVec[i].x;
            // }
            // if (doLJPME) {
            //     paramsDefines["INCLUDE_LJPME"] = "1";
            //     paramsDefines["LJPME_SELF_ENERGY_SCALE"] = cc.doubleToString(pow(dispersionAlpha, 6)/3.0);
            //     for (int i = 0; i < numParticles; i++)
            //         ewaldSelfEnergy += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
            // }
            if (doLJPME) {
                paramsDefines["INCLUDE_LJPME"] = "1";
                paramsDefines["LJPME_SELF_ENERGY_SCALE"] = cc.doubleToString(pow(dispersionAlpha, 6)/3.0);
                // for (int i = 0; i < numParticles; i++)
                //     ewaldSelfEnergy += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
            }
            pmeDefines["PME_ORDER"] = cc.intToString(PmeOrder);
            pmeDefines["NUM_ATOMS"] = cc.intToString(numParticles);
            pmeDefines["NUM_SUBSETS"] = cc.intToString(numSubsets);
            pmeDefines["NUM_SLICES"] = cc.intToString(numSlices);
            pmeDefines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());
            pmeDefines["RECIP_EXP_FACTOR"] = cc.doubleToString(M_PI*M_PI/(alpha*alpha));
            pmeDefines["GRID_SIZE_X"] = cc.intToString(gridSizeX);
            pmeDefines["GRID_SIZE_Y"] = cc.intToString(gridSizeY);
            pmeDefines["GRID_SIZE_Z"] = cc.intToString(gridSizeZ);
            pmeDefines["EPSILON_FACTOR"] = cc.doubleToString(sqrt(ONE_4PI_EPS0));
            pmeDefines["M_PI"] = cc.doubleToString(M_PI);
            pmeDefines["HAS_DERIVATIVES"] = hasDerivatives ? "1" : "0";
            if (useFixedPointChargeSpreading)
                pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";
            if (deviceIsCpu)
                pmeDefines["DEVICE_IS_CPU"] = "1";
            // if (useCpuPme && !doLJPME && usePosqCharges) {
            //     // Create the CPU PME kernel.

            //     try {
            //         cpuPme = getPlatform().createKernel(CalcPmeReciprocalForceKernel::Name(), cc.getContextImpl());
            //         cpuPme.getAs<CalcPmeReciprocalForceKernel>().initialize(gridSizeX, gridSizeY, gridSizeZ, numParticles, alpha, false);
            //         ComputeProgram program = cc.compileProgram(CommonNonbondedSlicingKernelSources::pme, pmeDefines);
            //         ComputeKernel addForcesKernel = program->createKernel("addForces");
            //         pmeio = new PmeIO(cc, addForcesKernel);
            //         cc.addPreComputation(new PmePreComputation(cc, cpuPme, *pmeio));
            //         cc.addPostComputation(new PmePostComputation(cpuPme, *pmeio));
            //     }
            //     catch (OpenMMException& ex) {
            //         // The CPU PME plugin isn't available.
            //     }
            // }
            if (pmeio == NULL) {
                // Create required data structures.

                int elementSize = (cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
                int gridElements = numSubsets*gridSizeX*gridSizeY*gridSizeZ;
                if (doLJPME) {
                    gridElements = max(gridElements, numSubsets*dispersionGridSizeX*dispersionGridSizeY*dispersionGridSizeZ);
                }
                pmeGrid1.initialize(cc, gridElements, 2*elementSize, "pmeGrid1");
                pmeGrid2.initialize(cc, gridElements, 2*elementSize, "pmeGrid2");
                if (useFixedPointChargeSpreading)
                    cc.addAutoclearBuffer(pmeGrid2);
                else
                    cc.addAutoclearBuffer(pmeGrid1);
                pmeBsplineModuliX.initialize(cc, gridSizeX, elementSize, "pmeBsplineModuliX");
                pmeBsplineModuliY.initialize(cc, gridSizeY, elementSize, "pmeBsplineModuliY");
                pmeBsplineModuliZ.initialize(cc, gridSizeZ, elementSize, "pmeBsplineModuliZ");
                if (doLJPME) {
                    pmeDispersionBsplineModuliX.initialize(cc, dispersionGridSizeX, elementSize, "pmeDispersionBsplineModuliX");
                    pmeDispersionBsplineModuliY.initialize(cc, dispersionGridSizeY, elementSize, "pmeDispersionBsplineModuliY");
                    pmeDispersionBsplineModuliZ.initialize(cc, dispersionGridSizeZ, elementSize, "pmeDispersionBsplineModuliZ");
                }
                pmeAtomGridIndex.initialize<mm_int2>(cc, numParticles, "pmeAtomGridIndex");
                int energyElementSize = (cc.getUseDoublePrecision() || cc.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
                pmeEnergyBuffer.initialize(cc, cc.getNumThreadBlocks()*ComputeContext::ThreadBlockSize, energyElementSize, "pmeEnergyBuffer");
                cc.clearBuffer(pmeEnergyBuffer);
                sort = cc.createSort(new SortTrait(), cc.getNumAtoms());
                fft = fftFactory.createFFT3D(cc, gridSizeX, gridSizeY, gridSizeZ, numSubsets, true);
                if (doLJPME)
                    dispersionFft = fftFactory.createFFT3D(cc, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, numSubsets, true);
                this->usePmeQueue = usePmeQueue;
                if (usePmeQueue) {
                    pmeDefines["USE_PME_STREAM"] = "1";
                    pmeQueue = cc.createQueue();
                    int recipForceGroup = force.getReciprocalSpaceForceGroup();
                    if (recipForceGroup < 0)
                        recipForceGroup = force.getForceGroup();
                    pmeSyncEvent = cc.createEvent();
                    paramsSyncEvent = cc.createEvent();
                    cc.addPreComputation(new SyncQueuePreComputation(cc, pmeQueue, pmeSyncEvent, recipForceGroup));
                    cc.addPostComputation(syncQueue = new SyncQueuePostComputation(cc, pmeSyncEvent, pmeEnergyBuffer, recipForceGroup));
                }

                // Initialize the b-spline moduli.

                for (int grid = 0; grid < 2; grid++) {
                    int xsize, ysize, zsize;
                    ComputeArray *xmoduli, *ymoduli, *zmoduli;
                    if (grid == 0) {
                        xsize = gridSizeX;
                        ysize = gridSizeY;
                        zsize = gridSizeZ;
                        xmoduli = &pmeBsplineModuliX;
                        ymoduli = &pmeBsplineModuliY;
                        zmoduli = &pmeBsplineModuliZ;
                    }
                    else {
                        if (!doLJPME)
                            continue;
                        xsize = dispersionGridSizeX;
                        ysize = dispersionGridSizeY;
                        zsize = dispersionGridSizeZ;
                        xmoduli = &pmeDispersionBsplineModuliX;
                        ymoduli = &pmeDispersionBsplineModuliY;
                        zmoduli = &pmeDispersionBsplineModuliZ;
                    }
                    int maxSize = max(max(xsize, ysize), zsize);
                    vector<double> data(PmeOrder);
                    vector<double> ddata(PmeOrder);
                    vector<double> bsplines_data(maxSize);
                    data[PmeOrder-1] = 0.0;
                    data[1] = 0.0;
                    data[0] = 1.0;
                    for (int i = 3; i < PmeOrder; i++) {
                        double div = 1.0/(i-1.0);
                        data[i-1] = 0.0;
                        for (int j = 1; j < (i-1); j++)
                            data[i-j-1] = div*(j*data[i-j-2]+(i-j)*data[i-j-1]);
                        data[0] = div*data[0];
                    }

                    // Differentiate.

                    ddata[0] = -data[0];
                    for (int i = 1; i < PmeOrder; i++)
                        ddata[i] = data[i-1]-data[i];
                    double div = 1.0/(PmeOrder-1);
                    data[PmeOrder-1] = 0.0;
                    for (int i = 1; i < (PmeOrder-1); i++)
                        data[PmeOrder-i-1] = div*(i*data[PmeOrder-i-2]+(PmeOrder-i)*data[PmeOrder-i-1]);
                    data[0] = div*data[0];
                    for (int i = 0; i < maxSize; i++)
                        bsplines_data[i] = 0.0;
                    for (int i = 1; i <= PmeOrder; i++)
                        bsplines_data[i] = data[i-1];

                    // Evaluate the actual bspline moduli for X/Y/Z.

                    for (int dim = 0; dim < 3; dim++) {
                        int ndata = (dim == 0 ? xsize : dim == 1 ? ysize : zsize);
                        vector<double> moduli(ndata);
                        for (int i = 0; i < ndata; i++) {
                            double sc = 0.0;
                            double ss = 0.0;
                            for (int j = 0; j < ndata; j++) {
                                double arg = (2.0*M_PI*i*j)/ndata;
                                sc += bsplines_data[j]*cos(arg);
                                ss += bsplines_data[j]*sin(arg);
                            }
                            moduli[i] = sc*sc+ss*ss;
                        }
                        for (int i = 0; i < ndata; i++)
                            if (moduli[i] < 1.0e-7)
                                moduli[i] = (moduli[(i-1+ndata)%ndata]+moduli[(i+1)%ndata])*0.5;
                        if (dim == 0)
                            xmoduli->upload(moduli, true);
                        else if (dim == 1)
                            ymoduli->upload(moduli, true);
                        else
                            zmoduli->upload(moduli, true);
                    }
                }
            }
        }
    }

    // Add code to subtract off the reciprocal part of excluded interactions.

    if ((nonbondedMethod == Ewald || nonbondedMethod == PME || nonbondedMethod == LJPME) && pmeio == NULL) {
        int numContexts = cc.getNumContexts();
        int startIndex = cc.getContextIndex()*force.getNumExceptions()/numContexts;
        int endIndex = (cc.getContextIndex()+1)*force.getNumExceptions()/numContexts;
        int numExclusions = endIndex-startIndex;
        if (numExclusions > 0) {
            paramsDefines["HAS_EXCLUSIONS"] = "1";
            vector<vector<int> > atoms(numExclusions, vector<int>(2));
            exclusionAtoms.initialize<mm_int2>(cc, numExclusions, "exclusionAtoms");
            exclusionParams.initialize<mm_float4>(cc, numExclusions, "exclusionParams");
            vector<mm_int2> exclusionAtomsVec(numExclusions);
            for (int i = 0; i < numExclusions; i++) {
                int j = i+startIndex;
                exclusionAtomsVec[i] = mm_int2(exclusions[j].first, exclusions[j].second);
                atoms[i][0] = exclusions[j].first;
                atoms[i][1] = exclusions[j].second;
            }
            exclusionAtoms.upload(exclusionAtomsVec);
            map<string, string> replacements;
            replacements["PARAMS"] = cc.getBondedUtilities().addArgument(exclusionParams, "float4");
            replacements["EWALD_ALPHA"] = cc.doubleToString(alpha);
            replacements["TWO_OVER_SQRT_PI"] = cc.doubleToString(2.0/sqrt(M_PI));
            replacements["DO_LJPME"] = doLJPME ? "1" : "0";
            replacements["USE_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
            if (doLJPME)
                replacements["EWALD_DISPERSION_ALPHA"] = cc.doubleToString(dispersionAlpha);
            replacements["LAMBDAS"] = cc.getBondedUtilities().addArgument(sliceLambdas, "real2");
            stringstream code;
            for (string param : requestedDerivatives) {
                string variableName = cc.getBondedUtilities().addEnergyParameterDerivative(param);
                string expression = getDerivativeExpression(param, hasCoulomb, doLJPME);
                if (expression.length() > 0)
                    code<<variableName<<" += "<<expression<<";"<<endl;
            }
            replacements["COMPUTE_DERIVATIVES"] = code.str();
            if (force.getIncludeDirectSpace())
                cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonNonbondedSlicingKernelSources::pmeExclusions, replacements), force.getForceGroup());
        }
    }

    // Add the interaction to the default nonbonded kernel.

    string source = cc.replaceStrings(CommonNonbondedSlicingKernelSources::coulombLennardJones, defines);
    charges.initialize(cc, cc.getPaddedNumAtoms(), cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "charges");
    baseParticleParams.initialize<mm_float4>(cc, cc.getPaddedNumAtoms(), "baseParticleParams");
    baseParticleParams.upload(baseParticleParamVec);
    map<string, string> replacements;
    replacements["ONE_4PI_EPS0"] = cc.doubleToString(ONE_4PI_EPS0);
    if (usePosqCharges) {
        replacements["CHARGE1"] = "posq1.w";
        replacements["CHARGE2"] = "posq2.w";
    }
    else {
        replacements["CHARGE1"] = prefix+"charge1";
        replacements["CHARGE2"] = prefix+"charge2";
    }
    if (hasCoulomb && !usePosqCharges)
        cc.getNonbondedUtilities().addParameter(ComputeParameterInfo(charges, prefix+"charge", "real", 1));
    sigmaEpsilon.initialize<mm_float2>(cc, cc.getPaddedNumAtoms(), "sigmaEpsilon");
    if (hasLJ) {
        replacements["SIGMA_EPSILON1"] = prefix+"sigmaEpsilon1";
        replacements["SIGMA_EPSILON2"] = prefix+"sigmaEpsilon2";
        cc.getNonbondedUtilities().addParameter(ComputeParameterInfo(sigmaEpsilon, prefix+"sigmaEpsilon", "float", 2));
    }
    replacements["SUBSET1"] = prefix+"subset1";
    replacements["SUBSET2"] = prefix+"subset2";
    cc.getNonbondedUtilities().addParameter(ComputeParameterInfo(subsets, prefix+"subset", "int", 1));
    replacements["LAMBDA"] = prefix+"lambda";
    cc.getNonbondedUtilities().addArgument(ComputeParameterInfo(sliceLambdas, prefix+"lambda", "real", 2));
    stringstream code;
    for (string param : requestedDerivatives) {
        string variableName = cc.getNonbondedUtilities().addEnergyParameterDerivative(param);
        string expression = getDerivativeExpression(param, hasCoulomb, hasLJ);
        if (expression.length() > 0)
            code<<variableName<<" += interactionScale*("<<expression<<");"<<endl;
    }
    replacements["COMPUTE_DERIVATIVES"] = code.str();
    source = cc.replaceStrings(source, replacements);
    if (force.getIncludeDirectSpace())
        cc.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup(), numParticles > 3000, true);

    // Initialize the exceptions.

    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0) {
        paramsDefines["HAS_EXCEPTIONS"] = "1";
        exceptionAtoms.resize(numExceptions);
        vector<vector<int> > atoms(numExceptions, vector<int>(2));
        exceptionParams.initialize<mm_float4>(cc, numExceptions, "exceptionParams");
        baseExceptionParams.initialize<mm_float4>(cc, numExceptions, "baseExceptionParams");
        vector<mm_float4> baseExceptionParamsVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], atoms[i][0], atoms[i][1], chargeProd, sigma, epsilon);
            int subset1 = force.getParticleSubset(atoms[i][0]);
            int subset2 = force.getParticleSubset(atoms[i][1]);
            union {int i; float f;} slice;
            slice.i = sliceIndex(subset1, subset2);
            baseExceptionParamsVec[i] = mm_float4(chargeProd, sigma, epsilon, slice.f);
            exceptionAtoms[i] = make_pair(atoms[i][0], atoms[i][1]);
        }
        baseExceptionParams.upload(baseExceptionParamsVec);
        map<string, string> replacements;
        replacements["APPLY_PERIODIC"] = (usePeriodic && force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0");
        replacements["PARAMS"] = cc.getBondedUtilities().addArgument(exceptionParams, "float4");
        replacements["LAMBDAS"] = cc.getBondedUtilities().addArgument(sliceLambdas, "real2");
        stringstream code;
        for (string param : requestedDerivatives) {
            string variableName = cc.getBondedUtilities().addEnergyParameterDerivative(param);
            string expression = getDerivativeExpression(param, hasCoulomb, hasLJ);
            if (expression.length() > 0)
                code<<variableName<<" += "<<expression<<";"<<endl;
        }
        replacements["COMPUTE_DERIVATIVES"] = code.str();
        if (force.getIncludeDirectSpace())
            cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonNonbondedSlicingKernelSources::nonbondedExceptions, replacements), force.getForceGroup());
    }

    // Initialize parameter offsets.

    vector<vector<mm_float4> > particleOffsetVec(force.getNumParticles());
    vector<vector<mm_float4> > exceptionOffsetVec(numExceptions);
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        auto paramPos = find(paramNames.begin(), paramNames.end(), param);
        int paramIndex;
        if (paramPos == paramNames.end()) {
            paramIndex = paramNames.size();
            paramNames.push_back(param);
        }
        else
            paramIndex = paramPos-paramNames.begin();
        particleOffsetVec[particle].push_back(mm_float4(charge, sigma, epsilon, paramIndex));
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        int index = exceptionIndex[exception];
        if (index < startIndex || index >= endIndex)
            continue;
        auto paramPos = find(paramNames.begin(), paramNames.end(), param);
        int paramIndex;
        if (paramPos == paramNames.end()) {
            paramIndex = paramNames.size();
            paramNames.push_back(param);
        }
        else
            paramIndex = paramPos-paramNames.begin();
        exceptionOffsetVec[index-startIndex].push_back(mm_float4(charge, sigma, epsilon, paramIndex));
    }
    paramValues.resize(paramNames.size(), 0.0);
    particleParamOffsets.initialize<mm_float4>(cc, max(force.getNumParticleParameterOffsets(), 1), "particleParamOffsets");
    particleOffsetIndices.initialize<int>(cc, cc.getPaddedNumAtoms()+1, "particleOffsetIndices");
    vector<int> particleOffsetIndicesVec, exceptionOffsetIndicesVec;
    vector<mm_float4> p, e;
    for (int i = 0; i < particleOffsetVec.size(); i++) {
        particleOffsetIndicesVec.push_back(p.size());
        for (int j = 0; j < particleOffsetVec[i].size(); j++)
            p.push_back(particleOffsetVec[i][j]);
    }
    while (particleOffsetIndicesVec.size() < particleOffsetIndices.getSize())
        particleOffsetIndicesVec.push_back(p.size());
    for (int i = 0; i < exceptionOffsetVec.size(); i++) {
        exceptionOffsetIndicesVec.push_back(e.size());
        for (int j = 0; j < exceptionOffsetVec[i].size(); j++)
            e.push_back(exceptionOffsetVec[i][j]);
    }
    exceptionOffsetIndicesVec.push_back(e.size());
    if (force.getNumParticleParameterOffsets() > 0) {
        particleParamOffsets.upload(p);
        particleOffsetIndices.upload(particleOffsetIndicesVec);
    }
    exceptionParamOffsets.initialize<mm_float4>(cc, max((int) e.size(), 1), "exceptionParamOffsets");
    exceptionOffsetIndices.initialize<int>(cc, exceptionOffsetIndicesVec.size(), "exceptionOffsetIndices");
    if (e.size() > 0) {
        exceptionParamOffsets.upload(e);
        exceptionOffsetIndices.upload(exceptionOffsetIndicesVec);
    }
    globalParams.initialize(cc, max((int) paramValues.size(), 1), cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "globalParams");
    if (paramValues.size() > 0)
        globalParams.upload(paramValues, true);
    chargeBuffer.initialize(cc, numSubsets*cc.getNumThreadBlocks(), cc.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "chargeBuffer");
    cc.clearBuffer(chargeBuffer);
    recomputeParams = true;

    // Add post-computation for dispersion correction.

    if (dispersionCoefficients.size() > 0 && force.getIncludeDirectSpace())
        cc.addPostComputation(new DispersionCorrectionPostComputation(cc, dispersionCoefficients, sliceLambdasVec, sliceScalingParams, force.getForceGroup()));

    // Initialize the kernel for updating parameters.

    ComputeProgram program = cc.compileProgram(CommonNonbondedSlicingKernelSources::nonbondedParameters, paramsDefines);
    computeParamsKernel = program->createKernel("computeParameters");
    computeExclusionParamsKernel = program->createKernel("computeExclusionParameters");
    computePlasmaCorrectionKernel = program->createKernel("computePlasmaCorrection");
    info = new ForceInfo(force);
    cc.addForce(info);
}

double CommonCalcSlicedNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    ContextSelector selector(cc);
    if (!hasInitializedKernel) {
        hasInitializedKernel = true;
        computeParamsKernel->addArg(cc.getEnergyBuffer());
        computeParamsKernel->addArg();
        computeParamsKernel->addArg(globalParams);
        computeParamsKernel->addArg(cc.getPaddedNumAtoms());
        computeParamsKernel->addArg(baseParticleParams);
        computeParamsKernel->addArg(cc.getPosq());
        computeParamsKernel->addArg(charges);
        computeParamsKernel->addArg(sigmaEpsilon);
        computeParamsKernel->addArg(particleParamOffsets);
        computeParamsKernel->addArg(particleOffsetIndices);
        computeParamsKernel->addArg(chargeBuffer);
        computeParamsKernel->addArg(subsets);
        computeParamsKernel->addArg(sliceLambdas);
        if (exceptionParams.isInitialized()) {
            computeParamsKernel->addArg((int) exceptionParams.getSize());
            computeParamsKernel->addArg(baseExceptionParams);
            computeParamsKernel->addArg(exceptionParams);
            computeParamsKernel->addArg(exceptionParamOffsets);
            computeParamsKernel->addArg(exceptionOffsetIndices);
        }
        if (exclusionParams.isInitialized()) {
            computeExclusionParamsKernel->addArg(cc.getPosq());
            computeExclusionParamsKernel->addArg(charges);
            computeExclusionParamsKernel->addArg(sigmaEpsilon);
            computeExclusionParamsKernel->addArg(subsets);
            computeExclusionParamsKernel->addArg((int) exclusionParams.getSize());
            computeExclusionParamsKernel->addArg(exclusionAtoms);
            computeExclusionParamsKernel->addArg(exclusionParams);
        }
        computePlasmaCorrectionKernel->addArg(chargeBuffer);
        computePlasmaCorrectionKernel->addArg(cc.getEnergyBuffer());
        if (cc.getUseDoublePrecision())
            computePlasmaCorrectionKernel->addArg(alpha);
        else
            computePlasmaCorrectionKernel->addArg((float) alpha);
        computePlasmaCorrectionKernel->addArg();
        computePlasmaCorrectionKernel->addArg(sliceLambdas);
        // TODO: Use the kernel above to compute slice background energy times volume for each slice

        if (usePmeQueue && hasDerivatives) {
            int energyElementSize = (cc.getUseDoublePrecision() || cc.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
            int bufferSize = pmeEnergyBuffer.getSize() * cc.getEnergyParamDerivNames().size();
            pmeEnergyParamDerivBuffer.initialize(cc, bufferSize, energyElementSize, "pmeEnergyParamDerivBuffer");
            cc.clearBuffer(pmeEnergyParamDerivBuffer);
        }

        stringstream coulombDerivativeCode;
        if (hasDerivatives && (cosSinSums.isInitialized() || (pmeGrid1.isInitialized() && hasCoulomb))) {
            bool assign = (pmeGrid1.isInitialized() && usePmeQueue && !doLJPME);
            const vector<string>& allDerivParams = cc.getEnergyParamDerivNames();
            for (int i = 0; i < allDerivParams.size(); i++) {
                stringstream expr;
                expr<<"energyParamDerivBuffer[GLOBAL_ID*"<<allDerivParams.size()<<"+"<<i<<"] "<<(assign ? "=" : "+=");
                bool empty = true;
                for (int slice = 0; slice < numSlices; slice++) {
                    ScalingParameterInfo info = sliceScalingParams[slice];
                    if (info.includeCoulomb && info.nameCoulomb == allDerivParams[i]) {
                        expr<<(empty ? " " : "+")<<"energy["<<slice<<"]";
                        empty = false;
                    }
                }
                if (!empty)
                    coulombDerivativeCode<<expr.str()<<";"<<endl;
                else if (assign)
                    coulombDerivativeCode<<expr.str()<<"0;"<<endl;
            }
        }

        stringstream ljDerivativeCode;
        if (hasDerivatives && (pmeGrid1.isInitialized() && doLJPME)) {
            const vector<string>& allDerivParams = cc.getEnergyParamDerivNames();
            for (int i = 0; i < allDerivParams.size(); i++) {
                stringstream expr;
                expr<<"energyParamDerivBuffer[GLOBAL_ID*"<<allDerivParams.size()<<"+"<<i<<"] += ";
                bool empty = true;
                for (int slice = 0; slice < numSlices; slice++) {
                    ScalingParameterInfo info = sliceScalingParams[slice];
                    if (info.includeLJ && info.nameLJ == allDerivParams[i]) {
                        expr<<(empty ? "" : "+")<<"energy["<<slice<<"]";
                        empty = false;
                    }
                }
                if (!empty)
                    ljDerivativeCode<<expr.str()<<";"<<endl;
            }
        }

        if (cosSinSums.isInitialized()) {
            map<string, string> replacements;
            replacements["ADD_DERIVATIVES"] = coulombDerivativeCode.str();
            ComputeProgram program = cc.compileProgram(cc.replaceStrings(CommonNonbondedSlicingKernelSources::ewald, replacements), ewaldDefines);
            ewaldSumsKernel = program->createKernel("calculateEwaldCosSinSums");
            ewaldForcesKernel = program->createKernel("calculateEwaldForces");
            ewaldSumsKernel->addArg(cc.getEnergyBuffer());
            ewaldSumsKernel->addArg(cc.getPosq());
            ewaldSumsKernel->addArg(cosSinSums);
            ewaldSumsKernel->addArg();
            ewaldSumsKernel->addArg(subsets);
            ewaldSumsKernel->addArg(sliceLambdas);
            if (hasDerivatives)
                ewaldSumsKernel->addArg(cc.getEnergyParamDerivBuffer());
            ewaldForcesKernel->addArg(cc.getLongForceBuffer());
            ewaldForcesKernel->addArg(cc.getPosq());
            ewaldForcesKernel->addArg(cosSinSums);
            ewaldForcesKernel->addArg();
            ewaldForcesKernel->addArg(subsets);
            ewaldForcesKernel->addArg(sliceLambdas);
        }
        if (pmeGrid1.isInitialized()) {
            // Create kernels for Coulomb PME.

            map<string, string> replacements;
            replacements["CHARGE"] = (usePosqCharges ? "pos.w" : "charges[atom]");
            replacements["ADD_DERIVATIVES"] = coulombDerivativeCode.str();
            ComputeProgram program = cc.compileProgram(cc.replaceStrings(CommonNonbondedSlicingKernelSources::pme, replacements), pmeDefines);
            pmeGridIndexKernel = program->createKernel("findAtomGridIndex");
            pmeSpreadChargeKernel = program->createKernel("gridSpreadCharge");
            pmeConvolutionKernel = program->createKernel("reciprocalConvolution");
            pmeEvalEnergyKernel = program->createKernel("gridEvaluateEnergy");
            pmeInterpolateForceKernel = program->createKernel("gridInterpolateForce");
            pmeGridIndexKernel->addArg(cc.getPosq());
            pmeGridIndexKernel->addArg(pmeAtomGridIndex);
            for (int i = 0; i < 8; i++)
                pmeGridIndexKernel->addArg();
            pmeGridIndexKernel->addArg(subsets);
            pmeSpreadChargeKernel->addArg(cc.getPosq());
            if (useFixedPointChargeSpreading)
                pmeSpreadChargeKernel->addArg(pmeGrid2);
            else
                pmeSpreadChargeKernel->addArg(pmeGrid1);
            for (int i = 0; i < 8; i++)
                pmeSpreadChargeKernel->addArg();
            pmeSpreadChargeKernel->addArg(pmeAtomGridIndex);
            pmeSpreadChargeKernel->addArg(charges);
            pmeConvolutionKernel->addArg(pmeGrid2);
            pmeConvolutionKernel->addArg(pmeBsplineModuliX);
            pmeConvolutionKernel->addArg(pmeBsplineModuliY);
            pmeConvolutionKernel->addArg(pmeBsplineModuliZ);
            for (int i = 0; i < 3; i++)
                pmeConvolutionKernel->addArg();
            pmeEvalEnergyKernel->addArg(pmeGrid2);
            if (usePmeQueue)
                pmeEvalEnergyKernel->addArg(pmeEnergyBuffer);
            else
                pmeEvalEnergyKernel->addArg(cc.getEnergyBuffer());
            pmeEvalEnergyKernel->addArg(pmeBsplineModuliX);
            pmeEvalEnergyKernel->addArg(pmeBsplineModuliY);
            pmeEvalEnergyKernel->addArg(pmeBsplineModuliZ);
            for (int i = 0; i < 3; i++)
                pmeEvalEnergyKernel->addArg();
            pmeEvalEnergyKernel->addArg(sliceLambdas);
            if (hasDerivatives) {
                if (usePmeQueue)
                    pmeEvalEnergyKernel->addArg(pmeEnergyParamDerivBuffer);
                else
                    pmeEvalEnergyKernel->addArg(cc.getEnergyParamDerivBuffer());
            }
            pmeInterpolateForceKernel->addArg(cc.getPosq());
            pmeInterpolateForceKernel->addArg(cc.getLongForceBuffer());
            pmeInterpolateForceKernel->addArg(pmeGrid1);
            for (int i = 0; i < 8; i++)
                pmeInterpolateForceKernel->addArg();
            pmeInterpolateForceKernel->addArg(pmeAtomGridIndex);
            pmeInterpolateForceKernel->addArg(charges);
            pmeInterpolateForceKernel->addArg(subsets);
            pmeInterpolateForceKernel->addArg(sliceLambdas);
            if (useFixedPointChargeSpreading) {
                pmeFinishSpreadChargeKernel = program->createKernel("finishSpreadCharge");
                pmeFinishSpreadChargeKernel->addArg(pmeGrid2);
                pmeFinishSpreadChargeKernel->addArg(pmeGrid1);
            }
            if (usePmeQueue) {
                syncQueue->setKernel(program->createKernel("addEnergy"));
                if (hasDerivatives)
                    syncQueue->addDerivatives(pmeEnergyParamDerivBuffer, cc.getEnergyParamDerivNames().size());
            }

            if (doLJPME) {
                // Create kernels for LJ PME.

                pmeDefines["EWALD_ALPHA"] = cc.doubleToString(dispersionAlpha);
                pmeDefines["GRID_SIZE_X"] = cc.intToString(dispersionGridSizeX);
                pmeDefines["GRID_SIZE_Y"] = cc.intToString(dispersionGridSizeY);
                pmeDefines["GRID_SIZE_Z"] = cc.intToString(dispersionGridSizeZ);
                pmeDefines["EPSILON_FACTOR"] = "1";
                pmeDefines["RECIP_EXP_FACTOR"] = cc.doubleToString(M_PI*M_PI/(dispersionAlpha*dispersionAlpha));
                pmeDefines["USE_LJPME"] = "1";
                pmeDefines["CHARGE_FROM_SIGEPS"] = "1";
                map<string, string> replacements;
                replacements["ADD_DERIVATIVES"] = ljDerivativeCode.str();
                program = cc.compileProgram(cc.replaceStrings(CommonNonbondedSlicingKernelSources::pme, replacements), pmeDefines);
                pmeDispersionGridIndexKernel = program->createKernel("findAtomGridIndex");
                pmeDispersionSpreadChargeKernel = program->createKernel("gridSpreadCharge");
                pmeDispersionConvolutionKernel = program->createKernel("reciprocalConvolution");
                pmeDispersionEvalEnergyKernel = program->createKernel("gridEvaluateEnergy");
                pmeDispersionInterpolateForceKernel = program->createKernel("gridInterpolateForce");
                pmeDispersionGridIndexKernel->addArg(cc.getPosq());
                pmeDispersionGridIndexKernel->addArg(pmeAtomGridIndex);
                for (int i = 0; i < 8; i++)
                    pmeDispersionGridIndexKernel->addArg();
                pmeDispersionGridIndexKernel->addArg(subsets);
                pmeDispersionSpreadChargeKernel->addArg(cc.getPosq());
                if (useFixedPointChargeSpreading)
                    pmeDispersionSpreadChargeKernel->addArg(pmeGrid2);
                else
                    pmeDispersionSpreadChargeKernel->addArg(pmeGrid1);
                for (int i = 0; i < 8; i++)
                    pmeDispersionSpreadChargeKernel->addArg();
                pmeDispersionSpreadChargeKernel->addArg(pmeAtomGridIndex);
                pmeDispersionSpreadChargeKernel->addArg(sigmaEpsilon);
                pmeDispersionConvolutionKernel->addArg(pmeGrid2);
                pmeDispersionConvolutionKernel->addArg(pmeDispersionBsplineModuliX);
                pmeDispersionConvolutionKernel->addArg(pmeDispersionBsplineModuliY);
                pmeDispersionConvolutionKernel->addArg(pmeDispersionBsplineModuliZ);
                for (int i = 0; i < 3; i++)
                    pmeDispersionConvolutionKernel->addArg();
                pmeDispersionEvalEnergyKernel->addArg(pmeGrid2);
                if (usePmeQueue)
                    pmeDispersionEvalEnergyKernel->addArg(pmeEnergyBuffer);
                else
                    pmeDispersionEvalEnergyKernel->addArg(cc.getEnergyBuffer());
                pmeDispersionEvalEnergyKernel->addArg(pmeDispersionBsplineModuliX);
                pmeDispersionEvalEnergyKernel->addArg(pmeDispersionBsplineModuliY);
                pmeDispersionEvalEnergyKernel->addArg(pmeDispersionBsplineModuliZ);
                for (int i = 0; i < 3; i++)
                    pmeDispersionEvalEnergyKernel->addArg();
                pmeDispersionEvalEnergyKernel->addArg(sliceLambdas);
                if (hasDerivatives) {
                    if (usePmeQueue)
                        pmeDispersionEvalEnergyKernel->addArg(pmeEnergyParamDerivBuffer);
                    else
                        pmeDispersionEvalEnergyKernel->addArg(cc.getEnergyParamDerivBuffer());
                }
                pmeDispersionInterpolateForceKernel->addArg(cc.getPosq());
                pmeDispersionInterpolateForceKernel->addArg(cc.getLongForceBuffer());
                pmeDispersionInterpolateForceKernel->addArg(pmeGrid1);
                for (int i = 0; i < 8; i++)
                    pmeDispersionInterpolateForceKernel->addArg();
                pmeDispersionInterpolateForceKernel->addArg(pmeAtomGridIndex);
                pmeDispersionInterpolateForceKernel->addArg(sigmaEpsilon);
                pmeDispersionInterpolateForceKernel->addArg(subsets);
                pmeDispersionInterpolateForceKernel->addArg(sliceLambdas);
                if (useFixedPointChargeSpreading) {
                    pmeDispersionFinishSpreadChargeKernel = program->createKernel("finishSpreadCharge");
                    pmeDispersionFinishSpreadChargeKernel->addArg(pmeGrid2);
                    pmeDispersionFinishSpreadChargeKernel->addArg(pmeGrid1);
                }
            }
        }
    }

    // Update scaling parameters if needed.

    bool scalingParamChanged = false;
    for (int slice = 0; slice < numSlices; slice++) {
        ScalingParameterInfo info = sliceScalingParams[slice];
        if (info.includeCoulomb) {
            double value = context.getParameter(info.nameCoulomb);
            if (value != sliceLambdasVec[slice].x) {
                sliceLambdasVec[slice].x = value;
                scalingParamChanged = true;
            }
        }
        if (info.includeLJ) {
            double value = context.getParameter(info.nameLJ);
            if (value != sliceLambdasVec[slice].y) {
                sliceLambdasVec[slice].y = value;
                scalingParamChanged = true;
            }
        }
    }
    if (scalingParamChanged) {
        if (hasReciprocal && cc.getContextIndex() == 0) {
            ewaldSelfEnergy = 0.0;
            for (int i = 0; i < numSubsets; i++) {
                int slice = sliceIndex(i, i);
                ewaldSelfEnergy += sliceLambdasVec[slice].x*subsetSelfEnergy[i].x + sliceLambdasVec[slice].y*subsetSelfEnergy[i].y;
            }
            backgroundEnergyVolume = 0.0;
            for (int slice = 0; slice < numSlices; slice++)
                backgroundEnergyVolume += sliceLambdasVec[slice].x*sliceBackgroundEnergyVolume[slice];
        }
        if (cc.getUseDoublePrecision())
            sliceLambdas.upload(sliceLambdasVec);
        else {
            vector<mm_float2> sliceLambdasVecFloat(numSlices);
            for (size_t i = 0; i < numSlices; i++)
                sliceLambdasVecFloat[i] = mm_float2(sliceLambdasVec[i].x, sliceLambdasVec[i].y);
            sliceLambdas.upload(sliceLambdasVecFloat);
        }
    }

    // Update particle and exception parameters.

    bool paramChanged = false;
    for (int i = 0; i < paramNames.size(); i++) {
        double value = context.getParameter(paramNames[i]);
        if (value != paramValues[i]) {
            paramValues[i] = value;;
            paramChanged = true;
        }
    }
    if (paramChanged) {
        recomputeParams = true;
        globalParams.upload(paramValues, true);
    }
    double energy = 0.0;
    if (includeReciprocal && (pmeGrid1.isInitialized() || cosSinSums.isInitialized())) {
        Vec3 a, b, c;
        cc.getPeriodicBoxVectors(a, b, c);
        double volume = a[0]*b[1]*c[2];
        energy = ewaldSelfEnergy + backgroundEnergyVolume/volume;
    }
    if (recomputeParams || hasOffsets) {
        computeParamsKernel->setArg(1, (int) (includeEnergy && includeReciprocal));
        computeParamsKernel->execute(cc.getNumAtoms());
        if (exclusionParams.isInitialized())
            computeExclusionParamsKernel->execute(exclusionParams.getSize());
        if (usePmeQueue) {
            paramsSyncEvent->enqueue();
            paramsSyncEvent->queueWait(pmeQueue);
        }
        if (hasOffsets) {
            // The Ewald self energy was computed in the kernel.

            energy = 0.0;
            if (pmeGrid1.isInitialized() || cosSinSums.isInitialized()) {
                // Invoke a kernel to compute the correction for the neutralizing plasma.

                Vec3 a, b, c;
                cc.getPeriodicBoxVectors(a, b, c);
                double volume = a[0]*b[1]*c[2];
                if (cc.getUseDoublePrecision())
                    computePlasmaCorrectionKernel->setArg(3, volume);
                else
                    computePlasmaCorrectionKernel->setArg(3, (float) volume);
                computePlasmaCorrectionKernel->execute(ComputeContext::ThreadBlockSize, ComputeContext::ThreadBlockSize);
            }
        }
        recomputeParams = false;
    }

    // Do reciprocal space calculations.

    if (cosSinSums.isInitialized() && includeReciprocal) {
        Vec3 a, b, c;
        cc.getPeriodicBoxVectors(a, b, c);
        if (cc.getUseDoublePrecision()) {
            ewaldSumsKernel->setArg(3, mm_double4(a[0], b[1], c[2], 0));
            ewaldForcesKernel->setArg(3, mm_double4(a[0], b[1], c[2], 0));
        }
        else {
            ewaldSumsKernel->setArg(3, mm_float4((float) a[0], (float) b[1], (float) c[2], 0));
            ewaldForcesKernel->setArg(3, mm_float4((float) a[0], (float) b[1], (float) c[2], 0));
        }
        ewaldSumsKernel->execute(cosSinSums.getSize());
        ewaldForcesKernel->execute(cc.getNumAtoms());
    }
    if (pmeGrid1.isInitialized() && includeReciprocal) {
        if (usePmeQueue)
            cc.setCurrentQueue(pmeQueue);

        // Invert the periodic box vectors.

        Vec3 boxVectors[3];
        cc.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
        double scale = 1.0/determinant;
        mm_double4 recipBoxVectors[3];
        recipBoxVectors[0] = mm_double4(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0, 0);
        recipBoxVectors[1] = mm_double4(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0, 0);
        recipBoxVectors[2] = mm_double4((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale, -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale, 0);
        mm_float4 recipBoxVectorsFloat[3];
        for (int i = 0; i < 3; i++)
            recipBoxVectorsFloat[i] = mm_float4((float) recipBoxVectors[i].x, (float) recipBoxVectors[i].y, (float) recipBoxVectors[i].z, 0);

        // Execute the reciprocal space kernels.

        if (hasCoulomb) {
            setPeriodicBoxArgs(cc, pmeGridIndexKernel, 2);
            if (cc.getUseDoublePrecision()) {
                pmeGridIndexKernel->setArg(7, recipBoxVectors[0]);
                pmeGridIndexKernel->setArg(8, recipBoxVectors[1]);
                pmeGridIndexKernel->setArg(9, recipBoxVectors[2]);
            }
            else {
                pmeGridIndexKernel->setArg(7, recipBoxVectorsFloat[0]);
                pmeGridIndexKernel->setArg(8, recipBoxVectorsFloat[1]);
                pmeGridIndexKernel->setArg(9, recipBoxVectorsFloat[2]);
            }
            pmeGridIndexKernel->execute(cc.getNumAtoms());
            sort->sort(pmeAtomGridIndex);
            setPeriodicBoxArgs(cc, pmeSpreadChargeKernel, 2);
            if (cc.getUseDoublePrecision()) {
                pmeSpreadChargeKernel->setArg(7, recipBoxVectors[0]);
                pmeSpreadChargeKernel->setArg(8, recipBoxVectors[1]);
                pmeSpreadChargeKernel->setArg(9, recipBoxVectors[2]);
            }
            else {
                pmeSpreadChargeKernel->setArg(7, recipBoxVectorsFloat[0]);
                pmeSpreadChargeKernel->setArg(8, recipBoxVectorsFloat[1]);
                pmeSpreadChargeKernel->setArg(9, recipBoxVectorsFloat[2]);
            }
            pmeSpreadChargeKernel->execute(cc.getNumAtoms());
            if (useFixedPointChargeSpreading)
                pmeFinishSpreadChargeKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            fft->execFFT(pmeGrid1, pmeGrid2, true);
            if (cc.getUseDoublePrecision()) {
                pmeConvolutionKernel->setArg<mm_double4>(4, recipBoxVectors[0]);
                pmeConvolutionKernel->setArg<mm_double4>(5, recipBoxVectors[1]);
                pmeConvolutionKernel->setArg<mm_double4>(6, recipBoxVectors[2]);
                pmeEvalEnergyKernel->setArg<mm_double4>(5, recipBoxVectors[0]);
                pmeEvalEnergyKernel->setArg<mm_double4>(6, recipBoxVectors[1]);
                pmeEvalEnergyKernel->setArg<mm_double4>(7, recipBoxVectors[2]);
            }
            else {
                pmeConvolutionKernel->setArg<mm_float4>(4, recipBoxVectorsFloat[0]);
                pmeConvolutionKernel->setArg<mm_float4>(5, recipBoxVectorsFloat[1]);
                pmeConvolutionKernel->setArg<mm_float4>(6, recipBoxVectorsFloat[2]);
                pmeEvalEnergyKernel->setArg<mm_float4>(5, recipBoxVectorsFloat[0]);
                pmeEvalEnergyKernel->setArg<mm_float4>(6, recipBoxVectorsFloat[1]);
                pmeEvalEnergyKernel->setArg<mm_float4>(7, recipBoxVectorsFloat[2]);
            }
            if (includeEnergy || hasDerivatives)
                pmeEvalEnergyKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            pmeConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            fft->execFFT(pmeGrid2, pmeGrid1, false);
            setPeriodicBoxArgs(cc, pmeInterpolateForceKernel, 3);
            if (cc.getUseDoublePrecision()) {
                pmeInterpolateForceKernel->setArg(8, recipBoxVectors[0]);
                pmeInterpolateForceKernel->setArg(9, recipBoxVectors[1]);
                pmeInterpolateForceKernel->setArg(10, recipBoxVectors[2]);
            }
            else {
                pmeInterpolateForceKernel->setArg(8, recipBoxVectorsFloat[0]);
                pmeInterpolateForceKernel->setArg(9, recipBoxVectorsFloat[1]);
                pmeInterpolateForceKernel->setArg(10, recipBoxVectorsFloat[2]);
            }
            if (deviceIsCpu)
                pmeInterpolateForceKernel->execute(cc.getNumThreadBlocks(), 1);
            else
                pmeInterpolateForceKernel->execute(cc.getNumAtoms());
        }

        if (doLJPME && hasLJ) {
            setPeriodicBoxArgs(cc, pmeDispersionGridIndexKernel, 2);
            if (cc.getUseDoublePrecision()) {
                pmeDispersionGridIndexKernel->setArg(7, recipBoxVectors[0]);
                pmeDispersionGridIndexKernel->setArg(8, recipBoxVectors[1]);
                pmeDispersionGridIndexKernel->setArg(9, recipBoxVectors[2]);
            }
            else {
                pmeDispersionGridIndexKernel->setArg(7, recipBoxVectorsFloat[0]);
                pmeDispersionGridIndexKernel->setArg(8, recipBoxVectorsFloat[1]);
                pmeDispersionGridIndexKernel->setArg(9, recipBoxVectorsFloat[2]);
            }
            pmeDispersionGridIndexKernel->execute(cc.getNumAtoms());
            if (!hasCoulomb)
                sort->sort(pmeAtomGridIndex);
            if (useFixedPointChargeSpreading)
                cc.clearBuffer(pmeGrid2);
            else
                cc.clearBuffer(pmeGrid1);
            setPeriodicBoxArgs(cc, pmeDispersionSpreadChargeKernel, 2);
            if (cc.getUseDoublePrecision()) {
                pmeDispersionSpreadChargeKernel->setArg(7, recipBoxVectors[0]);
                pmeDispersionSpreadChargeKernel->setArg(8, recipBoxVectors[1]);
                pmeDispersionSpreadChargeKernel->setArg(9, recipBoxVectors[2]);
            }
            else {
                pmeDispersionSpreadChargeKernel->setArg(7, recipBoxVectorsFloat[0]);
                pmeDispersionSpreadChargeKernel->setArg(8, recipBoxVectorsFloat[1]);
                pmeDispersionSpreadChargeKernel->setArg(9, recipBoxVectorsFloat[2]);
            }
            pmeDispersionSpreadChargeKernel->execute(cc.getNumAtoms());
            if (useFixedPointChargeSpreading)
                pmeDispersionFinishSpreadChargeKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            dispersionFft->execFFT(pmeGrid1, pmeGrid2, true);
            if (cc.getUseDoublePrecision()) {
                pmeDispersionConvolutionKernel->setArg(4, recipBoxVectors[0]);
                pmeDispersionConvolutionKernel->setArg(5, recipBoxVectors[1]);
                pmeDispersionConvolutionKernel->setArg(6, recipBoxVectors[2]);
                pmeDispersionEvalEnergyKernel->setArg(5, recipBoxVectors[0]);
                pmeDispersionEvalEnergyKernel->setArg(6, recipBoxVectors[1]);
                pmeDispersionEvalEnergyKernel->setArg(7, recipBoxVectors[2]);
            }
            else {
                pmeDispersionConvolutionKernel->setArg(4, recipBoxVectorsFloat[0]);
                pmeDispersionConvolutionKernel->setArg(5, recipBoxVectorsFloat[1]);
                pmeDispersionConvolutionKernel->setArg(6, recipBoxVectorsFloat[2]);
                pmeDispersionEvalEnergyKernel->setArg(5, recipBoxVectorsFloat[0]);
                pmeDispersionEvalEnergyKernel->setArg(6, recipBoxVectorsFloat[1]);
                pmeDispersionEvalEnergyKernel->setArg(7, recipBoxVectorsFloat[2]);
            }
            if (!hasCoulomb) {
                cc.clearBuffer(pmeEnergyBuffer);
                if (hasDerivatives)
                    cc.clearBuffer(pmeEnergyParamDerivBuffer);
            }
            if (includeEnergy || hasDerivatives)
                pmeDispersionEvalEnergyKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            pmeDispersionConvolutionKernel->execute(gridSizeX*gridSizeY*gridSizeZ);
            dispersionFft->execFFT(pmeGrid2, pmeGrid1, false);
            setPeriodicBoxArgs(cc, pmeDispersionInterpolateForceKernel, 3);
            if (cc.getUseDoublePrecision()) {
                pmeDispersionInterpolateForceKernel->setArg(8, recipBoxVectors[0]);
                pmeDispersionInterpolateForceKernel->setArg(9, recipBoxVectors[1]);
                pmeDispersionInterpolateForceKernel->setArg(10, recipBoxVectors[2]);
            }
            else {
                pmeDispersionInterpolateForceKernel->setArg(8, recipBoxVectorsFloat[0]);
                pmeDispersionInterpolateForceKernel->setArg(9, recipBoxVectorsFloat[1]);
                pmeDispersionInterpolateForceKernel->setArg(10, recipBoxVectorsFloat[2]);
            }
            if (deviceIsCpu)
                pmeDispersionInterpolateForceKernel->execute(cc.getNumThreadBlocks(), 1);
            else
                pmeDispersionInterpolateForceKernel->execute(cc.getNumAtoms());
        }
        if (usePmeQueue) {
            pmeSyncEvent->enqueue();
            cc.restoreDefaultQueue();
        }
    }
    // if (dispersionCoefficient != 0.0 && includeDirect) {
    //     Vec3 a, b, c;
    //     cc.getPeriodicBoxVectors(a, b, c);
    //     energy += dispersionCoefficient/(a[0]*b[1]*c[2]);
    // }
    if (!hasOffsets && hasReciprocal && includeReciprocal) {
        map<string, double>& energyParamDerivs = cc.getEnergyParamDerivWorkspace();
        for (int i = 0; i < numSubsets; i++) {
            ScalingParameterInfo info = sliceScalingParams[sliceIndex(i, i)];
            if (info.hasDerivativeCoulomb)
                energyParamDerivs[info.nameCoulomb] += subsetSelfEnergy[i].x;
            if (doLJPME && info.hasDerivativeLJ)
                energyParamDerivs[info.nameLJ] += subsetSelfEnergy[i].y;
        }
        Vec3 a, b, c;
        cc.getPeriodicBoxVectors(a, b, c);
        double volume = a[0]*b[1]*c[2];
        for (int slice = 0; slice < numSlices; slice++) {
            ScalingParameterInfo info = sliceScalingParams[slice];
            if (info.hasDerivativeCoulomb)
                energyParamDerivs[info.nameCoulomb] += sliceBackgroundEnergyVolume[slice]/volume;
        }
    }
    return energy;
}

void CommonCalcSlicedNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const SlicedNonbondedForce& force
    // , int firstParticle, int lastParticle, int firstException, int lastException
) {
    // Make sure the new parameters are acceptable.

    ContextSelector selector(cc);
    if (force.getNumParticles() != cc.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (!hasCoulomb || !hasLJ) {
        for (int i = 0; i < force.getNumParticles(); i++) {
            double charge, sigma, epsilon;
            force.getParticleParameters(i, charge, sigma, epsilon);
            if (!hasCoulomb && charge != 0.0)
                throw OpenMMException("updateParametersInContext: The nonbonded force kernel does not include Coulomb interactions, because all charges were originally 0");
            if (!hasLJ && epsilon != 0.0)
                throw OpenMMException("updateParametersInContext: The nonbonded force kernel does not include Lennard-Jones interactions, because all epsilons were originally 0");
        }
    }
    for (int i = 0; i < force.getNumParticles(); i++)
        subsetsVec[i] = force.getParticleSubset(i);
    subsets.upload(subsetsVec);
    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        exceptionsWithOffsets.insert(exception);
    }
    vector<int> exceptions;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if (exceptionIndex.find(i) == exceptionIndex.end()) {
            if (chargeProd != 0.0 || epsilon != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end())
                throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");
        }
        else
            exceptions.push_back(i);
    }
    int numContexts = cc.getNumContexts();
    int startIndex = cc.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cc.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions != exceptionAtoms.size())
        throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");

    // TODO: Remove lines below and implement the approach added to OpenMM in commit 78902be (PR #4610)

    int firstParticle = 0;
    int lastParticle = force.getNumParticles() - 1;
    int firstException = 0;
    int lastException = numExceptions - 1;
    
    // Record the per-particle parameters.

    if (firstParticle <= lastParticle) {
        vector<mm_float4> baseParticleParamVec(cc.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
        for (int i = 0; i < force.getNumParticles(); i++) {
            double charge, sigma, epsilon;
            force.getParticleParameters(i, charge, sigma, epsilon);
            baseParticleParamVec[i] = mm_float4(charge, sigma, epsilon, 0);
        }
        baseParticleParams.uploadSubArray(&baseParticleParamVec[firstParticle], firstParticle, lastParticle-firstParticle+1);

        // Compute the self energy.
    
        // totalCharge = 0.0;
        ewaldSelfEnergy = 0.0;
        backgroundEnergyVolume = 0.0;
        subsetSelfEnergy.assign(numSubsets, mm_double2(0, 0));
        sliceBackgroundEnergyVolume.assign(numSlices, 0.0);
        vector<double> subsetCharges(numSubsets, 0.0);
        if (nonbondedMethod == Ewald || nonbondedMethod == PME || nonbondedMethod == LJPME) {
            if (cc.getContextIndex() == 0) {
                for (int i = 0; i < force.getNumParticles(); i++) {
                    int subset = subsetsVec[i];
                    double charge = baseParticleParamVec[i].x;
                    subsetSelfEnergy[subset].x -= charge*charge*ONE_4PI_EPS0*alpha/sqrt(M_PI);
                    subsetCharges[subset] += charge;
                    if (doLJPME)
                        subsetSelfEnergy[subset].y += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
                }
                for (int i = 0; i < numSubsets; i++) {
                    int slice = sliceIndex(i, i);
                    ewaldSelfEnergy += sliceLambdasVec[slice].x*subsetSelfEnergy[i].x;
                    if (doLJPME)
                        ewaldSelfEnergy += sliceLambdasVec[slice].y*subsetSelfEnergy[i].y;
                    double factor = -subsetCharges[i]/(8*EPSILON0*alpha*alpha);
                    sliceBackgroundEnergyVolume[slice] += subsetCharges[i]*factor;
                    for (int j = i + 1; j < numSubsets; j++)
                        sliceBackgroundEnergyVolume[sliceIndex(i, j)] += 2.0*subsetCharges[j]*factor;
                }
            }
        }
    }

    // Record the exceptions.

    if (firstException <= lastException) {
        vector<mm_float4> baseExceptionParamsVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            int particle1, particle2;
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], particle1, particle2, chargeProd, sigma, epsilon);
            if (make_pair(particle1, particle2) != exceptionAtoms[i])
                throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");
            baseExceptionParamsVec[i] = mm_float4(chargeProd, sigma, epsilon, 0);
        }
        baseExceptionParams.upload(baseExceptionParamsVec);
    }

    // Record parameter offsets.

    vector<vector<mm_float4> > particleOffsetVec(force.getNumParticles());
    vector<vector<mm_float4> > exceptionOffsetVec(numExceptions);
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge, sigma, epsilon;
        force.getParticleParameterOffset(i, param, particle, charge, sigma, epsilon);
        auto paramPos = find(paramNames.begin(), paramNames.end(), param);
        if (paramPos == paramNames.end())
            throw OpenMMException("updateParametersInContext: The parameter of a particle parameter offset has changed");
        int paramIndex = paramPos-paramNames.begin();
        particleOffsetVec[particle].push_back(mm_float4(charge, sigma, epsilon, paramIndex));
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge, sigma, epsilon;
        force.getExceptionParameterOffset(i, param, exception, charge, sigma, epsilon);
        int index = exceptionIndex[exception];
        if (index < startIndex || index >= endIndex)
            continue;
        auto paramPos = find(paramNames.begin(), paramNames.end(), param);
        if (paramPos == paramNames.end())
            throw OpenMMException("updateParametersInContext: The parameter of an exception parameter offset has changed");
        int paramIndex = paramPos-paramNames.begin();
        exceptionOffsetVec[index-startIndex].push_back(mm_float4(charge, sigma, epsilon, paramIndex));
    }
    if (max(force.getNumParticleParameterOffsets(), 1) != particleParamOffsets.getSize())
        throw OpenMMException("updateParametersInContext: The number of particle parameter offsets has changed");
    vector<mm_float4> p, e;
    for (int i = 0; i < particleOffsetVec.size(); i++)
        for (int j = 0; j < particleOffsetVec[i].size(); j++)
            p.push_back(particleOffsetVec[i][j]);
    for (int i = 0; i < exceptionOffsetVec.size(); i++)
        for (int j = 0; j < exceptionOffsetVec[i].size(); j++)
            e.push_back(exceptionOffsetVec[i][j]);
    if (force.getNumParticleParameterOffsets() > 0)
        particleParamOffsets.upload(p);
    if (max((int) e.size(), 1) != exceptionParamOffsets.getSize())
        throw OpenMMException("updateParametersInContext: The number of exception parameter offsets has changed");
    if (e.size() > 0)
        exceptionParamOffsets.upload(e);

    // Compute other values.

    if (force.getUseDispersionCorrection() && cc.getContextIndex() == 0 && (nonbondedMethod == CutoffPeriodic || nonbondedMethod == Ewald || nonbondedMethod == PME))
        dispersionCoefficients = SlicedNonbondedForceImpl::calcDispersionCorrections(context.getSystem(), force);
    cc.invalidateMolecules(info, firstParticle <= lastParticle || force.getNumParticleParameterOffsets() > 0,
                           firstException <= lastException || force.getNumExceptionParameterOffsets() > 0);
    recomputeParams = true;
}

void CommonCalcSlicedNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != PME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    // if (useCpuPme)
    //     cpuPme.getAs<CalcPmeReciprocalForceKernel>().getPMEParameters(alpha, nx, ny, nz);
    // else {
        alpha = this->alpha;
        nx = gridSizeX;
        ny = gridSizeY;
        nz = gridSizeZ;
    // }
}

void CommonCalcSlicedNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != LJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    // if (useCpuPme)
    //     //cpuPme.getAs<CalcPmeReciprocalForceKernel>().getLJPMEParameters(alpha, nx, ny, nz);
    //     throw OpenMMException("getPMEParametersInContext: CPUPME has not been implemented for LJPME yet.");
    // else {
        alpha = this->dispersionAlpha;
        nx = dispersionGridSizeX;
        ny = dispersionGridSizeY;
        nz = dispersionGridSizeZ;
    // }
}
