/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "CudaNonbondedSlicingKernels.h"
#include "CudaNonbondedSlicingKernelSources.h"
#include "CommonNonbondedSlicingKernelSources.h"
#include "SlicedNonbondedForce.h"
#include "internal/SlicedNonbondedForceImpl.h"
#include "openmm/NonbondedForce.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/common/ContextSelector.h"
#include <cstring>
#include <map>
#include <algorithm>
#include <iostream>

#define CHECK_RESULT(result, prefix) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        throw OpenMMException(m.str());\
    }

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

class CudaCalcSlicedNonbondedForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const SlicedNonbondedForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
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
        int particle1, particle2, subset1, subset2;
        double chargeProd1, chargeProd2, sigma1, sigma2, epsilon1, epsilon2;
        force.getExceptionParameters(group1, particle1, particle2, chargeProd1, sigma1, epsilon1);
        subset1 = force.getParticleSubset(particle1);
        subset2 = force.getParticleSubset(particle2);
        int slice1 = sliceIndex(subset1, subset2);
        force.getExceptionParameters(group2, particle1, particle2, chargeProd2, sigma2, epsilon2);
        subset1 = force.getParticleSubset(particle1);
        subset2 = force.getParticleSubset(particle2);
        int slice2 = sliceIndex(subset1, subset2);
        return (chargeProd1 == chargeProd2 && sigma1 == sigma2 && epsilon1 == epsilon2 && slice1 == slice2);
    }
private:
    const SlicedNonbondedForce& force;
};

class CudaCalcSlicedNonbondedForceKernel::SyncStreamPreComputation : public CudaContext::ForcePreComputation {
public:
    SyncStreamPreComputation(CudaContext& cu, CUstream stream, CUevent event, int forceGroup) : cu(cu), stream(stream), event(event), forceGroup(forceGroup) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            cuEventRecord(event, cu.getCurrentStream());
            cuStreamWaitEvent(stream, event, 0);
        }
    }
private:
    CudaContext& cu;
    CUstream stream;
    CUevent event;
    int forceGroup;
};

class CudaCalcSlicedNonbondedForceKernel::SyncStreamPostComputation : public CudaContext::ForcePostComputation {
public:
    SyncStreamPostComputation(CudaContext& cu, CUevent event, int forceGroup) : cu(cu), event(event), forceGroup(forceGroup) {}
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0)
            cuStreamWaitEvent(cu.getCurrentStream(), event, 0);
        return 0.0;
    }
private:
    CudaContext& cu;
    CUevent event;
    int forceGroup;
};

class CudaCalcSlicedNonbondedForceKernel::AddEnergyPostComputation : public CudaContext::ForcePostComputation {
public:
    AddEnergyPostComputation(CudaContext& cu, int forceGroup) : cu(cu), forceGroup(forceGroup), initialized(false) {
    }
    void initialize(CudaArray& pmeEnergyBuffer, CudaArray& ljpmeEnergyBuffer, CudaArray& sliceLambdas, vector<ScalingParameterInfo> sliceScalingParams) {
        int numSlices = sliceLambdas.getSize();
        bool doLJPME = ljpmeEnergyBuffer.isInitialized();
        bufferSize = pmeEnergyBuffer.getSize()/numSlices;
        set<string> requestedDerivs;
        for (ScalingParameterInfo info : sliceScalingParams) {
            if (info.hasDerivativeCoulomb)
                requestedDerivs.insert(info.nameCoulomb);
            if (doLJPME && info.hasDerivativeLJ)
                requestedDerivs.insert(info.nameLJ);
        }
        hasDerivatives = requestedDerivs.size() > 0;
        stringstream code;
        if (hasDerivatives) {
            const vector<string>& allDerivs = cu.getEnergyParamDerivNames();
            for (string param : requestedDerivs) {
                int position = find(allDerivs.begin(), allDerivs.end(), param) - allDerivs.begin();
                code<<"energyParamDerivs[index*"<<allDerivs.size()<<"+"<<position<<"] += ";
                for (int slice = 0; slice < numSlices; slice++) {
                    ScalingParameterInfo info = sliceScalingParams[slice];
                    if (info.nameCoulomb == param)
                        code<<"+clEnergy["<<slice<<"]";
                    if (doLJPME && info.nameLJ == param)
                        code<<"+ljEnergy["<<slice<<"]";
                }
                code<<";"<<endl;
            }
        }
        map<string, string> replacements, defines;
        replacements["NUM_SLICES"] = cu.intToString(numSlices);
        replacements["USE_LJPME"] = doLJPME ? "1" : "0";
        replacements["HAS_DERIVATIVES"] = hasDerivatives ? "1" : "0";
        replacements["ADD_DERIVATIVES"] = code.str();
        string source = cu.replaceStrings(CommonNonbondedSlicingKernelSources::pmeAddEnergy, replacements);
        CUmodule module = cu.createModule(source, defines);
        addEnergyKernel = cu.getKernel(module, "addEnergy");
        arguments.clear();
        arguments.push_back(&cu.getEnergyBuffer().getDevicePointer());
        if (hasDerivatives)
            arguments.push_back(&cu.getEnergyParamDerivBuffer().getDevicePointer());
        arguments.push_back(&pmeEnergyBuffer.getDevicePointer());
        if (doLJPME)
            arguments.push_back(&ljpmeEnergyBuffer.getDevicePointer());
        arguments.push_back(&sliceLambdas.getDevicePointer());
        arguments.push_back(&bufferSize);
        initialized = true;
    }
    bool isInitialized() {
        return initialized;
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((includeEnergy || hasDerivatives) && (groups&(1<<forceGroup)) != 0)
            cu.executeKernel(addEnergyKernel, &arguments[0], bufferSize);
        return 0.0;
    }
private:
    CudaContext& cu;
    CUfunction addEnergyKernel;
    vector<void*> arguments;
    int forceGroup;
    int bufferSize;
    bool initialized;
    bool hasDerivatives;
};

class CudaCalcSlicedNonbondedForceKernel::DispersionCorrectionPostComputation : public CudaContext::ForcePostComputation {
public:
    DispersionCorrectionPostComputation(CudaContext& cu, vector<double>& coefficients, vector<double2>& sliceLambdas, vector<ScalingParameterInfo>& sliceScalingParams, int forceGroup) :
                                        cu(cu), coefficients(coefficients), sliceLambdas(sliceLambdas), sliceScalingParams(sliceScalingParams), forceGroup(forceGroup) {
        numSlices = coefficients.size();
        hasDerivatives = false;
        for (auto info : sliceScalingParams)
            hasDerivatives = hasDerivatives || info.hasDerivativeLJ;
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        double energy = 0.0;
        if ((includeEnergy || hasDerivatives) && (groups&(1<<forceGroup)) != 0) {
            double4 boxSize = cu.getPeriodicBoxSize();
            double volume = boxSize.x*boxSize.y*boxSize.z;
            if (includeEnergy)
                for (int slice = 0; slice < numSlices; slice++)
                    energy += sliceLambdas[slice].y*coefficients[slice]/volume;
            if (hasDerivatives) {
                map<string, double>& energyParamDerivs = cu.getEnergyParamDerivWorkspace();
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
    CudaContext& cu;
    vector<double>& coefficients;
    vector<double2>& sliceLambdas;
    vector<ScalingParameterInfo>& sliceScalingParams;
    int forceGroup;
    int numSlices;
    bool hasDerivatives;
};

CudaCalcSlicedNonbondedForceKernel::~CudaCalcSlicedNonbondedForceKernel() {
    ContextSelector selector(cu);
    if (sort != NULL)
        delete sort;
    if (fft != NULL)
        delete fft;
    if (dispersionFft != NULL)
        delete dispersionFft;
    if (hasInitializedFFT && usePmeStream) {
        cuStreamDestroy(pmeStream);
        cuEventDestroy(pmeSyncEvent);
        cuEventDestroy(paramsSyncEvent);
    }
}

string CudaCalcSlicedNonbondedForceKernel::getDerivativeExpression(string param, bool conditionCoulomb, bool conditionLJ) {
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

void CudaCalcSlicedNonbondedForceKernel::initialize(const System& system, const SlicedNonbondedForce& force) {
    ContextSelector selector(cu);
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "slicedNonbonded"+cu.intToString(forceIndex)+"_";

    string realToFixedPoint = Platform::getOpenMMVersion()[0] == '7' ? CudaNonbondedSlicingKernelSources::realToFixedPoint : "";

    int numParticles = force.getNumParticles();
    numSubsets = force.getNumSubsets();
    numSlices = force.getNumSlices();
    sliceLambdasVec.resize(numSlices, make_double2(1, 1));
    subsetSelfEnergy.resize(numSlices, make_double2(0, 0));
    sliceScalingParams.resize(numSlices, ScalingParameterInfo());

    subsetsVec.resize(cu.getPaddedNumAtoms(), 0);
    for (int i = 0; i < numParticles; i++)
        subsetsVec[i] = force.getParticleSubset(i);
    subsets.initialize<int>(cu, cu.getPaddedNumAtoms(), "subsets");
    subsets.upload(subsetsVec);

    int numDerivs = force.getNumScalingParameterDerivatives();
    hasDerivatives = numDerivs > 0;
    set<string> requestedDerivatives;
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

    size_t sizeOfReal = cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float);
    sliceLambdas.initialize(cu, numSlices, 2*sizeOfReal, "sliceLambdas");
    if (cu.getUseDoublePrecision())
        sliceLambdas.upload(sliceLambdasVec);
    else
        sliceLambdas.upload(double2Tofloat2(sliceLambdasVec));

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
    map<int, int> exceptionIndex;
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

    vector<float4> baseParticleParamVec(cu.getPaddedNumAtoms(), make_float4(0, 0, 0, 0));
    vector<vector<int> > exclusionList(numParticles);
    hasCoulomb = false;
    hasLJ = false;
    for (int i = 0; i < numParticles; i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        baseParticleParamVec[i] = make_float4(charge, sigma, epsilon, 0);
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
    usePosqCharges = hasCoulomb ? cu.requestPosqCharges() : false;

    map<string, string> defines;
    defines["HAS_COULOMB"] = (hasCoulomb ? "1" : "0");
    defines["HAS_LENNARD_JONES"] = (hasLJ ? "1" : "0");
    defines["USE_LJ_SWITCH"] = (useCutoff && force.getUseSwitchingFunction() ? "1" : "0");
    if (useCutoff) {
        // Compute the reaction field constants.

        double reactionFieldK = pow(force.getCutoffDistance(), -3.0)*(force.getReactionFieldDielectric()-1.0)/(2.0*force.getReactionFieldDielectric()+1.0);
        double reactionFieldC = (1.0 / force.getCutoffDistance())*(3.0*force.getReactionFieldDielectric())/(2.0*force.getReactionFieldDielectric()+1.0);
        defines["REACTION_FIELD_K"] = cu.doubleToString(reactionFieldK);
        defines["REACTION_FIELD_C"] = cu.doubleToString(reactionFieldC);

        // Compute the switching coefficients.

        if (force.getUseSwitchingFunction()) {
            defines["LJ_SWITCH_CUTOFF"] = cu.doubleToString(force.getSwitchingDistance());
            defines["LJ_SWITCH_C3"] = cu.doubleToString(10/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 3.0));
            defines["LJ_SWITCH_C4"] = cu.doubleToString(15/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 4.0));
            defines["LJ_SWITCH_C5"] = cu.doubleToString(6/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 5.0));
        }
    }
    if (force.getUseDispersionCorrection() && cu.getContextIndex() == 0 && hasLJ && useCutoff && usePeriodic && !doLJPME)
        dispersionCoefficients = SlicedNonbondedForceImpl::calcDispersionCorrections(system, force);
    alpha = 0;
    ewaldSelfEnergy = 0.0;
    map<string, string> paramsDefines;
    paramsDefines["NUM_SUBSETS"] = cu.intToString(numSubsets);
    paramsDefines["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
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
        defines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        if (cu.getContextIndex() == 0) {
            paramsDefines["INCLUDE_EWALD"] = "1";
            paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cu.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
            for (int i = 0; i < numParticles; i++)
                subsetSelfEnergy[subsetsVec[i]].x -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
            for (int i = 0; i < numSubsets; i++)
                ewaldSelfEnergy += sliceLambdasVec[sliceIndex(i, i)].x*subsetSelfEnergy[i].x;

            // Create the reciprocal space kernels.

            map<string, string> replacements;
            replacements["NUM_ATOMS"] = cu.intToString(numParticles);
            replacements["NUM_SUBSETS"] = cu.intToString(numSubsets);
            replacements["NUM_SLICES"] = cu.intToString(numSlices);
            replacements["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
            replacements["KMAX_X"] = cu.intToString(kmaxx);
            replacements["KMAX_Y"] = cu.intToString(kmaxy);
            replacements["KMAX_Z"] = cu.intToString(kmaxz);
            replacements["EXP_COEFFICIENT"] = cu.doubleToString(-1.0/(4.0*alpha*alpha));
            replacements["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
            replacements["M_PI"] = cu.doubleToString(M_PI);
            CUmodule module = cu.createModule(realToFixedPoint+CudaNonbondedSlicingKernelSources::vectorOps+CommonNonbondedSlicingKernelSources::ewald, replacements);
            ewaldSumsKernel = cu.getKernel(module, "calculateEwaldCosSinSums");
            ewaldForcesKernel = cu.getKernel(module, "calculateEwaldForces");
            int elementSize = (cu.getUseDoublePrecision() ? sizeof(double2) : sizeof(float2));
            cosSinSums.initialize(cu, (2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1)*numSubsets, elementSize, "cosSinSums");
            int bufferSize = cu.getNumThreadBlocks()*CudaContext::ThreadBlockSize;
            pmeEnergyBuffer.initialize(cu, numSlices*bufferSize, elementSize, "pmeEnergyBuffer");
            cu.clearBuffer(pmeEnergyBuffer);
            int recipForceGroup = force.getReciprocalSpaceForceGroup();
            cu.addPostComputation(addEnergy = new AddEnergyPostComputation(cu, recipForceGroup >= 0 ? recipForceGroup : force.getForceGroup()));
        }
    }
    else if (((nonbondedMethod == PME || nonbondedMethod == LJPME) && hasCoulomb) || doLJPME) {
        // Compute the PME parameters.

        SlicedNonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSizeX, gridSizeY, gridSizeZ, false);
        gridSizeX = CudaFFT3D::findLegalDimension(gridSizeX);
        gridSizeY = CudaFFT3D::findLegalDimension(gridSizeY);
        gridSizeZ = CudaFFT3D::findLegalDimension(gridSizeZ);
        if (doLJPME) {
            SlicedNonbondedForceImpl::calcPMEParameters(system, force, dispersionAlpha, dispersionGridSizeX,
                                                  dispersionGridSizeY, dispersionGridSizeZ, true);
            dispersionGridSizeX = CudaFFT3D::findLegalDimension(dispersionGridSizeX);
            dispersionGridSizeY = CudaFFT3D::findLegalDimension(dispersionGridSizeY);
            dispersionGridSizeZ = CudaFFT3D::findLegalDimension(dispersionGridSizeZ);
        }
        defines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        defines["DO_LJPME"] = doLJPME ? "1" : "0";
        if (doLJPME) {
            defines["EWALD_DISPERSION_ALPHA"] = cu.doubleToString(dispersionAlpha);
            double invRCut6 = pow(force.getCutoffDistance(), -6);
            double dalphaR = dispersionAlpha * force.getCutoffDistance();
            double dar2 = dalphaR*dalphaR;
            double dar4 = dar2*dar2;
            double multShift6 = -invRCut6*(1.0 - exp(-dar2) * (1.0 + dar2 + 0.5*dar4));
            defines["INVCUT6"] = cu.doubleToString(invRCut6);
            defines["MULTSHIFT6"] = cu.doubleToString(multShift6);
        }
        if (cu.getContextIndex() == 0) {
            paramsDefines["INCLUDE_EWALD"] = "1";
            paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cu.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
            for (int i = 0; i < numParticles; i++)
                subsetSelfEnergy[subsetsVec[i]].x -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
            if (doLJPME) {
                paramsDefines["INCLUDE_LJPME"] = "1";
                paramsDefines["LJPME_SELF_ENERGY_SCALE"] = cu.doubleToString(pow(dispersionAlpha, 6)/3.0);
                for (int i = 0; i < numParticles; i++)
                    subsetSelfEnergy[subsetsVec[i]].y += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
            }
            for (int i = 0; i < numSubsets; i++) {
                int slice = sliceIndex(i, i);
                ewaldSelfEnergy += sliceLambdasVec[slice].x*subsetSelfEnergy[i].x + sliceLambdasVec[slice].y*subsetSelfEnergy[i].y;
            }
            char deviceName[100];
            cuDeviceGetName(deviceName, 100, cu.getDevice());
            usePmeStream = (!cu.getPlatformData().disablePmeStream && string(deviceName) != "GeForce GTX 980"); // Using a separate stream is slower on GTX 980
            map<string, string> pmeDefines;
            pmeDefines["PME_ORDER"] = cu.intToString(PmeOrder);
            pmeDefines["NUM_ATOMS"] = cu.intToString(numParticles);
            pmeDefines["NUM_SUBSETS"] = cu.intToString(numSubsets);
            pmeDefines["NUM_SLICES"] = cu.intToString(numSlices);
            pmeDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
            pmeDefines["RECIP_EXP_FACTOR"] = cu.doubleToString(M_PI*M_PI/(alpha*alpha));
            pmeDefines["GRID_SIZE_X"] = cu.intToString(gridSizeX);
            pmeDefines["GRID_SIZE_Y"] = cu.intToString(gridSizeY);
            pmeDefines["GRID_SIZE_Z"] = cu.intToString(gridSizeZ);
            pmeDefines["EPSILON_FACTOR"] = cu.doubleToString(sqrt(ONE_4PI_EPS0));
            pmeDefines["M_PI"] = cu.doubleToString(M_PI);
            if (cu.getUseDoublePrecision() || cu.getPlatformData().deterministicForces)
                pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";
            if (usePmeStream)
                pmeDefines["USE_PME_STREAM"] = "1";
            map<string, string> replacements;
            replacements["CHARGE"] = (usePosqCharges ? "pos.w" : "charges[atom]");
            CUmodule module = cu.createModule(realToFixedPoint+CudaNonbondedSlicingKernelSources::vectorOps+cu.replaceStrings(CommonNonbondedSlicingKernelSources::pme, replacements), pmeDefines);
            pmeGridIndexKernel = cu.getKernel(module, "findAtomGridIndex");
            pmeSpreadChargeKernel = cu.getKernel(module, "gridSpreadCharge");
            pmeConvolutionKernel = cu.getKernel(module, "reciprocalConvolution");
            pmeInterpolateForceKernel = cu.getKernel(module, "gridInterpolateForce");
            pmeEvalEnergyKernel = cu.getKernel(module, "gridEvaluateEnergy");
            pmeFinishSpreadChargeKernel = cu.getKernel(module, "finishSpreadCharge");
            cuFuncSetCacheConfig(pmeSpreadChargeKernel, CU_FUNC_CACHE_PREFER_SHARED);
            cuFuncSetCacheConfig(pmeInterpolateForceKernel, CU_FUNC_CACHE_PREFER_L1);
            if (doLJPME) {
                pmeDefines["EWALD_ALPHA"] = cu.doubleToString(dispersionAlpha);
                pmeDefines["GRID_SIZE_X"] = cu.intToString(dispersionGridSizeX);
                pmeDefines["GRID_SIZE_Y"] = cu.intToString(dispersionGridSizeY);
                pmeDefines["GRID_SIZE_Z"] = cu.intToString(dispersionGridSizeZ);
                pmeDefines["RECIP_EXP_FACTOR"] = cu.doubleToString(M_PI*M_PI/(dispersionAlpha*dispersionAlpha));
                pmeDefines["USE_LJPME"] = "1";
                pmeDefines["CHARGE_FROM_SIGEPS"] = "1";
                if (cu.getUseDoublePrecision() || cu.getPlatformData().deterministicForces)
                    pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";
                module = cu.createModule(realToFixedPoint+CudaNonbondedSlicingKernelSources::vectorOps+CommonNonbondedSlicingKernelSources::pme, pmeDefines);
                pmeDispersionFinishSpreadChargeKernel = cu.getKernel(module, "finishSpreadCharge");
                pmeDispersionGridIndexKernel = cu.getKernel(module, "findAtomGridIndex");
                pmeDispersionSpreadChargeKernel = cu.getKernel(module, "gridSpreadCharge");
                pmeDispersionConvolutionKernel = cu.getKernel(module, "reciprocalConvolution");
                pmeEvalDispersionEnergyKernel = cu.getKernel(module, "gridEvaluateEnergy");
                pmeInterpolateDispersionForceKernel = cu.getKernel(module, "gridInterpolateForce");
                cuFuncSetCacheConfig(pmeDispersionSpreadChargeKernel, CU_FUNC_CACHE_PREFER_L1);
            }

            // Create required data structures.

            int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
            int roundedZSize = PmeOrder*(int) ceil(gridSizeZ/(double) PmeOrder);
            int gridElements = gridSizeX*gridSizeY*roundedZSize*numSubsets;
            if (doLJPME) {
                roundedZSize = PmeOrder*(int) ceil(dispersionGridSizeZ/(double) PmeOrder);
                gridElements = max(gridElements, dispersionGridSizeX*dispersionGridSizeY*roundedZSize*numSubsets);
            }
            pmeGrid1.initialize(cu, gridElements, 2*elementSize, "pmeGrid1");
            pmeGrid2.initialize(cu, gridElements, 2*elementSize, "pmeGrid2");
            cu.addAutoclearBuffer(pmeGrid2);
            pmeBsplineModuliX.initialize(cu, gridSizeX, elementSize, "pmeBsplineModuliX");
            pmeBsplineModuliY.initialize(cu, gridSizeY, elementSize, "pmeBsplineModuliY");
            pmeBsplineModuliZ.initialize(cu, gridSizeZ, elementSize, "pmeBsplineModuliZ");
            if (doLJPME) {
                pmeDispersionBsplineModuliX.initialize(cu, dispersionGridSizeX, elementSize, "pmeDispersionBsplineModuliX");
                pmeDispersionBsplineModuliY.initialize(cu, dispersionGridSizeY, elementSize, "pmeDispersionBsplineModuliY");
                pmeDispersionBsplineModuliZ.initialize(cu, dispersionGridSizeZ, elementSize, "pmeDispersionBsplineModuliZ");
            }
            pmeAtomGridIndex.initialize<int2>(cu, numParticles, "pmeAtomGridIndex");
            int energyElementSize = (cu.getUseDoublePrecision() || cu.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
            int bufferSize = cu.getNumThreadBlocks()*CudaContext::ThreadBlockSize;
            pmeEnergyBuffer.initialize(cu, numSlices*bufferSize, energyElementSize, "pmeEnergyBuffer");
            cu.clearBuffer(pmeEnergyBuffer);
            sort = new CudaSort(cu, new SortTrait(), cu.getNumAtoms());

            // Prepare for doing PME on its own stream.

            int recipForceGroup = force.getReciprocalSpaceForceGroup();
            if (recipForceGroup < 0)
                recipForceGroup = force.getForceGroup();
            if (usePmeStream) {
                pmeDefines["USE_PME_STREAM"] = "1";
                cuStreamCreate(&pmeStream, CU_STREAM_NON_BLOCKING);
                // CHECK_RESULT(cuEventCreate(&pmeSyncEvent, cu.getEventFlags()), "Error creating event for SlicedNonbondedForce");  // OpenMM 8.0
                // CHECK_RESULT(cuEventCreate(&paramsSyncEvent, cu.getEventFlags()), "Error creating event for SlicedNonbondedForce");  // OpenMM 8.0
                CHECK_RESULT(cuEventCreate(&pmeSyncEvent, CU_EVENT_DISABLE_TIMING), "Error creating event for SlicedNonbondedForce");
                CHECK_RESULT(cuEventCreate(&paramsSyncEvent, CU_EVENT_DISABLE_TIMING), "Error creating event for SlicedNonbondedForce");
                cu.addPreComputation(new SyncStreamPreComputation(cu, pmeStream, pmeSyncEvent, recipForceGroup));
                cu.addPostComputation(new SyncStreamPostComputation(cu, pmeSyncEvent, recipForceGroup));
            }
            else
                pmeStream = cu.getCurrentStream();

            cu.addPostComputation(addEnergy = new AddEnergyPostComputation(cu, recipForceGroup));

            int cufftVersion;
            cufftGetVersion(&cufftVersion);
            useCudaFFT = force.getUseCudaFFT() && (cufftVersion >= 7050); // There was a critical bug in version 7.0
            if (useCudaFFT)
                fft = (CudaFFT3D*) new CudaCuFFT3D(cu, pmeStream, gridSizeX, gridSizeY, gridSizeZ, numSubsets, true, pmeGrid1, pmeGrid2);
            else
                fft = (CudaFFT3D*) new CudaVkFFT3D(cu, pmeStream, gridSizeX, gridSizeY, gridSizeZ, numSubsets, true, pmeGrid1, pmeGrid2);
            if (doLJPME) {
                ljpmeEnergyBuffer.initialize(cu, numSlices*bufferSize, energyElementSize, "ljpmeEnergyBuffer");
                cu.clearBuffer(ljpmeEnergyBuffer);
                if (useCudaFFT)
                    dispersionFft = (CudaFFT3D*) new CudaCuFFT3D(cu, pmeStream, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, numSubsets, true, pmeGrid1, pmeGrid2);
                else
                    dispersionFft = (CudaFFT3D*) new CudaVkFFT3D(cu, pmeStream, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, numSubsets, true, pmeGrid1, pmeGrid2);
            }
            hasInitializedFFT = true;

            // Initialize the b-spline moduli.

            for (int grid = 0; grid < 2; grid++) {
                int xsize, ysize, zsize;
                CudaArray *xmoduli, *ymoduli, *zmoduli;
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

    // Add code to subtract off the reciprocal part of excluded interactions.

    if (nonbondedMethod == Ewald || nonbondedMethod == PME || nonbondedMethod == LJPME) {
        int numContexts = cu.getPlatformData().contexts.size();
        int startIndex = cu.getContextIndex()*force.getNumExceptions()/numContexts;
        int endIndex = (cu.getContextIndex()+1)*force.getNumExceptions()/numContexts;
        int numExclusions = endIndex-startIndex;
        if (numExclusions > 0) {
            paramsDefines["HAS_EXCLUSIONS"] = "1";
            vector<vector<int> > atoms(numExclusions, vector<int>(2));
            exclusionAtoms.initialize<int2>(cu, numExclusions, "exclusionAtoms");
            exclusionParams.initialize<float4>(cu, numExclusions, "exclusionParams");
            vector<int2> exclusionAtomsVec(numExclusions);
            for (int i = 0; i < numExclusions; i++) {
                int j = i+startIndex;
                exclusionAtomsVec[i] = make_int2(exclusions[j].first, exclusions[j].second);
                atoms[i][0] = exclusions[j].first;
                atoms[i][1] = exclusions[j].second;
            }
            exclusionAtoms.upload(exclusionAtomsVec);
            map<string, string> replacements;
            replacements["PARAMS"] = cu.getBondedUtilities().addArgument(exclusionParams.getDevicePointer(), "float4");
            replacements["EWALD_ALPHA"] = cu.doubleToString(alpha);
            replacements["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
            replacements["DO_LJPME"] = doLJPME ? "1" : "0";
            replacements["USE_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
            if (doLJPME)
                replacements["EWALD_DISPERSION_ALPHA"] = cu.doubleToString(dispersionAlpha);
            replacements["LAMBDAS"] = cu.getBondedUtilities().addArgument(sliceLambdas.getDevicePointer(), "real2");
            stringstream code;
            for (string param : requestedDerivatives) {
                string variableName = cu.getBondedUtilities().addEnergyParameterDerivative(param);
                string expression = getDerivativeExpression(param, hasCoulomb, doLJPME);
                if (expression.length() > 0)
                    code<<variableName<<" += "<<expression<<";"<<endl;
            }
            replacements["COMPUTE_DERIVATIVES"] = code.str();
            if (force.getIncludeDirectSpace())
                cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CommonNonbondedSlicingKernelSources::pmeExclusions, replacements), force.getForceGroup());
        }
    }

    // Add the interaction to the default nonbonded kernel.

    string source = cu.replaceStrings(CommonNonbondedSlicingKernelSources::coulombLennardJones, defines);
    charges.initialize(cu, cu.getPaddedNumAtoms(), cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "charges");
    baseParticleParams.initialize<float4>(cu, cu.getPaddedNumAtoms(), "baseParticleParams");
    baseParticleParams.upload(baseParticleParamVec);
    map<string, string> replacements;
    replacements["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
    if (usePosqCharges) {
        replacements["CHARGE1"] = "posq1.w";
        replacements["CHARGE2"] = "posq2.w";
    }
    else {
        replacements["CHARGE1"] = prefix+"charge1";
        replacements["CHARGE2"] = prefix+"charge2";
    }
    if (hasCoulomb && !usePosqCharges)
        cu.getNonbondedUtilities().addParameter(CudaNonbondedUtilities::ParameterInfo(prefix+"charge", "real", 1, charges.getElementSize(), charges.getDevicePointer()));
    sigmaEpsilon.initialize<float2>(cu, cu.getPaddedNumAtoms(), "sigmaEpsilon");
    if (hasLJ) {
        replacements["SIGMA_EPSILON1"] = prefix+"sigmaEpsilon1";
        replacements["SIGMA_EPSILON2"] = prefix+"sigmaEpsilon2";
        cu.getNonbondedUtilities().addParameter(CudaNonbondedUtilities::ParameterInfo(prefix+"sigmaEpsilon", "float", 2, sizeof(float2), sigmaEpsilon.getDevicePointer()));
    }
    replacements["SUBSET1"] = prefix+"subset1";
    replacements["SUBSET2"] = prefix+"subset2";
    cu.getNonbondedUtilities().addParameter(CudaNonbondedUtilities::ParameterInfo(prefix+"subset", "int", 1, sizeof(int), subsets.getDevicePointer()));
    replacements["LAMBDA"] = prefix+"lambda";
    cu.getNonbondedUtilities().addArgument(CudaNonbondedUtilities::ParameterInfo(prefix+"lambda", "real", 2, 2*sizeOfReal, sliceLambdas.getDevicePointer()));
    stringstream code;
    for (string param : requestedDerivatives) {
        string variableName = cu.getNonbondedUtilities().addEnergyParameterDerivative(param);
        string expression = getDerivativeExpression(param, hasCoulomb, hasLJ);
        if (expression.length() > 0)
            code<<variableName<<" += interactionScale*("<<expression<<");"<<endl;
    }
    replacements["COMPUTE_DERIVATIVES"] = code.str();
    source = cu.replaceStrings(source, replacements);
    if (force.getIncludeDirectSpace())
        cu.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup(), true);

    // Initialize the exceptions.

    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0) {
        paramsDefines["HAS_EXCEPTIONS"] = "1";
        exceptionAtoms.resize(numExceptions);
        vector<vector<int> > atoms(numExceptions, vector<int>(2));
        exceptionParams.initialize<float4>(cu, numExceptions, "exceptionParams");
        baseExceptionParams.initialize<float4>(cu, numExceptions, "baseExceptionParams");
        exceptionPairs.initialize<int2>(cu, numExceptions, "exceptionPairs");
        exceptionSlices.initialize<int>(cu, numExceptions, "exceptionSlices");
        vector<float4> baseExceptionParamsVec(numExceptions);
        vector<int> exceptionSlicesVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], atoms[i][0], atoms[i][1], chargeProd, sigma, epsilon);
            baseExceptionParamsVec[i] = make_float4(chargeProd, sigma, epsilon, 0);
            exceptionAtoms[i] = make_pair(atoms[i][0], atoms[i][1]);
            int subset1 = force.getParticleSubset(atoms[i][0]);
            int subset2 = force.getParticleSubset(atoms[i][1]);
            exceptionSlicesVec[i] = sliceIndex(subset1, subset2);
        }
        baseExceptionParams.upload(baseExceptionParamsVec);
        exceptionPairs.upload(exceptionAtoms);
        exceptionSlices.upload(exceptionSlicesVec);
        map<string, string> replacements;
        replacements["APPLY_PERIODIC"] = (usePeriodic && force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0");
        replacements["PARAMS"] = cu.getBondedUtilities().addArgument(exceptionParams.getDevicePointer(), "float4");
        replacements["LAMBDAS"] = cu.getBondedUtilities().addArgument(sliceLambdas.getDevicePointer(), "real2");
        stringstream code;
        for (string param : requestedDerivatives) {
            string variableName = cu.getBondedUtilities().addEnergyParameterDerivative(param);
            string expression = getDerivativeExpression(param, hasCoulomb, hasLJ);
            if (expression.length() > 0)
                code<<variableName<<" += "<<expression<<";"<<endl;
        }
        replacements["COMPUTE_DERIVATIVES"] = code.str();
        if (force.getIncludeDirectSpace())
            cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CommonNonbondedSlicingKernelSources::nonbondedExceptions, replacements), force.getForceGroup());
    }

    // Initialize parameter offsets.

    vector<vector<float4> > particleOffsetVec(force.getNumParticles());
    vector<vector<float4> > exceptionOffsetVec(numExceptions);
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
        particleOffsetVec[particle].push_back(make_float4(charge, sigma, epsilon, paramIndex));
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
        exceptionOffsetVec[index-startIndex].push_back(make_float4(charge, sigma, epsilon, paramIndex));
    }
    paramValues.resize(paramNames.size(), 0.0);
    particleParamOffsets.initialize<float4>(cu, max(force.getNumParticleParameterOffsets(), 1), "particleParamOffsets");
    particleOffsetIndices.initialize<int>(cu, cu.getPaddedNumAtoms()+1, "particleOffsetIndices");
    vector<int> particleOffsetIndicesVec, exceptionOffsetIndicesVec;
    vector<float4> p, e;
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
    exceptionParamOffsets.initialize<float4>(cu, max((int) e.size(), 1), "exceptionParamOffsets");
    exceptionOffsetIndices.initialize<int>(cu, exceptionOffsetIndicesVec.size(), "exceptionOffsetIndices");
    if (e.size() > 0) {
        exceptionParamOffsets.upload(e);
        exceptionOffsetIndices.upload(exceptionOffsetIndicesVec);
    }
    globalParams.initialize(cu, max((int) paramValues.size(), 1), cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "globalParams");
    if (paramValues.size() > 0)
        globalParams.upload(paramValues, true);
    recomputeParams = true;

    // Add post-computation for dispersion correction.

    if (dispersionCoefficients.size() > 0 && force.getIncludeDirectSpace())
        cu.addPostComputation(new DispersionCorrectionPostComputation(cu, dispersionCoefficients, sliceLambdasVec, sliceScalingParams, force.getForceGroup()));

    // Initialize the kernel for updating parameters.

    CUmodule module = cu.createModule(CommonNonbondedSlicingKernelSources::nonbondedParameters, paramsDefines);
    computeParamsKernel = cu.getKernel(module, "computeParameters");
    computeExclusionParamsKernel = cu.getKernel(module, "computeExclusionParameters");
    info = new ForceInfo(force);
    cu.addForce(info);
}

double CudaCalcSlicedNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    ContextSelector selector(cu);

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
        ewaldSelfEnergy = 0.0;
        for (int i = 0; i < numSubsets; i++) {
            int slice = sliceIndex(i, i);
            ewaldSelfEnergy += sliceLambdasVec[slice].x*subsetSelfEnergy[i].x + sliceLambdasVec[slice].y*subsetSelfEnergy[i].y;
        }
        if (cu.getUseDoublePrecision())
            sliceLambdas.upload(sliceLambdasVec);
        else
            sliceLambdas.upload(double2Tofloat2(sliceLambdasVec));
    }

    // Update particle and exception parameters.

    bool paramChanged = false;
    for (int i = 0; i < paramNames.size(); i++) {
        double value = context.getParameter(paramNames[i]);
        if (value != paramValues[i]) {
            paramValues[i] = value;
            paramChanged = true;
        }
    }
    if (paramChanged) {
        recomputeParams = true;
        globalParams.upload(paramValues, true);
    }
    double energy = (includeReciprocal ? ewaldSelfEnergy : 0.0);
    if (recomputeParams || hasOffsets) {
        int computeSelfEnergy = (includeEnergy && includeReciprocal);
        int numAtoms = cu.getPaddedNumAtoms();
        vector<void*> paramsArgs = {&cu.getEnergyBuffer().getDevicePointer(), &computeSelfEnergy, &globalParams.getDevicePointer(), &numAtoms,
                &baseParticleParams.getDevicePointer(), &cu.getPosq().getDevicePointer(), &charges.getDevicePointer(), &sigmaEpsilon.getDevicePointer(),
                &particleParamOffsets.getDevicePointer(), &particleOffsetIndices.getDevicePointer(), &subsets.getDevicePointer(), &sliceLambdas.getDevicePointer()};
        int numExceptions;
        if (exceptionParams.isInitialized()) {
            numExceptions = exceptionParams.getSize();
            paramsArgs.push_back(&numExceptions);
            paramsArgs.push_back(&exceptionPairs.getDevicePointer());
            paramsArgs.push_back(&baseExceptionParams.getDevicePointer());
            paramsArgs.push_back(&exceptionSlices.getDevicePointer());
            paramsArgs.push_back(&exceptionParams.getDevicePointer());
            paramsArgs.push_back(&exceptionParamOffsets.getDevicePointer());
            paramsArgs.push_back(&exceptionOffsetIndices.getDevicePointer());
        }
        cu.executeKernel(computeParamsKernel, &paramsArgs[0], cu.getPaddedNumAtoms());
        if (exclusionParams.isInitialized()) {
            int numExclusions = exclusionParams.getSize();
            vector<void*> exclusionParamsArgs = {&cu.getPosq().getDevicePointer(), &charges.getDevicePointer(), &sigmaEpsilon.getDevicePointer(),
                    &subsets.getDevicePointer(), &numExclusions, &exclusionAtoms.getDevicePointer(), &exclusionParams.getDevicePointer()};
            cu.executeKernel(computeExclusionParamsKernel, &exclusionParamsArgs[0], numExclusions);
        }
        if (usePmeStream) {
            cuEventRecord(paramsSyncEvent, cu.getCurrentStream());
            cuStreamWaitEvent(pmeStream, paramsSyncEvent, 0);
        }
        if (hasOffsets)
            energy = 0.0; // The Ewald self energy was computed in the kernel.
        recomputeParams = false;
    }

    // Do reciprocal space calculations.

    if (cosSinSums.isInitialized() && includeReciprocal) {
        if (!addEnergy->isInitialized())
            addEnergy->initialize(pmeEnergyBuffer, ljpmeEnergyBuffer, sliceLambdas, sliceScalingParams);
        void* sumsArgs[] = {&pmeEnergyBuffer.getDevicePointer(), &cu.getPosq().getDevicePointer(),
                &subsets.getDevicePointer(), &cosSinSums.getDevicePointer(), cu.getPeriodicBoxSizePointer()};
        cu.executeKernel(ewaldSumsKernel, sumsArgs, cosSinSums.getSize()/numSubsets);
        void* forcesArgs[] = {&cu.getForce().getDevicePointer(), &cu.getPosq().getDevicePointer(), &cosSinSums.getDevicePointer(),
                &subsets.getDevicePointer(), &sliceLambdas.getDevicePointer(), cu.getPeriodicBoxSizePointer()};
        cu.executeKernel(ewaldForcesKernel, forcesArgs, cu.getNumAtoms());
    }
    if (pmeGrid1.isInitialized() && includeReciprocal) {
        if (!addEnergy->isInitialized())
            addEnergy->initialize(pmeEnergyBuffer, ljpmeEnergyBuffer, sliceLambdas, sliceScalingParams);

        if (usePmeStream)
            cu.setCurrentStream(pmeStream);

        // Invert the periodic box vectors.

        Vec3 boxVectors[3];
        cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
        double scale = 1.0/determinant;
        double4 recipBoxVectors[3];
        recipBoxVectors[0] = make_double4(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0, 0);
        recipBoxVectors[1] = make_double4(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0, 0);
        recipBoxVectors[2] = make_double4((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale, -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale, 0);
        float4 recipBoxVectorsFloat[3];
        void* recipBoxVectorPointer[3];
        if (cu.getUseDoublePrecision()) {
            recipBoxVectorPointer[0] = &recipBoxVectors[0];
            recipBoxVectorPointer[1] = &recipBoxVectors[1];
            recipBoxVectorPointer[2] = &recipBoxVectors[2];
        }
        else {
            recipBoxVectorsFloat[0] = make_float4((float) recipBoxVectors[0].x, 0, 0, 0);
            recipBoxVectorsFloat[1] = make_float4((float) recipBoxVectors[1].x, (float) recipBoxVectors[1].y, 0, 0);
            recipBoxVectorsFloat[2] = make_float4((float) recipBoxVectors[2].x, (float) recipBoxVectors[2].y, (float) recipBoxVectors[2].z, 0);
            recipBoxVectorPointer[0] = &recipBoxVectorsFloat[0];
            recipBoxVectorPointer[1] = &recipBoxVectorsFloat[1];
            recipBoxVectorPointer[2] = &recipBoxVectorsFloat[2];
        }

        // Execute the reciprocal space kernels.

        if (hasCoulomb) {
            void* gridIndexArgs[] = {&cu.getPosq().getDevicePointer(), &pmeAtomGridIndex.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &subsets.getDevicePointer()};
            cu.executeKernel(pmeGridIndexKernel, gridIndexArgs, cu.getNumAtoms());

            sort->sort(pmeAtomGridIndex);

            void* spreadArgs[] = {&cu.getPosq().getDevicePointer(), &pmeGrid2.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                    &charges.getDevicePointer()};
            cu.executeKernel(pmeSpreadChargeKernel, spreadArgs, cu.getNumAtoms(), 128);

            void* finishSpreadArgs[] = {&pmeGrid2.getDevicePointer(), &pmeGrid1.getDevicePointer()};
            cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, gridSizeX*gridSizeY*gridSizeZ, 256);

            fft->execFFT(true);

            if (includeEnergy || hasDerivatives) {
                void* computeEnergyArgs[] = {&pmeGrid2.getDevicePointer(), &pmeEnergyBuffer.getDevicePointer(),
                        &pmeBsplineModuliX.getDevicePointer(), &pmeBsplineModuliY.getDevicePointer(), &pmeBsplineModuliZ.getDevicePointer(),
                        recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
                cu.executeKernel(pmeEvalEnergyKernel, computeEnergyArgs, gridSizeX*gridSizeY*gridSizeZ);
            }

            void* convolutionArgs[] = {&pmeGrid2.getDevicePointer(), &pmeBsplineModuliX.getDevicePointer(),
                    &pmeBsplineModuliY.getDevicePointer(), &pmeBsplineModuliZ.getDevicePointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
            cu.executeKernel(pmeConvolutionKernel, convolutionArgs, gridSizeX*gridSizeY*gridSizeZ, 256);

            fft->execFFT(false);

            void* interpolateArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &pmeGrid1.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                    &charges.getDevicePointer(), &subsets.getDevicePointer(), &sliceLambdas.getDevicePointer()};
            cu.executeKernel(pmeInterpolateForceKernel, interpolateArgs, cu.getNumAtoms(), 128);
        }

        if (doLJPME && hasLJ) {
            void* gridIndexArgs[] = {&cu.getPosq().getDevicePointer(), &pmeAtomGridIndex.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &subsets.getDevicePointer()};
            cu.executeKernel(pmeDispersionGridIndexKernel, gridIndexArgs, cu.getNumAtoms());
            sort->sort(pmeAtomGridIndex);
            cu.clearBuffer(pmeGrid2);
            void* spreadArgs[] = {&cu.getPosq().getDevicePointer(), &pmeGrid2.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                    &sigmaEpsilon.getDevicePointer()};
            cu.executeKernel(pmeDispersionSpreadChargeKernel, spreadArgs, cu.getNumAtoms(), 128);

            void* finishSpreadArgs[] = {&pmeGrid2.getDevicePointer(), &pmeGrid1.getDevicePointer()};
            cu.executeKernel(pmeDispersionFinishSpreadChargeKernel, finishSpreadArgs, dispersionGridSizeX*dispersionGridSizeY*dispersionGridSizeZ, 256);

            dispersionFft->execFFT(true);

            if (includeEnergy || hasDerivatives) {
                void* computeEnergyArgs[] = {&pmeGrid2.getDevicePointer(), &ljpmeEnergyBuffer.getDevicePointer(),
                        &pmeDispersionBsplineModuliX.getDevicePointer(), &pmeDispersionBsplineModuliY.getDevicePointer(), &pmeDispersionBsplineModuliZ.getDevicePointer(),
                        recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
                cu.executeKernel(pmeEvalDispersionEnergyKernel, computeEnergyArgs, dispersionGridSizeX*dispersionGridSizeY*dispersionGridSizeZ);
            }

            void* convolutionArgs[] = {&pmeGrid2.getDevicePointer(), &pmeDispersionBsplineModuliX.getDevicePointer(),
                    &pmeDispersionBsplineModuliY.getDevicePointer(), &pmeDispersionBsplineModuliZ.getDevicePointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
            cu.executeKernel(pmeDispersionConvolutionKernel, convolutionArgs, dispersionGridSizeX*dispersionGridSizeY*dispersionGridSizeZ, 256);

            dispersionFft->execFFT(false);

            void* interpolateArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &pmeGrid1.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                    &sigmaEpsilon.getDevicePointer(), &subsets.getDevicePointer(), &sliceLambdas.getDevicePointer()};
            cu.executeKernel(pmeInterpolateDispersionForceKernel, interpolateArgs, cu.getNumAtoms(), 128);
        }
        if (usePmeStream) {
            cuEventRecord(pmeSyncEvent, pmeStream);
            cu.restoreDefaultStream();
        }
    }
    if (!hasOffsets && includeReciprocal) {
        map<string, double>& energyParamDerivs = cu.getEnergyParamDerivWorkspace();
        for (int i = 0; i < numSubsets; i++) {
            ScalingParameterInfo info = sliceScalingParams[sliceIndex(i, i)];
            if (info.hasDerivativeCoulomb)
                energyParamDerivs[info.nameCoulomb] += subsetSelfEnergy[i].x;
            if (doLJPME && info.hasDerivativeLJ)
                energyParamDerivs[info.nameLJ] += subsetSelfEnergy[i].y;
        }
    }
    return energy;
}

void CudaCalcSlicedNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const SlicedNonbondedForce& force) {
    // Make sure the new parameters are acceptable.

    ContextSelector selector(cu);
    if (force.getNumParticles() != cu.getNumAtoms())
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
        if (chargeProd != 0.0 || epsilon != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end())
            exceptions.push_back(i);
    }
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions != exceptionAtoms.size())
        throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");

    // Record the per-particle parameters.

    vector<float4> baseParticleParamVec(cu.getPaddedNumAtoms(), make_float4(0, 0, 0, 0));
    const vector<int>& order = cu.getAtomIndex();
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        baseParticleParamVec[i] = make_float4(charge, sigma, epsilon, 0);
    }
    baseParticleParams.upload(baseParticleParamVec);

    // Record the exceptions.

    if (numExceptions > 0) {
        vector<float4> baseExceptionParamsVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            int particle1, particle2;
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], particle1, particle2, chargeProd, sigma, epsilon);
            if (make_pair(particle1, particle2) != exceptionAtoms[i])
                throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");
            baseExceptionParamsVec[i] = make_float4(chargeProd, sigma, epsilon, 0);
        }
        baseExceptionParams.upload(baseExceptionParamsVec);
    }

    // Compute other values.

    ewaldSelfEnergy = 0.0;
    subsetSelfEnergy.assign(numSubsets, make_double2(0, 0));
    if (nonbondedMethod == Ewald || nonbondedMethod == PME || nonbondedMethod == LJPME) {
        if (cu.getContextIndex() == 0) {
            for (int i = 0; i < force.getNumParticles(); i++) {
                subsetSelfEnergy[subsetsVec[i]].x -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
                if (doLJPME)
                    subsetSelfEnergy[subsetsVec[i]].y += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
            }
            for (int i = 0; i < force.getNumSubsets(); i++) {
                int slice = sliceIndex(i, i);
                ewaldSelfEnergy += sliceLambdasVec[slice].x*subsetSelfEnergy[i].x + sliceLambdasVec[slice].y*subsetSelfEnergy[i].y;
            }
        }
    }
    if (force.getUseDispersionCorrection() && cu.getContextIndex() == 0 && (nonbondedMethod == CutoffPeriodic || nonbondedMethod == Ewald || nonbondedMethod == PME))
        dispersionCoefficients = SlicedNonbondedForceImpl::calcDispersionCorrections(context.getSystem(), force);
    cu.invalidateMolecules();
    recomputeParams = true;
}

void CudaCalcSlicedNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != PME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = this->alpha;
    nx = gridSizeX;
    ny = gridSizeY;
    nz = gridSizeZ;
}

void CudaCalcSlicedNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!doLJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = this->dispersionAlpha;
    nx = dispersionGridSizeX;
    ny = dispersionGridSizeY;
    nz = dispersionGridSizeZ;
}
