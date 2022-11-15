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

#include "CudaPmeSlicingKernels.h"
#include "CudaPmeSlicingKernelSources.h"
#include "CommonPmeSlicingKernelSources.h"
#include "SlicedPmeForce.h"
#include "internal/SlicedPmeForceImpl.h"
#include "openmm/NonbondedForce.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/common/ContextSelector.h"
#include <cstring>
#include <algorithm>
#include <iostream>

#define CHECK_RESULT(result, prefix) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        throw OpenMMException(m.str());\
    }

using namespace PmeSlicing;
using namespace OpenMM;
using namespace std;

class CudaCalcSlicedPmeForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const SlicedPmeForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1 = force.getParticleCharge(particle1);
        double charge2 = force.getParticleCharge(particle2);
        return (charge1 == charge2);
    }
    int getNumParticleGroups() {
        return force.getNumExceptions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        double chargeProd;
        force.getExceptionParameters(index, particle1, particle2, chargeProd);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2;
        double chargeProd1, chargeProd2;
        force.getExceptionParameters(group1, particle1, particle2, chargeProd1);
        force.getExceptionParameters(group2, particle1, particle2, chargeProd2);
        return (chargeProd1 == chargeProd2);
    }
private:
    const SlicedPmeForce& force;
};

class CudaCalcSlicedPmeForceKernel::SyncStreamPreComputation : public CudaContext::ForcePreComputation {
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

class CudaCalcSlicedPmeForceKernel::SyncStreamPostComputation : public CudaContext::ForcePostComputation {
public:
    SyncStreamPostComputation(CudaContext& cu, CUevent event, CUfunction addEnergyKernel, CudaArray& pmeEnergyBuffer, int forceGroup) : cu(cu), event(event),
            addEnergyKernel(addEnergyKernel), pmeEnergyBuffer(pmeEnergyBuffer), forceGroup(forceGroup) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            cuStreamWaitEvent(cu.getCurrentStream(), event, 0);
            if (includeEnergy) {
                int bufferSize = pmeEnergyBuffer.getSize();
                void* args[] = {&pmeEnergyBuffer.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(), &bufferSize};
                cu.executeKernel(addEnergyKernel, args, bufferSize);
            }
        }
        return 0.0;
    }
private:
    CudaContext& cu;
    CUevent event;
    CUfunction addEnergyKernel;
    CudaArray& pmeEnergyBuffer;
    int forceGroup;
};

CudaCalcSlicedPmeForceKernel::~CudaCalcSlicedPmeForceKernel() {
    ContextSelector selector(cu);
    if (sort != NULL)
        delete sort;
    if (fft != NULL)
        delete fft;
    if (hasInitializedFFT) {
        if (usePmeStream) {
            cuStreamDestroy(pmeStream);
            cuEventDestroy(pmeSyncEvent);
            cuEventDestroy(paramsSyncEvent);
        }
    }
}

void CudaCalcSlicedPmeForceKernel::initialize(const System& system, const SlicedPmeForce& force) {
    ContextSelector selector(cu);
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "pme"+cu.intToString(forceIndex)+"_";

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionChargeOffsets(); i++) {
        string param;
        int exception;
        double charge;
        force.getExceptionChargeOffset(i, param, exception, charge);
        exceptionsWithOffsets.insert(exception);
    }
    vector<pair<int, int> > exclusions;
    vector<int> exceptions;
    map<int, int> exceptionIndex;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd;
        force.getExceptionParameters(i, particle1, particle2, chargeProd);
        exclusions.push_back(pair<int, int>(particle1, particle2));
        if (chargeProd != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end()) {
            exceptionIndex[i] = exceptions.size();
            exceptions.push_back(i);
        }
    }

    // Initialize nonbonded interactions.

    int numParticles = force.getNumParticles();
    numSubsets = force.getNumSubsets();
    numSlices = numSubsets*(numSubsets + 1)/2;
    vector<float> baseParticleChargeVec(cu.getPaddedNumAtoms(), 0.0);
    vector<vector<int> > exclusionList(numParticles);
    for (int i = 0; i < numParticles; i++) {
        baseParticleChargeVec[i] = force.getParticleCharge(i);
        exclusionList[i].push_back(i);
    }
    for (auto exclusion : exclusions) {
        exclusionList[exclusion.first].push_back(exclusion.second);
        exclusionList[exclusion.second].push_back(exclusion.first);
    }
    usePosqCharges = cu.requestPosqCharges();
    size_t sizeOfReal = cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float);
    size_t sizeOfMixed = (cu.getUseMixedPrecision() ? sizeof(double) : sizeOfReal);

    alpha = 0;
    ewaldSelfEnergy = 0.0;
    subsetSelfEnergy.resize(numSubsets, 0.0);
    map<string, string> paramsDefines;
    paramsDefines["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
    hasOffsets = (force.getNumParticleChargeOffsets() > 0 || force.getNumExceptionChargeOffsets() > 0);
    if (hasOffsets)
        paramsDefines["HAS_OFFSETS"] = "1";
    if (force.getNumParticleChargeOffsets() > 0)
        paramsDefines["HAS_PARTICLE_OFFSETS"] = "1";
    if (force.getNumExceptionChargeOffsets() > 0)
        paramsDefines["HAS_EXCEPTION_OFFSETS"] = "1";
    if (usePosqCharges)
        paramsDefines["USE_POSQ_CHARGES"] = "1";

    // Initialize subsets.

    subsets.initialize<int>(cu, cu.getPaddedNumAtoms(), "subsets");
    vector<int> subsetVec(cu.getPaddedNumAtoms());
    for (int i = 0; i < numParticles; i++)
        subsetVec[i] = force.getParticleSubset(i);
    subsets.upload(subsetVec);

    // Identify requested derivatives.

    set<string> requestedDerivs;
    for (int index = 0; index < force.getNumSwitchingParameterDerivatives(); index++) {
        string param = force.getSwitchingParameterDerivativeName(index);
        requestedDerivs.insert(param);
        cu.addEnergyParameterDerivative(param);
    }
    bool hasDerivatives = requestedDerivs.size() > 0;
    const vector<string>& allDerivs = cu.getEnergyParamDerivNames();

    // Initialize switching parameters and derivative indices.

    sliceSwitchParamIndices.resize(numSlices, -1);
    vector<int> sliceDerivIndexVec(numSlices, -1);
    for (int i = 0; i < force.getNumSwitchingParameters(); i++) {
        string param;
        int s1, s2;
        force.getSwitchingParameter(i, param, s1, s2);
        int index = find(switchParamNames.begin(), switchParamNames.end(), param) - switchParamNames.begin();
        if (index == switchParamNames.size()) {
            switchParamNames.push_back(param);
            switchParamValues.push_back(1.0);
        }
        int slice = s1 > s2 ? s1*(s1+1)/2+s2 : s2*(s2+1)/2+s1;
        sliceSwitchParamIndices[slice] = index;
        if (requestedDerivs.find(param) != requestedDerivs.end())
            sliceDerivIndexVec[slice] = find(allDerivs.begin(), allDerivs.end(), param) - allDerivs.begin();
    }
    sliceLambdaVec.resize(numSlices, 1.0);
    sliceLambda.initialize(cu, numSlices, sizeOfReal, "sliceLambda");
    if (cu.getUseDoublePrecision())
        sliceLambda.upload(sliceLambdaVec);
    else
        sliceLambda.upload(floatVector(sliceLambdaVec));
    sliceDerivIndices.initialize<int>(cu, numSlices, "sliceDerivIndices");
    sliceDerivIndices.upload(sliceDerivIndexVec);

    // Compute the PME parameters.

    int cufftVersion;
    cufftGetVersion(&cufftVersion);
    useCudaFFT = force.getUseCudaFFT() && (cufftVersion >= 7050); // There was a critical bug in version 7.0

    SlicedPmeForceImpl::calcPMEParameters(system, force, alpha, gridSizeX, gridSizeY, gridSizeZ, false);

    gridSizeX = CudaFFT3D::findLegalDimension(gridSizeX);
    gridSizeY = CudaFFT3D::findLegalDimension(gridSizeY);
    gridSizeZ = CudaFFT3D::findLegalDimension(gridSizeZ);
    int roundedZSize = PmeOrder*(int) ceil(gridSizeZ/(double) PmeOrder);

    if (cu.getContextIndex() == 0) {
        paramsDefines["INCLUDE_EWALD"] = "1";
        for (int i = 0; i < numParticles; i++)
            subsetSelfEnergy[subsetVec[i]] += baseParticleChargeVec[i]*baseParticleChargeVec[i];
        for (int j = 0; j < numSubsets; j++) {
            subsetSelfEnergy[j] *= -ONE_4PI_EPS0*alpha/sqrt(M_PI);
            ewaldSelfEnergy += subsetSelfEnergy[j];
        }
        char deviceName[100];
        cuDeviceGetName(deviceName, 100, cu.getDevice());
        usePmeStream = (!cu.getPlatformData().disablePmeStream && !cu.getPlatformData().useCpuPme && string(deviceName) != "GeForce GTX 980"); // Using a separate stream is slower on GTX 980
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
        pmeDefines["ROUNDED_Z_SIZE"] = cu.intToString(roundedZSize);
        pmeDefines["EPSILON_FACTOR"] = cu.doubleToString(sqrt(ONE_4PI_EPS0));
        pmeDefines["M_PI"] = cu.doubleToString(M_PI);
        pmeDefines["EWALD_SELF_ENERGY_SCALE"] = cu.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
        pmeDefines["USE_POSQ_CHARGES"] = usePosqCharges ? "1" : "0";
        if (cu.getUseDoublePrecision() || cu.getPlatformData().deterministicForces)
            pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";
        if (usePmeStream)
            pmeDefines["USE_PME_STREAM"] = "1";
        map<string, string> replacements;
        replacements["CHARGE"] = (usePosqCharges ? "pos.w" : "charges[atom]");
        CUmodule module = cu.createModule(CudaPmeSlicingKernelSources::vectorOps+
                                          cu.replaceStrings(CommonPmeSlicingKernelSources::slicedPme, replacements), pmeDefines);

        pmeGridIndexKernel = cu.getKernel(module, "findAtomGridIndex");
        pmeSpreadChargeKernel = cu.getKernel(module, "gridSpreadCharge");
        pmeConvolutionKernel = cu.getKernel(module, "reciprocalConvolution");
        pmeInterpolateForceKernel = cu.getKernel(module, "gridInterpolateForce");
        pmeEvalEnergyKernel = cu.getKernel(module, "gridEvaluateEnergy");
        pmeFinishSpreadChargeKernel = cu.getKernel(module, "finishSpreadCharge");
        if (hasOffsets)
            pmeAddSelfEnergyKernel = cu.getKernel(module, "addSelfEnergy");
        cuFuncSetCacheConfig(pmeSpreadChargeKernel, CU_FUNC_CACHE_PREFER_SHARED);
        cuFuncSetCacheConfig(pmeInterpolateForceKernel, CU_FUNC_CACHE_PREFER_L1);

        // Create required data structures.

        int gridElements = gridSizeX*gridSizeY*roundedZSize*numSubsets;
        pmeGrid1.initialize(cu, gridElements, 2*sizeOfReal, "pmeGrid1");
        pmeGrid2.initialize(cu, gridElements, 2*sizeOfReal, "pmeGrid2");
        cu.addAutoclearBuffer(pmeGrid2);
        pmeBsplineModuliX.initialize(cu, gridSizeX, sizeOfReal, "pmeBsplineModuliX");
        pmeBsplineModuliY.initialize(cu, gridSizeY, sizeOfReal, "pmeBsplineModuliY");
        pmeBsplineModuliZ.initialize(cu, gridSizeZ, sizeOfReal, "pmeBsplineModuliZ");
        pmeAtomGridIndex.initialize<int2>(cu, numParticles, "pmeAtomGridIndex");
        if (usePmeStream) {
            int bufferSize = cu.getNumThreadBlocks()*CudaContext::ThreadBlockSize;
            pmeEnergyBuffer.initialize(cu, bufferSize, sizeOfMixed, "pmeEnergyBuffer");
            cu.clearBuffer(pmeEnergyBuffer);
        }
        sort = new CudaSort(cu, new SortTrait(), cu.getNumAtoms());

        // Prepare for doing PME on its own stream or not.

        int recipForceGroup = force.getReciprocalSpaceForceGroup();
        if (recipForceGroup < 0)
            recipForceGroup = force.getForceGroup();
        if (usePmeStream) {
            cuStreamCreate(&pmeStream, CU_STREAM_NON_BLOCKING);
            CHECK_RESULT(cuEventCreate(&pmeSyncEvent, CU_EVENT_DISABLE_TIMING), "Error creating event for NonbondedForce");
            CHECK_RESULT(cuEventCreate(&paramsSyncEvent, CU_EVENT_DISABLE_TIMING), "Error creating event for NonbondedForce");
            cu.addPreComputation(new SyncStreamPreComputation(cu, pmeStream, pmeSyncEvent, recipForceGroup));
            cu.addPostComputation(new SyncStreamPostComputation(cu, pmeSyncEvent, cu.getKernel(module, "addEnergy"), pmeEnergyBuffer, recipForceGroup));
        }
        else
            pmeStream = cu.getCurrentStream();

        if (useCudaFFT)
            fft = (CudaFFT3D*) new CudaCuFFT3D(cu, pmeStream, gridSizeX, gridSizeY, gridSizeZ, numSubsets, true, pmeGrid1, pmeGrid2);
        else
            fft = (CudaFFT3D*) new CudaVkFFT3D(cu, pmeStream, gridSizeX, gridSizeY, gridSizeZ, numSubsets, true, pmeGrid1, pmeGrid2);
        hasInitializedFFT = true;

        // Initialize the b-spline moduli.

        int xsize, ysize, zsize;
        CudaArray *xmoduli, *ymoduli, *zmoduli;

        xsize = gridSizeX;
        ysize = gridSizeY;
        zsize = gridSizeZ;
        xmoduli = &pmeBsplineModuliX;
        ymoduli = &pmeBsplineModuliY;
        zmoduli = &pmeBsplineModuliZ;

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

    // Add the interaction to the default nonbonded kernel.

    charges.initialize(cu, cu.getPaddedNumAtoms(), sizeOfReal, "charges");
    baseParticleCharges.initialize<float>(cu, cu.getPaddedNumAtoms(), "baseParticleCharges");
    baseParticleCharges.upload(baseParticleChargeVec);

    if (force.getIncludeDirectSpace()) {
        CudaNonbondedUtilities* nb = &cu.getNonbondedUtilities();
        map<string, string> replacements;
        replacements["NUM_REQUESTED_DERIVS"] = cu.intToString(requestedDerivs.size());
        replacements["NUM_ALL_DERIVS"] = cu.intToString(allDerivs.size());
        replacements["LAMBDA"] = prefix+"lambda";
        replacements["EWALD_ALPHA"] = cu.doubleToString(alpha);
        replacements["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        replacements["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
        replacements["CHARGE1"] = usePosqCharges ? "posq1.w" : prefix+"charge1";
        replacements["CHARGE2"] = usePosqCharges ? "posq2.w" : prefix+"charge2";
        replacements["SUBSET1"] = prefix+"subset1";
        replacements["SUBSET2"] = prefix+"subset2";
        replacements["DERIV_INDICES"] = prefix+"derivIndices";
        if (!usePosqCharges)
            nb->addParameter(ComputeParameterInfo(charges, prefix+"charge", "real", 1));
        nb->addParameter(ComputeParameterInfo(subsets, prefix+"subset", "int", 1));
        nb->addArgument(ComputeParameterInfo(sliceLambda, prefix+"lambda", "real", 1));
        if (hasDerivatives) {
            for (string param : requestedDerivs)
                nb->addEnergyParameterDerivative(param);
            nb->addArgument(ComputeParameterInfo(sliceDerivIndices, prefix+"derivIndices", "int", 1));
        }
        string source = cu.replaceStrings(CommonPmeSlicingKernelSources::coulomb, replacements);
        nb->addInteraction(true, true, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup(), true);
    }

    // Add code to subtract off the reciprocal part of excluded interactions.

    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumExceptions()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumExceptions()/numContexts;
    int numExclusions = endIndex-startIndex;
    hasExclusions = numExclusions > 0;
    if (hasExclusions) {
        exclusionAtoms.initialize<int2>(cu, numExclusions, "exclusionAtoms");
        exclusionSlices.initialize<int>(cu, numExclusions, "exclusionSlices");
        exclusionChargeProds.initialize<float>(cu, numExclusions, "exclusionChargeProds");
        vector<int2> exclusionAtomsVec(numExclusions);
        vector<int> exclusionSlicesVec(numExclusions);
        for (int k = 0; k < numExclusions; k++) {
            int atom1 = exclusions[k+startIndex].first;
            int atom2 = exclusions[k+startIndex].second;
            exclusionAtomsVec[k] = make_int2(atom1, atom2);
            int i = subsetVec[atom1];
            int j = subsetVec[atom2];
            exclusionSlicesVec[k] = i > j ? i*(i+1)/2+j : j*(j+1)/2+i;
        }
        exclusionAtoms.upload(exclusionAtomsVec);
        exclusionSlices.upload(exclusionSlicesVec);
    }

    // Initialize the exceptions.

    startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0) {
        paramsDefines["HAS_EXCEPTIONS"] = "1";
        exceptionPairs.resize(numExceptions);
        exceptionAtoms.initialize<int2>(cu, numExceptions, "exceptionAtoms");
        exceptionSlices.initialize<int>(cu, numExceptions, "exceptionSlices");
        exceptionChargeProds.initialize<float>(cu, numExceptions, "exceptionChargeProds");
        baseExceptionChargeProds.initialize<float>(cu, numExceptions, "baseExceptionChargeProds");
        vector<int2> exceptionAtomsVec(numExceptions);
        vector<int> exceptionSlicesVec(numExceptions);
        vector<float> baseExceptionChargeProdsVec(numExceptions);
        for (int k = 0; k < numExceptions; k++) {
            double chargeProd;
            int atom1, atom2;
            force.getExceptionParameters(exceptions[startIndex+k], atom1, atom2, chargeProd);
            exceptionPairs[k] = (vector<int>) {atom1, atom2};
            baseExceptionChargeProdsVec[k] = chargeProd;
            exceptionAtomsVec[k] = make_int2(atom1, atom2);
            int i = subsetVec[atom1];
            int j = subsetVec[atom2];
            exceptionSlicesVec[k] = i > j ? i*(i+1)/2+j : j*(j+1)/2+i;
        }
        exceptionAtoms.upload(exceptionAtomsVec);
        exceptionSlices.upload(exceptionSlicesVec);
        baseExceptionChargeProds.upload(baseExceptionChargeProdsVec);
        CudaBondedUtilities* bonded = &cu.getBondedUtilities();
        map<string, string> replacements;
        replacements["NUM_ALL_DERIVS"] = cu.intToString(allDerivs.size());
        replacements["HAS_DERIVATIVES"] = hasDerivatives ? "1" : "0";
        replacements["APPLY_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
        replacements["CHARGE_PRODS"] = bonded->addArgument(exceptionChargeProds.getDevicePointer(), "float");
        replacements["SLICES"] = bonded->addArgument(exceptionSlices.getDevicePointer(), "int");
        replacements["LAMBDAS"] = bonded->addArgument(sliceLambda.getDevicePointer(), "real");
        if (hasDerivatives) {
            for (string param : requestedDerivs)
                bonded->addEnergyParameterDerivative(param);
            replacements["DERIV_INDICES"] = bonded->addArgument(sliceDerivIndices, "int");
        }
        if (force.getIncludeDirectSpace())
            bonded->addInteraction(exceptionPairs, cu.replaceStrings(CommonPmeSlicingKernelSources::slicedPmeExceptions, replacements), force.getForceGroup());
    }

    if (hasExclusions) {
        map<string, string> bondDefines;
        bondDefines["NUM_EXCLUSIONS"] = cu.intToString(numExclusions);
        bondDefines["NUM_EXCEPTIONS"] = cu.intToString(numExceptions);
        bondDefines["NUM_SLICES"] = cu.intToString(numSlices);
        bondDefines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        bondDefines["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        bondDefines["USE_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
        bondDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        CUmodule bondModule = cu.createModule(CudaPmeSlicingKernelSources::vectorOps+
                                              CommonPmeSlicingKernelSources::slicedPmeBonds, bondDefines);
        computeBondsKernel = cu.getKernel(bondModule, "computeBonds");
    }

    // Initialize charge offsets.

    vector<vector<float2> > particleOffsetVec(force.getNumParticles());
    vector<vector<float2> > exceptionOffsetVec(numExceptions);
    for (int i = 0; i < force.getNumParticleChargeOffsets(); i++) {
        string param;
        int particle;
        double charge;
        force.getParticleChargeOffset(i, param, particle, charge);
        auto paramPos = find(paramNames.begin(), paramNames.end(), param);
        int paramIndex;
        if (paramPos == paramNames.end()) {
            paramIndex = paramNames.size();
            paramNames.push_back(param);
        }
        else
            paramIndex = paramPos-paramNames.begin();
        particleOffsetVec[particle].push_back(make_float2(charge, paramIndex));
    }
    for (int i = 0; i < force.getNumExceptionChargeOffsets(); i++) {
        string param;
        int exception;
        double charge;
        force.getExceptionChargeOffset(i, param, exception, charge);
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
        exceptionOffsetVec[index-startIndex].push_back(make_float2(charge, paramIndex));
    }
    paramValues.resize(paramNames.size(), 0.0);
    particleParamOffsets.initialize<float2>(cu, max(force.getNumParticleChargeOffsets(), 1), "particleParamOffsets");
    particleOffsetIndices.initialize<int>(cu, cu.getPaddedNumAtoms()+1, "particleOffsetIndices");
    vector<int> particleOffsetIndicesVec, exceptionOffsetIndicesVec;
    vector<float2> p, e;
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
    if (force.getNumParticleChargeOffsets() > 0) {
        particleParamOffsets.upload(p);
        particleOffsetIndices.upload(particleOffsetIndicesVec);
    }
    exceptionParamOffsets.initialize<float2>(cu, max((int) e.size(), 1), "exceptionParamOffsets");
    exceptionOffsetIndices.initialize<int>(cu, exceptionOffsetIndicesVec.size(), "exceptionOffsetIndices");
    if (e.size() > 0) {
        exceptionParamOffsets.upload(e);
        exceptionOffsetIndices.upload(exceptionOffsetIndicesVec);
    }
    globalParams.initialize(cu, max((int) paramValues.size(), 1), sizeOfReal, "globalParams");
    if (paramValues.size() > 0)
        globalParams.upload(paramValues, true);
    recomputeParams = true;

    // Initialize the kernel for updating parameters.

    CUmodule module = cu.createModule(CommonPmeSlicingKernelSources::slicedPmeParameters, paramsDefines);
    computeParamsKernel = cu.getKernel(module, "computeParameters");
    computeExclusionParamsKernel = cu.getKernel(module, "computeExclusionParameters");
    info = new ForceInfo(force);
    cu.addForce(info);
}

double CudaCalcSlicedPmeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    ContextSelector selector(cu);
    double energy = 0.0;

    // Update switching parameters if needed.

    bool switchParamChanged = false;
    for (int i = 0; i < switchParamNames.size(); i++) {
        double value = context.getParameter(switchParamNames[i]);
        if (value != switchParamValues[i]) {
            switchParamValues[i] = value;
            switchParamChanged = true;
        }
    }
    if (switchParamChanged) {
        for (int slice = 0; slice < numSlices; slice++) {
            int index = sliceSwitchParamIndices[slice];
            if (index != -1)
                sliceLambdaVec[slice] = switchParamValues[index];
        }
        ewaldSelfEnergy = 0.0;
        for (int j = 0; j < numSubsets; j++)
            ewaldSelfEnergy += sliceLambdaVec[j*(j+3)/2]*subsetSelfEnergy[j];
        if (cu.getUseDoublePrecision())
            sliceLambda.upload(sliceLambdaVec);
        else
            sliceLambda.upload(floatVector(sliceLambdaVec));
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
    if (recomputeParams) {
        int numAtoms = cu.getPaddedNumAtoms();
        vector<void*> paramsArgs = {&globalParams.getDevicePointer(), &numAtoms,
                &baseParticleCharges.getDevicePointer(), &cu.getPosq().getDevicePointer(), &charges.getDevicePointer(),
                &particleParamOffsets.getDevicePointer(), &particleOffsetIndices.getDevicePointer(),
                &subsets.getDevicePointer()};
        int numExceptions;
        if (exceptionChargeProds.isInitialized()) {
            numExceptions = exceptionChargeProds.getSize();
            paramsArgs.push_back(&numExceptions);
            paramsArgs.push_back(&baseExceptionChargeProds.getDevicePointer());
            paramsArgs.push_back(&exceptionChargeProds.getDevicePointer());
            paramsArgs.push_back(&exceptionParamOffsets.getDevicePointer());
            paramsArgs.push_back(&exceptionOffsetIndices.getDevicePointer());
            paramsArgs.push_back(&exceptionAtoms.getDevicePointer());
            paramsArgs.push_back(&exceptionSlices.getDevicePointer());
        }
        cu.executeKernel(computeParamsKernel, &paramsArgs[0], cu.getPaddedNumAtoms());
        if (exclusionChargeProds.isInitialized()) {
            int numExclusions = exclusionChargeProds.getSize();
            vector<void*> exclusionChargeProdsArgs = {&cu.getPosq().getDevicePointer(), &charges.getDevicePointer(),
                    &numExclusions, &exclusionAtoms.getDevicePointer(), &subsets.getDevicePointer(),
                    &exclusionSlices.getDevicePointer(), &exclusionChargeProds.getDevicePointer()};
            cu.executeKernel(computeExclusionParamsKernel, &exclusionChargeProdsArgs[0], numExclusions);
        }
        if (usePmeStream) {
            cuEventRecord(paramsSyncEvent, cu.getCurrentStream());
            cuStreamWaitEvent(pmeStream, paramsSyncEvent, 0);
        }
        ewaldSelfEnergy = 0.0;
        for (int j = 0; j < numSubsets; j++)
            ewaldSelfEnergy += sliceLambdaVec[j*(j+3)/2]*subsetSelfEnergy[j];
        recomputeParams = false;
    }

    // Do exclusion and exception calculations.

    if (hasExclusions && includeDirect) {
        void* computeBondsArgs[] = {
            &cu.getPosq().getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(), &cu.getForce().getDevicePointer(),
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &exclusionAtoms.getDevicePointer(), &exclusionSlices.getDevicePointer(), &exclusionChargeProds.getDevicePointer(),
            &exceptionAtoms.getDevicePointer(), &exceptionSlices.getDevicePointer(), &exceptionChargeProds.getDevicePointer(),
            &sliceLambda.getDevicePointer()
        };
        cu.executeKernel(computeBondsKernel, computeBondsArgs, exclusionChargeProds.getSize());
    }

    // Do reciprocal space calculations.

    if (pmeGrid1.isInitialized() && includeReciprocal) {
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

        void* gridIndexArgs[] = {&cu.getPosq().getDevicePointer(), &subsets.getDevicePointer(), &pmeAtomGridIndex.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeGridIndexKernel, gridIndexArgs, cu.getNumAtoms());

        sort->sort(pmeAtomGridIndex);

        void* spreadArgs[] = {&cu.getPosq().getDevicePointer(), &pmeGrid2.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                &charges.getDevicePointer()};
        cu.executeKernel(pmeSpreadChargeKernel, spreadArgs, cu.getNumAtoms(), 128);

        void* finishSpreadArgs[] = {&pmeGrid2.getDevicePointer(), &pmeGrid1.getDevicePointer()};
        cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, numSubsets*gridSizeX*gridSizeY*gridSizeZ, 256);

        fft->execFFT(true);

        if (includeEnergy) {
            void* computeEnergyArgs[] = {
                &pmeGrid2.getDevicePointer(),
                usePmeStream ? &pmeEnergyBuffer.getDevicePointer() : &cu.getEnergyBuffer().getDevicePointer(),
                &sliceLambda.getDevicePointer(),
                &pmeBsplineModuliX.getDevicePointer(),
                &pmeBsplineModuliY.getDevicePointer(),
                &pmeBsplineModuliZ.getDevicePointer(),
                recipBoxVectorPointer[0],
                recipBoxVectorPointer[1],
                recipBoxVectorPointer[2]
            };
            cu.executeKernel(pmeEvalEnergyKernel, computeEnergyArgs, gridSizeX*gridSizeY*gridSizeZ);

            if (hasOffsets) {
                vector<void*> addSelfEnergyArgs = {
                    usePmeStream ? &pmeEnergyBuffer.getDevicePointer() : &cu.getEnergyBuffer().getDevicePointer(),
                    &cu.getPosq().getDevicePointer(),
                    &charges.getDevicePointer(),
                    &subsets.getDevicePointer(),
                    &sliceLambda.getDevicePointer()
                };
                cu.executeKernel(pmeAddSelfEnergyKernel, &addSelfEnergyArgs[0], cu.getPaddedNumAtoms());
            }
            else
                energy = ewaldSelfEnergy;
        }

        void* convolutionArgs[] = {&pmeGrid2.getDevicePointer(), &pmeBsplineModuliX.getDevicePointer(),
                &pmeBsplineModuliY.getDevicePointer(), &pmeBsplineModuliZ.getDevicePointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeConvolutionKernel, convolutionArgs, gridSizeX*gridSizeY*(gridSizeZ/2+1), 256);

        fft->execFFT(false);

        void* interpolateArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &pmeGrid1.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                &charges.getDevicePointer(), &subsets.getDevicePointer(), &sliceLambda.getDevicePointer()};
        cu.executeKernel(pmeInterpolateForceKernel, interpolateArgs, cu.getNumAtoms(), 128);

        if (usePmeStream) {
            cuEventRecord(pmeSyncEvent, pmeStream);
            cu.restoreDefaultStream();
        }
    }

    return energy;
}

void CudaCalcSlicedPmeForceKernel::copyParametersToContext(ContextImpl& context, const SlicedPmeForce& force) {
    // Make sure the new parameters are acceptable.

    ContextSelector selector(cu);
    if (force.getNumParticles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionChargeOffsets(); i++) {
        string param;
        int exception;
        double charge;
        force.getExceptionChargeOffset(i, param, exception, charge);
        exceptionsWithOffsets.insert(exception);
    }
    vector<int> exceptions;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd;
        force.getExceptionParameters(i, particle1, particle2, chargeProd);
        if (chargeProd != 0.0 || exceptionsWithOffsets.find(i) != exceptionsWithOffsets.end())
            exceptions.push_back(i);
    }
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions != exceptionPairs.size())
        throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");

    // Record the per-particle parameters.

    vector<float> baseParticleChargeVec(cu.getPaddedNumAtoms(), 0.0);
    vector<int> subsetVec(cu.getPaddedNumAtoms());
    const vector<int>& order = cu.getAtomIndex();
    for (int i = 0; i < force.getNumParticles(); i++) {
        baseParticleChargeVec[i] = force.getParticleCharge(i);
        subsetVec[i] = force.getParticleSubset(i);
    }
    baseParticleCharges.upload(baseParticleChargeVec);
    subsets.upload(subsetVec);

    // Record the exceptions.

    if (numExceptions > 0) {
        vector<float> baseExceptionChargeProdsVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            int particle1, particle2;
            double chargeProd;
            force.getExceptionParameters(exceptions[startIndex+i], particle1, particle2, chargeProd);
            if (exceptionPairs[i][0] != particle1 || exceptionPairs[i][1] != particle2)
                throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");
            baseExceptionChargeProdsVec[i] = chargeProd;
        }
        baseExceptionChargeProds.upload(baseExceptionChargeProdsVec);
    }

    // Compute other values.

    ewaldSelfEnergy = 0.0;
    subsetSelfEnergy.assign(numSubsets, 0.0);
    if (cu.getContextIndex() == 0) {
        for (int i = 0; i < cu.getNumAtoms(); i++)
            subsetSelfEnergy[subsetVec[i]] += baseParticleChargeVec[i]*baseParticleChargeVec[i];
        for (int j = 0; j < numSubsets; j++)
            subsetSelfEnergy[j] *= -ONE_4PI_EPS0*alpha/sqrt(M_PI);
    }
    cu.invalidateMolecules();
    recomputeParams = true;
}

void CudaCalcSlicedPmeForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (cu.getPlatformData().useCpuPme)
        cpuPme.getAs<CalcPmeReciprocalForceKernel>().getPMEParameters(alpha, nx, ny, nz);
    else {
        alpha = this->alpha;
        nx = gridSizeX;
        ny = gridSizeY;
        nz = gridSizeZ;
    }
}

