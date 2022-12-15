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
#include "SlicedNonbondedForce.h"
#include "internal/SlicedPmeForceImpl.h"
#include "internal/SlicedNonbondedForceImpl.h"
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
        int subset1 = force.getParticleSubset(particle1);
        int subset2 = force.getParticleSubset(particle2);
        return (charge1 == charge2 && subset1 == subset2);
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
        int particle1, particle2, i, j, slice1, slice2;
        double chargeProd1, chargeProd2;
        force.getExceptionParameters(group1, particle1, particle2, chargeProd1);
        i = force.getParticleSubset(particle1);
        j = force.getParticleSubset(particle2);
        slice1 = i > j ? i*(i+1)/2+j : j*(j+1)/2+i;
        force.getExceptionParameters(group2, particle1, particle2, chargeProd2);
        i = force.getParticleSubset(particle1);
        j = force.getParticleSubset(particle2);
        slice2 = i > j ? i*(i+1)/2+j : j*(j+1)/2+i;
        return (chargeProd1 == chargeProd2 && slice1 == slice2);
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

class CudaCalcSlicedPmeForceKernel::AddEnergyPostComputation : public CudaContext::ForcePostComputation {
public:
    AddEnergyPostComputation(CudaContext& cu, int forceGroup) : cu(cu), forceGroup(forceGroup), initialized(false) { }
    void initialize(CudaArray& pmeEnergyBuffer, CudaArray& sliceLambda, vector<string> requestedDerivs, CudaArray& sliceDerivIndices) {
        int numSlices = sliceDerivIndices.getSize();
        hasDerivatives = requestedDerivs.size() > 0;
        stringstream code;
        if (hasDerivatives) {
            vector<int> sliceDerivIndexVec(numSlices);
            sliceDerivIndices.download(sliceDerivIndexVec);
            const vector<string>& allDerivs = cu.getEnergyParamDerivNames();
            for (int index = 0; index < requestedDerivs.size(); index++) {
                string param = requestedDerivs[index];
                int derivIndex = find(allDerivs.begin(), allDerivs.end(), param) - allDerivs.begin();
                code<<"energyParamDerivs[index*"<<allDerivs.size()<<"+"<<derivIndex<<"] +=";
                int position = 0;
                for (int slice = 0; slice < numSlices; slice++)
                    if (sliceDerivIndexVec[slice] == index)
                        code<<(position++ > 0 ? " + " : " ")<<"sliceEnergy["<<slice<<"]";
                code<<";"<<endl;
            }
        }
        map<string, string> replacements, defines;
        replacements["UPDATE_DERIVATIVE_BUFFER"] = code.str();
        replacements["NUM_SLICES"] = cu.intToString(numSlices);
        string source = cu.replaceStrings(CommonPmeSlicingKernelSources::slicedPmeAddEnergy, replacements);
        CUmodule module = cu.createModule(CudaPmeSlicingKernelSources::vectorOps + source, defines);
        addEnergyKernel = cu.getKernel(module, "addEnergy");
        bufferSize = pmeEnergyBuffer.getSize()/numSlices;
        arguments = {&pmeEnergyBuffer.getDevicePointer(),
                     &cu.getEnergyBuffer().getDevicePointer(),
                     &cu.getEnergyParamDerivBuffer().getDevicePointer(),
                     &sliceLambda.getDevicePointer(),
                     &bufferSize};
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
    bool hasDerivatives;
    int bufferSize;
    int forceGroup;
    vector<void*> arguments;
    bool initialized;
};

CudaCalcSlicedPmeForceKernel::CudaCalcSlicedPmeForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) :
        CalcSlicedPmeForceKernel(name, platform), cu(cu), hasInitializedFFT(false), sort(NULL), fft(NULL), usePmeStream(false) {
    stringstream code;
    if (Platform::getOpenMMVersion().at(0) == '7') {
        code<<"__device__ inline long long realToFixedPoint(real x) {"<<endl;
        code<<"return static_cast<long long>(x * 0x100000000);"<<endl;
        code<<"}"<<endl;
    }
    realToFixedPoint = code.str();
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

    for (int index = 0; index < force.getNumSwitchingParameterDerivatives(); index++) {
        string param = force.getSwitchingParameterDerivativeName(index);
        requestedDerivs.push_back(param);
        cu.addEnergyParameterDerivative(param);
    }
    hasDerivatives = requestedDerivs.size() > 0;
    const vector<string>& allDerivs = cu.getEnergyParamDerivNames();
    int numAllDerivs = allDerivs.size();

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
        index = find(requestedDerivs.begin(), requestedDerivs.end(), param) - requestedDerivs.begin();
        if (index < requestedDerivs.size())
            sliceDerivIndexVec[slice] = index;
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
        pmeDefines["HAS_DERIVATIVES"] = hasDerivatives ? "1" : "0";
        pmeDefines["NUM_ALL_DERIVS"] = cu.intToString(numAllDerivs);
        if (cu.getUseDoublePrecision() || cu.getPlatformData().deterministicForces)
            pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";
        if (usePmeStream)
            pmeDefines["USE_PME_STREAM"] = "1";
        map<string, string> replacements;
        replacements["CHARGE"] = (usePosqCharges ? "pos.w" : "charges[atom]");
        CUmodule module = cu.createModule(realToFixedPoint+CudaPmeSlicingKernelSources::vectorOps+
                                          cu.replaceStrings(CommonPmeSlicingKernelSources::slicedPme, replacements), pmeDefines);

        pmeGridIndexKernel = cu.getKernel(module, "findAtomGridIndex");
        pmeSpreadChargeKernel = cu.getKernel(module, "gridSpreadCharge");
        pmeConvolutionKernel = cu.getKernel(module, "reciprocalConvolution");
        pmeInterpolateForceKernel = cu.getKernel(module, "gridInterpolateForce");
        pmeEvalEnergyKernel = cu.getKernel(module, "gridEvaluateEnergy");
        pmeFinishSpreadChargeKernel = cu.getKernel(module, "finishSpreadCharge");
        if (hasOffsets || hasDerivatives)
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
        int bufferSize = cu.getNumThreadBlocks()*CudaContext::ThreadBlockSize;
        pmeEnergyBuffer.initialize(cu, numSlices*bufferSize, sizeOfMixed, "pmeEnergyBuffer");
        cu.clearBuffer(pmeEnergyBuffer);
        sort = new CudaSort(cu, new SortTrait(), cu.getNumAtoms());

        // Prepare for doing PME on its own stream or not.

        int recipForceGroup = force.getReciprocalSpaceForceGroup();
        if (recipForceGroup < 0)
            recipForceGroup = force.getForceGroup();
        if (usePmeStream) {
            cuStreamCreate(&pmeStream, CU_STREAM_NON_BLOCKING);
            CHECK_RESULT(cuEventCreate(&pmeSyncEvent, CU_EVENT_DISABLE_TIMING), "Error creating event for SlicedNonbondedForce");
            CHECK_RESULT(cuEventCreate(&paramsSyncEvent, CU_EVENT_DISABLE_TIMING), "Error creating event for SlicedNonbondedForce");
            cu.addPreComputation(new SyncStreamPreComputation(cu, pmeStream, pmeSyncEvent, recipForceGroup));
            cu.addPostComputation(new SyncStreamPostComputation(cu, pmeSyncEvent, recipForceGroup));
        }
        else
            pmeStream = cu.getCurrentStream();

        cu.addPostComputation(addEnergy = new AddEnergyPostComputation(cu, recipForceGroup));

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
        replacements["LAMBDA"] = prefix+"lambda";
        replacements["EWALD_ALPHA"] = cu.doubleToString(alpha);
        replacements["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        replacements["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
        replacements["CHARGE1"] = usePosqCharges ? "posq1.w" : prefix+"charge1";
        replacements["CHARGE2"] = usePosqCharges ? "posq2.w" : prefix+"charge2";
        replacements["SUBSET1"] = prefix+"subset1";
        replacements["SUBSET2"] = prefix+"subset2";
        if (!usePosqCharges)
            nb->addParameter(ComputeParameterInfo(charges, prefix+"charge", "real", 1));
        nb->addParameter(ComputeParameterInfo(subsets, prefix+"subset", "int", 1));
        nb->addArgument(ComputeParameterInfo(sliceLambda, prefix+"lambda", "real", 1));
        stringstream code;
        if (hasDerivatives) {
            string derivIndices = prefix+"derivIndices";
            nb->addArgument(ComputeParameterInfo(sliceDerivIndices, derivIndices, "int", 1));
            code<<"int which = "<<derivIndices<<"[slice];"<<endl;
            for (int i = 0; i < requestedDerivs.size(); i++) {
                string paramDeriv = nb->addEnergyParameterDerivative(requestedDerivs[i]);
                code<<paramDeriv<<" += which == "<<i<<" ? interactionScale*tempEnergy : 0;"<<endl;
            }
        }
        replacements["COMPUTE_DERIVATIVES"] = code.str();
        string source = cu.replaceStrings(CommonPmeSlicingKernelSources::coulomb, replacements);
        nb->addInteraction(true, true, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup(), true);
    }

    // Add code to subtract off the reciprocal part of excluded interactions.

    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumExceptions()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumExceptions()/numContexts;
    int numExclusions = endIndex-startIndex;
    if (numExclusions > 0 && force.getIncludeDirectSpace()) {
        exclusionPairs.resize(numExclusions);
        exclusionAtoms.initialize<int2>(cu, numExclusions, "exclusionAtoms");
        exclusionSlices.initialize<int>(cu, numExclusions, "exclusionSlices");
        exclusionChargeProds.initialize<float>(cu, numExclusions, "exclusionChargeProds");
        vector<int2> exclusionAtomsVec(numExclusions);
        vector<int> exclusionSlicesVec(numExclusions);
        for (int k = 0; k < numExclusions; k++) {
            int atom1 = exclusions[k+startIndex].first;
            int atom2 = exclusions[k+startIndex].second;
            exclusionAtomsVec[k] = make_int2(atom1, atom2);
            exclusionPairs[k] = (vector<int>) {atom1, atom2};
            int i = subsetVec[atom1];
            int j = subsetVec[atom2];
            exclusionSlicesVec[k] = i > j ? i*(i+1)/2+j : j*(j+1)/2+i;
        }
        exclusionAtoms.upload(exclusionAtomsVec);
        exclusionSlices.upload(exclusionSlicesVec);
        CudaBondedUtilities* bonded = &cu.getBondedUtilities();
        map<string, string> replacements;
        replacements["APPLY_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
        replacements["EWALD_ALPHA"] = cu.doubleToString(alpha);
        replacements["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        replacements["CHARGE_PRODS"] = bonded->addArgument(exclusionChargeProds.getDevicePointer(), "float");
        replacements["SLICES"] = bonded->addArgument(exclusionSlices.getDevicePointer(), "int");
        replacements["LAMBDAS"] = bonded->addArgument(sliceLambda.getDevicePointer(), "real");
        stringstream code;
        if (hasDerivatives) {
            string derivIndices = bonded->addArgument(sliceDerivIndices, "int");
            code<<"int which = "<<derivIndices<<"[slice];"<<endl;
            for (int i = 0; i < requestedDerivs.size(); i++) {
                string paramDeriv = bonded->addEnergyParameterDerivative(requestedDerivs[i]);
                code<<paramDeriv<<" += which == "<<i<<" ? tempEnergy : 0;"<<endl;
            }
        }
        replacements["COMPUTE_DERIVATIVES"] = code.str();
        bonded->addInteraction(exclusionPairs, cu.replaceStrings(CommonPmeSlicingKernelSources::slicedPmeExclusions, replacements), force.getForceGroup());
    }

    // Initialize the exceptions.

    startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0 && force.getIncludeDirectSpace()) {
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
        replacements["APPLY_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
        replacements["CHARGE_PRODS"] = bonded->addArgument(exceptionChargeProds.getDevicePointer(), "float");
        replacements["SLICES"] = bonded->addArgument(exceptionSlices.getDevicePointer(), "int");
        replacements["LAMBDAS"] = bonded->addArgument(sliceLambda.getDevicePointer(), "real");
        stringstream code;
        if (hasDerivatives) {
            string derivIndices = bonded->addArgument(sliceDerivIndices, "int");
            code<<"int which = "<<derivIndices<<"[slice];"<<endl;
            for (int i = 0; i < requestedDerivs.size(); i++) {
                string paramDeriv = bonded->addEnergyParameterDerivative(requestedDerivs[i]);
                code<<paramDeriv<<" += which == "<<i<<" ? tempEnergy : 0;"<<endl;
            }
        }
        replacements["COMPUTE_DERIVATIVES"] = code.str();
        bonded->addInteraction(exceptionPairs, cu.replaceStrings(CommonPmeSlicingKernelSources::slicedPmeExceptions, replacements), force.getForceGroup());
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

    // Do reciprocal space calculations.

    if (pmeGrid1.isInitialized() && includeReciprocal) {
        if (!addEnergy->isInitialized())
            addEnergy->initialize(pmeEnergyBuffer, sliceLambda, requestedDerivs, sliceDerivIndices);

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

        if (includeEnergy || hasDerivatives) {
            void* computeEnergyArgs[] = {
                &pmeGrid2.getDevicePointer(),
                &pmeEnergyBuffer.getDevicePointer(),
                &pmeBsplineModuliX.getDevicePointer(),
                &pmeBsplineModuliY.getDevicePointer(),
                &pmeBsplineModuliZ.getDevicePointer(),
                recipBoxVectorPointer[0],
                recipBoxVectorPointer[1],
                recipBoxVectorPointer[2]
            };
            cu.executeKernel(pmeEvalEnergyKernel, computeEnergyArgs, gridSizeX*gridSizeY*gridSizeZ);

            if (hasOffsets || hasDerivatives) {
                void* addSelfEnergyArgs[] = {
                    &pmeEnergyBuffer.getDevicePointer(),
                    &cu.getPosq().getDevicePointer(),
                    &charges.getDevicePointer(),
                    &subsets.getDevicePointer()
                };
                cu.executeKernel(pmeAddSelfEnergyKernel, addSelfEnergyArgs, cu.getPaddedNumAtoms());
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
        int particle1, particle2;
        double chargeProd1, chargeProd2, sigma1, sigma2, epsilon1, epsilon2;
        force.getExceptionParameters(group1, particle1, particle2, chargeProd1, sigma1, epsilon1);
        int slice1 = force.getSliceIndex(force.getParticleSubset(particle1), force.getParticleSubset(particle2));
        force.getExceptionParameters(group2, particle1, particle2, chargeProd2, sigma2, epsilon2);
        int slice2 = force.getSliceIndex(force.getParticleSubset(particle1), force.getParticleSubset(particle2));
        return (chargeProd1 == chargeProd2 && sigma1 == sigma2 && epsilon1 == epsilon2 && slice1 == slice2);
    }
private:
    const SlicedNonbondedForce& force;
};

class CudaCalcSlicedNonbondedForceKernel::PmeIO : public CalcPmeReciprocalForceKernel::IO {
public:
    PmeIO(CudaContext& cu, CUfunction addForcesKernel) : cu(cu), addForcesKernel(addForcesKernel) {
        forceTemp.initialize<float4>(cu, cu.getNumAtoms(), "PmeForce");
    }
    float* getPosq() {
        ContextSelector selector(cu);
        cu.getPosq().download(posq);
        return (float*) &posq[0];
    }
    void setForce(float* force) {
        forceTemp.upload(force);
        void* args[] = {&forceTemp.getDevicePointer(), &cu.getForce().getDevicePointer()};
        cu.executeKernel(addForcesKernel, args, cu.getNumAtoms());
    }
private:
    CudaContext& cu;
    vector<float4> posq;
    CudaArray forceTemp;
    CUfunction addForcesKernel;
};

class CudaCalcSlicedNonbondedForceKernel::PmePreComputation : public CudaContext::ForcePreComputation {
public:
    PmePreComputation(CudaContext& cu, Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : cu(cu), pme(pme), io(io) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        Vec3 boxVectors[3] = {Vec3(cu.getPeriodicBoxSize().x, 0, 0), Vec3(0, cu.getPeriodicBoxSize().y, 0), Vec3(0, 0, cu.getPeriodicBoxSize().z)};
        pme.getAs<CalcPmeReciprocalForceKernel>().beginComputation(io, boxVectors, includeEnergy);
    }
private:
    CudaContext& cu;
    Kernel pme;
    CalcPmeReciprocalForceKernel::IO& io;
};

class CudaCalcSlicedNonbondedForceKernel::PmePostComputation : public CudaContext::ForcePostComputation {
public:
    PmePostComputation(Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : pme(pme), io(io) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        return pme.getAs<CalcPmeReciprocalForceKernel>().finishComputation(io);
    }
private:
    Kernel pme;
    CalcPmeReciprocalForceKernel::IO& io;
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

CudaCalcSlicedNonbondedForceKernel::CudaCalcSlicedNonbondedForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) : CalcSlicedNonbondedForceKernel(name, platform),
        cu(cu), hasInitializedFFT(false), sort(NULL), dispersionFft(NULL), fft(NULL), pmeio(NULL), usePmeStream(false) {
    string version = Platform::getOpenMMVersion();
    stringstream code;
    if (stoi(version.substr(0, version.find("."))) < 8) {
        code<<"__device__ inline long long realToFixedPoint(real x) {"<<endl;
        code<<"return static_cast<long long>(x * 0x100000000);"<<endl;
        code<<"}"<<endl;
    }
    realToFixedPoint = code.str();
}

CudaCalcSlicedNonbondedForceKernel::~CudaCalcSlicedNonbondedForceKernel() {
    ContextSelector selector(cu);
    if (sort != NULL)
        delete sort;
    if (fft != NULL)
        delete fft;
    if (dispersionFft != NULL)
        delete dispersionFft;
    if (pmeio != NULL)
        delete pmeio;
    if (hasInitializedFFT) {
        if (useCudaFFT) {
            cufftDestroy(fftForward);
            cufftDestroy(fftBackward);
            if (doLJPME) {
                cufftDestroy(dispersionFftForward);
                cufftDestroy(dispersionFftBackward);
            }
        }
        if (usePmeStream) {
            cuStreamDestroy(pmeStream);
            cuEventDestroy(pmeSyncEvent);
            cuEventDestroy(paramsSyncEvent);
        }
    }
}

void CudaCalcSlicedNonbondedForceKernel::initialize(const System& system, const SlicedNonbondedForce& force) {
    ContextSelector selector(cu);
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "slicedNonbonded"+cu.intToString(forceIndex)+"_";

    int numParticles = force.getNumParticles();
    numSubsets = force.getNumSubsets();
    numSlices = force.getNumSlices();
    sliceLambdasVec.resize(numSlices, make_double2(1, 1));
    sliceScalingParams.resize(numSlices, make_int2(-1, -1));
    sliceScalingParamDerivs.resize(numSlices, make_int2(-1, -1));
    subsetSelfEnergy.resize(numSlices, make_double2(0, 0));

    subsetsVec.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        subsetsVec[i] = force.getParticleSubset(i);
    subsets.initialize<int>(cu, numParticles, "subsets");
    subsets.upload(subsetsVec);

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
        sliceScalingParams[slice] = make_int2(includeCoulomb ? index : -1, includeLJ ? index : -1);
        int pos = find(derivs.begin(), derivs.end(), scalingParams[index]) - derivs.begin();
        if (pos < numDerivs)
            sliceScalingParamDerivs[slice] = make_int2(includeCoulomb ? pos : -1, includeLJ ? pos : -1);
    }

    size_t sizeOfReal = cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float);
    sliceLambdas.initialize(cu, numSlices, 2*sizeOfReal, "sliceLambdas");
    if (cu.getUseDoublePrecision())
        sliceLambdas.upload(sliceLambdasVec);
    else
        sliceLambdas.upload(double2float(sliceLambdasVec));

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
    if (force.getUseDispersionCorrection() && cu.getContextIndex() == 0 && !doLJPME)
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
                ewaldSelfEnergy += sliceLambdasVec[i*(i+3)/2].x*subsetSelfEnergy[i].x;

            // Create the reciprocal space kernels.

            map<string, string> replacements;
            replacements["NUM_ATOMS"] = cu.intToString(numParticles);
            replacements["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
            replacements["KMAX_X"] = cu.intToString(kmaxx);
            replacements["KMAX_Y"] = cu.intToString(kmaxy);
            replacements["KMAX_Z"] = cu.intToString(kmaxz);
            replacements["EXP_COEFFICIENT"] = cu.doubleToString(-1.0/(4.0*alpha*alpha));
            replacements["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
            replacements["M_PI"] = cu.doubleToString(M_PI);
            CUmodule module = cu.createModule(realToFixedPoint+CudaPmeSlicingKernelSources::vectorOps+CommonPmeSlicingKernelSources::ewald, replacements);
            ewaldSumsKernel = cu.getKernel(module, "calculateEwaldCosSinSums");
            ewaldForcesKernel = cu.getKernel(module, "calculateEwaldForces");
            int elementSize = (cu.getUseDoublePrecision() ? sizeof(double2) : sizeof(float2));
            cosSinSums.initialize(cu, (2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1), elementSize, "cosSinSums");
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
            for (int i = 0; i < numSubsets; i++)
                ewaldSelfEnergy += sliceLambdasVec[i*(i+3)/2].x*subsetSelfEnergy[i].x + sliceLambdasVec[i*(i+3)/2].y*subsetSelfEnergy[i].y;
            char deviceName[100];
            cuDeviceGetName(deviceName, 100, cu.getDevice());
            usePmeStream = (!cu.getPlatformData().disablePmeStream && !cu.getPlatformData().useCpuPme && string(deviceName) != "GeForce GTX 980"); // Using a separate stream is slower on GTX 980
            map<string, string> pmeDefines;
            pmeDefines["PME_ORDER"] = cu.intToString(PmeOrder);
            pmeDefines["NUM_ATOMS"] = cu.intToString(numParticles);
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
            CUmodule module = cu.createModule(realToFixedPoint+CudaPmeSlicingKernelSources::vectorOps+cu.replaceStrings(CommonPmeSlicingKernelSources::pme, replacements), pmeDefines);
            if (cu.getPlatformData().useCpuPme && !doLJPME && usePosqCharges) {
                // Create the CPU PME kernel.

                try {
                    cpuPme = getPlatform().createKernel(CalcPmeReciprocalForceKernel::Name(), *cu.getPlatformData().context);
                    cpuPme.getAs<CalcPmeReciprocalForceKernel>().initialize(gridSizeX, gridSizeY, gridSizeZ, numParticles, alpha, cu.getPlatformData().deterministicForces);
                    CUfunction addForcesKernel = cu.getKernel(module, "addForces");
                    pmeio = new PmeIO(cu, addForcesKernel);
                    cu.addPreComputation(new PmePreComputation(cu, cpuPme, *pmeio));
                    cu.addPostComputation(new PmePostComputation(cpuPme, *pmeio));
                }
                catch (OpenMMException& ex) {
                    // The CPU PME plugin isn't available.
                }
            }
            if (pmeio == NULL) {
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
                    module = cu.createModule(realToFixedPoint+CudaPmeSlicingKernelSources::vectorOps+CommonPmeSlicingKernelSources::pme, pmeDefines);
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
                int gridElements = gridSizeX*gridSizeY*roundedZSize;
                if (doLJPME) {
                    roundedZSize = PmeOrder*(int) ceil(dispersionGridSizeZ/(double) PmeOrder);
                    gridElements = max(gridElements, dispersionGridSizeX*dispersionGridSizeY*roundedZSize);
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
                pmeEnergyBuffer.initialize(cu, cu.getNumThreadBlocks()*CudaContext::ThreadBlockSize, energyElementSize, "pmeEnergyBuffer");
                cu.clearBuffer(pmeEnergyBuffer);
                sort = new CudaSort(cu, new SortTrait(), cu.getNumAtoms());
                int cufftVersion;
                cufftGetVersion(&cufftVersion);
                useCudaFFT = force.getUseCudaFFT() && (cufftVersion >= 7050); // There was a critical bug in version 7.0
                if (useCudaFFT) {
                    fft = (CudaFFT3D*) new CudaCuFFT3D(cu, pmeStream, gridSizeX, gridSizeY, gridSizeZ, 1, true, pmeGrid1, pmeGrid2);
                    if (doLJPME)
                        dispersionFft = (CudaFFT3D*) new CudaCuFFT3D(cu, pmeStream, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, 1, true, pmeGrid1, pmeGrid2);
                }
                else {
                    fft = (CudaFFT3D*) new CudaVkFFT3D(cu, pmeStream, gridSizeX, gridSizeY, gridSizeZ, 1, true, pmeGrid1, pmeGrid2);
                    if (doLJPME)
                        dispersionFft = (CudaFFT3D*) new CudaVkFFT3D(cu, pmeStream, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, 1, true, pmeGrid1, pmeGrid2);
                }

                // Prepare for doing PME on its own stream.

                if (usePmeStream) {
                    cuStreamCreate(&pmeStream, CU_STREAM_NON_BLOCKING);
                    if (useCudaFFT) {
                        cufftSetStream(fftForward, pmeStream);
                        cufftSetStream(fftBackward, pmeStream);
                        if (doLJPME) {
                            cufftSetStream(dispersionFftForward, pmeStream);
                            cufftSetStream(dispersionFftBackward, pmeStream);
                        }
                    }
                    // CHECK_RESULT(cuEventCreate(&pmeSyncEvent, cu.getEventFlags()), "Error creating event for SlicedNonbondedForce");  // OpenMM 8.0
                    // CHECK_RESULT(cuEventCreate(&paramsSyncEvent, cu.getEventFlags()), "Error creating event for SlicedNonbondedForce");  // OpenMM 8.0
                    CHECK_RESULT(cuEventCreate(&pmeSyncEvent, CU_EVENT_DISABLE_TIMING), "Error creating event for SlicedNonbondedForce");
                    CHECK_RESULT(cuEventCreate(&paramsSyncEvent, CU_EVENT_DISABLE_TIMING), "Error creating event for SlicedNonbondedForce");
                    int recipForceGroup = force.getReciprocalSpaceForceGroup();
                    if (recipForceGroup < 0)
                        recipForceGroup = force.getForceGroup();
                    cu.addPreComputation(new SyncStreamPreComputation(cu, pmeStream, pmeSyncEvent, recipForceGroup));
                    cu.addPostComputation(new SyncStreamPostComputation(cu, pmeSyncEvent, cu.getKernel(module, "addEnergy"), pmeEnergyBuffer, recipForceGroup));
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
    }

    // Add code to subtract off the reciprocal part of excluded interactions.

    if ((nonbondedMethod == Ewald || nonbondedMethod == PME || nonbondedMethod == LJPME) && pmeio == NULL) {
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
            if (force.getIncludeDirectSpace())
                cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CommonPmeSlicingKernelSources::pmeExclusions, replacements), force.getForceGroup());
        }
    }

    // Add the interaction to the default nonbonded kernel.

    string source = cu.replaceStrings(CommonPmeSlicingKernelSources::coulombLennardJones, defines);
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
            exceptionSlicesVec[i] = force.getSliceIndex(atoms[i][0], atoms[i][1]);
        }
        baseExceptionParams.upload(baseExceptionParamsVec);
        exceptionPairs.upload(exceptionAtoms);
        exceptionSlices.upload(exceptionSlicesVec);
        map<string, string> replacements;
        replacements["APPLY_PERIODIC"] = (usePeriodic && force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0");
        replacements["PARAMS"] = cu.getBondedUtilities().addArgument(exceptionParams.getDevicePointer(), "float4");
        replacements["LAMBDAS"] = cu.getBondedUtilities().addArgument(sliceLambdas.getDevicePointer(), "real2");
        cu.getBondedUtilities().addPrefixCode(CommonPmeSlicingKernelSources::bitcast);
        if (force.getIncludeDirectSpace())
            cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CommonPmeSlicingKernelSources::nonbondedExceptions, replacements), force.getForceGroup());
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

    // Initialize the kernel for updating parameters.

    CUmodule module = cu.createModule(CommonPmeSlicingKernelSources::bitcast+CommonPmeSlicingKernelSources::nonbondedParameters, paramsDefines);
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
        int2 indices = sliceScalingParams[slice];
        int index = max(indices.x, indices.y);
        if (index != -1) {
            double paramValue = context.getParameter(scalingParams[index]);
            double oldValue = indices.x != -1 ? sliceLambdasVec[slice].x : sliceLambdasVec[slice].y;
            if (oldValue != paramValue) {
                sliceLambdasVec[slice] = make_double2(indices.x == -1 ? 1.0 : paramValue, indices.y == -1 ? 1.0 : paramValue);
                scalingParamChanged = true;
            }
        }
    }
    if (scalingParamChanged) {
        ewaldSelfEnergy = 0.0;
        for (int i = 0; i < numSubsets; i++)
            ewaldSelfEnergy += sliceLambdasVec[i*(i+3)/2].x*subsetSelfEnergy[i].x + sliceLambdasVec[i*(i+3)/2].y*subsetSelfEnergy[i].y;
        if (cu.getUseDoublePrecision())
            sliceLambdas.upload(sliceLambdasVec);
        else
            sliceLambdas.upload(double2float(sliceLambdasVec));
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
                    &numExclusions, &exclusionAtoms.getDevicePointer(), &exclusionParams.getDevicePointer()};
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
        void* sumsArgs[] = {&cu.getEnergyBuffer().getDevicePointer(), &cu.getPosq().getDevicePointer(), &cosSinSums.getDevicePointer(), cu.getPeriodicBoxSizePointer()};
        cu.executeKernel(ewaldSumsKernel, sumsArgs, cosSinSums.getSize());
        void* forcesArgs[] = {&cu.getForce().getDevicePointer(), &cu.getPosq().getDevicePointer(), &cosSinSums.getDevicePointer(), cu.getPeriodicBoxSizePointer()};
        cu.executeKernel(ewaldForcesKernel, forcesArgs, cu.getNumAtoms());
    }
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

        if (hasCoulomb) {
            void* gridIndexArgs[] = {&cu.getPosq().getDevicePointer(), &pmeAtomGridIndex.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
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
            cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, gridSizeX*gridSizeY*gridSizeZ, 256);

            fft->execFFT(true);

            if (includeEnergy) {
                void* computeEnergyArgs[] = {&pmeGrid2.getDevicePointer(), usePmeStream ? &pmeEnergyBuffer.getDevicePointer() : &cu.getEnergyBuffer().getDevicePointer(),
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
                    &charges.getDevicePointer()};
            cu.executeKernel(pmeInterpolateForceKernel, interpolateArgs, cu.getNumAtoms(), 128);
        }

        if (doLJPME && hasLJ) {
            if (!hasCoulomb) {
                void* gridIndexArgs[] = {&cu.getPosq().getDevicePointer(), &pmeAtomGridIndex.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                        cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                        recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
                cu.executeKernel(pmeDispersionGridIndexKernel, gridIndexArgs, cu.getNumAtoms());

                sort->sort(pmeAtomGridIndex);
                cu.clearBuffer(pmeEnergyBuffer);
            }

            cu.clearBuffer(pmeGrid2);
            void* spreadArgs[] = {&cu.getPosq().getDevicePointer(), &pmeGrid2.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                    cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                    &sigmaEpsilon.getDevicePointer()};
            cu.executeKernel(pmeDispersionSpreadChargeKernel, spreadArgs, cu.getNumAtoms(), 128);

            void* finishSpreadArgs[] = {&pmeGrid2.getDevicePointer(), &pmeGrid1.getDevicePointer()};
            cu.executeKernel(pmeDispersionFinishSpreadChargeKernel, finishSpreadArgs, dispersionGridSizeX*dispersionGridSizeY*dispersionGridSizeZ, 256);

            dispersionFft->execFFT(true);

            if (includeEnergy) {
                void* computeEnergyArgs[] = {&pmeGrid2.getDevicePointer(), usePmeStream ? &pmeEnergyBuffer.getDevicePointer() : &cu.getEnergyBuffer().getDevicePointer(),
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
                    &sigmaEpsilon.getDevicePointer()};
            cu.executeKernel(pmeInterpolateDispersionForceKernel, interpolateArgs, cu.getNumAtoms(), 128);
        }
        if (usePmeStream) {
            cuEventRecord(pmeSyncEvent, pmeStream);
            cu.restoreDefaultStream();
        }
    }

    if (dispersionCoefficients.size() != 0 && includeDirect) {
        double4 boxSize = cu.getPeriodicBoxSize();
        double volume = boxSize.x*boxSize.y*boxSize.z;
        for (int slice = 0; slice < numSlices; slice++)
            energy += sliceLambdasVec[slice].y*dispersionCoefficients[slice]/volume;
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
            for (int i = 0; i < force.getNumSubsets(); i++)
                ewaldSelfEnergy += sliceLambdasVec[i*(i+3)/2].x*subsetSelfEnergy[i].x + sliceLambdasVec[i*(i+3)/2].y*subsetSelfEnergy[i].y;
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
    if (cu.getPlatformData().useCpuPme)
        cpuPme.getAs<CalcPmeReciprocalForceKernel>().getPMEParameters(alpha, nx, ny, nz);
    else {
        alpha = this->alpha;
        nx = gridSizeX;
        ny = gridSizeY;
        nz = gridSizeZ;
    }
}

void CudaCalcSlicedNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!doLJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    if (cu.getPlatformData().useCpuPme)
        //cpuPme.getAs<CalcPmeReciprocalForceKernel>().getLJPMEParameters(alpha, nx, ny, nz);
        throw OpenMMException("getPMEParametersInContext: CPUPME has not been implemented for LJPME yet.");
    else {
        alpha = this->dispersionAlpha;
        nx = dispersionGridSizeX;
        ny = dispersionGridSizeY;
        nz = dispersionGridSizeZ;
    }
}
