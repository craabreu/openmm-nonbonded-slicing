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

#include "OpenCLPmeSlicingKernels.h"
#include "OpenCLPmeSlicingKernelSources.h"
#include "CommonPmeSlicingKernelSources.h"
#include "SlicedPmeForce.h"
#include "SlicedNonbondedForce.h"
#include "internal/SlicedPmeForceImpl.h"
#include "internal/SlicedNonbondedForceImpl.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/opencl/OpenCLBondedUtilities.h"
#include "openmm/opencl/OpenCLForceInfo.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include <cstring>
#include <map>
#include <algorithm>

using namespace PmeSlicing;
using namespace OpenMM;
using namespace std;

static void setPeriodicBoxSizeArg(OpenCLContext& cl, cl::Kernel& kernel, int index) {
    if (cl.getUseDoublePrecision())
        kernel.setArg<mm_double4>(index, cl.getPeriodicBoxSizeDouble());
    else
        kernel.setArg<mm_float4>(index, cl.getPeriodicBoxSize());
}

static void setPeriodicBoxArgs(OpenCLContext& cl, cl::Kernel& kernel, int index) {
    if (cl.getUseDoublePrecision()) {
        kernel.setArg<mm_double4>(index++, cl.getPeriodicBoxSizeDouble());
        kernel.setArg<mm_double4>(index++, cl.getInvPeriodicBoxSizeDouble());
        kernel.setArg<mm_double4>(index++, cl.getPeriodicBoxVecXDouble());
        kernel.setArg<mm_double4>(index++, cl.getPeriodicBoxVecYDouble());
        kernel.setArg<mm_double4>(index, cl.getPeriodicBoxVecZDouble());
    }
    else {
        kernel.setArg<mm_float4>(index++, cl.getPeriodicBoxSize());
        kernel.setArg<mm_float4>(index++, cl.getInvPeriodicBoxSize());
        kernel.setArg<mm_float4>(index++, cl.getPeriodicBoxVecX());
        kernel.setArg<mm_float4>(index++, cl.getPeriodicBoxVecY());
        kernel.setArg<mm_float4>(index, cl.getPeriodicBoxVecZ());
    }
}

class OpenCLCalcSlicedPmeForceKernel::ForceInfo : public OpenCLForceInfo {
public:
    ForceInfo(int requiredBuffers, const SlicedPmeForce& force) : OpenCLForceInfo(requiredBuffers), force(force) {
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

class OpenCLCalcSlicedPmeForceKernel::SyncQueuePreComputation : public OpenCLContext::ForcePreComputation {
public:
    SyncQueuePreComputation(OpenCLContext& cl, cl::CommandQueue queue, int forceGroup) : cl(cl), queue(queue), forceGroup(forceGroup) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            vector<cl::Event> events(1);
            cl.getQueue().enqueueMarkerWithWaitList(NULL, &events[0]);
            queue.enqueueBarrierWithWaitList(&events);
        }
    }
private:
    OpenCLContext& cl;
    cl::CommandQueue queue;
    int forceGroup;
};

class OpenCLCalcSlicedPmeForceKernel::SyncQueuePostComputation : public OpenCLContext::ForcePostComputation {
public:
    SyncQueuePostComputation(OpenCLContext& cl, cl::Event& event, int forceGroup) :
        cl(cl), event(event), forceGroup(forceGroup) {}
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            vector<cl::Event> events(1);
            events[0] = event;
            event = cl::Event();
            cl.getQueue().enqueueBarrierWithWaitList(&events);
        }
        return 0.0;
    }
private:
    OpenCLContext& cl;
    cl::Event& event;
    int forceGroup;
};

class OpenCLCalcSlicedPmeForceKernel::AddEnergyPostComputation : public OpenCLContext::ForcePostComputation {
public:
    AddEnergyPostComputation(OpenCLContext& cl, int forceGroup) : cl(cl), forceGroup(forceGroup), initialized(false) { }
    void initialize(OpenCLArray& pmeEnergyBuffer, OpenCLArray& sliceLambda, vector<string> requestedDerivs, OpenCLArray& sliceDerivIndices) {
        int numSlices = sliceDerivIndices.getSize();
        hasDerivatives = requestedDerivs.size() > 0;
        stringstream code;
        if (hasDerivatives) {
            vector<int> sliceDerivIndexVec(numSlices);
            sliceDerivIndices.download(sliceDerivIndexVec);
            const vector<string>& allDerivs = cl.getEnergyParamDerivNames();
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
        replacements["NUM_SLICES"] = cl.intToString(numSlices);
        string source = cl.replaceStrings(CommonPmeSlicingKernelSources::slicedPmeAddEnergy, replacements);
        cl::Program program = cl.createProgram(source, defines);
        addEnergyKernel = cl::Kernel(program, "addEnergy");
        bufferSize = pmeEnergyBuffer.getSize()/numSlices;
        addEnergyKernel.setArg<cl::Buffer>(0, pmeEnergyBuffer.getDeviceBuffer());
        addEnergyKernel.setArg<cl::Buffer>(1, cl.getEnergyBuffer().getDeviceBuffer());
        addEnergyKernel.setArg<cl::Buffer>(2, cl.getEnergyParamDerivBuffer().getDeviceBuffer());
        addEnergyKernel.setArg<cl::Buffer>(3, sliceLambda.getDeviceBuffer());
        addEnergyKernel.setArg<cl_int>(4, bufferSize);
        initialized = true;
    }
    bool isInitialized() {
        return initialized;
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((includeEnergy || hasDerivatives) && (groups&(1<<forceGroup)) != 0)
            cl.executeKernel(addEnergyKernel, bufferSize);
        return 0.0;
    }
private:
    OpenCLContext& cl;
    cl::Kernel addEnergyKernel;
    bool hasDerivatives;
    int bufferSize;
    int forceGroup;
    bool initialized;
};

OpenCLCalcSlicedPmeForceKernel::OpenCLCalcSlicedPmeForceKernel(std::string name, const Platform& platform, OpenCLContext& cl, const System& system) :
    CalcSlicedPmeForceKernel(name, platform), hasInitializedKernel(false), cl(cl), sort(NULL), fft(NULL), usePmeQueue(false) {
    stringstream code;
    if (Platform::getOpenMMVersion().at(0) == '7') {
        code<<"inline long realToFixedPoint(real x) {"<<endl;
        code<<"    return (long) (x * 0x100000000);"<<endl;
        code<<"}"<<endl;
    }
    realToFixedPoint = code.str();
};

OpenCLCalcSlicedPmeForceKernel::~OpenCLCalcSlicedPmeForceKernel() {
    if (sort != NULL)
        delete sort;
    if (fft != NULL)
        delete fft;
}

void OpenCLCalcSlicedPmeForceKernel::initialize(const System& system, const SlicedPmeForce& force) {
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "pme"+cl.intToString(forceIndex)+"_";
    deviceIsCpu = (cl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU);

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
    vector<float> baseParticleChargeVec(cl.getPaddedNumAtoms(), 0.0);
    vector<vector<int> > exclusionList(numParticles);
    for (int i = 0; i < numParticles; i++) {
        baseParticleChargeVec[i] = force.getParticleCharge(i);
        exclusionList[i].push_back(i);
    }
    for (auto exclusion : exclusions) {
        exclusionList[exclusion.first].push_back(exclusion.second);
        exclusionList[exclusion.second].push_back(exclusion.first);
    }
    usePosqCharges = cl.requestPosqCharges();
    size_t sizeOfReal = cl.getUseDoublePrecision() ? sizeof(double) : sizeof(float);
    size_t sizeOfMixed = (cl.getUseMixedPrecision() ? sizeof(double) : sizeOfReal);

    alpha = 0;
    ewaldSelfEnergy = 0.0;
    subsetSelfEnergy.resize(numSubsets, 0.0);
    map<string, string> paramsDefines;
    paramsDefines["ONE_4PI_EPS0"] = cl.doubleToString(ONE_4PI_EPS0);
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

    subsets.initialize<int>(cl, cl.getPaddedNumAtoms(), "subsets");
    vector<int> subsetVec(cl.getPaddedNumAtoms());
    for (int i = 0; i < numParticles; i++)
        subsetVec[i] = force.getParticleSubset(i);
    subsets.upload(subsetVec);

    // Identify requested derivatives.

    for (int index = 0; index < force.getNumSwitchingParameterDerivatives(); index++) {
        string param = force.getSwitchingParameterDerivativeName(index);
        requestedDerivs.push_back(param);
        cl.addEnergyParameterDerivative(param);
    }
    hasDerivatives = requestedDerivs.size() > 0;
    const vector<string>& allDerivs = cl.getEnergyParamDerivNames();
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
    sliceLambda.initialize(cl, numSlices, sizeOfReal, "sliceLambda");
    if (cl.getUseDoublePrecision())
        sliceLambda.upload(sliceLambdaVec);
    else
        sliceLambda.upload(floatVector(sliceLambdaVec));
    sliceDerivIndices.initialize<int>(cl, numSlices, "sliceDerivIndices");
    sliceDerivIndices.upload(sliceDerivIndexVec);

    // Compute the PME parameters.

    SlicedPmeForceImpl::calcPMEParameters(system, force, alpha, gridSizeX, gridSizeY, gridSizeZ, false);

    gridSizeX = OpenCLVkFFT3D::findLegalDimension(gridSizeX);
    gridSizeY = OpenCLVkFFT3D::findLegalDimension(gridSizeY);
    gridSizeZ = OpenCLVkFFT3D::findLegalDimension(gridSizeZ);
    int roundedZSize = (int) ceil(gridSizeZ/(double) PmeOrder)*PmeOrder;

    if (cl.getContextIndex() == 0) {
        paramsDefines["INCLUDE_EWALD"] = "1";
        for (int i = 0; i < numParticles; i++)
            subsetSelfEnergy[subsetVec[i]] += baseParticleChargeVec[i]*baseParticleChargeVec[i];
        for (int j = 0; j < numSubsets; j++) {
            subsetSelfEnergy[j] *= -ONE_4PI_EPS0*alpha/sqrt(M_PI);
            ewaldSelfEnergy += subsetSelfEnergy[j];
        }
        string vendor = cl.getDevice().getInfo<CL_DEVICE_VENDOR>();
        bool isNvidia = (vendor.size() >= 6 && vendor.substr(0, 6) == "NVIDIA");
        usePmeQueue = (!cl.getPlatformData().disablePmeStream && !cl.getPlatformData().useCpuPme && isNvidia);
        pmeDefines["PME_ORDER"] = cl.intToString(PmeOrder);
        pmeDefines["NUM_ATOMS"] = cl.intToString(numParticles);
        pmeDefines["NUM_SUBSETS"] = cl.intToString(numSubsets);
        pmeDefines["NUM_SLICES"] = cl.intToString(numSlices);
        pmeDefines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
        pmeDefines["RECIP_EXP_FACTOR"] = cl.doubleToString(M_PI*M_PI/(alpha*alpha));
        pmeDefines["GRID_SIZE_X"] = cl.intToString(gridSizeX);
        pmeDefines["GRID_SIZE_Y"] = cl.intToString(gridSizeY);
        pmeDefines["GRID_SIZE_Z"] = cl.intToString(gridSizeZ);
        pmeDefines["ROUNDED_Z_SIZE"] = cl.intToString(roundedZSize);
        pmeDefines["EPSILON_FACTOR"] = cl.doubleToString(sqrt(ONE_4PI_EPS0));
        pmeDefines["M_PI"] = cl.doubleToString(M_PI);
        pmeDefines["EWALD_SELF_ENERGY_SCALE"] = cl.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
        pmeDefines["USE_POSQ_CHARGES"] = usePosqCharges ? "1" : "0";
        pmeDefines["HAS_DERIVATIVES"] = hasDerivatives ? "1" : "0";
        pmeDefines["NUM_ALL_DERIVS"] = cl.intToString(numAllDerivs);
        pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";
        if (usePmeQueue)
            pmeDefines["USE_PME_STREAM"] = "1";

        // Create required data structures.

        int gridElements = gridSizeX*gridSizeY*roundedZSize*numSubsets;
        pmeGrid1.initialize(cl, gridElements, 2*sizeOfReal, "pmeGrid1");
        pmeGrid2.initialize(cl, gridElements, 2*sizeOfReal, "pmeGrid2");
        cl.addAutoclearBuffer(pmeGrid2);
        pmeBsplineModuliX.initialize(cl, gridSizeX, sizeOfReal, "pmeBsplineModuliX");
        pmeBsplineModuliY.initialize(cl, gridSizeY, sizeOfReal, "pmeBsplineModuliY");
        pmeBsplineModuliZ.initialize(cl, gridSizeZ, sizeOfReal, "pmeBsplineModuliZ");
        pmeBsplineTheta.initialize(cl, PmeOrder*numParticles, 4*sizeOfReal, "pmeBsplineTheta");
        pmeAtomRange.initialize<cl_int>(cl, gridSizeX*gridSizeY*gridSizeZ+1, "pmeAtomRange");
        pmeAtomGridIndex.initialize<mm_int2>(cl, numParticles, "pmeAtomGridIndex");
        int bufferSize = cl.getNumThreadBlocks()*OpenCLContext::ThreadBlockSize;
        pmeEnergyBuffer.initialize(cl, numSlices*bufferSize, sizeOfMixed, "pmeEnergyBuffer");
        cl.clearBuffer(pmeEnergyBuffer);
        sort = new OpenCLSort(cl, new SortTrait(), cl.getNumAtoms());

        // Prepare for doing PME on its own queue or not.

        int recipForceGroup = force.getReciprocalSpaceForceGroup();
        if (recipForceGroup < 0)
            recipForceGroup = force.getForceGroup();
        if (usePmeQueue) {
            pmeQueue = cl::CommandQueue(cl.getContext(), cl.getDevice());
            cl.addPreComputation(new SyncQueuePreComputation(cl, pmeQueue, recipForceGroup));
            cl.addPostComputation(new SyncQueuePostComputation(cl, pmeSyncEvent, recipForceGroup));
        }
        cl.addPostComputation(addEnergy = new AddEnergyPostComputation(cl, recipForceGroup));

        fft = new OpenCLVkFFT3D(cl, gridSizeX, gridSizeY, gridSizeZ, numSubsets, true, pmeGrid1, pmeGrid2);

        // Initialize the b-spline moduli.

        int xsize, ysize, zsize;
        OpenCLArray *xmoduli, *ymoduli, *zmoduli;

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
            vector<cl_double> moduli(ndata);
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
            {
                if (moduli[i] < 1.0e-7)
                    moduli[i] = (moduli[(i-1+ndata)%ndata]+moduli[(i+1)%ndata])*0.5;
            }
            if (dim == 0)
                xmoduli->upload(moduli, true);
            else if (dim == 1)
                ymoduli->upload(moduli, true);
            else
                zmoduli->upload(moduli, true);
        }
    }

    // Add the interaction to the default nonbonded kernel.

    charges.initialize(cl, cl.getPaddedNumAtoms(), sizeOfReal, "charges");
    baseParticleCharges.initialize<float>(cl, cl.getPaddedNumAtoms(), "baseParticleCharges");
    baseParticleCharges.upload(baseParticleChargeVec);

    if (force.getIncludeDirectSpace()) {
        OpenCLNonbondedUtilities* nb = &cl.getNonbondedUtilities();
        map<string, string> replacements;
        replacements["LAMBDA"] = prefix+"lambda";
        replacements["EWALD_ALPHA"] = cl.doubleToString(alpha);
        replacements["TWO_OVER_SQRT_PI"] = cl.doubleToString(2.0/sqrt(M_PI));
        replacements["ONE_4PI_EPS0"] = cl.doubleToString(ONE_4PI_EPS0);
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
        string source = cl.replaceStrings(CommonPmeSlicingKernelSources::coulomb, replacements);
        nb->addInteraction(true, true, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup());
    }

    // Add code to subtract off the reciprocal part of excluded interactions.

    int numContexts = cl.getPlatformData().contexts.size();
    int startIndex = cl.getContextIndex()*force.getNumExceptions()/numContexts;
    int endIndex = (cl.getContextIndex()+1)*force.getNumExceptions()/numContexts;
    int numExclusions = endIndex-startIndex;
    if (numExclusions > 0 && force.getIncludeDirectSpace()) {
        exclusionPairs.resize(numExclusions);
        exclusionAtoms.initialize<mm_int2>(cl, numExclusions, "exclusionAtoms");
        exclusionSlices.initialize<int>(cl, numExclusions, "exclusionSlices");
        exclusionChargeProds.initialize<float>(cl, numExclusions, "exclusionChargeProds");
        vector<mm_int2> exclusionAtomsVec(numExclusions);
        vector<int> exclusionSlicesVec(numExclusions);
       for (int k = 0; k < numExclusions; k++) {
            int atom1 = exclusions[k+startIndex].first;
            int atom2 = exclusions[k+startIndex].second;
            exclusionAtomsVec[k] = mm_int2(atom1, atom2);
            exclusionPairs[k] = (vector<int>) {atom1, atom2};
            int i = subsetVec[atom1];
            int j = subsetVec[atom2];
            exclusionSlicesVec[k] = i > j ? i*(i+1)/2+j : j*(j+1)/2+i;
        }
        exclusionAtoms.upload(exclusionAtomsVec);
        exclusionSlices.upload(exclusionSlicesVec);
        OpenCLBondedUtilities* bonded = &cl.getBondedUtilities();
        map<string, string> replacements;
        replacements["APPLY_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
        replacements["EWALD_ALPHA"] = cl.doubleToString(alpha);
        replacements["TWO_OVER_SQRT_PI"] = cl.doubleToString(2.0/sqrt(M_PI));
        replacements["CHARGE_PRODS"] = bonded->addArgument(exclusionChargeProds.getDeviceBuffer(), "float");
        replacements["SLICES"] = bonded->addArgument(exclusionSlices.getDeviceBuffer(), "int");
        replacements["LAMBDAS"] = bonded->addArgument(sliceLambda.getDeviceBuffer(), "real");
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
        bonded->addInteraction(exclusionPairs, cl.replaceStrings(CommonPmeSlicingKernelSources::slicedPmeExclusions, replacements), force.getForceGroup());
    }

    // Initialize the exceptions.

    startIndex = cl.getContextIndex()*exceptions.size()/numContexts;
    endIndex = (cl.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0 && force.getIncludeDirectSpace()) {
        paramsDefines["HAS_EXCEPTIONS"] = "1";
        exceptionPairs.resize(numExceptions);
        exceptionAtoms.initialize<mm_int2>(cl, numExceptions, "exceptionAtoms");
        exceptionSlices.initialize<int>(cl, numExceptions, "exceptionSlices");
        exceptionChargeProds.initialize<float>(cl, numExceptions, "exceptionChargeProds");
        baseExceptionChargeProds.initialize<float>(cl, numExceptions, "baseExceptionChargeProds");
        vector<mm_int2> exceptionAtomsVec(numExceptions);
        vector<int> exceptionSlicesVec(numExceptions);
        vector<float> baseExceptionChargeProdsVec(numExceptions);
        for (int k = 0; k < numExceptions; k++) {
            double chargeProd;
            int atom1, atom2;
            force.getExceptionParameters(exceptions[startIndex+k], atom1, atom2, chargeProd);
            exceptionPairs[k] = (vector<int>) {atom1, atom2};
            baseExceptionChargeProdsVec[k] = chargeProd;
            exceptionAtomsVec[k] = mm_int2(atom1, atom2);
            int i = subsetVec[atom1];
            int j = subsetVec[atom2];
            exceptionSlicesVec[k] = i > j ? i*(i+1)/2+j : j*(j+1)/2+i;
        }
        exceptionAtoms.upload(exceptionAtomsVec);
        exceptionSlices.upload(exceptionSlicesVec);
        baseExceptionChargeProds.upload(baseExceptionChargeProdsVec);
        OpenCLBondedUtilities* bonded = &cl.getBondedUtilities();
        map<string, string> replacements;
        replacements["APPLY_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
        replacements["CHARGE_PRODS"] = bonded->addArgument(exceptionChargeProds.getDeviceBuffer(), "float");
        replacements["SLICES"] = bonded->addArgument(exceptionSlices.getDeviceBuffer(), "int");
        replacements["LAMBDAS"] = bonded->addArgument(sliceLambda.getDeviceBuffer(), "real");
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
        bonded->addInteraction(exceptionPairs, cl.replaceStrings(CommonPmeSlicingKernelSources::slicedPmeExceptions, replacements), force.getForceGroup());
    }

    // Initialize charge offsets.

    vector<vector<mm_float2> > particleOffsetVec(force.getNumParticles());
    vector<vector<mm_float2> > exceptionOffsetVec(numExceptions);
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
        particleOffsetVec[particle].push_back(mm_float2(charge, paramIndex));
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
        exceptionOffsetVec[index-startIndex].push_back(mm_float2(charge, paramIndex));
    }
    paramValues.resize(paramNames.size(), 0.0);
    particleParamOffsets.initialize<mm_float2>(cl, max(force.getNumParticleChargeOffsets(), 1), "particleParamOffsets");
    particleOffsetIndices.initialize<cl_int>(cl, cl.getPaddedNumAtoms()+1, "particleOffsetIndices");
    vector<cl_int> particleOffsetIndicesVec, exceptionOffsetIndicesVec;
    vector<mm_float2> p, e;
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
    exceptionParamOffsets.initialize<mm_float2>(cl, max((int) e.size(), 1), "exceptionParamOffsets");
    exceptionOffsetIndices.initialize<cl_int>(cl, exceptionOffsetIndicesVec.size(), "exceptionOffsetIndices");
    if (e.size() > 0) {
        exceptionParamOffsets.upload(e);
        exceptionOffsetIndices.upload(exceptionOffsetIndicesVec);
    }
    globalParams.initialize(cl, max((int) paramValues.size(), 1), sizeOfReal, "globalParams");
    if (paramValues.size() > 0)
        globalParams.upload(paramValues, true);
    recomputeParams = true;

    // Initialize the kernel for updating parameters.

    cl::Program program = cl.createProgram(CommonPmeSlicingKernelSources::slicedPmeParameters, paramsDefines);
    computeParamsKernel = cl::Kernel(program, "computeParameters");
    computeExclusionParamsKernel = cl::Kernel(program, "computeExclusionParameters");
    info = new ForceInfo(cl.getNonbondedUtilities().getNumForceBuffers(), force);
    cl.addForce(info);
}

double OpenCLCalcSlicedPmeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    double energy = 0.0;

    if (!hasInitializedKernel) {
        hasInitializedKernel = true;
        int index = 0;
        computeParamsKernel.setArg<cl::Buffer>(index++, globalParams.getDeviceBuffer());
        computeParamsKernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
        computeParamsKernel.setArg<cl::Buffer>(index++, baseParticleCharges.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, cl.getPosq().getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, charges.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, particleParamOffsets.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, particleOffsetIndices.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, subsets.getDeviceBuffer());
        if (exceptionChargeProds.isInitialized()) {
            computeParamsKernel.setArg<cl_int>(index++, exceptionChargeProds.getSize());
            computeParamsKernel.setArg<cl::Buffer>(index++, baseExceptionChargeProds.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionChargeProds.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionParamOffsets.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionOffsetIndices.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionAtoms.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionSlices.getDeviceBuffer());
        }
        if (exclusionChargeProds.isInitialized()) {
            computeExclusionParamsKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl::Buffer>(1, charges.getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl_int>(2, exclusionChargeProds.getSize());
            computeExclusionParamsKernel.setArg<cl::Buffer>(3, exclusionAtoms.getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl::Buffer>(4, subsets.getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl::Buffer>(5, exclusionSlices.getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl::Buffer>(6, exclusionChargeProds.getDeviceBuffer());
        }
        if (pmeGrid1.isInitialized()) {
            // Create kernels for Coulomb PME.

            map<string, string> replacements;
            replacements["CHARGE"] = (usePosqCharges ? "pos.w" : "charges[atom]");
            cl::Program program = cl.createProgram(realToFixedPoint+cl.replaceStrings(CommonPmeSlicingKernelSources::slicedPme, replacements), pmeDefines);
            pmeGridIndexKernel = cl::Kernel(program, "findAtomGridIndex");
            pmeSpreadChargeKernel = cl::Kernel(program, "gridSpreadCharge");
            pmeConvolutionKernel = cl::Kernel(program, "reciprocalConvolution");
            pmeEvalEnergyKernel = cl::Kernel(program, "gridEvaluateEnergy");
            pmeInterpolateForceKernel = cl::Kernel(program, "gridInterpolateForce");
            int sizeOfReal = (cl.getUseDoublePrecision() ? sizeof(mm_double4) : sizeof(mm_float4));
            pmeGridIndexKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
            pmeGridIndexKernel.setArg<cl::Buffer>(1, subsets.getDeviceBuffer());
            pmeGridIndexKernel.setArg<cl::Buffer>(2, pmeAtomGridIndex.getDeviceBuffer());
            pmeSpreadChargeKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
            pmeSpreadChargeKernel.setArg<cl::Buffer>(1, pmeGrid2.getDeviceBuffer());
            pmeSpreadChargeKernel.setArg<cl::Buffer>(10, pmeAtomGridIndex.getDeviceBuffer());
            pmeSpreadChargeKernel.setArg<cl::Buffer>(11, charges.getDeviceBuffer());
            pmeConvolutionKernel.setArg<cl::Buffer>(0, pmeGrid2.getDeviceBuffer());
            pmeConvolutionKernel.setArg<cl::Buffer>(1, pmeBsplineModuliX.getDeviceBuffer());
            pmeConvolutionKernel.setArg<cl::Buffer>(2, pmeBsplineModuliY.getDeviceBuffer());
            pmeConvolutionKernel.setArg<cl::Buffer>(3, pmeBsplineModuliZ.getDeviceBuffer());
            pmeEvalEnergyKernel.setArg<cl::Buffer>(0, pmeGrid2.getDeviceBuffer());
            pmeEvalEnergyKernel.setArg<cl::Buffer>(1, pmeEnergyBuffer.getDeviceBuffer());
            pmeEvalEnergyKernel.setArg<cl::Buffer>(2, pmeBsplineModuliX.getDeviceBuffer());
            pmeEvalEnergyKernel.setArg<cl::Buffer>(3, pmeBsplineModuliY.getDeviceBuffer());
            pmeEvalEnergyKernel.setArg<cl::Buffer>(4, pmeBsplineModuliZ.getDeviceBuffer());
            if (hasOffsets || hasDerivatives) {
                pmeAddSelfEnergyKernel = cl::Kernel(program, "addSelfEnergy");
                pmeAddSelfEnergyKernel.setArg<cl::Buffer>(0, pmeEnergyBuffer.getDeviceBuffer());
                pmeAddSelfEnergyKernel.setArg<cl::Buffer>(1, cl.getPosq().getDeviceBuffer());
                pmeAddSelfEnergyKernel.setArg<cl::Buffer>(2, charges.getDeviceBuffer());
                pmeAddSelfEnergyKernel.setArg<cl::Buffer>(3, subsets.getDeviceBuffer());
            }
            pmeInterpolateForceKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
            pmeInterpolateForceKernel.setArg<cl::Buffer>(1, cl.getLongForceBuffer().getDeviceBuffer());
            pmeInterpolateForceKernel.setArg<cl::Buffer>(2, pmeGrid1.getDeviceBuffer());
            pmeInterpolateForceKernel.setArg<cl::Buffer>(11, pmeAtomGridIndex.getDeviceBuffer());
            pmeInterpolateForceKernel.setArg<cl::Buffer>(12, charges.getDeviceBuffer());
            pmeInterpolateForceKernel.setArg<cl::Buffer>(13, subsets.getDeviceBuffer());
            pmeInterpolateForceKernel.setArg<cl::Buffer>(14, sliceLambda.getDeviceBuffer());
            pmeFinishSpreadChargeKernel = cl::Kernel(program, "finishSpreadCharge");
            pmeFinishSpreadChargeKernel.setArg<cl::Buffer>(0, pmeGrid2.getDeviceBuffer());
            pmeFinishSpreadChargeKernel.setArg<cl::Buffer>(1, pmeGrid1.getDeviceBuffer());
            addEnergy->initialize(pmeEnergyBuffer, sliceLambda, requestedDerivs, sliceDerivIndices);
       }
    }

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
        if (cl.getUseDoublePrecision())
            sliceLambda.upload(sliceLambdaVec);
        else
            sliceLambda.upload(floatVector(sliceLambdaVec));
    }

    // Update particle and exception parameters if needed.

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
        cl.executeKernel(computeParamsKernel, cl.getPaddedNumAtoms());
        if (exclusionChargeProds.isInitialized())
            cl.executeKernel(computeExclusionParamsKernel, exclusionChargeProds.getSize());
        if (usePmeQueue) {
            vector<cl::Event> events(1);
            cl.getQueue().enqueueMarkerWithWaitList(NULL, &events[0]);
            pmeQueue.enqueueBarrierWithWaitList(&events);
        }
        ewaldSelfEnergy = 0.0;
        for (int j = 0; j < numSubsets; j++)
            ewaldSelfEnergy += sliceLambdaVec[j*(j+3)/2]*subsetSelfEnergy[j];
        recomputeParams = false;
    }

    // Do reciprocal space calculations.

    if (pmeGrid1.isInitialized() && includeReciprocal) {
        if (usePmeQueue && !includeEnergy)
            cl.setQueue(pmeQueue);

        // Invert the periodic box vectors.

        Vec3 boxVectors[3];
        cl.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
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

        setPeriodicBoxArgs(cl, pmeGridIndexKernel, 3);
        if (cl.getUseDoublePrecision()) {
            pmeGridIndexKernel.setArg<mm_double4>(8, recipBoxVectors[0]);
            pmeGridIndexKernel.setArg<mm_double4>(9, recipBoxVectors[1]);
            pmeGridIndexKernel.setArg<mm_double4>(10, recipBoxVectors[2]);
        }
        else {
            pmeGridIndexKernel.setArg<mm_float4>(8, recipBoxVectorsFloat[0]);
            pmeGridIndexKernel.setArg<mm_float4>(9, recipBoxVectorsFloat[1]);
            pmeGridIndexKernel.setArg<mm_float4>(10, recipBoxVectorsFloat[2]);
        }
        cl.executeKernel(pmeGridIndexKernel, cl.getNumAtoms());
        sort->sort(pmeAtomGridIndex);
        setPeriodicBoxArgs(cl, pmeSpreadChargeKernel, 2);
        if (cl.getUseDoublePrecision()) {
            pmeSpreadChargeKernel.setArg<mm_double4>(7, recipBoxVectors[0]);
            pmeSpreadChargeKernel.setArg<mm_double4>(8, recipBoxVectors[1]);
            pmeSpreadChargeKernel.setArg<mm_double4>(9, recipBoxVectors[2]);
        }
        else {
            pmeSpreadChargeKernel.setArg<mm_float4>(7, recipBoxVectorsFloat[0]);
            pmeSpreadChargeKernel.setArg<mm_float4>(8, recipBoxVectorsFloat[1]);
            pmeSpreadChargeKernel.setArg<mm_float4>(9, recipBoxVectorsFloat[2]);
        }
        cl.executeKernel(pmeSpreadChargeKernel, cl.getNumAtoms());
        cl.executeKernel(pmeFinishSpreadChargeKernel, numSubsets*gridSizeX*gridSizeY*gridSizeZ);

        fft->execFFT(true, cl.getQueue());

        mm_double4 boxSize = cl.getPeriodicBoxSizeDouble();
        if (cl.getUseDoublePrecision()) {
            pmeConvolutionKernel.setArg<mm_double4>(4, recipBoxVectors[0]);
            pmeConvolutionKernel.setArg<mm_double4>(5, recipBoxVectors[1]);
            pmeConvolutionKernel.setArg<mm_double4>(6, recipBoxVectors[2]);
            pmeEvalEnergyKernel.setArg<mm_double4>(5, recipBoxVectors[0]);
            pmeEvalEnergyKernel.setArg<mm_double4>(6, recipBoxVectors[1]);
            pmeEvalEnergyKernel.setArg<mm_double4>(7, recipBoxVectors[2]);
        }
        else {
            pmeConvolutionKernel.setArg<mm_float4>(4, recipBoxVectorsFloat[0]);
            pmeConvolutionKernel.setArg<mm_float4>(5, recipBoxVectorsFloat[1]);
            pmeConvolutionKernel.setArg<mm_float4>(6, recipBoxVectorsFloat[2]);
            pmeEvalEnergyKernel.setArg<mm_float4>(5, recipBoxVectorsFloat[0]);
            pmeEvalEnergyKernel.setArg<mm_float4>(6, recipBoxVectorsFloat[1]);
            pmeEvalEnergyKernel.setArg<mm_float4>(7, recipBoxVectorsFloat[2]);
        }
        if (includeEnergy || hasDerivatives) {
            cl.executeKernel(pmeEvalEnergyKernel, gridSizeX*gridSizeY*gridSizeZ);
            if (hasOffsets || hasDerivatives)
                cl.executeKernel(pmeAddSelfEnergyKernel, cl.getNumAtoms());
            else
                energy = ewaldSelfEnergy;
        }

        cl.executeKernel(pmeConvolutionKernel, gridSizeX*gridSizeY*(gridSizeZ/2+1));
        fft->execFFT(false, cl.getQueue());
        setPeriodicBoxArgs(cl, pmeInterpolateForceKernel, 3);
        if (cl.getUseDoublePrecision()) {
            pmeInterpolateForceKernel.setArg<mm_double4>(8, recipBoxVectors[0]);
            pmeInterpolateForceKernel.setArg<mm_double4>(9, recipBoxVectors[1]);
            pmeInterpolateForceKernel.setArg<mm_double4>(10, recipBoxVectors[2]);
        }
        else {
            pmeInterpolateForceKernel.setArg<mm_float4>(8, recipBoxVectorsFloat[0]);
            pmeInterpolateForceKernel.setArg<mm_float4>(9, recipBoxVectorsFloat[1]);
            pmeInterpolateForceKernel.setArg<mm_float4>(10, recipBoxVectorsFloat[2]);
        }
        if (deviceIsCpu)
            cl.executeKernel(pmeInterpolateForceKernel, 2*cl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(), 1);
        else
            cl.executeKernel(pmeInterpolateForceKernel, cl.getNumAtoms());
        if (usePmeQueue) {
            pmeQueue.enqueueMarkerWithWaitList(NULL, &pmeSyncEvent);
            cl.restoreDefaultQueue();
        }
    }
    return energy;
}

void OpenCLCalcSlicedPmeForceKernel::copyParametersToContext(ContextImpl& context, const SlicedPmeForce& force) {
    // Make sure the new parameters are acceptable.

    if (force.getNumParticles() != cl.getNumAtoms())
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
    int numContexts = cl.getPlatformData().contexts.size();
    int startIndex = cl.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cl.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions != exceptionPairs.size())
        throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");

    // Record the per-particle parameters.

    vector<float> baseParticleChargeVec(cl.getPaddedNumAtoms(), 0.0);
    vector<int> subsetVec(cl.getPaddedNumAtoms());
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
    if (cl.getContextIndex() == 0) {
        for (int i = 0; i < cl.getNumAtoms(); i++)
            subsetSelfEnergy[subsetVec[i]] += baseParticleChargeVec[i]*baseParticleChargeVec[i];
        for (int j = 0; j < numSubsets; j++)
            subsetSelfEnergy[j] *= -ONE_4PI_EPS0*alpha/sqrt(M_PI);
    }
    cl.invalidateMolecules(info);
    recomputeParams = true;
}

void OpenCLCalcSlicedPmeForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (cl.getPlatformData().useCpuPme)
        cpuPme.getAs<CalcPmeReciprocalForceKernel>().getPMEParameters(alpha, nx, ny, nz);
    else {
        alpha = this->alpha;
        nx = gridSizeX;
        ny = gridSizeY;
        nz = gridSizeZ;
    }
}

class OpenCLCalcSlicedNonbondedForceKernel::ForceInfo : public OpenCLForceInfo {
public:
    ForceInfo(int requiredBuffers, const SlicedNonbondedForce& force) : OpenCLForceInfo(requiredBuffers), force(force) {
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

class OpenCLCalcSlicedNonbondedForceKernel::PmeIO : public CalcPmeReciprocalForceKernel::IO {
public:
    PmeIO(OpenCLContext& cl, cl::Kernel addForcesKernel) : cl(cl), addForcesKernel(addForcesKernel) {
        forceTemp.initialize<mm_float4>(cl, cl.getNumAtoms(), "PmeForce");
        addForcesKernel.setArg<cl::Buffer>(0, forceTemp.getDeviceBuffer());
    }
    float* getPosq() {
        cl.getPosq().download(posq);
        return (float*) &posq[0];
    }
    void setForce(float* force) {
        forceTemp.upload(force);
        addForcesKernel.setArg<cl::Buffer>(1, cl.getLongForceBuffer().getDeviceBuffer());
        cl.executeKernel(addForcesKernel, cl.getNumAtoms());
    }
private:
    OpenCLContext& cl;
    vector<mm_float4> posq;
    OpenCLArray forceTemp;
    cl::Kernel addForcesKernel;
};

class OpenCLCalcSlicedNonbondedForceKernel::PmePreComputation : public OpenCLContext::ForcePreComputation {
public:
    PmePreComputation(OpenCLContext& cl, Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : cl(cl), pme(pme), io(io) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        Vec3 boxVectors[3] = {Vec3(cl.getPeriodicBoxSize().x, 0, 0), Vec3(0, cl.getPeriodicBoxSize().y, 0), Vec3(0, 0, cl.getPeriodicBoxSize().z)};
        pme.getAs<CalcPmeReciprocalForceKernel>().beginComputation(io, boxVectors, includeEnergy);
    }
private:
    OpenCLContext& cl;
    Kernel pme;
    CalcPmeReciprocalForceKernel::IO& io;
};

class OpenCLCalcSlicedNonbondedForceKernel::PmePostComputation : public OpenCLContext::ForcePostComputation {
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

class OpenCLCalcSlicedNonbondedForceKernel::SyncQueuePreComputation : public OpenCLContext::ForcePreComputation {
public:
    SyncQueuePreComputation(OpenCLContext& cl, cl::CommandQueue queue, int forceGroup) : cl(cl), queue(queue), forceGroup(forceGroup) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            vector<cl::Event> events(1);
            cl.getQueue().enqueueMarkerWithWaitList(NULL, &events[0]);
            queue.enqueueBarrierWithWaitList(&events);
        }
    }
private:
    OpenCLContext& cl;
    cl::CommandQueue queue;
    int forceGroup;
};

class OpenCLCalcSlicedNonbondedForceKernel::SyncQueuePostComputation : public OpenCLContext::ForcePostComputation {
public:
    SyncQueuePostComputation(OpenCLContext& cl, cl::Event& event, int forceGroup) : cl(cl), event(event), forceGroup(forceGroup) {}
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            vector<cl::Event> events(1);
            events[0] = event;
            event = cl::Event();
            cl.getQueue().enqueueBarrierWithWaitList(&events);
        }
        return 0.0;
    }
private:
    OpenCLContext& cl;
    cl::Event& event;
    int forceGroup;
};

class OpenCLCalcSlicedNonbondedForceKernel::AddEnergyPostComputation : public OpenCLContext::ForcePostComputation {
public:
    AddEnergyPostComputation(OpenCLContext& cl, OpenCLArray& pmeEnergyBuffer, OpenCLArray& ljpmeEnergyBuffer, OpenCLArray& sliceLambdas, int forceGroup) : cl(cl),
            pmeEnergyBuffer(pmeEnergyBuffer), ljpmeEnergyBuffer(ljpmeEnergyBuffer), sliceLambdas(sliceLambdas), forceGroup(forceGroup) {
    }
    void setKernel(cl::Kernel kernel) {
        addEnergyKernel = kernel;
        addEnergyKernel.setArg<cl::Buffer>(0, pmeEnergyBuffer.getDeviceBuffer());
        addEnergyKernel.setArg<cl::Buffer>(1, ljpmeEnergyBuffer.getDeviceBuffer());
        addEnergyKernel.setArg<cl::Buffer>(2, cl.getEnergyBuffer().getDeviceBuffer());
        addEnergyKernel.setArg<cl::Buffer>(3, sliceLambdas.getDeviceBuffer());
        addEnergyKernel.setArg<cl_int>(4, pmeEnergyBuffer.getSize()/sliceLambdas.getSize());
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            if (includeEnergy)
                cl.executeKernel(addEnergyKernel, pmeEnergyBuffer.getSize());
        }
        return 0.0;
    }
private:
    OpenCLContext& cl;
    cl::Kernel addEnergyKernel;
    OpenCLArray& pmeEnergyBuffer;
    OpenCLArray& ljpmeEnergyBuffer;
    OpenCLArray& sliceLambdas;
    int forceGroup;
};

class OpenCLCalcSlicedNonbondedForceKernel::DispersionCorrectionPostComputation : public OpenCLContext::ForcePostComputation {
public:
    DispersionCorrectionPostComputation(OpenCLContext& cl, vector<double>& dispersionCoefficients, vector<mm_double2>& sliceLambdas, vector<string>& scalingParams, vector<mm_int2>& sliceScalingParamDerivs, int forceGroup) :
                cl(cl), dispersionCoefficients(dispersionCoefficients), sliceLambdas(sliceLambdas), scalingParams(scalingParams), sliceScalingParamDerivs(sliceScalingParamDerivs), forceGroup(forceGroup) {
        numSlices = dispersionCoefficients.size();
        hasDerivs = false;
        for (int slice = 0; slice < numSlices; slice++)
            hasDerivs = hasDerivs || sliceScalingParamDerivs[slice].y != -1;
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) == 0)
            return 0;
        // if (!cl.getWorkThread().isCurrentThread())  // OpenMM 8.0
            cl.getWorkThread().flush();
        double energy = 0.0;
        mm_double4 boxSize = cl.getPeriodicBoxSizeDouble();
        double volume = boxSize.x*boxSize.y*boxSize.z;
        for (int slice = 0; slice < numSlices; slice++)
            energy += sliceLambdas[slice].y*dispersionCoefficients[slice]/volume;
        if (hasDerivs) {
            map<string, double>& energyParamDerivs = cl.getEnergyParamDerivWorkspace();
            for (int slice = 0; slice < numSlices; slice++) {
                int index = sliceScalingParamDerivs[slice].y;
                if (index != -1)
                    energyParamDerivs[scalingParams[index]] += dispersionCoefficients[slice]/volume;
            }
        }
        return energy;
    }
private:
    OpenCLContext& cl;
    vector<double>& dispersionCoefficients;
    vector<mm_double2>& sliceLambdas;
    vector<string>& scalingParams;
    bool hasDerivs;
    vector<mm_int2>& sliceScalingParamDerivs;
    int forceGroup;
    int numSlices;
};

OpenCLCalcSlicedNonbondedForceKernel::~OpenCLCalcSlicedNonbondedForceKernel() {
    if (sort != NULL)
        delete sort;
    if (fft != NULL)
        delete fft;
    if (dispersionFft != NULL)
        delete dispersionFft;
    if (pmeio != NULL)
        delete pmeio;
}

void OpenCLCalcSlicedNonbondedForceKernel::initialize(const System& system, const SlicedNonbondedForce& force) {
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "slicedNonbonded"+cl.intToString(forceIndex)+"_";

    realToFixedPoint = Platform::getOpenMMVersion()[0] == '7' ? OpenCLPmeSlicingKernelSources::realToFixedPoint : "";

    int numParticles = force.getNumParticles();
    numSubsets = force.getNumSubsets();
    numSlices = force.getNumSlices();
    sliceLambdasVec.resize(numSlices, mm_double2(1, 1));
    sliceScalingParams.resize(numSlices, mm_int2(-1, -1));
    sliceScalingParamDerivsVec.resize(numSlices, mm_int2(-1, -1));
    subsetSelfEnergy.resize(numSlices, mm_double2(0, 0));

    subsetsVec.resize(cl.getPaddedNumAtoms(), 0);
    for (int i = 0; i < numParticles; i++)
        subsetsVec[i] = force.getParticleSubset(i);
    subsets.initialize<int>(cl, cl.getPaddedNumAtoms(), "subsets");
    subsets.upload(subsetsVec);

    int numDerivs = force.getNumScalingParameterDerivatives();
    set<string> derivs;
    for (int i = 0; i < numDerivs; i++)
        derivs.insert(force.getScalingParameterDerivativeName(i));

    int numScalingParams = force.getNumScalingParameters();
    scalingParams.resize(numScalingParams);
    for (int index = 0; index < numScalingParams; index++) {
        int subset1, subset2;
        bool includeLJ, includeCoulomb;
        force.getScalingParameter(index, scalingParams[index], subset1, subset2, includeLJ, includeCoulomb);
        int slice = force.getSliceIndex(subset1, subset2);
        sliceScalingParams[slice] = mm_int2(includeCoulomb ? index : -1, includeLJ ? index : -1);
        if (derivs.find(scalingParams[index]) != derivs.end())
            sliceScalingParamDerivsVec[slice] = mm_int2(includeCoulomb ? index : -1, includeLJ ? index : -1);
    }

    size_t sizeOfReal = cl.getUseDoublePrecision() ? sizeof(double) : sizeof(float);
    sliceLambdas.initialize(cl, numSlices, 2*sizeOfReal, "sliceLambdas");
    if (cl.getUseDoublePrecision())
        sliceLambdas.upload(sliceLambdasVec);
    else
        sliceLambdas.upload(double2Tofloat2(sliceLambdasVec));

    if (numDerivs > 0) {
        sliceScalingParamDerivs.initialize<mm_int2>(cl, numSlices, "sliceScalingParamDerivs");
        sliceScalingParamDerivs.upload(sliceScalingParamDerivsVec);
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

    vector<mm_float4> baseParticleParamVec(cl.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
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
    usePosqCharges = hasCoulomb ? cl.requestPosqCharges() : false;
    map<string, string> defines;
    defines["HAS_COULOMB"] = (hasCoulomb ? "1" : "0");
    defines["HAS_LENNARD_JONES"] = (hasLJ ? "1" : "0");
    defines["USE_LJ_SWITCH"] = (useCutoff && force.getUseSwitchingFunction() ? "1" : "0");
    if (useCutoff) {
        // Compute the reaction field constants.

        double reactionFieldK = pow(force.getCutoffDistance(), -3.0)*(force.getReactionFieldDielectric()-1.0)/(2.0*force.getReactionFieldDielectric()+1.0);
        double reactionFieldC = (1.0 / force.getCutoffDistance())*(3.0*force.getReactionFieldDielectric())/(2.0*force.getReactionFieldDielectric()+1.0);
        defines["REACTION_FIELD_K"] = cl.doubleToString(reactionFieldK);
        defines["REACTION_FIELD_C"] = cl.doubleToString(reactionFieldC);

        // Compute the switching coefficients.

        if (force.getUseSwitchingFunction()) {
            defines["LJ_SWITCH_CUTOFF"] = cl.doubleToString(force.getSwitchingDistance());
            defines["LJ_SWITCH_C3"] = cl.doubleToString(10/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 3.0));
            defines["LJ_SWITCH_C4"] = cl.doubleToString(15/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 4.0));
            defines["LJ_SWITCH_C5"] = cl.doubleToString(6/pow(force.getSwitchingDistance()-force.getCutoffDistance(), 5.0));
        }
    }
    if (force.getUseDispersionCorrection() && cl.getContextIndex() == 0 && hasLJ && useCutoff && usePeriodic && !doLJPME)
        dispersionCoefficients = SlicedNonbondedForceImpl::calcDispersionCorrections(system, force);
    alpha = 0;
    ewaldSelfEnergy = 0.0;
    map<string, string> paramsDefines;
    paramsDefines["NUM_SUBSETS"] = cl.intToString(numSubsets);
    paramsDefines["ONE_4PI_EPS0"] = cl.doubleToString(ONE_4PI_EPS0);
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
        defines["EWALD_ALPHA"] = cl.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cl.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        if (cl.getContextIndex() == 0) {
            paramsDefines["INCLUDE_EWALD"] = "1";
            paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cl.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
            for (int i = 0; i < numParticles; i++)
                subsetSelfEnergy[subsetsVec[i]].x -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
            for (int i = 0; i < numSubsets; i++)
                ewaldSelfEnergy += sliceLambdasVec[i*(i+3)/2].x*subsetSelfEnergy[i].x;

            // Create the reciprocal space kernels.

            map<string, string> replacements;
            replacements["NUM_ATOMS"] = cl.intToString(numParticles);
            replacements["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
            replacements["KMAX_X"] = cl.intToString(kmaxx);
            replacements["KMAX_Y"] = cl.intToString(kmaxy);
            replacements["KMAX_Z"] = cl.intToString(kmaxz);
            replacements["EXP_COEFFICIENT"] = cl.doubleToString(-1.0/(4.0*alpha*alpha));
            replacements["ONE_4PI_EPS0"] = cl.doubleToString(ONE_4PI_EPS0);
            replacements["M_PI"] = cl.doubleToString(M_PI);
            cl::Program program = cl.createProgram(realToFixedPoint+CommonPmeSlicingKernelSources::ewald, replacements);
            ewaldSumsKernel = cl::Kernel(program, "calculateEwaldCosSinSums");
            ewaldForcesKernel = cl::Kernel(program, "calculateEwaldForces");
            int elementSize = (cl.getUseDoublePrecision() ? sizeof(mm_double2) : sizeof(mm_float2));
            cosSinSums.initialize(cl, (2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1), elementSize, "cosSinSums");
        }
    }
    else if (((nonbondedMethod == PME || nonbondedMethod == LJPME) && hasCoulomb) || doLJPME) {
        // Compute the PME parameters.

        SlicedNonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSizeX, gridSizeY, gridSizeZ, false);
        gridSizeX = OpenCLVkFFT3D::findLegalDimension(gridSizeX);
        gridSizeY = OpenCLVkFFT3D::findLegalDimension(gridSizeY);
        gridSizeZ = OpenCLVkFFT3D::findLegalDimension(gridSizeZ);
        if (doLJPME) {
            SlicedNonbondedForceImpl::calcPMEParameters(system, force, dispersionAlpha, dispersionGridSizeX,
                                                  dispersionGridSizeY, dispersionGridSizeZ, true);
            dispersionGridSizeX = OpenCLVkFFT3D::findLegalDimension(dispersionGridSizeX);
            dispersionGridSizeY = OpenCLVkFFT3D::findLegalDimension(dispersionGridSizeY);
            dispersionGridSizeZ = OpenCLVkFFT3D::findLegalDimension(dispersionGridSizeZ);
        }
        defines["EWALD_ALPHA"] = cl.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cl.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        defines["DO_LJPME"] = doLJPME ? "1" : "0";
        if (doLJPME) {
            defines["EWALD_DISPERSION_ALPHA"] = cl.doubleToString(dispersionAlpha);
            double invRCut6 = pow(force.getCutoffDistance(), -6);
            double dalphaR = dispersionAlpha * force.getCutoffDistance();
            double dar2 = dalphaR*dalphaR;
            double dar4 = dar2*dar2;
            double multShift6 = -invRCut6*(1.0 - exp(-dar2) * (1.0 + dar2 + 0.5*dar4));
            defines["INVCUT6"] = cl.doubleToString(invRCut6);
            defines["MULTSHIFT6"] = cl.doubleToString(multShift6);
        }
        if (cl.getContextIndex() == 0) {
            paramsDefines["INCLUDE_EWALD"] = "1";
            paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cl.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
            for (int i = 0; i < numParticles; i++)
                subsetSelfEnergy[subsetsVec[i]].x -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
            if (doLJPME) {
                paramsDefines["INCLUDE_LJPME"] = "1";
                paramsDefines["LJPME_SELF_ENERGY_SCALE"] = cl.doubleToString(pow(dispersionAlpha, 6)/3.0);
                for (int i = 0; i < numParticles; i++)
                    subsetSelfEnergy[subsetsVec[i]].y += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
            }
            for (int i = 0; i < numSubsets; i++)
                ewaldSelfEnergy += sliceLambdasVec[i*(i+3)/2].x*subsetSelfEnergy[i].x + sliceLambdasVec[i*(i+3)/2].y*subsetSelfEnergy[i].y;
            pmeDefines["PME_ORDER"] = cl.intToString(PmeOrder);
            pmeDefines["NUM_ATOMS"] = cl.intToString(numParticles);
            pmeDefines["NUM_SUBSETS"] = cl.intToString(numSubsets);
            pmeDefines["NUM_SLICES"] = cl.intToString(numSlices);
            pmeDefines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
            pmeDefines["RECIP_EXP_FACTOR"] = cl.doubleToString(M_PI*M_PI/(alpha*alpha));
            pmeDefines["GRID_SIZE_X"] = cl.intToString(gridSizeX);
            pmeDefines["GRID_SIZE_Y"] = cl.intToString(gridSizeY);
            pmeDefines["GRID_SIZE_Z"] = cl.intToString(gridSizeZ);
            pmeDefines["EPSILON_FACTOR"] = cl.doubleToString(sqrt(ONE_4PI_EPS0));
            pmeDefines["M_PI"] = cl.doubleToString(M_PI);
            pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";
            bool deviceIsCpu = (cl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU);
            if (deviceIsCpu)
                pmeDefines["DEVICE_IS_CPU"] = "1";
            if (cl.getPlatformData().useCpuPme && !doLJPME && usePosqCharges) {
                // Create the CPU PME kernel.

                try {
                    cpuPme = getPlatform().createKernel(CalcPmeReciprocalForceKernel::Name(), *cl.getPlatformData().context);
                    cpuPme.getAs<CalcPmeReciprocalForceKernel>().initialize(gridSizeX, gridSizeY, gridSizeZ, numParticles, alpha, false);
                    cl::Program program = cl.createProgram(realToFixedPoint+CommonPmeSlicingKernelSources::pme, pmeDefines);
                    cl::Kernel addForcesKernel = cl::Kernel(program, "addForces");
                    pmeio = new PmeIO(cl, addForcesKernel);
                    cl.addPreComputation(new PmePreComputation(cl, cpuPme, *pmeio));
                    cl.addPostComputation(new PmePostComputation(cpuPme, *pmeio));
                }
                catch (OpenMMException& ex) {
                    // The CPU PME plugin isn't available.
                }
            }
            if (pmeio == NULL) {
                // Create required data structures.

                int elementSize = (cl.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
                int roundedZSize = PmeOrder*(int) ceil(gridSizeZ/(double) PmeOrder);
                int gridElements = gridSizeX*gridSizeY*roundedZSize*numSubsets;
                if (doLJPME) {
                    roundedZSize = PmeOrder*(int) ceil(dispersionGridSizeZ/(double) PmeOrder);
                    gridElements = max(gridElements, dispersionGridSizeX*dispersionGridSizeY*roundedZSize*numSubsets);
                }
                pmeGrid1.initialize(cl, gridElements, 2*elementSize, "pmeGrid1");
                pmeGrid2.initialize(cl, gridElements, 2*elementSize, "pmeGrid2");
                cl.addAutoclearBuffer(pmeGrid2);
                pmeBsplineModuliX.initialize(cl, gridSizeX, elementSize, "pmeBsplineModuliX");
                pmeBsplineModuliY.initialize(cl, gridSizeY, elementSize, "pmeBsplineModuliY");
                pmeBsplineModuliZ.initialize(cl, gridSizeZ, elementSize, "pmeBsplineModuliZ");
                if (doLJPME) {
                    pmeDispersionBsplineModuliX.initialize(cl, dispersionGridSizeX, elementSize, "pmeDispersionBsplineModuliX");
                    pmeDispersionBsplineModuliY.initialize(cl, dispersionGridSizeY, elementSize, "pmeDispersionBsplineModuliY");
                    pmeDispersionBsplineModuliZ.initialize(cl, dispersionGridSizeZ, elementSize, "pmeDispersionBsplineModuliZ");
                }
                pmeBsplineTheta.initialize(cl, PmeOrder*numParticles, 4*elementSize, "pmeBsplineTheta");
                pmeAtomRange.initialize<cl_int>(cl, gridSizeX*gridSizeY*gridSizeZ+1, "pmeAtomRange");
                pmeAtomGridIndex.initialize<mm_int2>(cl, numParticles, "pmeAtomGridIndex");
                int energyElementSize = (cl.getUseDoublePrecision() || cl.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
                int bufferSize = numSlices*cl.getNumThreadBlocks()*OpenCLContext::ThreadBlockSize;
                pmeEnergyBuffer.initialize(cl, bufferSize, energyElementSize, "pmeEnergyBuffer");
                cl.clearBuffer(pmeEnergyBuffer);
                sort = new OpenCLSort(cl, new SortTrait(), cl.getNumAtoms());
                fft = new OpenCLVkFFT3D(cl, gridSizeX, gridSizeY, gridSizeZ, numSubsets, true, pmeGrid1, pmeGrid2);
                if (doLJPME) {
                    ljpmeEnergyBuffer.initialize(cl, bufferSize, energyElementSize, "ljpmeEnergyBuffer");
                    cl.clearBuffer(ljpmeEnergyBuffer);
                    dispersionFft = new OpenCLVkFFT3D(cl, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, numSubsets, true, pmeGrid1, pmeGrid2);
                }

                string vendor = cl.getDevice().getInfo<CL_DEVICE_VENDOR>();
                bool isNvidia = (vendor.size() >= 6 && vendor.substr(0, 6) == "NVIDIA");
                usePmeQueue = (!cl.getPlatformData().disablePmeStream && !cl.getPlatformData().useCpuPme && isNvidia);
                int recipForceGroup = force.getReciprocalSpaceForceGroup();
                if (recipForceGroup < 0)
                    recipForceGroup = force.getForceGroup();
                if (usePmeQueue) {
                    pmeDefines["USE_PME_STREAM"] = "1";
                    pmeQueue = cl::CommandQueue(cl.getContext(), cl.getDevice());
                    cl.addPreComputation(new SyncQueuePreComputation(cl, pmeQueue, recipForceGroup));
                    cl.addPostComputation(new SyncQueuePostComputation(cl, pmeSyncEvent, recipForceGroup));
                }
                cl.addPostComputation(addEnergy = new AddEnergyPostComputation(cl, pmeEnergyBuffer, ljpmeEnergyBuffer, sliceLambdas, recipForceGroup));

                // Initialize the b-spline moduli.

                for (int grid = 0; grid < 2; grid++) {
                    int xsize, ysize, zsize;
                    OpenCLArray *xmoduli, *ymoduli, *zmoduli;
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
                        vector<cl_double> moduli(ndata);
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
                        {
                            if (moduli[i] < 1.0e-7)
                                moduli[i] = (moduli[(i-1+ndata)%ndata]+moduli[(i+1)%ndata])*0.5;
                        }
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
        int numContexts = cl.getPlatformData().contexts.size();
        int startIndex = cl.getContextIndex()*force.getNumExceptions()/numContexts;
        int endIndex = (cl.getContextIndex()+1)*force.getNumExceptions()/numContexts;
        int numExclusions = endIndex-startIndex;
        if (numExclusions > 0) {
            paramsDefines["HAS_EXCLUSIONS"] = "1";
            vector<vector<int> > atoms(numExclusions, vector<int>(2));
            exclusionAtoms.initialize<mm_int2>(cl, numExclusions, "exclusionAtoms");
            exclusionParams.initialize<mm_float4>(cl, numExclusions, "exclusionParams");
            vector<mm_int2> exclusionAtomsVec(numExclusions);
            for (int i = 0; i < numExclusions; i++) {
                int j = i+startIndex;
                exclusionAtomsVec[i] = mm_int2(exclusions[j].first, exclusions[j].second);
                atoms[i][0] = exclusions[j].first;
                atoms[i][1] = exclusions[j].second;
            }
            exclusionAtoms.upload(exclusionAtomsVec);
            map<string, string> replacements;
            replacements["PARAMS"] = cl.getBondedUtilities().addArgument(exclusionParams.getDeviceBuffer(), "float4");
            replacements["EWALD_ALPHA"] = cl.doubleToString(alpha);
            replacements["TWO_OVER_SQRT_PI"] = cl.doubleToString(2.0/sqrt(M_PI));
            replacements["DO_LJPME"] = doLJPME ? "1" : "0";
            replacements["USE_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
            if (doLJPME)
                replacements["EWALD_DISPERSION_ALPHA"] = cl.doubleToString(dispersionAlpha);
            replacements["LAMBDAS"] = cl.getBondedUtilities().addArgument(sliceLambdas.getDeviceBuffer(), "real2");
            stringstream code;
            if (numDerivs > 0) {
                string derivIndices = cl.getBondedUtilities().addArgument(sliceScalingParamDerivs.getDeviceBuffer(), "int2");
                code<<"int2 which = "<<derivIndices<<"[slice];"<<endl;
                for (int slice = 0; slice < numSlices; slice++) {
                    mm_int2 indices = sliceScalingParamDerivsVec[slice];
                    int index = max(indices.x, indices.y);
                    if (index != -1) {
                        string paramDeriv = cl.getBondedUtilities().addEnergyParameterDerivative(scalingParams[index]);
                        if (indices.x == index)
                            code<<paramDeriv<<" += (which.x == "<<index<<" ? clEnergy : 0);"<<endl;
                        if (doLJPME && indices.y == index)
                            code<<paramDeriv<<" += (which.y == "<<index<<" ? ljEnergy : 0);"<<endl;
                    }
                }
            }
            replacements["COMPUTE_DERIVATIVES"] = code.str();
            if (force.getIncludeDirectSpace())
                cl.getBondedUtilities().addInteraction(atoms, cl.replaceStrings(CommonPmeSlicingKernelSources::pmeExclusions, replacements), force.getForceGroup());
        }
    }

    // Add the interaction to the default nonbonded kernel.

    string source = cl.replaceStrings(CommonPmeSlicingKernelSources::coulombLennardJones, defines);
    charges.initialize(cl, cl.getPaddedNumAtoms(), cl.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "charges");
    baseParticleParams.initialize<mm_float4>(cl, cl.getPaddedNumAtoms(), "baseParticleParams");
    baseParticleParams.upload(baseParticleParamVec);
    map<string, string> replacements;
    replacements["ONE_4PI_EPS0"] = cl.doubleToString(ONE_4PI_EPS0);
    if (usePosqCharges) {
        replacements["CHARGE1"] = "posq1.w";
        replacements["CHARGE2"] = "posq2.w";
    }
    else {
        replacements["CHARGE1"] = prefix+"charge1";
        replacements["CHARGE2"] = prefix+"charge2";
    }
    if (hasCoulomb && !usePosqCharges)
        cl.getNonbondedUtilities().addParameter(OpenCLNonbondedUtilities::ParameterInfo(prefix+"charge", "real", 1, charges.getElementSize(), charges.getDeviceBuffer()));
    sigmaEpsilon.initialize<mm_float2>(cl, cl.getPaddedNumAtoms(), "sigmaEpsilon");
    if (hasLJ) {
        replacements["SIGMA_EPSILON1"] = prefix+"sigmaEpsilon1";
        replacements["SIGMA_EPSILON2"] = prefix+"sigmaEpsilon2";
        cl.getNonbondedUtilities().addParameter(OpenCLNonbondedUtilities::ParameterInfo(prefix+"sigmaEpsilon", "float", 2, sizeof(cl_float2), sigmaEpsilon.getDeviceBuffer()));
    }
    replacements["SUBSET1"] = prefix+"subset1";
    replacements["SUBSET2"] = prefix+"subset2";
    cl.getNonbondedUtilities().addParameter(OpenCLNonbondedUtilities::ParameterInfo(prefix+"subset", "int", 1, sizeof(int), subsets.getDeviceBuffer()));
    replacements["LAMBDA"] = prefix+"lambda";
    cl.getNonbondedUtilities().addArgument(OpenCLNonbondedUtilities::ParameterInfo(prefix+"lambda", "real", 2, 2*sizeOfReal, sliceLambdas.getDeviceBuffer()));
    stringstream code;
    if (numDerivs > 0) {
        string derivIndices = prefix+"derivIndices";
        cl.getNonbondedUtilities().addArgument(OpenCLNonbondedUtilities::ParameterInfo(derivIndices, "int", 2, 2*sizeof(int), sliceScalingParamDerivs.getDeviceBuffer()));
        code<<"int2 which = "<<derivIndices<<"[slice];"<<endl;
        for (int slice = 0; slice < numSlices; slice++) {
            mm_int2 indices = sliceScalingParamDerivsVec[slice];
            int index = max(indices.x, indices.y);
            if (index != -1) {
                string paramDeriv = cl.getNonbondedUtilities().addEnergyParameterDerivative(scalingParams[index]);
                if (hasCoulomb && indices.x == index)
                    code<<paramDeriv<<" += (which.x == "<<index<<" ? interactionScale*clEnergy : 0);"<<endl;
                if (hasLJ && indices.y == index)
                    code<<paramDeriv<<" += (which.y == "<<index<<" ? interactionScale*ljEnergy : 0);"<<endl;
            }
        }
    }
    replacements["COMPUTE_DERIVATIVES"] = code.str();
    source = cl.replaceStrings(source, replacements);
    if (force.getIncludeDirectSpace())
        cl.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup());

    // Initialize the exceptions.

    int numContexts = cl.getPlatformData().contexts.size();
    int startIndex = cl.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cl.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0) {
        paramsDefines["HAS_EXCEPTIONS"] = "1";
        exceptionAtoms.resize(numExceptions);
        vector<vector<int> > atoms(numExceptions, vector<int>(2));
        exceptionParams.initialize<mm_float4>(cl, numExceptions, "exceptionParams");
        baseExceptionParams.initialize<mm_float4>(cl, numExceptions, "baseExceptionParams");
        exceptionPairs.initialize<mm_int2>(cl, numExceptions, "exceptionPairs");
        exceptionSlices.initialize<int>(cl, numExceptions, "exceptionSlices");
        vector<mm_float4> baseExceptionParamsVec(numExceptions);
        vector<int> exceptionSlicesVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], atoms[i][0], atoms[i][1], chargeProd, sigma, epsilon);
            baseExceptionParamsVec[i] = mm_float4(chargeProd, sigma, epsilon, 0);
            exceptionAtoms[i] = make_pair(atoms[i][0], atoms[i][1]);
            exceptionSlicesVec[i] = force.getSliceIndex(atoms[i][0], atoms[i][1]);
        }
        baseExceptionParams.upload(baseExceptionParamsVec);
        exceptionPairs.upload(exceptionAtoms);
        exceptionSlices.upload(exceptionSlicesVec);
        map<string, string> replacements;
        replacements["APPLY_PERIODIC"] = (usePeriodic && force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0");
        replacements["PARAMS"] = cl.getBondedUtilities().addArgument(exceptionParams.getDeviceBuffer(), "float4");
        replacements["LAMBDAS"] = cl.getBondedUtilities().addArgument(sliceLambdas.getDeviceBuffer(), "real2");
        stringstream code;
        if (numDerivs > 0) {
            string derivIndices = cl.getBondedUtilities().addArgument(sliceScalingParamDerivs.getDeviceBuffer(), "int2");
            code<<"int2 which = "<<derivIndices<<"[slice];"<<endl;
            for (int slice = 0; slice < numSlices; slice++) {
                mm_int2 indices = sliceScalingParamDerivsVec[slice];
                int index = max(indices.x, indices.y);
                if (index != -1) {
                    string paramDeriv = cl.getBondedUtilities().addEnergyParameterDerivative(scalingParams[index]);
                    if (hasCoulomb && indices.x == index)
                        code<<paramDeriv<<" += (which.x == "<<index<<" ? clEnergy : 0);"<<endl;
                    if (hasLJ && indices.y == index)
                        code<<paramDeriv<<" += (which.y == "<<index<<" ? ljEnergy : 0);"<<endl;
                }
            }
        }
        replacements["COMPUTE_DERIVATIVES"] = code.str();
        if (force.getIncludeDirectSpace())
            cl.getBondedUtilities().addInteraction(atoms, cl.replaceStrings(CommonPmeSlicingKernelSources::nonbondedExceptions, replacements), force.getForceGroup());
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
    particleParamOffsets.initialize<mm_float4>(cl, max(force.getNumParticleParameterOffsets(), 1), "particleParamOffsets");
    particleOffsetIndices.initialize<cl_int>(cl, cl.getPaddedNumAtoms()+1, "particleOffsetIndices");
    vector<cl_int> particleOffsetIndicesVec, exceptionOffsetIndicesVec;
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
    exceptionParamOffsets.initialize<mm_float4>(cl, max((int) e.size(), 1), "exceptionParamOffsets");
    exceptionOffsetIndices.initialize<cl_int>(cl, exceptionOffsetIndicesVec.size(), "exceptionOffsetIndices");
    if (e.size() > 0) {
        exceptionParamOffsets.upload(e);
        exceptionOffsetIndices.upload(exceptionOffsetIndicesVec);
    }
    globalParams.initialize(cl, max((int) paramValues.size(), 1), cl.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "globalParams");
    if (paramValues.size() > 0)
        globalParams.upload(paramValues, true);
    recomputeParams = true;

    // Add post-computation for dispersion correction.

    if (dispersionCoefficients.size() > 0 && force.getIncludeDirectSpace())
        cl.addPostComputation(new DispersionCorrectionPostComputation(cl, dispersionCoefficients, sliceLambdasVec, scalingParams, sliceScalingParamDerivsVec, force.getForceGroup()));

    // Initialize the kernel for updating parameters.

    cl::Program program = cl.createProgram(CommonPmeSlicingKernelSources::nonbondedParameters, paramsDefines);
    computeParamsKernel = cl::Kernel(program, "computeParameters");
    computeExclusionParamsKernel = cl::Kernel(program, "computeExclusionParameters");
    info = new ForceInfo(0, force);
    cl.addForce(info);
}

double OpenCLCalcSlicedNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    bool deviceIsCpu = (cl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU);
    if (!hasInitializedKernel) {
        hasInitializedKernel = true;
        int index = 0;
        computeParamsKernel.setArg<cl::Buffer>(index++, cl.getEnergyBuffer().getDeviceBuffer());
        index++;
        computeParamsKernel.setArg<cl::Buffer>(index++, globalParams.getDeviceBuffer());
        computeParamsKernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
        computeParamsKernel.setArg<cl::Buffer>(index++, baseParticleParams.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, cl.getPosq().getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, charges.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, sigmaEpsilon.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, particleParamOffsets.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, particleOffsetIndices.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, subsets.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, sliceLambdas.getDeviceBuffer());
        if (exceptionParams.isInitialized()) {
            computeParamsKernel.setArg<cl_int>(index++, exceptionParams.getSize());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionPairs.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, baseExceptionParams.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionSlices.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionParams.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionParamOffsets.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionOffsetIndices.getDeviceBuffer());
        }
        if (exclusionParams.isInitialized()) {
            computeExclusionParamsKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl::Buffer>(1, charges.getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl::Buffer>(2, sigmaEpsilon.getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl::Buffer>(3, subsets.getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl_int>(4, exclusionParams.getSize());
            computeExclusionParamsKernel.setArg<cl::Buffer>(5, exclusionAtoms.getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl::Buffer>(6, exclusionParams.getDeviceBuffer());
        }
        if (cosSinSums.isInitialized()) {
            ewaldSumsKernel.setArg<cl::Buffer>(0, cl.getEnergyBuffer().getDeviceBuffer());
            ewaldSumsKernel.setArg<cl::Buffer>(1, cl.getPosq().getDeviceBuffer());
            ewaldSumsKernel.setArg<cl::Buffer>(2, cosSinSums.getDeviceBuffer());
            ewaldForcesKernel.setArg<cl::Buffer>(0, cl.getLongForceBuffer().getDeviceBuffer());
            ewaldForcesKernel.setArg<cl::Buffer>(1, cl.getPosq().getDeviceBuffer());
            ewaldForcesKernel.setArg<cl::Buffer>(2, cosSinSums.getDeviceBuffer());
        }
        if (pmeGrid1.isInitialized()) {
            // Create kernels for Coulomb PME.

            map<string, string> replacements;
            replacements["CHARGE"] = (usePosqCharges ? "pos.w" : "charges[atom]");
            cl::Program program = cl.createProgram(realToFixedPoint+cl.replaceStrings(CommonPmeSlicingKernelSources::pme, replacements), pmeDefines);
            pmeGridIndexKernel = cl::Kernel(program, "findAtomGridIndex");
            pmeSpreadChargeKernel = cl::Kernel(program, "gridSpreadCharge");
            pmeConvolutionKernel = cl::Kernel(program, "reciprocalConvolution");
            pmeEvalEnergyKernel = cl::Kernel(program, "gridEvaluateEnergy");
            pmeInterpolateForceKernel = cl::Kernel(program, "gridInterpolateForce");
            int elementSize = (cl.getUseDoublePrecision() ? sizeof(mm_double4) : sizeof(mm_float4));
            pmeGridIndexKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
            pmeGridIndexKernel.setArg<cl::Buffer>(1, pmeAtomGridIndex.getDeviceBuffer());
            pmeGridIndexKernel.setArg<cl::Buffer>(10, subsets.getDeviceBuffer());
            pmeSpreadChargeKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
            pmeSpreadChargeKernel.setArg<cl::Buffer>(1, pmeGrid2.getDeviceBuffer());
            pmeSpreadChargeKernel.setArg<cl::Buffer>(10, pmeAtomGridIndex.getDeviceBuffer());
            pmeSpreadChargeKernel.setArg<cl::Buffer>(11, charges.getDeviceBuffer());
            pmeConvolutionKernel.setArg<cl::Buffer>(0, pmeGrid2.getDeviceBuffer());
            pmeConvolutionKernel.setArg<cl::Buffer>(1, pmeBsplineModuliX.getDeviceBuffer());
            pmeConvolutionKernel.setArg<cl::Buffer>(2, pmeBsplineModuliY.getDeviceBuffer());
            pmeConvolutionKernel.setArg<cl::Buffer>(3, pmeBsplineModuliZ.getDeviceBuffer());
            pmeEvalEnergyKernel.setArg<cl::Buffer>(0, pmeGrid2.getDeviceBuffer());
            pmeEvalEnergyKernel.setArg<cl::Buffer>(1, pmeEnergyBuffer.getDeviceBuffer());
            pmeEvalEnergyKernel.setArg<cl::Buffer>(2, pmeBsplineModuliX.getDeviceBuffer());
            pmeEvalEnergyKernel.setArg<cl::Buffer>(3, pmeBsplineModuliY.getDeviceBuffer());
            pmeEvalEnergyKernel.setArg<cl::Buffer>(4, pmeBsplineModuliZ.getDeviceBuffer());
            pmeInterpolateForceKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
            pmeInterpolateForceKernel.setArg<cl::Buffer>(1, cl.getLongForceBuffer().getDeviceBuffer());
            pmeInterpolateForceKernel.setArg<cl::Buffer>(2, pmeGrid1.getDeviceBuffer());
            pmeInterpolateForceKernel.setArg<cl::Buffer>(11, pmeAtomGridIndex.getDeviceBuffer());
            pmeInterpolateForceKernel.setArg<cl::Buffer>(12, charges.getDeviceBuffer());
            pmeFinishSpreadChargeKernel = cl::Kernel(program, "finishSpreadCharge");
            pmeFinishSpreadChargeKernel.setArg<cl::Buffer>(0, pmeGrid2.getDeviceBuffer());
            pmeFinishSpreadChargeKernel.setArg<cl::Buffer>(1, pmeGrid1.getDeviceBuffer());
            addEnergy->setKernel(cl::Kernel(program, "addEnergy"));

            if (doLJPME) {
                // Create kernels for LJ PME.

                pmeDefines["EWALD_ALPHA"] = cl.doubleToString(dispersionAlpha);
                pmeDefines["GRID_SIZE_X"] = cl.intToString(dispersionGridSizeX);
                pmeDefines["GRID_SIZE_Y"] = cl.intToString(dispersionGridSizeY);
                pmeDefines["GRID_SIZE_Z"] = cl.intToString(dispersionGridSizeZ);
                pmeDefines["EPSILON_FACTOR"] = "1";
                pmeDefines["RECIP_EXP_FACTOR"] = cl.doubleToString(M_PI*M_PI/(dispersionAlpha*dispersionAlpha));
                pmeDefines["USE_LJPME"] = "1";
                pmeDefines["CHARGE_FROM_SIGEPS"] = "1";
                program = cl.createProgram(realToFixedPoint+CommonPmeSlicingKernelSources::pme, pmeDefines);
                pmeDispersionGridIndexKernel = cl::Kernel(program, "findAtomGridIndex");
                pmeDispersionSpreadChargeKernel = cl::Kernel(program, "gridSpreadCharge");
                pmeDispersionConvolutionKernel = cl::Kernel(program, "reciprocalConvolution");
                pmeDispersionEvalEnergyKernel = cl::Kernel(program, "gridEvaluateEnergy");
                pmeDispersionInterpolateForceKernel = cl::Kernel(program, "gridInterpolateForce");
                pmeDispersionGridIndexKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
                pmeDispersionGridIndexKernel.setArg<cl::Buffer>(1, pmeAtomGridIndex.getDeviceBuffer());
                pmeDispersionGridIndexKernel.setArg<cl::Buffer>(10, subsets.getDeviceBuffer());
                pmeDispersionSpreadChargeKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
                pmeDispersionSpreadChargeKernel.setArg<cl::Buffer>(1, pmeGrid2.getDeviceBuffer());
                pmeDispersionSpreadChargeKernel.setArg<cl::Buffer>(10, pmeAtomGridIndex.getDeviceBuffer());
                pmeDispersionSpreadChargeKernel.setArg<cl::Buffer>(11, sigmaEpsilon.getDeviceBuffer());
                pmeDispersionConvolutionKernel.setArg<cl::Buffer>(0, pmeGrid2.getDeviceBuffer());
                pmeDispersionConvolutionKernel.setArg<cl::Buffer>(1, pmeDispersionBsplineModuliX.getDeviceBuffer());
                pmeDispersionConvolutionKernel.setArg<cl::Buffer>(2, pmeDispersionBsplineModuliY.getDeviceBuffer());
                pmeDispersionConvolutionKernel.setArg<cl::Buffer>(3, pmeDispersionBsplineModuliZ.getDeviceBuffer());
                pmeDispersionEvalEnergyKernel.setArg<cl::Buffer>(0, pmeGrid2.getDeviceBuffer());
                pmeDispersionEvalEnergyKernel.setArg<cl::Buffer>(1, ljpmeEnergyBuffer.getDeviceBuffer());
                pmeDispersionEvalEnergyKernel.setArg<cl::Buffer>(2, pmeDispersionBsplineModuliX.getDeviceBuffer());
                pmeDispersionEvalEnergyKernel.setArg<cl::Buffer>(3, pmeDispersionBsplineModuliY.getDeviceBuffer());
                pmeDispersionEvalEnergyKernel.setArg<cl::Buffer>(4, pmeDispersionBsplineModuliZ.getDeviceBuffer());
                pmeDispersionInterpolateForceKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
                pmeDispersionInterpolateForceKernel.setArg<cl::Buffer>(1, cl.getLongForceBuffer().getDeviceBuffer());
                pmeDispersionInterpolateForceKernel.setArg<cl::Buffer>(2, pmeGrid1.getDeviceBuffer());
                pmeDispersionInterpolateForceKernel.setArg<cl::Buffer>(11, pmeAtomGridIndex.getDeviceBuffer());
                pmeDispersionInterpolateForceKernel.setArg<cl::Buffer>(12, sigmaEpsilon.getDeviceBuffer());
                pmeDispersionFinishSpreadChargeKernel = cl::Kernel(program, "finishSpreadCharge");
                pmeDispersionFinishSpreadChargeKernel.setArg<cl::Buffer>(0, pmeGrid2.getDeviceBuffer());
                pmeDispersionFinishSpreadChargeKernel.setArg<cl::Buffer>(1, pmeGrid1.getDeviceBuffer());
            }
       }
    }

    // Update scaling parameters if needed.

    bool scalingParamChanged = false;
    for (int slice = 0; slice < numSlices; slice++) {
        mm_int2 indices = sliceScalingParams[slice];
        int index = max(indices.x, indices.y);
        if (index != -1) {
            double paramValue = context.getParameter(scalingParams[index]);
            double oldValue = indices.x != -1 ? sliceLambdasVec[slice].x : sliceLambdasVec[slice].y;
            if (oldValue != paramValue) {
                sliceLambdasVec[slice] = mm_double2(indices.x == -1 ? 1.0 : paramValue, indices.y == -1 ? 1.0 : paramValue);
                scalingParamChanged = true;
            }
        }
    }
    if (scalingParamChanged) {
        ewaldSelfEnergy = 0.0;
        for (int i = 0; i < numSubsets; i++)
            ewaldSelfEnergy += sliceLambdasVec[i*(i+3)/2].x*subsetSelfEnergy[i].x + sliceLambdasVec[i*(i+3)/2].y*subsetSelfEnergy[i].y;
        if (cl.getUseDoublePrecision())
            sliceLambdas.upload(sliceLambdasVec);
        else
            sliceLambdas.upload(double2Tofloat2(sliceLambdasVec));
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
    double energy = (includeReciprocal ? ewaldSelfEnergy : 0.0);
    if (recomputeParams || hasOffsets) {
        computeParamsKernel.setArg<cl_int>(1, includeEnergy && includeReciprocal);
        cl.executeKernel(computeParamsKernel, cl.getPaddedNumAtoms());
        if (exclusionParams.isInitialized())
            cl.executeKernel(computeExclusionParamsKernel, exclusionParams.getSize());
        if (usePmeQueue) {
            vector<cl::Event> events(1);
            cl.getQueue().enqueueMarkerWithWaitList(NULL, &events[0]);
            pmeQueue.enqueueBarrierWithWaitList(&events);
        }
        if (hasOffsets)
            energy = 0.0; // The Ewald self energy was computed in the kernel.
        recomputeParams = false;
    }

    // Do reciprocal space calculations.

    if (cosSinSums.isInitialized() && includeReciprocal) {
        mm_double4 boxSize = cl.getPeriodicBoxSizeDouble();
        if (cl.getUseDoublePrecision()) {
            ewaldSumsKernel.setArg<mm_double4>(3, boxSize);
            ewaldForcesKernel.setArg<mm_double4>(3, boxSize);
        }
        else {
            ewaldSumsKernel.setArg<mm_float4>(3, mm_float4((float) boxSize.x, (float) boxSize.y, (float) boxSize.z, 0));
            ewaldForcesKernel.setArg<mm_float4>(3, mm_float4((float) boxSize.x, (float) boxSize.y, (float) boxSize.z, 0));
        }
        cl.executeKernel(ewaldSumsKernel, cosSinSums.getSize());
        cl.executeKernel(ewaldForcesKernel, cl.getNumAtoms());
    }
    if (pmeGrid1.isInitialized() && includeReciprocal) {
        if (usePmeQueue && !includeEnergy)
            cl.setQueue(pmeQueue);

        // Invert the periodic box vectors.

        Vec3 boxVectors[3];
        cl.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
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
            setPeriodicBoxArgs(cl, pmeGridIndexKernel, 2);
            if (cl.getUseDoublePrecision()) {
                pmeGridIndexKernel.setArg<mm_double4>(7, recipBoxVectors[0]);
                pmeGridIndexKernel.setArg<mm_double4>(8, recipBoxVectors[1]);
                pmeGridIndexKernel.setArg<mm_double4>(9, recipBoxVectors[2]);
            }
            else {
                pmeGridIndexKernel.setArg<mm_float4>(7, recipBoxVectorsFloat[0]);
                pmeGridIndexKernel.setArg<mm_float4>(8, recipBoxVectorsFloat[1]);
                pmeGridIndexKernel.setArg<mm_float4>(9, recipBoxVectorsFloat[2]);
            }
            cl.executeKernel(pmeGridIndexKernel, cl.getNumAtoms());
            sort->sort(pmeAtomGridIndex);
            setPeriodicBoxArgs(cl, pmeSpreadChargeKernel, 2);
            if (cl.getUseDoublePrecision()) {
                pmeSpreadChargeKernel.setArg<mm_double4>(7, recipBoxVectors[0]);
                pmeSpreadChargeKernel.setArg<mm_double4>(8, recipBoxVectors[1]);
                pmeSpreadChargeKernel.setArg<mm_double4>(9, recipBoxVectors[2]);
            }
            else {
                pmeSpreadChargeKernel.setArg<mm_float4>(7, recipBoxVectorsFloat[0]);
                pmeSpreadChargeKernel.setArg<mm_float4>(8, recipBoxVectorsFloat[1]);
                pmeSpreadChargeKernel.setArg<mm_float4>(9, recipBoxVectorsFloat[2]);
            }
            cl.executeKernel(pmeSpreadChargeKernel, cl.getNumAtoms());
            cl.executeKernel(pmeFinishSpreadChargeKernel, gridSizeX*gridSizeY*gridSizeZ);
            fft->execFFT(true, cl.getQueue());
            mm_double4 boxSize = cl.getPeriodicBoxSizeDouble();
            if (cl.getUseDoublePrecision()) {
                pmeConvolutionKernel.setArg<mm_double4>(4, recipBoxVectors[0]);
                pmeConvolutionKernel.setArg<mm_double4>(5, recipBoxVectors[1]);
                pmeConvolutionKernel.setArg<mm_double4>(6, recipBoxVectors[2]);
                pmeEvalEnergyKernel.setArg<mm_double4>(5, recipBoxVectors[0]);
                pmeEvalEnergyKernel.setArg<mm_double4>(6, recipBoxVectors[1]);
                pmeEvalEnergyKernel.setArg<mm_double4>(7, recipBoxVectors[2]);
            }
            else {
                pmeConvolutionKernel.setArg<mm_float4>(4, recipBoxVectorsFloat[0]);
                pmeConvolutionKernel.setArg<mm_float4>(5, recipBoxVectorsFloat[1]);
                pmeConvolutionKernel.setArg<mm_float4>(6, recipBoxVectorsFloat[2]);
                pmeEvalEnergyKernel.setArg<mm_float4>(5, recipBoxVectorsFloat[0]);
                pmeEvalEnergyKernel.setArg<mm_float4>(6, recipBoxVectorsFloat[1]);
                pmeEvalEnergyKernel.setArg<mm_float4>(7, recipBoxVectorsFloat[2]);
            }
            if (includeEnergy)
                cl.executeKernel(pmeEvalEnergyKernel, gridSizeX*gridSizeY*gridSizeZ);
            cl.executeKernel(pmeConvolutionKernel, gridSizeX*gridSizeY*gridSizeZ);
            fft->execFFT(false, cl.getQueue());
            setPeriodicBoxArgs(cl, pmeInterpolateForceKernel, 3);
            if (cl.getUseDoublePrecision()) {
                pmeInterpolateForceKernel.setArg<mm_double4>(8, recipBoxVectors[0]);
                pmeInterpolateForceKernel.setArg<mm_double4>(9, recipBoxVectors[1]);
                pmeInterpolateForceKernel.setArg<mm_double4>(10, recipBoxVectors[2]);
            }
            else {
                pmeInterpolateForceKernel.setArg<mm_float4>(8, recipBoxVectorsFloat[0]);
                pmeInterpolateForceKernel.setArg<mm_float4>(9, recipBoxVectorsFloat[1]);
                pmeInterpolateForceKernel.setArg<mm_float4>(10, recipBoxVectorsFloat[2]);
            }
            if (deviceIsCpu)
                cl.executeKernel(pmeInterpolateForceKernel, 2*cl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(), 1);
            else
                cl.executeKernel(pmeInterpolateForceKernel, cl.getNumAtoms());
        }

        if (doLJPME && hasLJ) {
            setPeriodicBoxArgs(cl, pmeDispersionGridIndexKernel, 2);
            if (cl.getUseDoublePrecision()) {
                pmeDispersionGridIndexKernel.setArg<mm_double4>(7, recipBoxVectors[0]);
                pmeDispersionGridIndexKernel.setArg<mm_double4>(8, recipBoxVectors[1]);
                pmeDispersionGridIndexKernel.setArg<mm_double4>(9, recipBoxVectors[2]);
            }
            else {
                pmeDispersionGridIndexKernel.setArg<mm_float4>(7, recipBoxVectorsFloat[0]);
                pmeDispersionGridIndexKernel.setArg<mm_float4>(8, recipBoxVectorsFloat[1]);
                pmeDispersionGridIndexKernel.setArg<mm_float4>(9, recipBoxVectorsFloat[2]);
            }
            cl.executeKernel(pmeDispersionGridIndexKernel, cl.getNumAtoms());
            if (!hasCoulomb)
                sort->sort(pmeAtomGridIndex);
            cl.clearBuffer(pmeGrid2);
            setPeriodicBoxArgs(cl, pmeDispersionSpreadChargeKernel, 2);
            if (cl.getUseDoublePrecision()) {
                pmeDispersionSpreadChargeKernel.setArg<mm_double4>(7, recipBoxVectors[0]);
                pmeDispersionSpreadChargeKernel.setArg<mm_double4>(8, recipBoxVectors[1]);
                pmeDispersionSpreadChargeKernel.setArg<mm_double4>(9, recipBoxVectors[2]);
            }
            else {
                pmeDispersionSpreadChargeKernel.setArg<mm_float4>(7, recipBoxVectorsFloat[0]);
                pmeDispersionSpreadChargeKernel.setArg<mm_float4>(8, recipBoxVectorsFloat[1]);
                pmeDispersionSpreadChargeKernel.setArg<mm_float4>(9, recipBoxVectorsFloat[2]);
            }
            cl.executeKernel(pmeDispersionSpreadChargeKernel, cl.getNumAtoms());
            cl.executeKernel(pmeDispersionFinishSpreadChargeKernel, gridSizeX*gridSizeY*gridSizeZ);
            dispersionFft->execFFT(true, cl.getQueue());
            if (cl.getUseDoublePrecision()) {
                pmeDispersionConvolutionKernel.setArg<mm_double4>(4, recipBoxVectors[0]);
                pmeDispersionConvolutionKernel.setArg<mm_double4>(5, recipBoxVectors[1]);
                pmeDispersionConvolutionKernel.setArg<mm_double4>(6, recipBoxVectors[2]);
                pmeDispersionEvalEnergyKernel.setArg<mm_double4>(5, recipBoxVectors[0]);
                pmeDispersionEvalEnergyKernel.setArg<mm_double4>(6, recipBoxVectors[1]);
                pmeDispersionEvalEnergyKernel.setArg<mm_double4>(7, recipBoxVectors[2]);
            }
            else {
                pmeDispersionConvolutionKernel.setArg<mm_float4>(4, recipBoxVectorsFloat[0]);
                pmeDispersionConvolutionKernel.setArg<mm_float4>(5, recipBoxVectorsFloat[1]);
                pmeDispersionConvolutionKernel.setArg<mm_float4>(6, recipBoxVectorsFloat[2]);
                pmeDispersionEvalEnergyKernel.setArg<mm_float4>(5, recipBoxVectorsFloat[0]);
                pmeDispersionEvalEnergyKernel.setArg<mm_float4>(6, recipBoxVectorsFloat[1]);
                pmeDispersionEvalEnergyKernel.setArg<mm_float4>(7, recipBoxVectorsFloat[2]);
            }
            // if (!hasCoulomb) cl.clearBuffer(ljpmeEnergyBuffer);  // Is this necessary?
            if (includeEnergy)
                cl.executeKernel(pmeDispersionEvalEnergyKernel, gridSizeX*gridSizeY*gridSizeZ);
            cl.executeKernel(pmeDispersionConvolutionKernel, gridSizeX*gridSizeY*gridSizeZ);
            dispersionFft->execFFT(false, cl.getQueue());
            setPeriodicBoxArgs(cl, pmeDispersionInterpolateForceKernel, 3);
            if (cl.getUseDoublePrecision()) {
                pmeDispersionInterpolateForceKernel.setArg<mm_double4>(8, recipBoxVectors[0]);
                pmeDispersionInterpolateForceKernel.setArg<mm_double4>(9, recipBoxVectors[1]);
                pmeDispersionInterpolateForceKernel.setArg<mm_double4>(10, recipBoxVectors[2]);
            }
            else {
                pmeDispersionInterpolateForceKernel.setArg<mm_float4>(8, recipBoxVectorsFloat[0]);
                pmeDispersionInterpolateForceKernel.setArg<mm_float4>(9, recipBoxVectorsFloat[1]);
                pmeDispersionInterpolateForceKernel.setArg<mm_float4>(10, recipBoxVectorsFloat[2]);
            }
            if (deviceIsCpu)
                cl.executeKernel(pmeDispersionInterpolateForceKernel, 2*cl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(), 1);
            else
                cl.executeKernel(pmeDispersionInterpolateForceKernel, cl.getNumAtoms());
        }
        if (usePmeQueue) {
            pmeQueue.enqueueMarkerWithWaitList(NULL, &pmeSyncEvent);
            cl.restoreDefaultQueue();
        }
    }

    return energy;
}

void OpenCLCalcSlicedNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const SlicedNonbondedForce& force) {
    // Make sure the new parameters are acceptable.

    if (force.getNumParticles() != cl.getNumAtoms())
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
    int numContexts = cl.getPlatformData().contexts.size();
    int startIndex = cl.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cl.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions != exceptionAtoms.size())
        throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");

    // Record the per-particle parameters.

    vector<mm_float4> baseParticleParamVec(cl.getPaddedNumAtoms(), mm_float4(0, 0, 0, 0));
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        baseParticleParamVec[i] = mm_float4(charge, sigma, epsilon, 0);
    }
    baseParticleParams.upload(baseParticleParamVec);

    // Record the exceptions.

    if (numExceptions > 0) {
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

    // Compute other values.

    ewaldSelfEnergy = 0.0;
    subsetSelfEnergy.assign(numSubsets, mm_double2(0, 0));
    if (nonbondedMethod == Ewald || nonbondedMethod == PME || nonbondedMethod == LJPME) {
        if (cl.getContextIndex() == 0) {
            for (int i = 0; i < force.getNumParticles(); i++) {
                subsetSelfEnergy[subsetsVec[i]].x -= baseParticleParamVec[i].x*baseParticleParamVec[i].x*ONE_4PI_EPS0*alpha/sqrt(M_PI);
                if (doLJPME)
                    subsetSelfEnergy[subsetsVec[i]].y += baseParticleParamVec[i].z*pow(baseParticleParamVec[i].y*dispersionAlpha, 6)/3.0;
            }
            for (int i = 0; i < force.getNumSubsets(); i++)
                ewaldSelfEnergy += sliceLambdasVec[i*(i+3)/2].x*subsetSelfEnergy[i].x + sliceLambdasVec[i*(i+3)/2].y*subsetSelfEnergy[i].y;
        }
    }
    if (force.getUseDispersionCorrection() && cl.getContextIndex() == 0 && (nonbondedMethod == CutoffPeriodic || nonbondedMethod == Ewald || nonbondedMethod == PME))
        dispersionCoefficients = SlicedNonbondedForceImpl::calcDispersionCorrections(context.getSystem(), force);
    cl.invalidateMolecules(info);
    recomputeParams = true;
}

void OpenCLCalcSlicedNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != PME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    if (cl.getPlatformData().useCpuPme)
        cpuPme.getAs<CalcPmeReciprocalForceKernel>().getPMEParameters(alpha, nx, ny, nz);
    else {
        alpha = this->alpha;
        nx = gridSizeX;
        ny = gridSizeY;
        nz = gridSizeZ;
    }
}

void OpenCLCalcSlicedNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != LJPME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    if (cl.getPlatformData().useCpuPme)
        //cpuPme.getAs<CalcPmeReciprocalForceKernel>().getLJPMEParameters(alpha, nx, ny, nz);
        throw OpenMMException("getPMEParametersInContext: CPUPME has not been implemented for LJPME yet.");
    else {
        alpha = this->dispersionAlpha;
        nx = dispersionGridSizeX;
        ny = dispersionGridSizeY;
        nz = dispersionGridSizeZ;
    }
}
