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
#include "internal/SlicedPmeForceImpl.h"
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
    AddEnergyPostComputation(OpenCLContext& cl, OpenCLArray& pmeEnergyBuffer, OpenCLArray& sliceLambda, int bufferSize, int forceGroup) :
        cl(cl), pmeEnergyBuffer(pmeEnergyBuffer), sliceLambda(sliceLambda), bufferSize(bufferSize), forceGroup(forceGroup) {}
    void setKernel(cl::Kernel kernel) {
        addEnergyKernel = kernel;
        addEnergyKernel.setArg<cl::Buffer>(0, pmeEnergyBuffer.getDeviceBuffer());
        addEnergyKernel.setArg<cl::Buffer>(1, cl.getEnergyBuffer().getDeviceBuffer());
        addEnergyKernel.setArg<cl::Buffer>(2, sliceLambda.getDeviceBuffer());
        addEnergyKernel.setArg<cl_int>(3, bufferSize);
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if (includeEnergy && (groups&(1<<forceGroup)) != 0)
            cl.executeKernel(addEnergyKernel, bufferSize);
        return 0.0;
    }
private:
    OpenCLContext& cl;
    cl::Kernel addEnergyKernel;
    OpenCLArray& pmeEnergyBuffer;
    OpenCLArray& sliceLambda;
    int bufferSize;
    int forceGroup;
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

    // Initialize switching parameters.

    sliceSwitchParamIndices.resize(numSlices, -1);
    for (int i = 0; i < force.getNumSwitchingParameters(); i++) {
        string param;
        int s1, s2;
        force.getSwitchingParameter(i, param, s1, s2);
        int index = find(switchParamNames.begin(), switchParamNames.end(), param) - switchParamNames.begin();
        if (index == switchParamNames.size()) {
            switchParamNames.push_back(param);
            switchParamValues.push_back(1.0);
        }
        sliceSwitchParamIndices[s2*(s2+1)/2+s1] = index;
    }
    sliceLambdaVec.resize(numSlices, 1.0);
    sliceLambda.initialize(cl, numSlices, sizeOfReal, "sliceLambda");
    if (cl.getUseDoublePrecision())
        sliceLambda.upload(sliceLambdaVec);
    else
        sliceLambda.upload(floatVector(sliceLambdaVec));

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
        pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";

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
        fft = new OpenCLVkFFT3D(cl, gridSizeX, gridSizeY, gridSizeZ, numSubsets, true, pmeGrid1, pmeGrid2);
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
        cl.addPostComputation(addEnergy = new AddEnergyPostComputation(cl, pmeEnergyBuffer, sliceLambda, bufferSize, recipForceGroup));

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

        int bufferSize = max(cl.getNumThreadBlocks()*OpenCLContext::ThreadBlockSize, nb->getNumEnergyBuffers());
        pairwiseEnergyBuffer.initialize(cl, numSlices*bufferSize, sizeOfMixed, "pairwiseEnergyBuffer");

        map<string, string> replacements;
        replacements["NUM_SLICES"] = cl.intToString(numSlices);
        replacements["BUFFER"] = prefix+"buffer";
        replacements["LAMBDA"] = prefix+"lambda";
        replacements["EWALD_ALPHA"] = cl.doubleToString(alpha);
        replacements["TWO_OVER_SQRT_PI"] = cl.doubleToString(2.0/sqrt(M_PI));
        replacements["ONE_4PI_EPS0"] = cl.doubleToString(ONE_4PI_EPS0);
        replacements["CHARGE1"] = usePosqCharges ? "posq1.w" : prefix+"charge1";
        replacements["CHARGE2"] = usePosqCharges ? "posq2.w" : prefix+"charge2";
        replacements["SUBSET1"] = prefix+"subset1";
        replacements["SUBSET2"] = prefix+"subset2";

        if (deviceIsCpu)
            nb->setKernelSource(cl.replaceStrings(OpenCLPmeSlicingKernelSources::nonbonded_cpu, replacements));
        else
            nb->setKernelSource(cl.replaceStrings(OpenCLPmeSlicingKernelSources::nonbonded, replacements));
        if (!usePosqCharges)
            nb->addParameter(ComputeParameterInfo(charges, prefix+"charge", "real", 1));
        nb->addParameter(ComputeParameterInfo(subsets, prefix+"subset", "int", 1));
        nb->addArgument(ComputeParameterInfo(sliceLambda, prefix+"lambda", "real", 1));
        nb->addArgument(ComputeParameterInfo(pairwiseEnergyBuffer, prefix+"buffer", "mixed", 1, false));
        string source = cl.replaceStrings(CommonPmeSlicingKernelSources::coulomb, replacements);
        nb->addInteraction(true, true, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup());
    }

    // Add code to subtract off the reciprocal part of excluded interactions.

    int numContexts = cl.getPlatformData().contexts.size();
    int startIndex = cl.getContextIndex()*force.getNumExceptions()/numContexts;
    int endIndex = (cl.getContextIndex()+1)*force.getNumExceptions()/numContexts;
    int numExclusions = endIndex-startIndex;
    hasExclusions = numExclusions > 0;
    if (hasExclusions) {
        exclusionAtoms.initialize<mm_int2>(cl, numExclusions, "exclusionAtoms");
        exclusionSlices.initialize<int>(cl, numExclusions, "exclusionSlices");
        exclusionChargeProds.initialize<float>(cl, numExclusions, "exclusionChargeProds");
        vector<mm_int2> exclusionAtomsVec(numExclusions);
        vector<int> exclusionSlicesVec(numExclusions);
       for (int k = 0; k < numExclusions; k++) {
            int atom1 = exclusions[k+startIndex].first;
            int atom2 = exclusions[k+startIndex].second;
            exclusionAtomsVec[k] = mm_int2(atom1, atom2);
            int i = subsetVec[atom1];
            int j = subsetVec[atom2];
            exclusionSlicesVec[k] = i > j ? i*(i+1)/2+j : j*(j+1)/2+i;
        }
        exclusionAtoms.upload(exclusionAtomsVec);
        exclusionSlices.upload(exclusionSlicesVec);
    }

    // Initialize the exceptions.

    startIndex = cl.getContextIndex()*exceptions.size()/numContexts;
    endIndex = (cl.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0) {
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
    }

    if (hasExclusions) {
        map<string, string> bondDefines;
        bondDefines["NUM_EXCLUSIONS"] = cl.intToString(numExclusions);
        bondDefines["NUM_EXCEPTIONS"] = cl.intToString(numExceptions);
        bondDefines["NUM_SLICES"] = cl.intToString(numSlices);
        bondDefines["EWALD_ALPHA"] = cl.doubleToString(alpha);
        bondDefines["TWO_OVER_SQRT_PI"] = cl.doubleToString(2.0/sqrt(M_PI));
        bondDefines["USE_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
        bondDefines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
        cl::Program bondProgram = cl.createProgram(CommonPmeSlicingKernelSources::slicedPmeBonds, bondDefines);
        computeBondsKernel = cl::Kernel(bondProgram, "computeBonds");
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
        if (hasExclusions) {
            computeBondsKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
            computeBondsKernel.setArg<cl::Buffer>(1, cl.getEnergyBuffer().getDeviceBuffer());
            computeBondsKernel.setArg<cl::Buffer>(2, cl.getLongForceBuffer().getDeviceBuffer());
            setPeriodicBoxArgs(cl, computeBondsKernel, 3);
            computeBondsKernel.setArg<cl::Buffer>(8, exclusionAtoms.getDeviceBuffer());
            computeBondsKernel.setArg<cl::Buffer>(9, exclusionSlices.getDeviceBuffer());
            computeBondsKernel.setArg<cl::Buffer>(10, exclusionChargeProds.getDeviceBuffer());
            computeBondsKernel.setArg<cl::Buffer>(11, exceptionAtoms.getDeviceBuffer());
            computeBondsKernel.setArg<cl::Buffer>(12, exceptionSlices.getDeviceBuffer());
            computeBondsKernel.setArg<cl::Buffer>(13, exceptionChargeProds.getDeviceBuffer());
            computeBondsKernel.setArg<cl::Buffer>(14, sliceLambda.getDeviceBuffer());
            computeBondsKernel.setArg<cl::Buffer>(15, pairwiseEnergyBuffer.getDeviceBuffer());
        }

        if (pmeGrid1.isInitialized()) {
            // Create kernels for Coulomb PME.

            map<string, string> replacements;
            replacements["CHARGE"] = (usePosqCharges ? "pos.w" : "charges[atom]");
            cl::Program program = cl.createProgram(cl.replaceStrings(CommonPmeSlicingKernelSources::slicedPme, replacements), pmeDefines);
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
            if (hasOffsets) {
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
            addEnergy->setKernel(cl::Kernel(program, "addEnergy"));
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

    // Do exclusion and exception calculations.

    if (hasExclusions && includeDirect)
       cl.executeKernel(computeBondsKernel, exclusionChargeProds.getSize());

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
        if (includeEnergy) {
            cl.executeKernel(pmeEvalEnergyKernel, gridSizeX*gridSizeY*gridSizeZ);
            if (hasOffsets)
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
