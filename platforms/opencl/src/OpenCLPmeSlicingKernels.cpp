/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2016 Stanford University and the Authors.           *
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

class OpenCLCalcSlicedPmeForceKernel::PmeIO : public CalcPmeReciprocalForceKernel::IO {
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

class OpenCLCalcSlicedPmeForceKernel::PmePreComputation : public OpenCLContext::ForcePreComputation {
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

class OpenCLCalcSlicedPmeForceKernel::PmePostComputation : public OpenCLContext::ForcePostComputation {
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
    SyncQueuePostComputation(OpenCLContext& cl, cl::Event& event, OpenCLArray& pmeEnergyBuffer, int forceGroup) : cl(cl), event(event),
            pmeEnergyBuffer(pmeEnergyBuffer), forceGroup(forceGroup) {
    }
    void setKernel(cl::Kernel kernel) {
        addEnergyKernel = kernel;
        addEnergyKernel.setArg<cl::Buffer>(0, pmeEnergyBuffer.getDeviceBuffer());
        addEnergyKernel.setArg<cl::Buffer>(1, cl.getEnergyBuffer().getDeviceBuffer());
        addEnergyKernel.setArg<cl_int>(2, pmeEnergyBuffer.getSize());
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            vector<cl::Event> events(1);
            events[0] = event;
            event = cl::Event();
            cl.getQueue().enqueueBarrierWithWaitList(&events);
            if (includeEnergy)
                cl.executeKernel(addEnergyKernel, pmeEnergyBuffer.getSize());
        }
        return 0.0;
    }
private:
    OpenCLContext& cl;
    cl::Event& event;
    cl::Kernel addEnergyKernel;
    OpenCLArray& pmeEnergyBuffer;
    int forceGroup;
};

OpenCLCalcSlicedPmeForceKernel::~OpenCLCalcSlicedPmeForceKernel() {
    if (sort != NULL)
        delete sort;
    if (fft != NULL)
        delete fft;
    if (pmeio != NULL)
        delete pmeio;
}

void OpenCLCalcSlicedPmeForceKernel::initialize(const System& system, const SlicedPmeForce& force) {
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "nonbonded"+cl.intToString(forceIndex)+"_";

    // Identify which exceptions are 1-4 interactions.

    set<int> exceptionsWithOffsets;
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge;
        force.getExceptionParameterOffset(i, param, exception, charge);
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
    int numSubsets = force.getNumSubsets();
    vector<float> baseParticleChargeVec(cl.getPaddedNumAtoms(), 0.0);
    vector<int> subsetVec(cl.getPaddedNumAtoms(), 0);
    vector<vector<int> > exclusionList(numParticles);
    for (int i = 0; i < numParticles; i++) {
        baseParticleChargeVec[i] = force.getParticleCharge(i);
        subsetVec[i] = force.getParticleSubset(i);
        exclusionList[i].push_back(i);
    }
    for (auto exclusion : exclusions) {
        exclusionList[exclusion.first].push_back(exclusion.second);
        exclusionList[exclusion.second].push_back(exclusion.first);
    }
    usePosqCharges = cl.requestPosqCharges();
    map<string, string> defines;
    defines["HAS_COULOMB"] = "1";
    defines["HAS_LENNARD_JONES"] = "0";
    alpha = 0;
    ewaldSelfEnergy = 0.0;
    map<string, string> paramsDefines;
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

    // Compute the PME parameters.

    SlicedPmeForceImpl::calcPMEParameters(system, force, alpha, gridSizeX, gridSizeY, gridSizeZ, false);
    gridSizeX = OpenCLVkFFT3D::findLegalDimension(gridSizeX);
    gridSizeY = OpenCLVkFFT3D::findLegalDimension(gridSizeY);
    gridSizeZ = OpenCLVkFFT3D::findLegalDimension(gridSizeZ);
    int roundedZSize = (int) ceil(gridSizeZ/(double) PmeOrder)*PmeOrder;

    defines["EWALD_ALPHA"] = cl.doubleToString(alpha);
    defines["TWO_OVER_SQRT_PI"] = cl.doubleToString(2.0/sqrt(M_PI));
    defines["USE_EWALD"] = "1";
    defines["DO_LJPME"] = "0";
    if (cl.getContextIndex() == 0) {
        paramsDefines["INCLUDE_EWALD"] = "1";
        paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cl.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
        for (int i = 0; i < numParticles; i++)
            ewaldSelfEnergy -= baseParticleChargeVec[i]*baseParticleChargeVec[i]*ONE_4PI_EPS0*alpha/sqrt(M_PI);
        pmeDefines["PME_ORDER"] = cl.intToString(PmeOrder);
        pmeDefines["NUM_ATOMS"] = cl.intToString(numParticles);
        pmeDefines["NUM_SUBSETS"] = cl.intToString(numSubsets);
        pmeDefines["PADDED_NUM_ATOMS"] = cl.intToString(cl.getPaddedNumAtoms());
        pmeDefines["RECIP_EXP_FACTOR"] = cl.doubleToString(M_PI*M_PI/(alpha*alpha));
        pmeDefines["GRID_SIZE_X"] = cl.intToString(gridSizeX);
        pmeDefines["GRID_SIZE_Y"] = cl.intToString(gridSizeY);
        pmeDefines["GRID_SIZE_Z"] = cl.intToString(gridSizeZ);
        pmeDefines["ROUNDED_Z_SIZE"] = cl.intToString(roundedZSize);
        pmeDefines["EPSILON_FACTOR"] = cl.doubleToString(sqrt(ONE_4PI_EPS0));
        pmeDefines["M_PI"] = cl.doubleToString(M_PI);
        pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";
        bool deviceIsCpu = (cl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU);
        if (deviceIsCpu)
            pmeDefines["DEVICE_IS_CPU"] = "1";
        if (cl.getPlatformData().useCpuPme && usePosqCharges) {
            // Create the CPU PME kernel.

            try {
                cpuPme = getPlatform().createKernel(CalcPmeReciprocalForceKernel::Name(), *cl.getPlatformData().context);
                cpuPme.getAs<CalcPmeReciprocalForceKernel>().initialize(gridSizeX, gridSizeY, gridSizeZ, numParticles, alpha, false);
                cl::Program program = cl.createProgram(CommonPmeSlicingKernelSources::slicedPme, pmeDefines);
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
            int gridElements = gridSizeX*gridSizeY*roundedZSize*numSubsets;
            pmeGrid1.initialize(cl, gridElements, 2*elementSize, "pmeGrid1");
            pmeGrid2.initialize(cl, gridElements, 2*elementSize, "pmeGrid2");
            cl.addAutoclearBuffer(pmeGrid2);
            pmeBsplineModuliX.initialize(cl, gridSizeX, elementSize, "pmeBsplineModuliX");
            pmeBsplineModuliY.initialize(cl, gridSizeY, elementSize, "pmeBsplineModuliY");
            pmeBsplineModuliZ.initialize(cl, gridSizeZ, elementSize, "pmeBsplineModuliZ");
            pmeBsplineTheta.initialize(cl, PmeOrder*numParticles, 4*elementSize, "pmeBsplineTheta");
            pmeAtomRange.initialize<cl_int>(cl, gridSizeX*gridSizeY*gridSizeZ+1, "pmeAtomRange");
            pmeAtomGridIndex.initialize<mm_int2>(cl, numParticles, "pmeAtomGridIndex");
            int energyElementSize = (cl.getUseDoublePrecision() || cl.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
            pmeEnergyBuffer.initialize(cl, cl.getNumThreadBlocks()*OpenCLContext::ThreadBlockSize, energyElementSize, "pmeEnergyBuffer");
            cl.clearBuffer(pmeEnergyBuffer);
            sort = new OpenCLSort(cl, new SortTrait(), cl.getNumAtoms());
            fft = new OpenCLVkFFT3D(cl, gridSizeX, gridSizeY, gridSizeZ, numSubsets, true, pmeGrid1, pmeGrid2);
            string vendor = cl.getDevice().getInfo<CL_DEVICE_VENDOR>();
            bool isNvidia = (vendor.size() >= 6 && vendor.substr(0, 6) == "NVIDIA");
            usePmeQueue = (!cl.getPlatformData().disablePmeStream && !cl.getPlatformData().useCpuPme && isNvidia);
            if (usePmeQueue) {
                pmeDefines["USE_PME_STREAM"] = "1";
                pmeQueue = cl::CommandQueue(cl.getContext(), cl.getDevice());
                int recipForceGroup = force.getReciprocalSpaceForceGroup();
                if (recipForceGroup < 0)
                    recipForceGroup = force.getForceGroup();
                cl.addPreComputation(new SyncQueuePreComputation(cl, pmeQueue, recipForceGroup));
                cl.addPostComputation(syncQueue = new SyncQueuePostComputation(cl, pmeSyncEvent, pmeEnergyBuffer, recipForceGroup));
            }

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
    }

    // Add code to subtract off the reciprocal part of excluded interactions.

    if (pmeio == NULL) {
        int numContexts = cl.getPlatformData().contexts.size();
        int startIndex = cl.getContextIndex()*force.getNumExceptions()/numContexts;
        int endIndex = (cl.getContextIndex()+1)*force.getNumExceptions()/numContexts;
        int numExclusions = endIndex-startIndex;
        if (numExclusions > 0) {
            paramsDefines["HAS_EXCLUSIONS"] = "1";
            vector<vector<int> > atoms(numExclusions, vector<int>(2));
            exclusionAtoms.initialize<mm_int2>(cl, numExclusions, "exclusionAtoms");
            exclusionChargeProds.initialize<float>(cl, numExclusions, "exclusionChargeProds");
            vector<mm_int2> exclusionAtomsVec(numExclusions);
            for (int i = 0; i < numExclusions; i++) {
                int j = i+startIndex;
                exclusionAtomsVec[i] = mm_int2(exclusions[j].first, exclusions[j].second);
                atoms[i][0] = exclusions[j].first;
                atoms[i][1] = exclusions[j].second;
            }
            exclusionAtoms.upload(exclusionAtomsVec);
            map<string, string> replacements;
            replacements["PARAMS"] = cl.getBondedUtilities().addArgument(exclusionChargeProds.getDeviceBuffer(), "float");
            replacements["EWALD_ALPHA"] = cl.doubleToString(alpha);
            replacements["TWO_OVER_SQRT_PI"] = cl.doubleToString(2.0/sqrt(M_PI));
            replacements["DO_LJPME"] = "0";
            replacements["USE_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
            if (force.getIncludeDirectSpace())
                cl.getBondedUtilities().addInteraction(atoms, cl.replaceStrings(CommonPmeSlicingKernelSources::slicedPmeExclusions, replacements), force.getForceGroup());
        }
    }

    // Add the interaction to the default nonbonded kernel.
    
    string source = cl.replaceStrings(CommonPmeSlicingKernelSources::coulombLennardJones, defines);
    charges.initialize(cl, cl.getPaddedNumAtoms(), cl.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "charges");
    baseParticleCharges.initialize<float>(cl, cl.getPaddedNumAtoms(), "baseParticleCharges");
    baseParticleCharges.upload(baseParticleChargeVec);
    subsets.initialize<int>(cl, cl.getPaddedNumAtoms(), "subsets");
    subsets.upload(subsetVec);
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
    if (!usePosqCharges)
        cl.getNonbondedUtilities().addParameter(OpenCLNonbondedUtilities::ParameterInfo(prefix+"charge", "real", 1, charges.getElementSize(), charges.getDeviceBuffer()));
    source = cl.replaceStrings(source, replacements);
    if (force.getIncludeDirectSpace())
        cl.getNonbondedUtilities().addInteraction(true, true, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup());

    // Initialize the exceptions.

    int numContexts = cl.getPlatformData().contexts.size();
    int startIndex = cl.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cl.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0) {
        paramsDefines["HAS_EXCEPTIONS"] = "1";
        exceptionAtoms.resize(numExceptions);
        vector<vector<int> > atoms(numExceptions, vector<int>(2));
        exceptionChargeProds.initialize<float>(cl, numExceptions, "exceptionChargeProds");
        baseExceptionChargeProds.initialize<float>(cl, numExceptions, "baseExceptionChargeProds");
        vector<float> baseExceptionChargeProdsVec(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            double chargeProd;
            force.getExceptionParameters(exceptions[startIndex+i], atoms[i][0], atoms[i][1], chargeProd);
            baseExceptionChargeProdsVec[i] = chargeProd;
            exceptionAtoms[i] = make_pair(atoms[i][0], atoms[i][1]);
        }
        baseExceptionChargeProds.upload(baseExceptionChargeProdsVec);
        map<string, string> replacements;
        replacements["APPLY_PERIODIC"] = (force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0");
        replacements["PARAMS"] = cl.getBondedUtilities().addArgument(exceptionChargeProds.getDeviceBuffer(), "float");
        if (force.getIncludeDirectSpace())
            cl.getBondedUtilities().addInteraction(atoms, cl.replaceStrings(CommonPmeSlicingKernelSources::slicedPmeExceptions, replacements), force.getForceGroup());
    }
    
    // Initialize parameter offsets.

    vector<vector<mm_float2> > particleOffsetVec(force.getNumParticles());
    vector<vector<mm_float2> > exceptionOffsetVec(numExceptions);
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string param;
        int particle;
        double charge;
        force.getParticleParameterOffset(i, param, particle, charge);
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
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge;
        force.getExceptionParameterOffset(i, param, exception, charge);
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
    particleParamOffsets.initialize<mm_float2>(cl, max(force.getNumParticleParameterOffsets(), 1), "particleParamOffsets");
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
    if (force.getNumParticleParameterOffsets() > 0) {
        particleParamOffsets.upload(p);
        particleOffsetIndices.upload(particleOffsetIndicesVec);
    }
    exceptionParamOffsets.initialize<mm_float2>(cl, max((int) e.size(), 1), "exceptionParamOffsets");
    exceptionOffsetIndices.initialize<cl_int>(cl, exceptionOffsetIndicesVec.size(), "exceptionOffsetIndices");
    if (e.size() > 0) {
        exceptionParamOffsets.upload(e);
        exceptionOffsetIndices.upload(exceptionOffsetIndicesVec);
    }
    globalParams.initialize(cl, max((int) paramValues.size(), 1), cl.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "globalParams");
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
    bool deviceIsCpu = (cl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU);
    if (!hasInitializedKernel) {
        hasInitializedKernel = true;
        int index = 0;
        computeParamsKernel.setArg<cl::Buffer>(index++, cl.getEnergyBuffer().getDeviceBuffer());
        index++;
        computeParamsKernel.setArg<cl::Buffer>(index++, globalParams.getDeviceBuffer());
        computeParamsKernel.setArg<cl_int>(index++, cl.getPaddedNumAtoms());
        computeParamsKernel.setArg<cl::Buffer>(index++, baseParticleCharges.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, cl.getPosq().getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, charges.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, particleParamOffsets.getDeviceBuffer());
        computeParamsKernel.setArg<cl::Buffer>(index++, particleOffsetIndices.getDeviceBuffer());
        if (exceptionChargeProds.isInitialized()) {
            computeParamsKernel.setArg<cl_int>(index++, exceptionChargeProds.getSize());
            computeParamsKernel.setArg<cl::Buffer>(index++, baseExceptionChargeProds.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionChargeProds.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionParamOffsets.getDeviceBuffer());
            computeParamsKernel.setArg<cl::Buffer>(index++, exceptionOffsetIndices.getDeviceBuffer());
        }
        if (exclusionChargeProds.isInitialized()) {
            computeExclusionParamsKernel.setArg<cl::Buffer>(0, cl.getPosq().getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl::Buffer>(1, charges.getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl_int>(2, exclusionChargeProds.getSize());
            computeExclusionParamsKernel.setArg<cl::Buffer>(3, exclusionAtoms.getDeviceBuffer());
            computeExclusionParamsKernel.setArg<cl::Buffer>(4, exclusionChargeProds.getDeviceBuffer());
        }
        if (pmeGrid1.isInitialized()) {
            // Create kernels for Coulomb PME.
            
            map<string, string> replacements;
            replacements["CHARGE"] = (usePosqCharges ? "pos.w" : "charges[atom]");
            cl::Program program = cl.createProgram(cl.replaceStrings(CommonPmeSlicingKernelSources::slicedPme, replacements), pmeDefines);
            pmeGridIndexKernel = cl::Kernel(program, "findAtomGridIndex");
            pmeSpreadChargeKernel = cl::Kernel(program, "gridSpreadCharge");
            pmeCollapseGridKernel = cl::Kernel(program, "collapseGrid");
            pmeConvolutionKernel = cl::Kernel(program, "reciprocalConvolution");
            pmeEvalEnergyKernel = cl::Kernel(program, "gridEvaluateEnergy");
            pmeInterpolateForceKernel = cl::Kernel(program, "gridInterpolateForce");
            int elementSize = (cl.getUseDoublePrecision() ? sizeof(mm_double4) : sizeof(mm_float4));
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
            pmeEvalEnergyKernel.setArg<cl::Buffer>(1, usePmeQueue ? pmeEnergyBuffer.getDeviceBuffer() : cl.getEnergyBuffer().getDeviceBuffer());
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
            if (usePmeQueue)
                syncQueue->setKernel(cl::Kernel(program, "addEnergy"));
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
    double energy = (includeReciprocal ? ewaldSelfEnergy : 0.0);
    if (recomputeParams || hasOffsets) {
        computeParamsKernel.setArg<cl_int>(1, includeEnergy && includeReciprocal);
        cl.executeKernel(computeParamsKernel, cl.getPaddedNumAtoms());
        if (exclusionChargeProds.isInitialized())
            cl.executeKernel(computeExclusionParamsKernel, exclusionChargeProds.getSize());
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
        cl.executeKernel(pmeFinishSpreadChargeKernel, gridSizeX*gridSizeY*gridSizeZ);
        fft->execFFT(true, cl.getQueue());

        pmeCollapseGridKernel.setArg<cl::Buffer>(0, pmeGrid2.getDeviceBuffer());
        cl.executeKernel(pmeCollapseGridKernel, gridSizeX*gridSizeY*gridSizeZ);

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
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string param;
        int exception;
        double charge;
        force.getExceptionParameterOffset(i, param, exception, charge);
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
    if (numExceptions != exceptionAtoms.size())
        throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");

    // Record the per-particle parameters.

    vector<float> baseParticleChargeVec(cl.getPaddedNumAtoms(), 0.0);
    vector<int> subsetVec(cl.getPaddedNumAtoms(), 0);
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
            if (make_pair(particle1, particle2) != exceptionAtoms[i])
                throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");
            baseExceptionChargeProdsVec[i] = chargeProd;
        }
        baseExceptionChargeProds.upload(baseExceptionChargeProdsVec);
    }
    
    // Compute other values.
    
    ewaldSelfEnergy = 0.0;
    if (cl.getContextIndex() == 0) {
        for (int i = 0; i < force.getNumParticles(); i++) {
            ewaldSelfEnergy -= baseParticleChargeVec[i]*baseParticleChargeVec[i]*ONE_4PI_EPS0*alpha/sqrt(M_PI);
        }
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
