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

class CudaCalcSlicedPmeForceKernel::PmeIO : public CalcPmeReciprocalForceKernel::IO {
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

class CudaCalcSlicedPmeForceKernel::PmePreComputation : public CudaContext::ForcePreComputation {
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

class CudaCalcSlicedPmeForceKernel::PmePostComputation : public CudaContext::ForcePostComputation {
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
    AddEnergyPostComputation(CudaContext& cu, CUfunction addEnergyKernel, CudaArray& pmeEnergyBuffer, CudaArray& pmeLambda, int bufferSize, int forceGroup) :
        cu(cu), addEnergyKernel(addEnergyKernel), pmeEnergyBuffer(pmeEnergyBuffer), pmeLambda(pmeLambda), bufferSize(bufferSize), forceGroup(forceGroup) {}
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if (includeEnergy && (groups&(1<<forceGroup)) != 0) {
            void* args[] = {&pmeEnergyBuffer.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(), &pmeLambda.getDevicePointer()};
            cu.executeKernel(addEnergyKernel, args, bufferSize);
        }
        return 0.0;
    }
private:
    CudaContext& cu;
    CUfunction addEnergyKernel;
    CudaArray& pmeEnergyBuffer;
    CudaArray& pmeLambda;
    int bufferSize;
    int forceGroup;
};

CudaCalcSlicedPmeForceKernel::~CudaCalcSlicedPmeForceKernel() {
    ContextSelector selector(cu);
    if (sort != NULL)
        delete sort;
    if (fft != NULL)
        delete fft;
    if (pmeio != NULL)
        delete pmeio;
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
    string prefix = "nonbonded"+cu.intToString(forceIndex)+"_";

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

    map<string, string> defines;
    defines["HAS_COULOMB"] = "1";
    defines["HAS_LENNARD_JONES"] = "0";
    alpha = 0;
    ewaldSelfEnergy = 0.0;
    map<string, string> paramsDefines;
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

    // Initialize subsets and coupling parameters.

    subsets.initialize<int>(cu, cu.getPaddedNumAtoms(), "subsets");
    vector<int> subsetVec(cu.getPaddedNumAtoms());
    for (int i = 0; i < numParticles; i++)
        subsetVec[i] = force.getParticleSubset(i);
    subsets.upload(subsetVec);

    if (cu.getUseDoublePrecision())
        uploadCouplingParameters<double>(force);
    else
        uploadCouplingParameters<float>(force);

    // Compute the PME parameters.

    int cufftVersion;
    cufftGetVersion(&cufftVersion);
    useCudaFFT = force.getUseCudaFFT() && (cufftVersion >= 7050); // There was a critical bug in version 7.0

    SlicedPmeForceImpl::calcPMEParameters(system, force, alpha, gridSizeX, gridSizeY, gridSizeZ, false);

    gridSizeX = CudaFFT3D::findLegalDimension(gridSizeX);
    gridSizeY = CudaFFT3D::findLegalDimension(gridSizeY);
    gridSizeZ = CudaFFT3D::findLegalDimension(gridSizeZ);
    int roundedZSize = PmeOrder*(int) ceil(gridSizeZ/(double) PmeOrder);
    int bufferSize = cu.getNumThreadBlocks()*CudaContext::ThreadBlockSize;

    defines["EWALD_ALPHA"] = cu.doubleToString(alpha);
    defines["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
    defines["USE_EWALD"] = "1";
    defines["DO_LJPME"] = "0";
    if (cu.getContextIndex() == 0) {
        paramsDefines["INCLUDE_EWALD"] = "1";
        paramsDefines["EWALD_SELF_ENERGY_SCALE"] = cu.doubleToString(ONE_4PI_EPS0*alpha/sqrt(M_PI));
        for (int i = 0; i < numParticles; i++)
            ewaldSelfEnergy -= baseParticleChargeVec[i]*baseParticleChargeVec[i]*ONE_4PI_EPS0*alpha/sqrt(M_PI);
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
        pmeDefines["BUFFER_SIZE"] = cu.intToString(bufferSize);
        pmeDefines["EPSILON_FACTOR"] = cu.doubleToString(sqrt(ONE_4PI_EPS0));
        pmeDefines["M_PI"] = cu.doubleToString(M_PI);
        if (cu.getUseDoublePrecision() || cu.getPlatformData().deterministicForces)
            pmeDefines["USE_FIXED_POINT_CHARGE_SPREADING"] = "1";
        if (usePmeStream)
            pmeDefines["USE_PME_STREAM"] = "1";
        map<string, string> replacements;
        replacements["CHARGE"] = (usePosqCharges ? "pos.w" : "charges[atom]");
        CUmodule module = cu.createModule(CudaPmeSlicingKernelSources::vectorOps+
                                          CommonPmeSlicingKernelSources::realtofixedpoint+
                                          cu.replaceStrings(CommonPmeSlicingKernelSources::slicedPme, replacements), pmeDefines);
        if (cu.getPlatformData().useCpuPme && usePosqCharges) {
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

            // Create required data structures.

            int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
            int gridElements = gridSizeX*gridSizeY*roundedZSize*numSubsets;
            pmeGrid1.initialize(cu, gridElements, 2*elementSize, "pmeGrid1");
            pmeGrid2.initialize(cu, gridElements, 2*elementSize, "pmeGrid2");
            cu.addAutoclearBuffer(pmeGrid2);
            pmeBsplineModuliX.initialize(cu, gridSizeX, elementSize, "pmeBsplineModuliX");
            pmeBsplineModuliY.initialize(cu, gridSizeY, elementSize, "pmeBsplineModuliY");
            pmeBsplineModuliZ.initialize(cu, gridSizeZ, elementSize, "pmeBsplineModuliZ");
            pmeAtomGridIndex.initialize<int2>(cu, numParticles, "pmeAtomGridIndex");
            int energyElementSize = (cu.getUseDoublePrecision() || cu.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
            pmeEnergyBuffer.initialize(cu, numSlices*bufferSize, energyElementSize, "pmeEnergyBuffer");
            cu.clearBuffer(pmeEnergyBuffer);
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
                cu.addPostComputation(new SyncStreamPostComputation(cu, pmeSyncEvent, recipForceGroup));
            }
            else
                pmeStream = cu.getCurrentStream();
            cu.addPostComputation(new AddEnergyPostComputation(cu, cu.getKernel(module, "addEnergy"), pmeEnergyBuffer, pmeLambda, bufferSize, recipForceGroup));

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
    }

    // Add code to subtract off the reciprocal part of excluded interactions.

    if (pmeio == NULL) {
        int numContexts = cu.getPlatformData().contexts.size();
        int startIndex = cu.getContextIndex()*force.getNumExceptions()/numContexts;
        int endIndex = (cu.getContextIndex()+1)*force.getNumExceptions()/numContexts;
        int numExclusions = endIndex-startIndex;
        if (numExclusions > 0) {
            paramsDefines["HAS_EXCLUSIONS"] = "1";
            vector<vector<int> > atoms(numExclusions, vector<int>(2));
            exclusionAtoms.initialize<int2>(cu, numExclusions, "exclusionAtoms");
            exclusionChargeProds.initialize<float>(cu, numExclusions, "exclusionChargeProds");
            vector<int2> exclusionAtomsVec(numExclusions);
            for (int i = 0; i < numExclusions; i++) {
                int j = i+startIndex;
                exclusionAtomsVec[i] = make_int2(exclusions[j].first, exclusions[j].second);
                atoms[i][0] = exclusions[j].first;
                atoms[i][1] = exclusions[j].second;
            }
            exclusionAtoms.upload(exclusionAtomsVec);
            map<string, string> replacements;
            replacements["PARAMS"] = cu.getBondedUtilities().addArgument(exclusionChargeProds.getDevicePointer(), "float");
            replacements["EWALD_ALPHA"] = cu.doubleToString(alpha);
            replacements["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
            replacements["DO_LJPME"] = "0";
            replacements["USE_PERIODIC"] = force.getExceptionsUsePeriodicBoundaryConditions() ? "1" : "0";
            if (force.getIncludeDirectSpace())
                cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CommonPmeSlicingKernelSources::slicedPmeExclusions, replacements), force.getForceGroup());
        }
    }

    // Add the interaction to the default nonbonded kernel.

    string source = cu.replaceStrings(CommonPmeSlicingKernelSources::coulombLennardJones, defines);
    charges.initialize(cu, cu.getPaddedNumAtoms(), cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float), "charges");
    baseParticleCharges.initialize<float>(cu, cu.getPaddedNumAtoms(), "baseParticleCharges");
    baseParticleCharges.upload(baseParticleChargeVec);
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
    if (!usePosqCharges)
        cu.getNonbondedUtilities().addParameter(CudaNonbondedUtilities::ParameterInfo(prefix+"charge", "real", 1, charges.getElementSize(), charges.getDevicePointer()));
    source = cu.replaceStrings(source, replacements);
    if (force.getIncludeDirectSpace())
        cu.getNonbondedUtilities().addInteraction(true, true, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup(), true);

    // Initialize the exceptions.

    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0) {
        paramsDefines["HAS_EXCEPTIONS"] = "1";
        exceptionAtoms.resize(numExceptions);
        vector<vector<int> > atoms(numExceptions, vector<int>(2));
        exceptionChargeProds.initialize<float>(cu, numExceptions, "exceptionChargeProds");
        baseExceptionChargeProds.initialize<float>(cu, numExceptions, "baseExceptionChargeProds");
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
        replacements["PARAMS"] = cu.getBondedUtilities().addArgument(exceptionChargeProds.getDevicePointer(), "float");
        if (force.getIncludeDirectSpace())
            cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CommonPmeSlicingKernelSources::slicedPmeExceptions, replacements), force.getForceGroup());
    }
    
    // Initialize parameter offsets.

    vector<vector<float2> > particleOffsetVec(force.getNumParticles());
    vector<vector<float2> > exceptionOffsetVec(numExceptions);
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
        particleOffsetVec[particle].push_back(make_float2(charge, paramIndex));
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
        exceptionOffsetVec[index-startIndex].push_back(make_float2(charge, paramIndex));
    }
    paramValues.resize(paramNames.size(), 0.0);
    particleParamOffsets.initialize<float2>(cu, max(force.getNumParticleParameterOffsets(), 1), "particleParamOffsets");
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
    if (force.getNumParticleParameterOffsets() > 0) {
        particleParamOffsets.upload(p);
        particleOffsetIndices.upload(particleOffsetIndicesVec);
    }
    exceptionParamOffsets.initialize<float2>(cu, max((int) e.size(), 1), "exceptionParamOffsets");
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
    
    CUmodule module = cu.createModule(CommonPmeSlicingKernelSources::slicedPmeParameters, paramsDefines);
    computeParamsKernel = cu.getKernel(module, "computeParameters");
    computeExclusionParamsKernel = cu.getKernel(module, "computeExclusionParameters");
    info = new ForceInfo(force);
    cu.addForce(info);
}

double CudaCalcSlicedPmeForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    // Update particle and exception parameters.

    ContextSelector selector(cu);
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
        int computeSelfEnergy = (includeEnergy && includeReciprocal);
        int numAtoms = cu.getPaddedNumAtoms();
        vector<void*> paramsArgs = {&cu.getEnergyBuffer().getDevicePointer(), &computeSelfEnergy, &globalParams.getDevicePointer(), &numAtoms,
                &baseParticleCharges.getDevicePointer(), &cu.getPosq().getDevicePointer(), &charges.getDevicePointer(),
                &particleParamOffsets.getDevicePointer(), &particleOffsetIndices.getDevicePointer()};
        int numExceptions;
        if (exceptionChargeProds.isInitialized()) {
            numExceptions = exceptionChargeProds.getSize();
            paramsArgs.push_back(&numExceptions);
            paramsArgs.push_back(&baseExceptionChargeProds.getDevicePointer());
            paramsArgs.push_back(&exceptionChargeProds.getDevicePointer());
            paramsArgs.push_back(&exceptionParamOffsets.getDevicePointer());
            paramsArgs.push_back(&exceptionOffsetIndices.getDevicePointer());
        }
        cu.executeKernel(computeParamsKernel, &paramsArgs[0], cu.getPaddedNumAtoms());
        if (exclusionChargeProds.isInitialized()) {
            int numExclusions = exclusionChargeProds.getSize();
            vector<void*> exclusionChargeProdsArgs = {&cu.getPosq().getDevicePointer(), &charges.getDevicePointer(),
                    &numExclusions, &exclusionAtoms.getDevicePointer(), &exclusionChargeProds.getDevicePointer()};
            cu.executeKernel(computeExclusionParamsKernel, &exclusionChargeProdsArgs[0], numExclusions);
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
            void* computeEnergyArgs[] = {&pmeGrid2.getDevicePointer(), &pmeEnergyBuffer.getDevicePointer(),
                    &pmeBsplineModuliX.getDevicePointer(), &pmeBsplineModuliY.getDevicePointer(), &pmeBsplineModuliZ.getDevicePointer(),
                    recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
            cu.executeKernel(pmeEvalEnergyKernel, computeEnergyArgs, gridSizeX*gridSizeY*gridSizeZ);
        }

        void* convolutionArgs[] = {&pmeGrid2.getDevicePointer(), &pmeBsplineModuliX.getDevicePointer(),
                &pmeBsplineModuliY.getDevicePointer(), &pmeBsplineModuliZ.getDevicePointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeConvolutionKernel, convolutionArgs, gridSizeX*gridSizeY*(gridSizeZ/2+1), 256);

        fft->execFFT(false);

        void* interpolateArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &pmeGrid1.getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex.getDevicePointer(),
                &charges.getDevicePointer(), &subsets.getDevicePointer(), &pmeLambda.getDevicePointer()};
        cu.executeKernel(pmeInterpolateForceKernel, interpolateArgs, cu.getNumAtoms(), 128);

        if (usePmeStream) {
            cuEventRecord(pmeSyncEvent, pmeStream);
            cu.restoreDefaultStream();
        }
    }

    return energy;
}

template <typename real>
void CudaCalcSlicedPmeForceKernel::uploadCouplingParameters(const SlicedPmeForce& force) {
    ContextSelector selector(cu);
    int numSubsets = force.getNumSubsets();
    if (!pmeLambda.isInitialized())
        pmeLambda.initialize<real>(cu, numSubsets*numSubsets, "pmeLambda");
    vector<real> pmeLambdaVec(numSubsets*numSubsets);
    for (int i = 0; i < numSubsets; i++)
        for (int j = 0; j < numSubsets; j++)
            pmeLambdaVec[j*numSubsets+i] = force.getCouplingParameter(i, j);
    pmeLambda.upload(pmeLambdaVec);
}

void CudaCalcSlicedPmeForceKernel::copyParametersToContext(ContextImpl& context, const SlicedPmeForce& force) {
    // Make sure the new parameters are acceptable.
    
    ContextSelector selector(cu);
    if (force.getNumParticles() != cu.getNumAtoms())
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
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions != exceptionAtoms.size())
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

    // Record coupling parameters.

    if (cu.getUseDoublePrecision())
        uploadCouplingParameters<double>(force);
    else
        uploadCouplingParameters<float>(force);

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
    if (cu.getContextIndex() == 0) {
        for (int i = 0; i < force.getNumParticles(); i++) {
            ewaldSelfEnergy -= baseParticleChargeVec[i]*baseParticleChargeVec[i]*ONE_4PI_EPS0*alpha/sqrt(M_PI);
        }
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

