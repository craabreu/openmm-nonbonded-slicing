#ifndef CUDA_NONBONDED_SLICING_KERNELS_H_
#define CUDA_NONBONDED_SLICING_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "NonbondedSlicingKernels.h"
#include "internal/CudaFFT3D.h"
#include "internal/CudaCuFFT3D.h"
#include "internal/CudaVkFFT3D.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaSort.h"
#include <vector>
#include <algorithm>

using namespace OpenMM;
using namespace std;

namespace NonbondedSlicing {

/**
 * This kernel is invoked by SlicedNonbondedForce to calculate the forces acting on the system.
 */
class CudaCalcSlicedNonbondedForceKernel : public CalcSlicedNonbondedForceKernel {
public:
    CudaCalcSlicedNonbondedForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) :
            CalcSlicedNonbondedForceKernel(name, platform), cu(cu), hasInitializedFFT(false), sort(NULL),
            dispersionFft(NULL), fft(NULL), usePmeStream(false) {};
    ~CudaCalcSlicedNonbondedForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the SlicedNonbondedForce this kernel will be used for
     */
    void initialize(const System& system, const SlicedNonbondedForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @param includeDirect  true if direct space interactions should be included
     * @param includeReciprocal  true if reciprocal space interactions should be included
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the SlicedNonbondedForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const SlicedNonbondedForce& force);
    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Get the dispersion parameters being used for the dispersion term in LJPME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    class SortTrait : public CudaSort::SortTrait {
        int getDataSize() const {return 8;}
        int getKeySize() const {return 4;}
        const char* getDataType() const {return "int2";}
        const char* getKeyType() const {return "int";}
        const char* getMinKey() const {return "(-2147483647-1)";}
        const char* getMaxKey() const {return "2147483647";}
        const char* getMaxValue() const {return "make_int2(2147483647, 2147483647)";}
        const char* getSortKey() const {return "value.y";}
    };
    class ForceInfo;
    class ScalingParameterInfo;
    class SyncStreamPreComputation;
    class AddEnergyPostComputation;
    class SyncStreamPostComputation;
    class DispersionCorrectionPostComputation;
    CudaContext& cu;
    ForceInfo* info;
    bool hasInitializedFFT;
    CudaArray charges;
    CudaArray sigmaEpsilon;
    CudaArray exceptionParams;
    CudaArray exclusionAtoms;
    CudaArray exclusionParams;
    CudaArray baseParticleParams;
    CudaArray baseExceptionParams;
    CudaArray particleParamOffsets;
    CudaArray exceptionParamOffsets;
    CudaArray particleOffsetIndices;
    CudaArray exceptionOffsetIndices;
    CudaArray globalParams;
    CudaArray cosSinSums;
    CudaArray pmeGrid1;
    CudaArray pmeGrid2;
    CudaArray pmeBsplineModuliX;
    CudaArray pmeBsplineModuliY;
    CudaArray pmeBsplineModuliZ;
    CudaArray pmeDispersionBsplineModuliX;
    CudaArray pmeDispersionBsplineModuliY;
    CudaArray pmeDispersionBsplineModuliZ;
    CudaArray pmeAtomGridIndex;
    CudaArray pmeEnergyBuffer;
    CudaArray ljpmeEnergyBuffer;
    CudaSort* sort;
    CUstream pmeStream;
    CUevent pmeSyncEvent, paramsSyncEvent;
    CudaFFT3D* fft;
    CudaFFT3D* dispersionFft;
    CUfunction computeParamsKernel, computeExclusionParamsKernel;
    CUfunction ewaldSumsKernel;
    CUfunction ewaldForcesKernel;
    CUfunction pmeGridIndexKernel;
    CUfunction pmeDispersionGridIndexKernel;
    CUfunction pmeSpreadChargeKernel;
    CUfunction pmeDispersionSpreadChargeKernel;
    CUfunction pmeFinishSpreadChargeKernel;
    CUfunction pmeDispersionFinishSpreadChargeKernel;
    CUfunction pmeEvalEnergyKernel;
    CUfunction pmeEvalDispersionEnergyKernel;
    CUfunction pmeConvolutionKernel;
    CUfunction pmeDispersionConvolutionKernel;
    CUfunction pmeInterpolateForceKernel;
    CUfunction pmeInterpolateDispersionForceKernel;
    AddEnergyPostComputation* addEnergy;
    std::vector<std::pair<int, int> > exceptionAtoms;
    CudaArray exceptionPairs;
    CudaArray exceptionSlices;
    std::vector<std::string> paramNames;
    std::vector<double> paramValues;
    double ewaldSelfEnergy, alpha, dispersionAlpha;
    int interpolateForceThreads;
    int gridSizeX, gridSizeY, gridSizeZ;
    int dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ;
    bool hasCoulomb, hasLJ, usePmeStream, useCudaFFT, doLJPME, usePosqCharges, recomputeParams, hasOffsets;
    NonbondedMethod nonbondedMethod;
    static const int PmeOrder = 5;

    int numSubsets, numSlices;
    bool hasDerivatives;
    vector<int> subsetsVec;
    vector<double> dispersionCoefficients;
    vector<double2> sliceLambdasVec, subsetSelfEnergy;
    vector<ScalingParameterInfo> sliceScalingParams;
    CudaArray subsets;
    CudaArray sliceLambdas;

    string getDerivativeExpression(string param, bool conditionCoulomb, bool conditionLJ);

    vector<float2> double2Tofloat2(vector<double2> input) {
        vector<float2> output(input.size());
        transform(
            input.begin(), input.end(), output.begin(),
            [](double2 v) -> float2 { return make_float2(v.x, v.y); }
        );
        return output;
    }
};

class CudaCalcSlicedNonbondedForceKernel::ScalingParameterInfo {
public:
    string nameCoulomb, nameLJ;
    bool includeCoulomb, includeLJ;
    bool hasDerivativeCoulomb, hasDerivativeLJ;
    ScalingParameterInfo() :
        nameCoulomb(""), nameLJ(""), includeCoulomb(false), includeLJ(false),
        hasDerivativeCoulomb(false), hasDerivativeLJ(false) {
    }
    void addInfo(string name, bool includeCoulomb, bool includeLJ, bool hasDerivative) {
        if (includeCoulomb) {
            this->includeCoulomb = true;
            nameCoulomb = name;
            hasDerivativeCoulomb = hasDerivative;
        }
        if (includeLJ) {
            this->includeLJ = true;
            nameLJ = name;
            hasDerivativeLJ = hasDerivative;
        }
    }
};

} // namespace NonbondedSlicing

#endif /*CUDA_NONBONDED_SLICING_KERNELS_H_*/
