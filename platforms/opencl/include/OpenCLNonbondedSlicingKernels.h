#ifndef OPENCL_NONBONDED_SLICING_KERNELS_H_
#define OPENCL_NONBONDED_SLICING_KERNELS_H_

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
#include "internal/OpenCLVkFFT3D.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/opencl/OpenCLArray.h"
#include "openmm/opencl/OpenCLSort.h"
#include <vector>
#include <algorithm>

namespace NonbondedSlicing {

/**
 * This kernel is invoked by SlicedNonbondedForce to calculate the forces acting on the system.
 */
class OpenCLCalcSlicedNonbondedForceKernel : public CalcSlicedNonbondedForceKernel {
public:
    OpenCLCalcSlicedNonbondedForceKernel(std::string name, const Platform& platform, OpenCLContext& cl, const System& system) :
            CalcSlicedNonbondedForceKernel(name, platform), hasInitializedKernel(false), cl(cl), sort(NULL),
            fft(NULL), dispersionFft(NULL), usePmeQueue(false) {};
    ~OpenCLCalcSlicedNonbondedForceKernel();
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
     * Get the parameters being used for the dispersion term in LJPME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    class SortTrait : public OpenCLSort::SortTrait {
        int getDataSize() const {return 8;}
        int getKeySize() const {return 4;}
        const char* getDataType() const {return "int2";}
        const char* getKeyType() const {return "int";}
        const char* getMinKey() const {return "INT_MIN";}
        const char* getMaxKey() const {return "INT_MAX";}
        const char* getMaxValue() const {return "(int2) (INT_MAX, INT_MAX)";}
        const char* getSortKey() const {return "value.y";}
    };
    class ForceInfo;
    class ScalingParameterInfo;
    class SyncQueuePreComputation;
    class SyncQueuePostComputation;
    class AddEnergyPostComputation;
    class DispersionCorrectionPostComputation;
    OpenCLContext& cl;
    ForceInfo* info;
    bool hasInitializedKernel;
    OpenCLArray charges;
    OpenCLArray sigmaEpsilon;
    OpenCLArray exceptionParams;
    OpenCLArray exclusionAtoms;
    OpenCLArray exclusionParams;
    OpenCLArray baseParticleParams;
    OpenCLArray baseExceptionParams;
    OpenCLArray particleParamOffsets;
    OpenCLArray exceptionParamOffsets;
    OpenCLArray particleOffsetIndices;
    OpenCLArray exceptionOffsetIndices;
    OpenCLArray globalParams;
    OpenCLArray cosSinSums;
    OpenCLArray pmeGrid1;
    OpenCLArray pmeGrid2;
    OpenCLArray pmeBsplineModuliX;
    OpenCLArray pmeBsplineModuliY;
    OpenCLArray pmeBsplineModuliZ;
    OpenCLArray pmeDispersionBsplineModuliX;
    OpenCLArray pmeDispersionBsplineModuliY;
    OpenCLArray pmeDispersionBsplineModuliZ;
    OpenCLArray pmeBsplineTheta;
    OpenCLArray pmeAtomRange;
    OpenCLArray pmeAtomGridIndex;
    OpenCLArray pmeEnergyBuffer;
    OpenCLArray ljpmeEnergyBuffer;
    OpenCLSort* sort;
    cl::CommandQueue pmeQueue;
    cl::Event pmeSyncEvent;
    OpenCLVkFFT3D* fft;
    OpenCLVkFFT3D* dispersionFft;
    AddEnergyPostComputation* addEnergy;
    cl::Kernel computeParamsKernel, computeExclusionParamsKernel;
    cl::Kernel ewaldSumsKernel;
    cl::Kernel ewaldForcesKernel;
    cl::Kernel pmeAtomRangeKernel;
    cl::Kernel pmeDispersionAtomRangeKernel;
    cl::Kernel pmeZIndexKernel;
    cl::Kernel pmeDispersionZIndexKernel;
    cl::Kernel pmeGridIndexKernel;
    cl::Kernel pmeDispersionGridIndexKernel;
    cl::Kernel pmeSpreadChargeKernel;
    cl::Kernel pmeDispersionSpreadChargeKernel;
    cl::Kernel pmeFinishSpreadChargeKernel;
    cl::Kernel pmeDispersionFinishSpreadChargeKernel;
    cl::Kernel pmeConvolutionKernel;
    cl::Kernel pmeDispersionConvolutionKernel;
    cl::Kernel pmeEvalEnergyKernel;
    cl::Kernel pmeDispersionEvalEnergyKernel;
    cl::Kernel pmeInterpolateForceKernel;
    cl::Kernel pmeDispersionInterpolateForceKernel;
    std::string realToFixedPoint;
    std::map<std::string, std::string> pmeDefines;
    std::vector<std::pair<int, int> > exceptionAtoms;
    OpenCLArray exceptionPairs;
    OpenCLArray exceptionSlices;
    std::vector<std::string> paramNames;
    std::vector<double> paramValues;
    double ewaldSelfEnergy, alpha, dispersionAlpha;
    int gridSizeX, gridSizeY, gridSizeZ;
    int dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ;
    bool hasCoulomb, hasLJ, usePmeQueue, doLJPME, usePosqCharges, recomputeParams, hasOffsets;
    NonbondedMethod nonbondedMethod;
    static const int PmeOrder = 5;

    int numSubsets, numSlices;
    bool hasDerivatives;
    vector<int> subsetsVec;
    vector<double> dispersionCoefficients;
    vector<mm_double2> sliceLambdasVec, subsetSelfEnergy;
    vector<ScalingParameterInfo> sliceScalingParams;
    OpenCLArray subsets;
    OpenCLArray sliceLambdas;

    string getDerivativeExpression(string param, bool conditionCoulomb, bool conditionLJ);

    vector<mm_float2> double2Tofloat2(vector<mm_double2> input) {
        vector<mm_float2> output(input.size());
        transform(
            input.begin(), input.end(), output.begin(),
            [](mm_double2 v) -> mm_float2 { return mm_float2(v.x, v.y); }
        );
        return output;
    }
};

class OpenCLCalcSlicedNonbondedForceKernel::ScalingParameterInfo {
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

#endif /*OPENCL_NONBONDED_SLICING_KERNELS_H_*/
