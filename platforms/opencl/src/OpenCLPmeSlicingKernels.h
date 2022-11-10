#ifndef OPENCL_PMESLICING_KERNELS_H_
#define OPENCL_PMESLICING_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                             OpenMM PME Slicing                             *
 *                             ==================                             *
 *                                                                            *
 * An OpenMM plugin for Smooth Particle Mesh Ewald electrostatic calculations *
 * with multiple coupling parameters.                                         *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-pme-slicing                             *
 * -------------------------------------------------------------------------- */

#include "PmeSlicingKernels.h"
#include "internal/OpenCLVkFFT3D.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/opencl/OpenCLArray.h"
#include "openmm/opencl/OpenCLSort.h"
#include <vector>

namespace PmeSlicing {

/**
 * This kernel is invoked by SlicedPmeForce to calculate the forces acting on the system.
 */
class OpenCLCalcSlicedPmeForceKernel : public CalcSlicedPmeForceKernel {
public:
    OpenCLCalcSlicedPmeForceKernel(std::string name, const Platform& platform, OpenCLContext& cl, const System& system) :
        CalcSlicedPmeForceKernel(name, platform), hasInitializedKernel(false), cl(cl), sort(NULL), fft(NULL), usePmeQueue(false) {
    };
    ~OpenCLCalcSlicedPmeForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the SlicedPmeForce this kernel will be used for
     */
    void initialize(const System& system, const SlicedPmeForce& force);
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
     * @param force      the SlicedPmeForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const SlicedPmeForce& force);
    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
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
    class SyncQueuePreComputation;
    class SyncQueuePostComputation;
    class AddEnergyPostComputation;
    OpenCLContext& cl;
    ForceInfo* info;
    bool hasInitializedKernel;
    bool deviceIsCpu;
    OpenCLArray charges;
    OpenCLArray subsets;
    OpenCLArray exceptionAtoms;
    OpenCLArray exceptionSlices;
    OpenCLArray exceptionChargeProds;
    OpenCLArray exclusionAtoms;
    OpenCLArray exclusionSlices;
    OpenCLArray exclusionChargeProds;
    OpenCLArray baseParticleCharges;
    OpenCLArray baseExceptionChargeProds;
    OpenCLArray particleParamOffsets;
    OpenCLArray exceptionParamOffsets;
    OpenCLArray particleOffsetIndices;
    OpenCLArray exceptionOffsetIndices;
    OpenCLArray globalParams;
    OpenCLArray pmeGrid1;
    OpenCLArray pmeGrid2;
    OpenCLArray pmeBsplineModuliX;
    OpenCLArray pmeBsplineModuliY;
    OpenCLArray pmeBsplineModuliZ;
    OpenCLArray pmeBsplineTheta;
    OpenCLArray pmeAtomRange;
    OpenCLArray pmeAtomGridIndex;
    OpenCLArray pmeEnergyBuffer;
    OpenCLArray pairwiseEnergyBuffer;
    OpenCLSort* sort;
    cl::CommandQueue pmeQueue;
    cl::Event pmeSyncEvent;
    OpenCLVkFFT3D* fft;
    Kernel cpuPme;
    AddEnergyPostComputation* addEnergy;
    cl::Kernel computeParamsKernel, computeExclusionParamsKernel;
    cl::Kernel ewaldSumsKernel;
    cl::Kernel ewaldForcesKernel;
    cl::Kernel pmeAtomRangeKernel;
    cl::Kernel pmeZIndexKernel;
    cl::Kernel pmeGridIndexKernel;
    cl::Kernel pmeSpreadChargeKernel;
    cl::Kernel pmeFinishSpreadChargeKernel;
    cl::Kernel pmeEvalEnergyKernel;
    cl::Kernel pmeAddSelfEnergyKernel;
    cl::Kernel pmeConvolutionKernel;
    cl::Kernel pmeInterpolateForceKernel;
    std::map<std::string, std::string> pmeDefines;
    std::vector<std::vector<int>> exceptionPairs;
    std::vector<std::string> paramNames;
    std::vector<double> paramValues;
    std::vector<double> subsetSelfEnergy;
    double ewaldSelfEnergy, alpha;
    int gridSizeX, gridSizeY, gridSizeZ;
    int numSubsets, numSlices;
    bool usePmeQueue, usePosqCharges, recomputeParams, hasOffsets;
    static const int PmeOrder = 5;

    bool hasExclusions;
    cl::Kernel computeBondsKernel;

    OpenCLArray sliceLambda;
    std::vector<double> sliceLambdaVec;
    std::vector<std::string> coupParamNames;
    std::vector<double> coupParamValues;
    std::vector<int> sliceCoupParamIndex;

    std::vector<float> floatVector(std::vector<double> input) {
        std::vector<float> output(input.begin(), input.end());
        return output;
    }
};

} // namespace PmeSlicing

#endif /*OPENCL_PMESLICING_KERNELS_H_*/
