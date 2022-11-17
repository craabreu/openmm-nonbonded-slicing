#ifndef CUDA_PMESLICING_KERNELS_H_
#define CUDA_PMESLICING_KERNELS_H_

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

#include "PmeSlicingKernels.h"
#include "internal/CudaFFT3D.h"
#include "internal/CudaCuFFT3D.h"
#include "internal/CudaVkFFT3D.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaSort.h"
#include <vector>

using namespace OpenMM;
using namespace std;

namespace PmeSlicing {

/**
 * This kernel is invoked by SlicedPmeForce to calculate the forces acting on the system.
 */
class CudaCalcSlicedPmeForceKernel : public CalcSlicedPmeForceKernel {
public:
    CudaCalcSlicedPmeForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) :
        CalcSlicedPmeForceKernel(name, platform), cu(cu), hasInitializedFFT(false), sort(NULL), fft(NULL), usePmeStream(false) {
    };
    ~CudaCalcSlicedPmeForceKernel();
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
    class SyncStreamPreComputation;
    class SyncStreamPostComputation;
    class AddEnergyPostComputation;
    CudaContext& cu;
    ForceInfo* info;
    bool hasInitializedFFT;
    CudaArray charges;
    CudaArray subsets;
    CudaArray exceptionAtoms;
    CudaArray exceptionSlices;
    CudaArray exceptionChargeProds;
    CudaArray exclusionAtoms;
    CudaArray exclusionSlices;
    CudaArray exclusionChargeProds;
    CudaArray baseParticleCharges;
    CudaArray baseExceptionChargeProds;
    CudaArray particleParamOffsets;
    CudaArray exceptionParamOffsets;
    CudaArray particleOffsetIndices;
    CudaArray exceptionOffsetIndices;
    CudaArray globalParams;
    CudaArray pmeGrid1;
    CudaArray pmeGrid2;
    CudaArray pmeBsplineModuliX;
    CudaArray pmeBsplineModuliY;
    CudaArray pmeBsplineModuliZ;
    CudaArray pmeAtomGridIndex;
    CudaArray pmeEnergyBuffer;
    CudaSort* sort;
    Kernel cpuPme;
    AddEnergyPostComputation* addEnergy;
    CUstream pmeStream;
    CUevent pmeSyncEvent, paramsSyncEvent;
    CudaFFT3D* fft;
    CUfunction computeParamsKernel, computeExclusionParamsKernel;
    CUfunction ewaldSumsKernel;
    CUfunction ewaldForcesKernel;
    CUfunction pmeGridIndexKernel;
    CUfunction pmeSpreadChargeKernel;
    CUfunction pmeFinishSpreadChargeKernel;
    CUfunction pmeEvalEnergyKernel;
    CUfunction pmeAddSelfEnergyKernel;
    CUfunction pmeConvolutionKernel;
    CUfunction pmeInterpolateForceKernel;
    std::vector<std::vector<int>> exclusionPairs, exceptionPairs;
    std::vector<std::string> paramNames;
    std::vector<double> paramValues;
    std::vector<double> subsetSelfEnergy;
    double ewaldSelfEnergy, alpha;
    int interpolateForceThreads;
    int gridSizeX, gridSizeY, gridSizeZ;
    int numSubsets, numSlices;
    bool usePmeStream, useCudaFFT, usePosqCharges, recomputeParams, hasOffsets;
    static const int PmeOrder = 5;

    bool hasDerivatives;
    CudaArray sliceLambda;
    CudaArray sliceDerivIndices;
    std::vector<std::string> requestedDerivs;
    std::vector<double> sliceLambdaVec;
    std::vector<std::string> switchParamNames;
    std::vector<double> switchParamValues;
    std::vector<int> sliceSwitchParamIndices;

    std::vector<float> floatVector(std::vector<double> input) {
        vector<float> output(input.begin(), input.end());
        return output;
    }
};

} // namespace PmeSlicing

#endif /*CUDA_PMESLICING_KERNELS_H_*/
