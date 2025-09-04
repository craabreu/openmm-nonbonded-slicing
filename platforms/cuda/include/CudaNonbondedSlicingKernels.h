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

#include "CommonNonbondedSlicingKernels.h"
#include "NonbondedSlicingKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <vector>
#include <algorithm>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

namespace NonbondedSlicing {

/**
 * This kernel is invoked by SlicedNonbondedForce to calculate the forces acting on the system.
 */
class CudaCalcSlicedNonbondedForceKernel : public CommonCalcSlicedNonbondedForceKernel {
public:
    CudaCalcSlicedNonbondedForceKernel(string name, const Platform& platform, CudaContext& cu, const System& system) :
            CommonCalcSlicedNonbondedForceKernel(name, platform, cu, system), cu(cu) {
    }
    /**
        * Initialize the kernel.
        *
        * @param system     the System this kernel will be applied to
        * @param force      the SlicedNonbondedForce this kernel will be used for
        */
    void initialize(const System& system, const SlicedNonbondedForce& force);
private:
    CudaContext& cu;
};

} // namespace NonbondedSlicing

#endif /*CUDA_NONBONDED_SLICING_KERNELS_H_*/
