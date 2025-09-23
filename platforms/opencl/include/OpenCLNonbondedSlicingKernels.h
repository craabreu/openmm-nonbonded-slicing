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

#include "CommonNonbondedSlicingKernels.h"
#include "NonbondedSlicingKernels.h"
#include "openmm/opencl/OpenCLContext.h"
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
class OpenCLCalcSlicedNonbondedForceKernel : public CommonCalcSlicedNonbondedForceKernel {
public:
    OpenCLCalcSlicedNonbondedForceKernel(string name, const Platform& platform, OpenCLContext& cl, const System& system) :
            CommonCalcSlicedNonbondedForceKernel(name, platform, cl, system), cl(cl) {
    }
    /**
        * Initialize the kernel.
        *
        * @param system     the System this kernel will be applied to
        * @param force      the SlicedNonbondedForce this kernel will be used for
        */
    void initialize(const System& system, const SlicedNonbondedForce& force);
private:
    OpenCLContext& cl;
};

} // namespace NonbondedSlicing

#endif /*OPENCL_NONBONDED_SLICING_KERNELS_H_*/
