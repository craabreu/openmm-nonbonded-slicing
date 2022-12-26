#ifndef OPENMM_OPENCLPARALLELNONBONDED_SLICINGKERNELS_H_
#define OPENMM_OPENCLPARALLELNONBONDED_SLICINGKERNELS_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "OpenCLNonbondedSlicingKernels.h"
#include "CommonNonbondedSlicingKernels.h"
#include "openmm/opencl/OpenCLPlatform.h"
#include "openmm/opencl/OpenCLContext.h"

namespace NonbondedSlicing {

/**
 * This kernel is invoked by SlicedNonbondedForce to calculate the forces acting on the system.
 */
class OpenCLParallelCalcSlicedNonbondedForceKernel : public CalcSlicedNonbondedForceKernel {
public:
    OpenCLParallelCalcSlicedNonbondedForceKernel(std::string name, const Platform& platform, OpenCLPlatform::PlatformData& data, const System& system);
    OpenCLCalcSlicedNonbondedForceKernel& getKernel(int index) {
        return dynamic_cast<OpenCLCalcSlicedNonbondedForceKernel&>(kernels[index].getImpl());
    }
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
     * @param includeReciprocal  true if reciprocal space interactions should be included
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
    class Task;
    OpenCLPlatform::PlatformData& data;
    std::vector<Kernel> kernels;
};

} // namespace NonbondedSlicing

#endif /*OPENMM_OPENCLPARALLELNONBONDED_SLICINGKERNELS_H_*/
