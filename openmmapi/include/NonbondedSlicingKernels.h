#ifndef NONBONDED_SLICING_KERNELS_H_
#define NONBONDED_SLICING_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "SlicedNonbondedForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

using namespace OpenMM;

namespace NonbondedSlicing {

/**
 * This kernel is invoked by SlicedNonbondedForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcSlicedNonbondedForceKernel : public KernelImpl {
public:
    enum NonbondedMethod {
        NoCutoff = 0,
        CutoffNonPeriodic = 1,
        CutoffPeriodic = 2,
        Ewald = 3,
        PME = 4,
        LJPME = 5
    };
    static std::string Name() {
        return "CalcSlicedNonbondedForce";
    }
    CalcSlicedNonbondedForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the SlicedNonbondedForce this kernel will be used for
     */
    virtual void initialize(const System& system, const SlicedNonbondedForce& force) = 0;
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
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the SlicedNonbondedForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const SlicedNonbondedForce& force) = 0;
    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    virtual void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;
    /**
     * Get the parameters being used for the dispersion terms in LJPME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    virtual void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;
};

} // namespace NonbondedSlicing

#endif /*NONBONDED_SLICING_KERNELS_H_*/
