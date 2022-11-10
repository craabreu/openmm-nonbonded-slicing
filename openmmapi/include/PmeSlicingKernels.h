#ifndef PMESLICING_KERNELS_H_
#define PMESLICING_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
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

#include "SlicedPmeForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

using namespace OpenMM;

namespace PmeSlicing {

/**
 * This kernel is invoked by SlicedPmeForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcSlicedPmeForceKernel : public KernelImpl {
public:
    static std::string Name() {
        return "CalcSlicedPmeForce";
    }
    CalcSlicedPmeForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the SlicedPmeForce this kernel will be used for
     */
    virtual void initialize(const System& system, const SlicedPmeForce& force) = 0;
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
     * @param force      the SlicedPmeForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const SlicedPmeForce& force) = 0;
    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    virtual void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;
};

/**
 * This kernel performs the reciprocal space calculation for PME.  In most cases, this
 * calculation is done directly by CalcNonbondedForceKernel so this kernel is unneeded.
 * In some cases it may want to outsource the work to a different kernel.  In particular,
 * GPU based platforms sometimes use a CPU based implementation provided by a separate
 * plugin.
 */
class CalcPmeReciprocalForceKernel : public KernelImpl {
public:
    class IO;
    static std::string Name() {
        return "CalcPmeReciprocalForce";
    }
    CalcPmeReciprocalForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param gridx        the x size of the PME grid
     * @param gridy        the y size of the PME grid
     * @param gridz        the z size of the PME grid
     * @param numParticles the number of particles in the system
     * @param alpha        the Ewald blending parameter
     * @param deterministic whether it should attempt to make the resulting forces deterministic
     */
    virtual void initialize(int gridx, int gridy, int gridz, int numParticles, double alpha, bool deterministic) = 0;
    /**
     * Begin computing the force and energy.
     *
     * @param io                  an object that coordinates data transfer
     * @param periodicBoxVectors  the vectors defining the periodic box (measured in nm)
     * @param includeEnergy       true if potential energy should be computed
     */
    virtual void beginComputation(IO& io, const Vec3* periodicBoxVectors, bool includeEnergy) = 0;
    /**
     * Finish computing the force and energy.
     *
     * @param io   an object that coordinates data transfer
     * @return the potential energy due to the PME reciprocal space interactions
     */
    virtual double finishComputation(IO& io) = 0;
    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    virtual void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;
};

/**
 * Any class that uses CalcPmeReciprocalForceKernel should create an implementation of this
 * class, then pass it to the kernel to manage communication with it.
 */
class CalcPmeReciprocalForceKernel::IO {
public:
    /**
     * Get a pointer to the atom charges and positions.  This array should contain four
     * elements for each atom: x, y, z, and q in that order.
     */
    virtual float* getPosq() = 0;
    /**
     * Record the forces calculated by the kernel.
     *
     * @param force    an array containing four elements for each atom.  The first three
     *                 are the x, y, and z components of the force, while the fourth element
     *                 should be ignored.
     */
    virtual void setForce(float* force) = 0;
};

} // namespace PmeSlicing

#endif /*PMESLICING_KERNELS_H_*/
