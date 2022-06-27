#ifndef REFERENCE_NATIVENONBONDED_KERNELS_H_
#define REFERENCE_NATIVENONBONDED_KERNELS_H_

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

#include "NativeNonbondedKernels.h"
#include "openmm/Platform.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include <vector>
#include <array>
#include <map>

namespace NativeNonbondedPlugin {

/**
 * This kernel is invoked by NativeNonbondedForce to calculate the forces acting on the system.
 */
class ReferenceCalcNativeNonbondedForceKernel : public CalcNativeNonbondedForceKernel {
public:
    ReferenceCalcNativeNonbondedForceKernel(std::string name, const OpenMM::Platform& platform) : CalcNativeNonbondedForceKernel(name, platform) {
    }
    ~ReferenceCalcNativeNonbondedForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the NativeNonbondedForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const NativeNonbondedForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @param includeReciprocal  true if reciprocal space interactions should be included
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the NativeNonbondedForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const NativeNonbondedForce& force);
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
    void computeParameters(OpenMM::ContextImpl& context);
    int numParticles, num14;
    std::vector<std::vector<int> >bonded14IndexArray;
    std::vector<std::vector<double> > particleParamArray, bonded14ParamArray;
    std::vector<std::array<double, 3> > baseParticleParams, baseExceptionParams;
    std::map<std::pair<std::string, int>, std::array<double, 3> > particleParamOffsets, exceptionParamOffsets;
    double nonbondedCutoff, switchingDistance, rfDielectric, ewaldAlpha, ewaldDispersionAlpha, dispersionCoefficient;
    int kmax[3], gridSize[3], dispersionGridSize[3];
    bool useSwitchingFunction, exceptionsArePeriodic;
    std::vector<std::set<int> > exclusions;
    NonbondedMethod nonbondedMethod;
    OpenMM::NeighborList* neighborList;
};

} // namespace NativeNonbondedPlugin

#endif /*REFERENCE_NATIVENONBONDED_KERNELS_H_*/
