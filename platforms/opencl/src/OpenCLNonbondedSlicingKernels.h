#ifndef OPENCL_NONBONDEDSLICING_KERNELS_H_
#define OPENCL_NONBONDEDSLICING_KERNELS_H_

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

#include "NonbondedSlicingKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/opencl/OpenCLArray.h"
#include "openmm/opencl/OpenCLFFT3D.h"
#include "openmm/opencl/OpenCLSort.h"
#include <vector>

namespace NonbondedSlicing {

/**
 * This kernel is invoked by SlicedNonbondedForce to calculate the forces acting on the system.
 */
class OpenCLCalcSlicedNonbondedForceKernel : public CalcSlicedNonbondedForceKernel {
public:
    OpenCLCalcSlicedNonbondedForceKernel(std::string name, const Platform& platform, OpenCLContext& cl, const System& system) : CalcSlicedNonbondedForceKernel(name, platform),
            hasInitializedKernel(false), cl(cl), sort(NULL), fft(NULL), dispersionFft(NULL), pmeio(NULL), usePmeQueue(false) {
    }
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
    class PmeIO;
    class PmePreComputation;
    class PmePostComputation;
    class SyncQueuePreComputation;
    class SyncQueuePostComputation;
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
    OpenCLSort* sort;
    cl::CommandQueue pmeQueue;
    cl::Event pmeSyncEvent;
    OpenCLFFT3D* fft;
    OpenCLFFT3D* dispersionFft;
    Kernel cpuPme;
    PmeIO* pmeio;
    SyncQueuePostComputation* syncQueue;
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
    std::map<std::string, std::string> pmeDefines;
    std::vector<std::pair<int, int> > exceptionAtoms;
    std::vector<std::string> paramNames;
    std::vector<double> paramValues;
    double ewaldSelfEnergy, alpha, dispersionAlpha;
    int gridSizeX, gridSizeY, gridSizeZ;
    int dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ;
    bool hasCoulomb, hasLJ, usePmeQueue, doLJPME, usePosqCharges, recomputeParams, hasOffsets;
    static const int PmeOrder = 5;
};

} // namespace NonbondedSlicing

#endif /*OPENCL_NONBONDEDSLICING_KERNELS_H_*/
