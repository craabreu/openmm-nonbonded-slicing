/* -------------------------------------------------------------------------- *
 *                              OpenMMNonbondedSlicing                                   *
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

#include <exception>

#include "OpenCLNonbondedSlicingKernelFactory.h"
#include "OpenCLNonbondedSlicingKernels.h"
#include "OpenCLParallelNonbondedSlicingKernels.h"
#include "CommonNonbondedSlicingKernels.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace NonbondedSlicing;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("OpenCL");
        OpenCLNonbondedSlicingKernelFactory* factory = new OpenCLNonbondedSlicingKernelFactory();
        platform.registerKernelFactory(CalcSlicedNonbondedForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerNonbondedSlicingOpenCLKernelFactories() {
    try {
        Platform::getPlatformByName("OpenCL");
    }
    catch (...) {
        Platform::registerPlatform(new OpenCLPlatform());
    }
    registerKernelFactories();
}

KernelImpl* OpenCLNonbondedSlicingKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    OpenCLPlatform::PlatformData& data = *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData());
    if (data.contexts.size() > 1) {
        if (name == CalcSlicedNonbondedForceKernel::Name())
            return new OpenCLParallelCalcSlicedNonbondedForceKernel(name, platform, data, context.getSystem());
        throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
    }
    OpenCLContext& cl = *data.contexts[0];
    if (name == CalcSlicedNonbondedForceKernel::Name())
        return new OpenCLCalcSlicedNonbondedForceKernel(name, platform, cl, context.getSystem());
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
