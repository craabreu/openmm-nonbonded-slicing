/* -------------------------------------------------------------------------- *
 *                              OpenMMNonbondedSlicing                                   *
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
