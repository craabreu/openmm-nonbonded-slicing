/* -------------------------------------------------------------------------- *
 *                              OpenMMPmeSlicing                                   *
 * -------------------------------------------------------------------------- */

#include <exception>

#include "OpenCLPmeSlicingKernelFactory.h"
#include "OpenCLPmeSlicingKernels.h"
#include "OpenCLParallelPmeSlicingKernels.h"
#include "CommonPmeSlicingKernels.h"
#include "openmm/opencl/OpenCLContext.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace PmeSlicing;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("OpenCL");
        OpenCLPmeSlicingKernelFactory* factory = new OpenCLPmeSlicingKernelFactory();
        platform.registerKernelFactory(CalcSlicedPmeForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcSlicedNonbondedForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerPmeSlicingOpenCLKernelFactories() {
    try {
        Platform::getPlatformByName("OpenCL");
    }
    catch (...) {
        Platform::registerPlatform(new OpenCLPlatform());
    }
    registerKernelFactories();
}

KernelImpl* OpenCLPmeSlicingKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    OpenCLPlatform::PlatformData& data = *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData());
    if (data.contexts.size() > 1) {
        if (name == CalcSlicedPmeForceKernel::Name())
            return new OpenCLParallelCalcSlicedPmeForceKernel(name, platform, data, context.getSystem());
        else if (name == CalcSlicedNonbondedForceKernel::Name())
            return new OpenCLParallelCalcSlicedNonbondedForceKernel(name, platform, data, context.getSystem());
        throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
    }
    OpenCLContext& cl = *data.contexts[0];
    if (name == CalcSlicedPmeForceKernel::Name())
        return new OpenCLCalcSlicedPmeForceKernel(name, platform, cl, context.getSystem());
    else if (name == CalcSlicedNonbondedForceKernel::Name())
        return new OpenCLCalcSlicedNonbondedForceKernel(name, platform, cl, context.getSystem());
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
