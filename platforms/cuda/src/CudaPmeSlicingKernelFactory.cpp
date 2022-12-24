/* -------------------------------------------------------------------------- *
 *                              OpenMMPmeSlicing                                   *
 * -------------------------------------------------------------------------- */

#include <exception>

#include "CudaPmeSlicingKernelFactory.h"
#include "CudaPmeSlicingKernels.h"
#include "CudaParallelPmeSlicingKernels.h"
#include "CommonPmeSlicingKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace NonbondedSlicing;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaPmeSlicingKernelFactory* factory = new CudaPmeSlicingKernelFactory();
        platform.registerKernelFactory(CalcSlicedPmeForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcSlicedNonbondedForceKernel::Name(), factory);
    }
    catch (std::exception ex) {
        // Ignore
    }
}

extern "C" OPENMM_EXPORT void registerPmeSlicingCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaPmeSlicingKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
    if (data.contexts.size() > 1) {
        // We are running in parallel on multiple devices, so we may want to create a parallel kernel.
        if (name == CalcSlicedPmeForceKernel::Name())
            return new CudaParallelCalcSlicedPmeForceKernel(name, platform, data, context.getSystem());
        else if (name == CalcSlicedNonbondedForceKernel::Name())
            return new CudaParallelCalcSlicedNonbondedForceKernel(name, platform, data, context.getSystem());
    }
    CudaContext& cu = *data.contexts[0];
    if (name == CalcSlicedPmeForceKernel::Name())
        return new CudaCalcSlicedPmeForceKernel(name, platform, cu, context.getSystem());
    else if (name == CalcSlicedNonbondedForceKernel::Name())
        return new CudaCalcSlicedNonbondedForceKernel(name, platform, cu, context.getSystem());
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
