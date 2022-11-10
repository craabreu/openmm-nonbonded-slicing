/* -------------------------------------------------------------------------- *
 *                              OpenMMPmeSlicing                                   *
 * -------------------------------------------------------------------------- */

#include "ReferencePmeSlicingKernelFactory.h"
#include "ReferencePmeSlicingKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace PmeSlicing;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferencePmeSlicingKernelFactory* factory = new ReferencePmeSlicingKernelFactory();
            platform.registerKernelFactory(CalcSlicedPmeForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerPmeSlicingReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferencePmeSlicingKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    ReferencePlatform::PlatformData& data = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcSlicedPmeForceKernel::Name())
        return new ReferenceCalcSlicedPmeForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
