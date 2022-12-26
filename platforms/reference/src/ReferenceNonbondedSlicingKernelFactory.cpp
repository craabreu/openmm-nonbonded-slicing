/* -------------------------------------------------------------------------- *
 *                              OpenMMNonbondedSlicing                                   *
 * -------------------------------------------------------------------------- */

#include "ReferenceNonbondedSlicingKernelFactory.h"
#include "ReferenceNonbondedSlicingKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace NonbondedSlicing;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferenceNonbondedSlicingKernelFactory* factory = new ReferenceNonbondedSlicingKernelFactory();
            platform.registerKernelFactory(CalcSlicedNonbondedForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerNonbondedSlicingReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferenceNonbondedSlicingKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    ReferencePlatform::PlatformData& data = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcSlicedNonbondedForceKernel::Name())
        return new ReferenceCalcSlicedNonbondedForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
