/* -------------------------------------------------------------------------- *
 *                             OpenMM PME Slicing                             *
 *                             ==================                             *
 *                                                                            *
 * An OpenMM plugin for Smooth Particle Mesh Ewald electrostatic calculations *
 * with multiple coupling parameters.                                         *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-pme-slicing                             *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "openmm/cuda/CudaPlatform.h"

extern "C" OPENMM_EXPORT void registerPmeSlicingCudaKernelFactories();

OpenMM::CudaPlatform platform;

void initializeTests(int argc, char* argv[]) {
    registerPmeSlicingCudaKernelFactories();
    platform = dynamic_cast<OpenMM::CudaPlatform&>(OpenMM::Platform::getPlatformByName("CUDA"));
    if (argc > 1)
        platform.setPropertyDefaultValue("Precision", std::string(argv[1]));
    if (argc > 2)
        platform.setPropertyDefaultValue("DeviceIndex", std::string(argv[2]));
}