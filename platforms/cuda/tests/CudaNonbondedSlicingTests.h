/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "openmm/cuda/CudaPlatform.h"

extern "C" OPENMM_EXPORT void registerNonbondedSlicingCudaKernelFactories();

OpenMM::CudaPlatform platform;

void initializeTests(int argc, char* argv[]) {
    registerNonbondedSlicingCudaKernelFactories();
    platform = dynamic_cast<OpenMM::CudaPlatform&>(OpenMM::Platform::getPlatformByName("CUDA"));
    if (argc > 1)
        platform.setPropertyDefaultValue("Precision", std::string(argv[1]));
    if (argc > 2)
        platform.setPropertyDefaultValue("DeviceIndex", std::string(argv[2]));
}