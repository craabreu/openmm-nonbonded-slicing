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
#include "openmm/opencl/OpenCLPlatform.h"

extern "C" OPENMM_EXPORT void registerNonbondedSlicingOpenCLKernelFactories();

OpenMM::OpenCLPlatform platform;

void initializeTests(int argc, char* argv[]) {
    registerNonbondedSlicingOpenCLKernelFactories();
    platform = dynamic_cast<OpenMM::OpenCLPlatform&>(OpenMM::Platform::getPlatformByName("OpenCL"));
    if (argc > 1)
        platform.setPropertyDefaultValue("Precision", std::string(argv[1]));
    if (argc > 2)
        platform.setPropertyDefaultValue("DeviceIndex", std::string(argv[2]));
}