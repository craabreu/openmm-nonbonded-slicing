/* -------------------------------------------------------------------------- *
 *                             OpenMM PME Slicing                             *
 *                             ==================                             *
 *                                                                            *
 * An OpenMM plugin for slicing Particle Mesh Ewald calculations on the basis *
 * of atom pairs and applying a different switching parameter to each slice.  *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-pme-slicing                             *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "openmm/opencl/OpenCLPlatform.h"

extern "C" OPENMM_EXPORT void registerPmeSlicingOpenCLKernelFactories();

OpenMM::OpenCLPlatform platform;

void initializeTests(int argc, char* argv[]) {
    registerPmeSlicingOpenCLKernelFactories();
    platform = dynamic_cast<OpenMM::OpenCLPlatform&>(OpenMM::Platform::getPlatformByName("OpenCL"));
    if (argc > 1)
        platform.setPropertyDefaultValue("Precision", std::string(argv[1]));
    if (argc > 2)
        platform.setPropertyDefaultValue("DeviceIndex", std::string(argv[2]));
}