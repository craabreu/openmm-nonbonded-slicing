/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

 #include "OpenCLNonbondedSlicingKernels.h"
 #include "OpenCLNonbondedSlicingKernelSources.h"
 #include "internal/OpenCLVkFFT3D.h"
 #include "SlicedNonbondedForce.h"
 #include "openmm/System.h"
 
 using namespace NonbondedSlicing;
 using namespace OpenMM;
 using namespace std;
 
 void OpenCLCalcSlicedNonbondedForceKernel::initialize(const System& system, const SlicedNonbondedForce& force) {
    string vendor = cl.getDevice().getInfo<CL_DEVICE_VENDOR>();
    bool isNvidia = (vendor.size() >= 6 && vendor.substr(0, 6) == "NVIDIA");
    bool usePmeQueue = (!cl.getPlatformData().disablePmeStream && !cl.getPlatformData().useCpuPme && isNvidia);
    bool deviceIsCpu = (cl.getDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU);
    OpenCLVkFFTFactory vkFFTFactory;
    commonInitialize(system, force, vkFFTFactory, usePmeQueue, deviceIsCpu, true, cl.getPlatformData().useCpuPme);
}
