/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "CudaNonbondedSlicingKernels.h"
#include "CudaNonbondedSlicingKernelSources.h"
#include "CudaBatchedFFT3D.h"
#include "SlicedNonbondedForce.h"
#include "openmm/System.h"

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

void CudaCalcSlicedNonbondedForceKernel::initialize(const System& system, const SlicedNonbondedForce& force) {
    bool usePmeQueue = (!cu.getPlatformData().disablePmeStream && !cu.getPlatformData().useCpuPme);
    bool useFixedPointChargeSpreading = cu.getUseDoublePrecision() || cu.getPlatformData().deterministicForces;
    CudaBatchedFFT3DFactory fftFactory;
    commonInitialize(system, force, fftFactory, usePmeQueue, false, useFixedPointChargeSpreading, cu.getPlatformData().useCpuPme);
}
