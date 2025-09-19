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
#include "internal/CudaCuFFT3D.h"
#include "internal/CudaVkFFT3D.h"
#include "SlicedNonbondedForce.h"
#include "openmm/System.h"

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

void CudaCalcSlicedNonbondedForceKernel::initialize(const System& system, const SlicedNonbondedForce& force) {
    bool usePmeQueue = !cu.getPlatformData().disablePmeStream;
    bool useFixedPointChargeSpreading = cu.getUseDoublePrecision() || cu.getPlatformData().deterministicForces;
    if (force.getUseCuFFT()) {
        CudaCuFFTFactory cuFFTFactory;
        commonInitialize(system, force, cuFFTFactory, usePmeQueue, false, useFixedPointChargeSpreading, false);
    }
    else {
        CudaVkFFTFactory vkFFTFactory;
        commonInitialize(system, force, vkFFTFactory, usePmeQueue, false, useFixedPointChargeSpreading, false);
    }
}
