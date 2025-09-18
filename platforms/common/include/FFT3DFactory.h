#ifndef COMMON_FFT3D_FACTORY_H_
#define COMMON_FFT3D_FACTORY_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2025 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "openmm/common/FFT3D.h"
#include "openmm/common/ComputeContext.h"

using namespace OpenMM;

namespace NonbondedSlicing {

class FFT3DFactory {
public:
    virtual FFT3D createFFT3D(ComputeContext& context, int xsize, int ysize, int zsize, int numBatches, bool realToComplex=false) = 0;
};

} // namespace NonbondedSlicing

#endif // COMMON_FFT3D_FACTORY_H_