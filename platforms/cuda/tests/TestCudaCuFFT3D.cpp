/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2025 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

/**
 * This tests the CUDA implementation of CudaCuFFT.
 */

#include "internal/CudaCuFFT3D.h"
#include "CudaFFT3DTests.h"

int main(int argc, char* argv[]) {
    return runFFT3DTests<NonbondedSlicing::CudaCuFFT>(argc, argv);
}
