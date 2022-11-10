#ifndef OPENMM_CUDAPMESLICINGKERNELFACTORY_H_
#define OPENMM_CUDAPMESLICINGKERNELFACTORY_H_

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

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the CUDA implementation of the PmeSlicing plugin.
 */

class CudaPmeSlicingKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_CUDAPMESLICINGKERNELFACTORY_H_*/
