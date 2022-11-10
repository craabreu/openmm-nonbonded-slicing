#ifndef OPENMM_REFERENCEPMESLICINGKERNELFACTORY_H_
#define OPENMM_REFERENCEPMESLICINGKERNELFACTORY_H_

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
#include <string.h>

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the reference implementation of the PmeSlicing plugin.
 */

class ReferencePmeSlicingKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_REFERENCEPMESLICINGKERNELFACTORY_H_*/
