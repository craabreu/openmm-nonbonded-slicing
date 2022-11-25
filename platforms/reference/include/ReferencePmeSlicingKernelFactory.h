#ifndef OPENMM_REFERENCEPMESLICINGKERNELFACTORY_H_
#define OPENMM_REFERENCEPMESLICINGKERNELFACTORY_H_

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

#include "openmm/KernelFactory.h"
#include <string.h>

using namespace OpenMM;

namespace PmeSlicing {

/**
 * This KernelFactory creates kernels for the reference implementation of the PmeSlicing plugin.
 */

class ReferencePmeSlicingKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace PmeSlicing

#endif /*OPENMM_REFERENCEPMESLICINGKERNELFACTORY_H_*/
