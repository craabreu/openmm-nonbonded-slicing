#ifndef OPENMM_OPENCLPMESLICINGKERNELFACTORY_H_
#define OPENMM_OPENCLPMESLICINGKERNELFACTORY_H_

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

namespace PmeSlicing {

/**
 * This KernelFactory creates kernels for the OpenCL implementation of the PmeSlicing plugin.
 */

class OpenCLPmeSlicingKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace PmeSlicing

#endif /*OPENMM_OPENCLPMESLICINGKERNELFACTORY_H_*/
