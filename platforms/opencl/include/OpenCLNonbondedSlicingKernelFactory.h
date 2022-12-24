#ifndef OPENMM_OPENCLPMESLICINGKERNELFACTORY_H_
#define OPENMM_OPENCLPMESLICINGKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential calculations on the basis *
 * of atom pairs and for applying scaling parameters to selected slices.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelFactory.h"

namespace NonbondedSlicing {

/**
 * This KernelFactory creates kernels for the OpenCL implementation of the NonbondedSlicing plugin.
 */

class OpenCLNonbondedSlicingKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace NonbondedSlicing

#endif /*OPENMM_OPENCLPMESLICINGKERNELFACTORY_H_*/
