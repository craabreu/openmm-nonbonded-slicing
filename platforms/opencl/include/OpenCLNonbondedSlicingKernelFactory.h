#ifndef OPENMM_OPENCLNONBONDED_SLICINGKERNELFACTORY_H_
#define OPENMM_OPENCLNONBONDED_SLICINGKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
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

#endif /*OPENMM_OPENCLNONBONDED_SLICINGKERNELFACTORY_H_*/
