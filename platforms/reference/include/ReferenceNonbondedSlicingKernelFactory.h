#ifndef OPENMM_REFERENCENONBONDED_SLICINGKERNELFACTORY_H_
#define OPENMM_REFERENCENONBONDED_SLICINGKERNELFACTORY_H_

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
#include <string.h>

using namespace OpenMM;

namespace NonbondedSlicing {

/**
 * This KernelFactory creates kernels for the reference implementation of the NonbondedSlicing plugin.
 */

class ReferenceNonbondedSlicingKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace NonbondedSlicing

#endif /*OPENMM_REFERENCENONBONDED_SLICINGKERNELFACTORY_H_*/
