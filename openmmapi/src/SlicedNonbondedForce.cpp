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

#include "SlicedNonbondedForce.h"
#include "internal/SlicedNonbondedForceImpl.h"

using namespace OpenMM;
using namespace PmeSlicing;

SlicedNonbondedForce::SlicedNonbondedForce(int numSubsets) : NonbondedForce(), numSubsets(numSubsets) {
}

ForceImpl* SlicedNonbondedForce::createImpl() const {
    return new SlicedNonbondedForceImpl(*this);
}