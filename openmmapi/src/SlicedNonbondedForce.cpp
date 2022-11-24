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
#include "openmm/internal/AssertionUtilities.h"

using namespace std;
using namespace OpenMM;
using namespace PmeSlicing;

#define ASSERT_VALID(name, value, number) {if (value < 0 || value >= number) throwException(__FILE__, __LINE__, name " out of range");};

SlicedNonbondedForce::SlicedNonbondedForce(int numSubsets) : NonbondedForce(), numSubsets(numSubsets) {
}

void SlicedNonbondedForce::setParticleSubset(int index, int subset) {
    ASSERT_VALID("Index", index, getNumParticles());
    ASSERT_VALID("Subset", subset, getNumSubsets());
    subsets[index] = subset;
}

int SlicedNonbondedForce::getParticleSubset(int index) const {
    ASSERT_VALID("Index", index, getNumParticles());
    auto element = subsets.find(index);
    return element == subsets.end() ? 0 : element->second;
}

ForceImpl* SlicedNonbondedForce::createImpl() const {
    return new SlicedNonbondedForceImpl(*this);
}