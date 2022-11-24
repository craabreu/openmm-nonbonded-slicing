#ifndef OPENMM_SLICEDNONBONDEDFORCE_H_
#define OPENMM_SLICEDNONBONDEDFORCE_H_

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

#include "openmm/Context.h"
#include "openmm/NonbondedForce.h"
#include "openmm/Force.h"
#include <map>
#include <set>
#include <utility>
#include <vector>
#include "internal/windowsExportPmeSlicing.h"

using namespace OpenMM;

namespace PmeSlicing {

class OPENMM_EXPORT_PMESLICING SlicedNonbondedForce : public NonbondedForce {
public:
    /**
     * Create a SlicedNonbondedForce.
     */
    SlicedNonbondedForce(int numSubsets);
protected:
    ForceImpl* createImpl() const;
private:
    int numSubsets;
};

} // namespace PmeSlicing

#endif /*OPENMM_SLICEDNONBONDEDFORCE_H_*/
