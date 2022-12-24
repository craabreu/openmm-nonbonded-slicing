#ifndef OPENMM_SLICEDNONBONDEDFORCE_PROXY_H_
#define OPENMM_SLICEDNONBONDEDFORCE_PROXY_H_

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

#include "internal/windowsExportPmeSlicing.h"
#include "openmm/serialization/SerializationProxy.h"

using namespace OpenMM;

namespace PmeSlicing {

/**
 * This is a proxy for serializing SlicedNonbondedForce objects.
 */

class OPENMM_EXPORT_PMESLICING SlicedNonbondedForceProxy : public SerializationProxy {
public:
    SlicedNonbondedForceProxy();
    void serialize(const void* object, SerializationNode& node) const;
    void* deserialize(const SerializationNode& node) const;
};

} // namespace OpenMM

#endif /*OPENMM_SLICEDNONBONDEDFORCE_PROXY_H_*/
