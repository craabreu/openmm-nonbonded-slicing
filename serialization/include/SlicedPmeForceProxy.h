#ifndef OPENMM_SLICEDPMEFORCE_PROXY_H_
#define OPENMM_SLICEDPMEFORCE_PROXY_H_

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

namespace OpenMM {

/**
 * This is a proxy for serializing SlicedPmeForce objects.
 */

class OPENMM_EXPORT_PMESLICING SlicedPmeForceProxy : public SerializationProxy {
public:
    SlicedPmeForceProxy();
    void serialize(const void* object, SerializationNode& node) const;
    void* deserialize(const SerializationNode& node) const;
};

} // namespace OpenMM

#endif /*OPENMM_SLICEDPMEFORCE_PROXY_H_*/
