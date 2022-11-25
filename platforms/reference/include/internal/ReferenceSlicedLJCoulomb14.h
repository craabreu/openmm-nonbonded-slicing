#ifndef __ReferenceSlicedLJCoulomb14_H__
#define __ReferenceSlicedLJCoulomb14_H__

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

#include "openmm/reference/ReferenceBondIxn.h"
#include "internal/windowsExportPmeSlicing.h"

using namespace OpenMM;

namespace PmeSlicing {

class OPENMM_EXPORT_PMESLICING ReferenceSlicedLJCoulomb14 : public ReferenceBondIxn {

public:

    /**---------------------------------------------------------------------------------------

       Constructor

       --------------------------------------------------------------------------------------- */

     ReferenceSlicedLJCoulomb14();

    /**---------------------------------------------------------------------------------------

       Destructor

       --------------------------------------------------------------------------------------- */

     ~ReferenceSlicedLJCoulomb14();

     /**---------------------------------------------------------------------------------------

       Set the force to use periodic boundary conditions.

       @param vectors    the vectors defining the periodic box

       --------------------------------------------------------------------------------------- */

    void setPeriodic(OpenMM::Vec3* vectors);

    /**---------------------------------------------------------------------------------------

       Calculate nonbonded 1-4 interactinos

       @param atomIndices      atom indices of the atoms in each pair
       @param atomCoordinates  atom coordinates
       @param parameters       (sigma, 4*epsilon, charge product) for each pair
       @param forces           force array (forces added to current values)
       @param totalEnergy      if not null, the energy will be added to this

       --------------------------------------------------------------------------------------- */

    void calculateBondIxn(std::vector<int>& atomIndices, std::vector<OpenMM::Vec3>& atomCoordinates,
                          std::vector<double>& parameters, std::vector<OpenMM::Vec3>& forces,
                          double* totalEnergy, double* energyParamDerivs);

private:
    bool periodic;
    OpenMM::Vec3 periodicBoxVectors[3];
};

} // namespace OpenMM

#endif // __ReferenceSlicedLJCoulomb14_H__
