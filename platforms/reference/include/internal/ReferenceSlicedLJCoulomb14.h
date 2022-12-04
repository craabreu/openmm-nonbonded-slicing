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

#include "openmm/Vec3.h"
#include "internal/windowsExportPmeSlicing.h"
#include <vector>

using namespace std;
using namespace OpenMM;

namespace PmeSlicing {

class OPENMM_EXPORT_PMESLICING ReferenceSlicedLJCoulomb14 {

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
       @param sliceLambdas     the scaling parameters of the slice
       @param sliceEnergies    the energies of the slice

       --------------------------------------------------------------------------------------- */

    void calculateBondIxn(vector<int>& atomIndices, vector<OpenMM::Vec3>& atomCoordinates,
                          vector<double>& parameters, vector<OpenMM::Vec3>& forces,
                          vector<double>& sliceLambdas, vector<double>& sliceEnergies);

private:
    bool periodic;
    OpenMM::Vec3 periodicBoxVectors[3];
};

} // namespace PmeSlicing

#endif // __ReferenceSlicedLJCoulomb14_H__
