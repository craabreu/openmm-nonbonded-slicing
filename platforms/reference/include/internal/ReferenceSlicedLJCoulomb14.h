#ifndef __ReferenceSlicedLJCoulomb14_H__
#define __ReferenceSlicedLJCoulomb14_H__
/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "openmm/Vec3.h"
#include "internal/windowsExportNonbondedSlicing.h"
#include "internal/InnerSplit.h"
#include "internal/CappedLJ.h"
#include <vector>

using namespace std;
using namespace OpenMM;

namespace NonbondedSlicing {

class OPENMM_EXPORT_NONBONDED_SLICING ReferenceSlicedLJCoulomb14 {

public:

    /**
     * Which part of an interaction to compute: all of it, only its inner part, or all but its inner part.
     */
    enum Mode {Full = 0, Inner = 1, Remainder = 2};

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

       Set the inner force group split used by the Inner and Remainder modes.

       @param split    the inner split

       --------------------------------------------------------------------------------------- */

    void setInnerSplit(const InnerSplit& split);

    /**---------------------------------------------------------------------------------------

       Set the capped Lennard-Jones potential of the decapped slices.

       @param capped   the capped potential

       --------------------------------------------------------------------------------------- */

    void setCappedLJ(const CappedLJ& capped);

    /**---------------------------------------------------------------------------------------

       Calculate nonbonded 1-4 interactinos

       @param atomIndices      atom indices of the atoms in each pair
       @param atomCoordinates  atom coordinates
       @param parameters       (sigma, 4*epsilon, charge product) for each pair
       @param forces           force array (forces added to current values)
       @param sliceLambdas     the scaling parameters of the slice
       @param sliceEnergies    the energies of the slice
       @param decapped         whether the slice is decapped
       @param mode             which part of the interactions to compute

       --------------------------------------------------------------------------------------- */

    void calculateBondIxn(vector<int>& atomIndices, vector<OpenMM::Vec3>& atomCoordinates,
                          vector<double>& parameters, vector<OpenMM::Vec3>& forces,
                          vector<double>& sliceLambdas, vector<double>& sliceEnergies,
                          bool decapped, Mode mode=Full);

private:
   static const int   Coul = 0;
   static const int   vdW = 1;
   static const int   decap = 2;
   bool periodic;
   OpenMM::Vec3 periodicBoxVectors[3];
   InnerSplit innerSplit;
   CappedLJ cappedLJ;
};

} // namespace NonbondedSlicing

#endif // __ReferenceSlicedLJCoulomb14_H__
