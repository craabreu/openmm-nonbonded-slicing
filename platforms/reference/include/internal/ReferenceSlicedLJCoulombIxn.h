#ifndef __ReferenceSlicedLJCoulombIxn_H__
#define __ReferenceSlicedLJCoulombIxn_H__

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "openmm/reference/ReferencePairIxn.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include "internal/InnerSplit.h"
#include "internal/CappedLJ.h"

using namespace std;
using namespace OpenMM;

namespace NonbondedSlicing {

class ReferenceSlicedLJCoulombIxn {

   private:

      bool cutoff;
      bool useSwitch;
      bool periodic, periodicExceptions, reciprocalExclusions;
      bool ewald;
      bool pme, ljpme;
      const OpenMM::NeighborList* neighborList;
      OpenMM::Vec3 periodicBoxVectors[3];
      double cutoffDistance, switchingDistance;
      InnerSplit innerSplit;
      CappedLJ cappedLJ;
      CappedCoulomb cappedCoulomb;
      bool coulombDecapping;
      double krf, crf;
      double alphaEwald, alphaDispersionEwald;
      int numRx, numRy, numRz;
      int meshDim[3], dispersionMeshDim[3];
      vector<bool> decappedSlices;

      // parameter indices

      static const int SigIndex = 0;
      static const int EpsIndex = 1;
      static const int   QIndex = 2;

      static const int   Coul = 0;
      static const int   vdW = 1;
      static const int   decap = 2;

      /**---------------------------------------------------------------------------------------

         Calculate LJ Coulomb pair ixn between two atoms

         @param atom1            the index of the first atom
         @param atom2            the index of the second atom
         @param atomCoordinates  atom coordinates
         @param atomSubsets      atom subsets
         @param atomParameters   atom parameters (charges, c6, c12, ...)     atomParameters[atomIndex][paramterIndex]
         @param sliceLambda      Coulomb and LJ scaling parameters for each slice
         @param forces           force array (forces added)
         @param sliceEnergies    the energy of each slice

         --------------------------------------------------------------------------------------- */

      void calculateOneIxn(int atom1, int atom2, vector<OpenMM::Vec3>& atomCoordinates, const vector<int>& atomSubsets,
                           const vector<vector<double>>& atomParameters, const vector<vector<double>>& sliceLambdas, vector<OpenMM::Vec3>& forces,
                           vector<vector<double>>& sliceEnergies) const;

      void splitCoulomb(double inverseR, double sig, double qq, double& coulEnergy, double& coulRdUdr,
                        double& decapEnergy, double& decapRdUdr) const;

      void calculateInnerPair(int ii, int jj, const vector<vector<double>>& atomParameters, double r, bool decapped,
                              double& forceFactorVdw, double& forceFactorCoul, double& vdwEnergy, double& coulEnergy,
                              double& forceFactorDecap, double& decapEnergy) const;


   public:

      /**---------------------------------------------------------------------------------------

         Constructor

         --------------------------------------------------------------------------------------- */

       ReferenceSlicedLJCoulombIxn();

      /**---------------------------------------------------------------------------------------

         Destructor

         --------------------------------------------------------------------------------------- */

       ~ReferenceSlicedLJCoulombIxn();

      /**---------------------------------------------------------------------------------------

         Set the force to use a cutoff.

         @param distance            the cutoff distance
         @param neighbors           the neighbor list to use
         @param solventDielectric   the dielectric constant of the bulk solvent

         --------------------------------------------------------------------------------------- */

      void setUseCutoff(double distance, const OpenMM::NeighborList& neighbors, double solventDielectric);

      /**---------------------------------------------------------------------------------------

         Set the force to use a switching function on the Lennard-Jones interaction.

         @param distance            the switching distance

         --------------------------------------------------------------------------------------- */

      void setUseSwitchingFunction(double distance);

      /**---------------------------------------------------------------------------------------

         Set the force to use periodic boundary conditions.  This requires that a cutoff has
         already been set, and the smallest side of the periodic box is at least twice the cutoff
         distance.

         @param vectors    the vectors defining the periodic box

         --------------------------------------------------------------------------------------- */

      void setPeriodic(OpenMM::Vec3* vectors);

      /**---------------------------------------------------------------------------------------

         Set the force to use Ewald summation.

         @param alpha  the Ewald separation parameter
         @param kmaxx  the largest wave vector in the x direction
         @param kmaxy  the largest wave vector in the y direction
         @param kmaxz  the largest wave vector in the z direction

         --------------------------------------------------------------------------------------- */

      void setUseEwald(double alpha, int kmaxx, int kmaxy, int kmaxz);


      /**---------------------------------------------------------------------------------------

         Set the force to use Particle-Mesh Ewald (PME) summation.

         @param alpha    the Ewald separation parameter
         @param gridSize the dimensions of the mesh

         --------------------------------------------------------------------------------------- */

      void setUsePME(double alpha, int meshSize[3]);

      /**---------------------------------------------------------------------------------------

         Set the force to use Particle-Mesh Ewald (PME) summation for dispersion.

         @param dalpha    the dispersion Ewald separation parameter
         @param dgridSize the dimensions of the dispersion mesh

         --------------------------------------------------------------------------------------- */

      void setUseLJPME(double dalpha, int dmeshSize[3]);

      /**---------------------------------------------------------------------------------------

         Set whether exceptions use periodic boundary conditions.

         --------------------------------------------------------------------------------------- */

      void setPeriodicExceptions(bool periodic);

      /**---------------------------------------------------------------------------------------

         Set whether the Ewald exclusion corrections go with the reciprocal space calculation
         instead of the direct space one.

         --------------------------------------------------------------------------------------- */

      void setReciprocalExclusions(bool reciprocal);

      /**---------------------------------------------------------------------------------------

         Set which slices are decapped. Must be called before calculatePairIxn.

         @param decapped  whether each slice is decapped

         --------------------------------------------------------------------------------------- */

      void setDecappedSlices(const vector<bool>& decapped);

      /**---------------------------------------------------------------------------------------

         Set the capped Lennard-Jones potential of the decapped slices.

         @param capped   the capped potential

         --------------------------------------------------------------------------------------- */

      void setCappedLJ(const CappedLJ& capped);

      /**---------------------------------------------------------------------------------------

         Set whether the decapping term of the decapped slices includes the Coulomb interaction, and
         the capped Coulomb potential.

         @param decapping  whether Coulomb interactions are decapped
         @param capped     the capped Coulomb potential

         --------------------------------------------------------------------------------------- */

      void setCoulombDecapping(bool decapping, const CappedCoulomb& capped);

      /**---------------------------------------------------------------------------------------

         Set the inner force group split.  When enabled, the direct-space interactions computed by
         calculatePairIxn exclude their inner part, which is computed by calculateInnerIxn instead.

         @param split    the inner split

         --------------------------------------------------------------------------------------- */

      void setInnerSplit(const InnerSplit& split);

      /**---------------------------------------------------------------------------------------

         Calculate the inner part of all direct-space pair interactions

         @param numberOfAtoms    number of atoms
         @param atomCoordinates  atom coordinates
         @param atomSubsets      atom subsets
         @param atomParameters   atom parameters (sigma/2, 2*sqrt(epsilon), charge)
         @param sliceLambdas     Coulomb and LJ scaling parameters for each slice
         @param exclusions       atom exclusion indices
         @param forces           force array (forces added)
         @param sliceEnergies    the energy of each slice

         --------------------------------------------------------------------------------------- */

      void calculateInnerIxn(int numberOfAtoms, vector<OpenMM::Vec3>& atomCoordinates, const vector<int>& atomSubsets,
                           const vector<vector<double>>& atomParameters, const vector<vector<double>>& sliceLambdas,
                           const vector<set<int> >& exclusions, vector<OpenMM::Vec3>& forces,
                           vector<vector<double>>& sliceEnergies) const;

      /**---------------------------------------------------------------------------------------

         Calculate LJ Coulomb pair ixn

         @param numberOfAtoms    number of atoms
         @param atomCoordinates  atom coordinates
         @param atomSubsets      atom subsets
         @param atomParameters   atom parameters (charges, c6, c12, ...)     atomParameters[atomIndex][paramterIndex]
         @param sliceLambda      Coulomb and LJ scaling parameters for each slice
         @param exclusions       atom exclusion indices
                                 exclusions[atomIndex] contains the list of exclusions for that atom
         @param forces           force array (forces added)
         @param sliceEnergies    the energy of each slice
         @param includeDirect      true if direct space interactions should be included
         @param includeReciprocal  true if reciprocal space interactions should be included

         --------------------------------------------------------------------------------------- */

      void calculatePairIxn(int numberOfAtoms, vector<OpenMM::Vec3>& atomCoordinates, int numberOfSubsets, const vector<int>& atomSubsets,
                           const vector<vector<double>>& atomParameters, const vector<vector<double>>& sliceLambdas, const vector<set<int> >& exclusions,
                           vector<OpenMM::Vec3>& forces, vector<vector<double>>& sliceEnergies, bool includeDirect, bool includeReciprocal) const;

private:
      /**---------------------------------------------------------------------------------------

         Calculate Ewald ixn

         @param numberOfAtoms    number of atoms
         @param atomCoordinates  atom coordinates
         @param atomSubsets      atom subsets
         @param atomParameters   atom parameters (charges, c6, c12, ...)     atomParameters[atomIndex][paramterIndex]
         @param sliceLambda      Coulomb and LJ scaling parameters for each slice
         @param exclusions       atom exclusion indices
                                 exclusions[atomIndex] contains the list of exclusions for that atom
         @param forces           force array (forces added)
         @param sliceEnergies    the energy of each slice
         @param includeDirect      true if direct space interactions should be included
         @param includeReciprocal  true if reciprocal space interactions should be included

         --------------------------------------------------------------------------------------- */

      void calculateEwaldIxn(int numberOfAtoms, vector<OpenMM::Vec3>& atomCoordinates, int numberOfSubsets, const vector<int>& atomSubsets,
                           const vector<vector<double>>& atomParameters, const vector<vector<double>>& sliceLambdas, const vector<set<int> >& exclusions,
                           vector<OpenMM::Vec3>& forces, vector<vector<double>>& sliceEnergies, bool includeDirect, bool includeReciprocal) const;
};

} // namespace OpenMM

#endif // __ReferenceSlicedLJCoulombIxn_H__
