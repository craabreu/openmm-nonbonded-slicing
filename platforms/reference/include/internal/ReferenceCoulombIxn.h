
/* Portions copyright (c) 2006-2020 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __ReferenceCoulombIxn_H__
#define __ReferenceCoulombIxn_H__

#include "openmm/reference/ReferencePairIxn.h"
#include "openmm/reference/ReferenceNeighborList.h"

using namespace std;

namespace PmeSlicing {

class ReferenceCoulombIxn {

   private:
       
      bool periodicExceptions;
      const OpenMM::NeighborList* neighborList;
      OpenMM::Vec3 periodicBoxVectors[3];
      double cutoffDistance;
      double alphaEwald;
      int meshDim[3];
            
   public:

      /**---------------------------------------------------------------------------------------
      
         Constructor
      
         --------------------------------------------------------------------------------------- */

       ReferenceCoulombIxn();

      /**---------------------------------------------------------------------------------------
      
         Destructor
      
         --------------------------------------------------------------------------------------- */

       ~ReferenceCoulombIxn();

      /**---------------------------------------------------------------------------------------
      
         Set the force to use a cutoff.
      
         @param distance            the cutoff distance
         @param neighbors           the neighbor list to use
      
         --------------------------------------------------------------------------------------- */
      
      void setCutoff(double distance, const OpenMM::NeighborList& neighbors);

      /**---------------------------------------------------------------------------------------
      
         Set the force to use periodic boundary conditions.  This requires that a cutoff has
         already been set, and the smallest side of the periodic box is at least twice the cutoff
         distance.
      
         @param vectors    the vectors defining the periodic box
      
         --------------------------------------------------------------------------------------- */
      
      void setPeriodic(OpenMM::Vec3* vectors);
       
      /**---------------------------------------------------------------------------------------

         Set the force to use Particle-Mesh Ewald (PME) summation.

         @param alpha    the Ewald separation parameter
         @param gridSize the dimensions of the mesh

         --------------------------------------------------------------------------------------- */
      
      void setPME(double alpha, int meshSize[3]);
      
      /**---------------------------------------------------------------------------------------

         Set whether exceptions use periodic boundary conditions.

         --------------------------------------------------------------------------------------- */

      void setPeriodicExceptions(bool periodic);

      /**---------------------------------------------------------------------------------------
      
         Calculate Ewald ixn
      
         @param numberOfAtoms    number of atoms
         @param atomCoordinates  atom coordinates
         @param atomCharges      atom charges
         @param exclusions       atom exclusion indices
                                 exclusions[atomIndex] contains the list of exclusions for that atom
         @param forces           force array (forces added)
         @param totalEnergy      total energy
         @param sliceEnergies    slice energies
         @param includeDirect      true if direct space interactions should be included
         @param includeReciprocal  true if reciprocal space interactions should be included
            
         --------------------------------------------------------------------------------------- */
          
      void calculateEwaldIxn(int numberOfAtoms, vector<OpenMM::Vec3>& atomCoordinates, vector<int> subsets, vector<double> sliceLambda,
                             vector<double>& atomCharges, vector<set<int> >& exclusions,
                             vector<OpenMM::Vec3>& forces, double* totalEnergy, vector<double> sliceEnergies, bool includeDirect, bool includeReciprocal) const;
};

} // namespace PmeSlicing

#endif // __ReferenceCoulombIxn_H__
