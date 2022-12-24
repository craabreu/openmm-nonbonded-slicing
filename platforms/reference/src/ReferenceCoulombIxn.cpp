
/* Portions copyright (c) 2006-2020 Stanford University and Simbios.
*Contributors: Pande Group
 *
*Permission is hereby granted, free of charge, to any person obtaining
*a copy of this software and associated documentation files (the
*"Software"), to deal in the Software without restriction, including
*without limitation the rights to use, copy, modify, merge, publish,
*distribute, sublicense, and/or sell copies of the Software, and to
*permit persons to whom the Software is furnished to do so, subject
*to the following conditions:
 *
*The above copyright notice and this permission notice shall be included
*in all copies or substantial portions of the Software.
 *
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
*OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
*IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
*LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
*OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
*WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <string.h>
#include <sstream>
#include <complex>
#include <algorithm>
#include <iostream>

#include "internal/ReferenceCoulombIxn.h"
#include "internal/ReferenceSlicedPME.h"
#include "openmm/reference/SimTKOpenMMUtilities.h"
#include "openmm/reference/ReferenceForce.h"
#include "openmm/OpenMMException.h"

// In case we're using some primitive version of Visual Studio this will
// make sure that erf() and erfc() are defined.
#include "openmm/internal/MSVC_erfc.h"

using std::set;
using std::vector;
using namespace OpenMM;
using namespace NonbondedSlicing;

/**---------------------------------------------------------------------------------------

   ReferenceCoulombIxn constructor

   --------------------------------------------------------------------------------------- */

ReferenceCoulombIxn::ReferenceCoulombIxn() : periodicExceptions(false) {
}

/**---------------------------------------------------------------------------------------

   ReferenceCoulombIxn destructor

   --------------------------------------------------------------------------------------- */

ReferenceCoulombIxn::~ReferenceCoulombIxn() {
}

/**---------------------------------------------------------------------------------------

     Set the force to use a cutoff.

     @param distance            the cutoff distance
     @param neighbors           the neighbor list to use
     @param solventDielectric   the dielectric constant of the bulk solvent

     --------------------------------------------------------------------------------------- */

void ReferenceCoulombIxn::setCutoff(double distance, const OpenMM::NeighborList& neighbors) {

    cutoffDistance = distance;
    neighborList = &neighbors;
}

/**---------------------------------------------------------------------------------------

     Set the force to use periodic boundary conditions.  This requires that a cutoff has
     also been set, and the smallest side of the periodic box is at least twice the cutoff
     distance.

     @param vectors    the vectors defining the periodic box

     --------------------------------------------------------------------------------------- */

void ReferenceCoulombIxn::setPeriodic(OpenMM::Vec3* vectors) {

    assert(vectors[0][0] >= 2.0*cutoffDistance);
    assert(vectors[1][1] >= 2.0*cutoffDistance);
    assert(vectors[2][2] >= 2.0*cutoffDistance);
    periodicBoxVectors[0] = vectors[0];
    periodicBoxVectors[1] = vectors[1];
    periodicBoxVectors[2] = vectors[2];
}

/**---------------------------------------------------------------------------------------

     Set the force to use Particle-Mesh Ewald (PME) summation.

     @param alpha  the Ewald separation parameter
     @param gridSize the dimensions of the mesh

     --------------------------------------------------------------------------------------- */

void ReferenceCoulombIxn::setPME(double alpha, int meshSize[3]) {
    alphaEwald = alpha;
    meshDim[0] = meshSize[0];
    meshDim[1] = meshSize[1];
    meshDim[2] = meshSize[2];
}

void ReferenceCoulombIxn::setPeriodicExceptions(bool periodic) {
    periodicExceptions = periodic;
}

/**---------------------------------------------------------------------------------------

   Calculate Ewald ixn

   @param numberOfAtoms    number of atoms
   @param coords  atom coordinates
   @param charges      atom charges
   @param exclusions       atom exclusion indices
                           exclusions[atomIndex] contains the list of exclusions for that atom
   @param forces           force array (forces added)
   @param sliceEnergies    slice energies
   @param includeDirect      true if direct space interactions should be included
   @param includeReciprocal  true if reciprocal space interactions should be included

   --------------------------------------------------------------------------------------- */

void ReferenceCoulombIxn::calculateEwaldIxn(int numberOfAtoms, vector<Vec3>& coords,
                                            int numSubsets, std::vector<int> subsets, std::vector<double> sliceLambda,
                                            vector<double>& charges, vector<set<int> >& exclusions,
                                            vector<Vec3>& forces, vector<double>& sliceEnergies,
                                            bool includeDirect, bool includeReciprocal) const {
    const double SQRT_PI = sqrt(PI_M);
    const double TWO_PI = 2.0*PI_M;

    // **************************************************************************************
    // SELF ENERGY
    // **************************************************************************************

    if (includeReciprocal) {

        for (int i = 0; i < numberOfAtoms; i++) {
            double selfEwaldEnergy = ONE_4PI_EPS0*charges[i]*charges[i]*alphaEwald/SQRT_PI;
            int si = subsets[i];
            int slice = si*(si+3)/2;
            sliceEnergies[slice] -= selfEwaldEnergy;
        }

        // **************************************************************************************
        // RECIPROCAL SPACE EWALD ENERGY AND FORCES
        // **************************************************************************************

        int numSlices = sliceEnergies.size();
        sliced_pme_t pmedata; /* abstract handle for PME data */
        pme_init(&pmedata, alphaEwald, numberOfAtoms, numSubsets, meshDim, 5, 1);
        pme_exec(pmedata, coords, subsets, sliceLambda, forces, charges, periodicBoxVectors, sliceEnergies);
        pme_destroy(pmedata);
    }

    // **************************************************************************************
    // SHORT-RANGE ENERGY AND FORCES
    // **************************************************************************************

    if (includeDirect) {
        for (auto& pair : *neighborList) {
            int i = pair.first;
            int j = pair.second;

            int si = subsets[i];
            int sj = subsets[j];
            int slice = si > sj ? si*(si+1)/2+sj : sj*(sj+1)/2+si;

            double deltaR[2][ReferenceForce::LastDeltaRIndex];
            ReferenceForce::getDeltaRPeriodic(coords[j], coords[i], periodicBoxVectors, deltaR[0]);
            double r = deltaR[0][ReferenceForce::RIndex];
            double inverseR = 1.0/deltaR[0][ReferenceForce::RIndex];
            double alphaR = alphaEwald*r;

            double dEdR = ONE_4PI_EPS0*charges[i]*charges[j]*inverseR*inverseR*inverseR;
            dEdR *= sliceLambda[slice]*(erfc(alphaR) + 2*alphaR*exp(-alphaR*alphaR)/SQRT_PI);

            // accumulate forces

            for (int k = 0; k < 3; k++) {
                double force = dEdR*deltaR[0][k];
                forces[i][k] += force;
                forces[j][k] -= force;
            }

            // accumulate energies

            sliceEnergies[slice] += ONE_4PI_EPS0*charges[i]*charges[j]*inverseR*erfc(alphaR);
        }

        // Now subtract off the exclusions, since they were implicitly included in the reciprocal space sum.

        double totalExclusionEnergy = 0.0;
        const double TWO_OVER_SQRT_PI = 2/sqrt(PI_M);
        for (int i = 0; i < numberOfAtoms; i++)
            for (int j : exclusions[i]) {
                if (j > i) {
                    int si = subsets[i];
                    int sj = subsets[j];
                    int slice = si > sj ? si*(si+1)/2+sj : sj*(sj+1)/2+si;

                    double deltaR[2][ReferenceForce::LastDeltaRIndex];
                    if (periodicExceptions)
                        ReferenceForce::getDeltaRPeriodic(coords[j], coords[i], periodicBoxVectors, deltaR[0]);
                    else
                        ReferenceForce::getDeltaR(coords[j], coords[i], deltaR[0]);
                    double r = deltaR[0][ReferenceForce::RIndex];
                    double inverseR = 1.0/(deltaR[0][ReferenceForce::RIndex]);
                    double alphaR = alphaEwald*r;
                    double realSpaceEwaldEnergy;
                    if (erf(alphaR) > 1e-6) {
                        double dEdR = ONE_4PI_EPS0*charges[i]*charges[j]*inverseR*inverseR*inverseR;
                        dEdR *= sliceLambda[slice]*(erf(alphaR) - 2*alphaR*exp(-alphaR*alphaR)/SQRT_PI);

                        // accumulate forces

                        for (int k = 0; k < 3; k++) {
                            double force = dEdR*deltaR[0][k];
                            forces[i][k] -= force;
                            forces[j][k] += force;
                        }

                        // accumulate energies

                        sliceEnergies[slice] -= ONE_4PI_EPS0*charges[i]*charges[j]*inverseR*erf(alphaR);
                    }
                    else
                        sliceEnergies[slice] -= alphaEwald*TWO_OVER_SQRT_PI*ONE_4PI_EPS0*charges[i]*charges[j];
                }
            }
    }
}
