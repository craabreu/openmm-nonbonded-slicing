
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

#include <string.h>
#include <sstream>
#include <complex>
#include <algorithm>
#include <iostream>

#include "ReferenceLJCoulombIxn.h"
#include "ReferencePME.h"
#include "openmm/reference/SimTKOpenMMUtilities.h"
#include "openmm/reference/ReferenceForce.h"
#include "openmm/OpenMMException.h"

// In case we're using some primitive version of Visual Studio this will
// make sure that erf() and erfc() are defined.
#include "openmm/internal/MSVC_erfc.h"

using std::set;
using std::vector;
using namespace NonbondedSlicing;
using namespace OpenMM;

/**---------------------------------------------------------------------------------------

   ReferenceLJCoulombIxn constructor

   --------------------------------------------------------------------------------------- */

ReferenceLJCoulombIxn::ReferenceLJCoulombIxn() : cutoff(false), periodic(false),
    periodicExceptions(false), pme(false) {
}

/**---------------------------------------------------------------------------------------

   ReferenceLJCoulombIxn destructor

   --------------------------------------------------------------------------------------- */

ReferenceLJCoulombIxn::~ReferenceLJCoulombIxn() {
}

/**---------------------------------------------------------------------------------------

     Set the force to use a cutoff.

     @param distance            the cutoff distance
     @param neighbors           the neighbor list to use

     --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::setUseCutoff(double distance, const OpenMM::NeighborList& neighbors) {

    cutoff = true;
    cutoffDistance = distance;
    neighborList = &neighbors;
}

/**---------------------------------------------------------------------------------------

     Set the force to use periodic boundary conditions.  This requires that a cutoff has
     also been set, and the smallest side of the periodic box is at least twice the cutoff
     distance.

     @param vectors    the vectors defining the periodic box

     --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::setPeriodic(OpenMM::Vec3* vectors) {

    assert(cutoff);
    assert(vectors[0][0] >= 2.0*cutoffDistance);
    assert(vectors[1][1] >= 2.0*cutoffDistance);
    assert(vectors[2][2] >= 2.0*cutoffDistance);
    periodic = true;
    periodicBoxVectors[0] = vectors[0];
    periodicBoxVectors[1] = vectors[1];
    periodicBoxVectors[2] = vectors[2];
}

/**---------------------------------------------------------------------------------------

     Set the force to use Particle-Mesh Ewald (PME) summation.

     @param alpha  the Ewald separation parameter
     @param gridSize the dimensions of the mesh

     --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::setUsePME(double alpha, int meshSize[3]) {
    alphaEwald = alpha;
    meshDim[0] = meshSize[0];
    meshDim[1] = meshSize[1];
    meshDim[2] = meshSize[2];
    pme = true;
}

void ReferenceLJCoulombIxn::setPeriodicExceptions(bool periodic) {
    periodicExceptions = periodic;
}

/**---------------------------------------------------------------------------------------

   Calculate Ewald ixn

   @param numberOfAtoms    number of atoms
   @param atomCoordinates  atom coordinates
   @param atomParameters   atom parameters                             atomParameters[atomIndex][paramterIndex]
   @param exclusions       atom exclusion indices
                           exclusions[atomIndex] contains the list of exclusions for that atom
   @param forces           force array (forces added)
   @param totalEnergy      total energy
   @param includeDirect      true if direct space interactions should be included
   @param includeReciprocal  true if reciprocal space interactions should be included

   --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::calculateEwaldIxn(int numberOfAtoms, vector<Vec3>& atomCoordinates,
                                              vector<vector<double> >& atomParameters, vector<set<int> >& exclusions,
                                              vector<Vec3>& forces, double* totalEnergy, bool includeDirect, bool includeReciprocal) const {
    typedef std::complex<double> d_complex;

    static const double epsilon     =  1.0;

    int kmax                        = 0;
    double factorEwald              = -1 / (4*alphaEwald*alphaEwald);
    double SQRT_PI                  = sqrt(PI_M);
    double TWO_PI                   = 2.0 * PI_M;
    double recipCoeff               = ONE_4PI_EPS0*4*PI_M/(periodicBoxVectors[0][0] * periodicBoxVectors[1][1] * periodicBoxVectors[2][2]) /epsilon;

    double totalSelfEwaldEnergy     = 0.0;
    double realSpaceEwaldEnergy     = 0.0;
    double recipEnergy              = 0.0;
    double totalRecipEnergy         = 0.0;
    double vdwEnergy                = 0.0;

    // **************************************************************************************
    // SELF ENERGY
    // **************************************************************************************

    if (includeReciprocal) {
        for (int atomID = 0; atomID < numberOfAtoms; atomID++) {
            double selfEwaldEnergy = ONE_4PI_EPS0*atomParameters[atomID][QIndex]*atomParameters[atomID][QIndex] * alphaEwald/SQRT_PI;
            totalSelfEwaldEnergy -= selfEwaldEnergy;
        }
    }

    if (totalEnergy) {
        *totalEnergy += totalSelfEwaldEnergy;
    }

    // **************************************************************************************
    // RECIPROCAL SPACE EWALD ENERGY AND FORCES
    // **************************************************************************************
    // PME

    if (pme && includeReciprocal) {
        pme_t          pmedata; /* abstract handle for PME data */

        pme_init(&pmedata,alphaEwald,numberOfAtoms,meshDim,5,1);

        vector<double> charges(numberOfAtoms);
        for (int i = 0; i < numberOfAtoms; i++)
            charges[i] = atomParameters[i][QIndex];
        pme_exec(pmedata,atomCoordinates,forces,charges,periodicBoxVectors,&recipEnergy);

        if (totalEnergy)
            *totalEnergy += recipEnergy;

        pme_destroy(pmedata);

    }

    // **************************************************************************************
    // SHORT-RANGE ENERGY AND FORCES
    // **************************************************************************************

    if (!includeDirect)
        return;
    double totalVdwEnergy            = 0.0f;
    double totalRealSpaceEwaldEnergy = 0.0f;


    for (auto& pair : *neighborList) {
        int ii = pair.first;
        int jj = pair.second;

        double deltaR[2][ReferenceForce::LastDeltaRIndex];
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
        double r         = deltaR[0][ReferenceForce::RIndex];
        double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
        double alphaR = alphaEwald * r;

        double dEdR = ONE_4PI_EPS0 * atomParameters[ii][QIndex] * atomParameters[jj][QIndex] * inverseR * inverseR * inverseR;
        dEdR = dEdR * (erfc(alphaR) + 2 * alphaR * exp (- alphaR * alphaR) / SQRT_PI);

        double sig = atomParameters[ii][SigIndex] +  atomParameters[jj][SigIndex];
        double sig2 = inverseR*sig;
        sig2 *= sig2;
        double sig6 = sig2*sig2*sig2;
        double eps = atomParameters[ii][EpsIndex]*atomParameters[jj][EpsIndex];
        dEdR += eps*(12.0*sig6 - 6.0)*sig6*inverseR*inverseR;
        vdwEnergy = eps*(sig6-1.0)*sig6;

        // accumulate forces

        for (int kk = 0; kk < 3; kk++) {
            double force  = dEdR*deltaR[0][kk];
            forces[ii][kk]   += force;
            forces[jj][kk]   -= force;
        }

        // accumulate energies

        realSpaceEwaldEnergy        = ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR*erfc(alphaR);

        totalVdwEnergy             += vdwEnergy;
        totalRealSpaceEwaldEnergy  += realSpaceEwaldEnergy;

    }

    if (totalEnergy)
        *totalEnergy += totalRealSpaceEwaldEnergy + totalVdwEnergy;

    // Now subtract off the exclusions, since they were implicitly included in the reciprocal space sum.

    double totalExclusionEnergy = 0.0f;
    const double TWO_OVER_SQRT_PI = 2/sqrt(PI_M);
    for (int i = 0; i < numberOfAtoms; i++)
        for (int exclusion : exclusions[i]) {
            if (exclusion > i) {
                int ii = i;
                int jj = exclusion;

                double deltaR[2][ReferenceForce::LastDeltaRIndex];
                if (periodicExceptions)
                    ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
                else
                    ReferenceForce::getDeltaR(atomCoordinates[jj], atomCoordinates[ii], deltaR[0]);
                double r         = deltaR[0][ReferenceForce::RIndex];
                double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
                double alphaR    = alphaEwald * r;
                if (erf(alphaR) > 1e-6) {
                    double dEdR = ONE_4PI_EPS0 * atomParameters[ii][QIndex] * atomParameters[jj][QIndex] * inverseR * inverseR * inverseR;
                    dEdR = dEdR * (erf(alphaR) - 2 * alphaR * exp (- alphaR * alphaR) / SQRT_PI);

                    // accumulate forces

                    for (int kk = 0; kk < 3; kk++) {
                        double force = dEdR*deltaR[0][kk];
                        forces[ii][kk] -= force;
                        forces[jj][kk] += force;
                    }

                    // accumulate energies

                    realSpaceEwaldEnergy = ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR*erf(alphaR);
                }
                else {
                    realSpaceEwaldEnergy = alphaEwald*TWO_OVER_SQRT_PI*ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex];
                }

                totalExclusionEnergy += realSpaceEwaldEnergy;
            }
        }

    if (totalEnergy)
        *totalEnergy -= totalExclusionEnergy;
}


/**---------------------------------------------------------------------------------------

   Calculate LJ Coulomb pair ixn

   @param numberOfAtoms    number of atoms
   @param atomCoordinates  atom coordinates
   @param atomParameters   atom parameters                             atomParameters[atomIndex][paramterIndex]
   @param exclusions       atom exclusion indices
                           exclusions[atomIndex] contains the list of exclusions for that atom
   @param forces           force array (forces added)
   @param totalEnergy      total energy
   @param includeDirect      true if direct space interactions should be included
   @param includeReciprocal  true if reciprocal space interactions should be included

   --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::calculatePairIxn(int numberOfAtoms, vector<Vec3>& atomCoordinates,
                                             vector<vector<double> >& atomParameters, vector<set<int> >& exclusions,
                                             vector<Vec3>& forces, double* totalEnergy, bool includeDirect, bool includeReciprocal) const {

    if (pme) {
        calculateEwaldIxn(numberOfAtoms, atomCoordinates, atomParameters, exclusions, forces,
                          totalEnergy, includeDirect, includeReciprocal);
        return;
    }
    if (!includeDirect)
        return;
    if (cutoff) {
        for (auto& pair : *neighborList)
            calculateOneIxn(pair.first, pair.second, atomCoordinates, atomParameters, forces, totalEnergy);
    }
    else {
        for (int ii = 0; ii < numberOfAtoms; ii++) {
            // loop over atom pairs

            for (int jj = ii+1; jj < numberOfAtoms; jj++)
                if (exclusions[jj].find(ii) == exclusions[jj].end())
                    calculateOneIxn(ii, jj, atomCoordinates, atomParameters, forces, totalEnergy);
        }
    }
}

/**---------------------------------------------------------------------------------------

     Calculate LJ Coulomb pair ixn between two atoms

     @param ii               the index of the first atom
     @param jj               the index of the second atom
     @param atomCoordinates  atom coordinates
     @param atomParameters   atom parameters (charges, c6, c12, ...)     atomParameters[atomIndex][paramterIndex]
     @param forces           force array (forces added)
     @param totalEnergy      total energy

     --------------------------------------------------------------------------------------- */

void ReferenceLJCoulombIxn::calculateOneIxn(int ii, int jj, vector<Vec3>& atomCoordinates,
                                            vector<vector<double> >& atomParameters, vector<Vec3>& forces,
                                            double* totalEnergy) const {
    double deltaR[2][ReferenceForce::LastDeltaRIndex];

    // get deltaR, R2, and R between 2 atoms

    if (periodic)
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
    else
        ReferenceForce::getDeltaR(atomCoordinates[jj], atomCoordinates[ii], deltaR[0]);

    double r2        = deltaR[0][ReferenceForce::R2Index];
    double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
    double sig = atomParameters[ii][SigIndex] +  atomParameters[jj][SigIndex];
    double sig2 = inverseR*sig;
    sig2 *= sig2;
    double sig6 = sig2*sig2*sig2;

    double eps = atomParameters[ii][EpsIndex]*atomParameters[jj][EpsIndex];
    double dEdR = eps*(12.0*sig6 - 6.0)*sig6;
    dEdR += ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR;
    dEdR *= inverseR*inverseR;
    double energy = eps*(sig6-1.0)*sig6;
    energy += ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR;

    // accumulate forces

    for (int kk = 0; kk < 3; kk++) {
        double force  = dEdR*deltaR[0][kk];
        forces[ii][kk]   += force;
        forces[jj][kk]   -= force;
    }

    // accumulate energies

    if (totalEnergy)
        *totalEnergy += energy;
}

