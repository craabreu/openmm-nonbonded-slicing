/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include <string.h>
#include <sstream>
#include <complex>
#include <algorithm>
#include <iostream>

#include "internal/ReferenceSlicedLJCoulombIxn.h"
#include "internal/ReferenceSlicedPME.h"
#include "openmm/reference/SimTKOpenMMUtilities.h"
#include "openmm/reference/ReferenceForce.h"
#include "openmm/OpenMMException.h"

// In case we're using some primitive version of Visual Studio this will
// make sure that erf() and erfc() are defined.
#include "openmm/internal/MSVC_erfc.h"

using namespace std;
using namespace NonbondedSlicing;
using namespace OpenMM;

/**---------------------------------------------------------------------------------------

   ReferenceSlicedLJCoulombIxn constructor

   --------------------------------------------------------------------------------------- */

ReferenceSlicedLJCoulombIxn::ReferenceSlicedLJCoulombIxn() : cutoff(false), useSwitch(false),
            periodic(false), periodicExceptions(false), ewald(false), pme(false), ljpme(false) {
}

/**---------------------------------------------------------------------------------------

   ReferenceSlicedLJCoulombIxn destructor

   --------------------------------------------------------------------------------------- */

ReferenceSlicedLJCoulombIxn::~ReferenceSlicedLJCoulombIxn() {
}

/**---------------------------------------------------------------------------------------

     Set the force to use a cutoff.

     @param distance            the cutoff distance
     @param neighbors           the neighbor list to use
     @param solventDielectric   the dielectric constant of the bulk solvent

     --------------------------------------------------------------------------------------- */

void ReferenceSlicedLJCoulombIxn::setUseCutoff(double distance, const OpenMM::NeighborList& neighbors, double solventDielectric) {
    cutoff = true;
    cutoffDistance = distance;
    neighborList = &neighbors;
    krf = pow(cutoffDistance, -3.0)*(solventDielectric-1.0)/(2.0*solventDielectric+1.0);
    crf = (1.0/cutoffDistance)*(3.0*solventDielectric)/(2.0*solventDielectric+1.0);
}

/**---------------------------------------------------------------------------------------

   Set the force to use a switching function on the Lennard-Jones interaction.

   @param distance            the switching distance

   --------------------------------------------------------------------------------------- */

void ReferenceSlicedLJCoulombIxn::setUseSwitchingFunction(double distance) {
    useSwitch = true;
    switchingDistance = distance;
}

/**---------------------------------------------------------------------------------------

     Set the force to use periodic boundary conditions.  This requires that a cutoff has
     also been set, and the smallest side of the periodic box is at least twice the cutoff
     distance.

     @param vectors    the vectors defining the periodic box

     --------------------------------------------------------------------------------------- */

void ReferenceSlicedLJCoulombIxn::setPeriodic(OpenMM::Vec3* vectors) {
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

     Set the force to use Ewald summation.

     @param alpha  the Ewald separation parameter
     @param kmaxx  the largest wave vector in the x direction
     @param kmaxy  the largest wave vector in the y direction
     @param kmaxz  the largest wave vector in the z direction

     --------------------------------------------------------------------------------------- */

void ReferenceSlicedLJCoulombIxn::setUseEwald(double alpha, int kmaxx, int kmaxy, int kmaxz) {
    alphaEwald = alpha;
    numRx = kmaxx;
    numRy = kmaxy;
    numRz = kmaxz;
    ewald = true;
}

/**---------------------------------------------------------------------------------------

     Set the force to use Particle-Mesh Ewald (PME) summation.

     @param alpha  the Ewald separation parameter
     @param gridSize the dimensions of the mesh

     --------------------------------------------------------------------------------------- */

void ReferenceSlicedLJCoulombIxn::setUsePME(double alpha, int meshSize[3]) {
    alphaEwald = alpha;
    meshDim[0] = meshSize[0];
    meshDim[1] = meshSize[1];
    meshDim[2] = meshSize[2];
    pme = true;
}

/**---------------------------------------------------------------------------------------

     Set the force to use Particle-Mesh Ewald (PME) summation for dispersion terms.

     @param alpha  the dispersion Ewald separation parameter
     @param gridSize the dimensions of the dispersion mesh

     --------------------------------------------------------------------------------------- */

void ReferenceSlicedLJCoulombIxn::setUseLJPME(double alpha, int meshSize[3]) {
    alphaDispersionEwald = alpha;
    dispersionMeshDim[0] = meshSize[0];
    dispersionMeshDim[1] = meshSize[1];
    dispersionMeshDim[2] = meshSize[2];
    ljpme = true;
}

void ReferenceSlicedLJCoulombIxn::setPeriodicExceptions(bool periodic) {
    periodicExceptions = periodic;
}

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

void ReferenceSlicedLJCoulombIxn::calculateEwaldIxn(int numberOfAtoms, vector<Vec3>& atomCoordinates, int numberOfSubsets, const vector<int>& atomSubsets,
                                            const vector<vector<double>>& atomParameters, const vector<vector<double>>& sliceLambdas, const vector<set<int>>& exclusions,
                                            vector<Vec3>& forces, vector<vector<double>>& sliceEnergies, bool includeDirect, bool includeReciprocal) const {
    typedef complex<double> d_complex;

    int kmax = ewald ? max(numRx, max(numRy, numRz)) : 0;
    double factorEwald = -1/(4*alphaEwald*alphaEwald);
    double SQRT_PI = sqrt(PI_M);
    double TWO_PI = 2.0*PI_M;
    double recipCoeff = ONE_4PI_EPS0*4*PI_M/(periodicBoxVectors[0][0]*periodicBoxVectors[1][1]*periodicBoxVectors[2][2]);

    // A couple of sanity checks
    if (ljpme && useSwitch)
        throw OpenMMException("Switching cannot be used with LJPME");
    if (ljpme && !pme)
        throw OpenMMException("LJPME has been set without PME being set");

    // **************************************************************************************
    // SELF ENERGY
    // **************************************************************************************

    if (includeReciprocal) {
        for (int atomID = 0; atomID < numberOfAtoms; atomID++) {
            int subset = atomSubsets[atomID];
            int slice = subset*(subset + 3)/2;
            sliceEnergies[slice][Coul] -= ONE_4PI_EPS0*atomParameters[atomID][QIndex]*atomParameters[atomID][QIndex]*alphaEwald/SQRT_PI;
            if (ljpme)
                sliceEnergies[slice][vdW] += pow(alphaDispersionEwald, 6.0)*64.0*pow(atomParameters[atomID][SigIndex], 6.0)*pow(atomParameters[atomID][EpsIndex], 2.0)/12.0;
        }
    }

    // **************************************************************************************
    // RECIPROCAL SPACE EWALD ENERGY AND FORCES
    // **************************************************************************************
    // PME

    if (pme && includeReciprocal) {
        pme_t pmedata; /* abstract handle for PME data */

        pme_init(&pmedata, alphaEwald, numberOfAtoms, numberOfSubsets, meshDim, 5, 1);

        vector<double> charges(numberOfAtoms);
        for (int i = 0; i < numberOfAtoms; i++)
            charges[i] = atomParameters[i][QIndex];
        pme_exec(pmedata, atomCoordinates, atomSubsets, sliceLambdas, forces, charges, periodicBoxVectors, sliceEnergies);

        pme_destroy(pmedata);

        if (ljpme) {
            // Dispersion reciprocal space terms
            pme_init(&pmedata, alphaDispersionEwald, numberOfAtoms, numberOfSubsets, dispersionMeshDim, 5, 1);

            vector<Vec3> dpmeforces(numberOfAtoms);
            for (int i = 0; i < numberOfAtoms; i++)
                charges[i] = 8.0*pow(atomParameters[i][SigIndex], 3.0)*atomParameters[i][EpsIndex];
            pme_exec_dpme(pmedata, atomCoordinates, atomSubsets, sliceLambdas, dpmeforces, charges, periodicBoxVectors, sliceEnergies);
            for (int i = 0; i < numberOfAtoms; i++)
                forces[i] += dpmeforces[i];
            pme_destroy(pmedata);
        }
    }
    // Ewald method

    else if (ewald && includeReciprocal) {

        // setup reciprocal box

        double recipBoxSize[3] = { TWO_PI/periodicBoxVectors[0][0], TWO_PI/periodicBoxVectors[1][1], TWO_PI/periodicBoxVectors[2][2]};

        // setup K-vectors

        #define EIR(subset, x, y, z) eir[(((subset)*kmax+(x))*numberOfAtoms+(y))*3+z]
        vector<d_complex> eir(numberOfSubsets*kmax*numberOfAtoms*3);
        vector<d_complex> tab_xy(numberOfAtoms);
        vector<d_complex> tab_qxyz(numberOfAtoms);

        if (kmax < 1)
            throw OpenMMException("kmax for Ewald summation < 1");

        for (int i = 0; (i < numberOfAtoms); i++) {
            int subset = atomSubsets[i];
            for (int m = 0; (m < 3); m++)
                EIR(subset, 0, i, m) = d_complex(1,0);

            for (int m=0; (m<3); m++)
                EIR(subset, 1, i, m) = d_complex(cos(atomCoordinates[i][m]*recipBoxSize[m]),
                                         sin(atomCoordinates[i][m]*recipBoxSize[m]));

            for (int j=2; (j<kmax); j++)
                for (int m=0; (m<3); m++)
                    EIR(subset, j, i, m) = EIR(subset, j-1, i, m)*EIR(subset, 1, i, m);
        }

        // calculate reciprocal space energy and forces

        int lowry = 0;
        int lowrz = 1;

        for (int rx = 0; rx < numRx; rx++) {

            double kx = rx*recipBoxSize[0];

            for (int ry = lowry; ry < numRy; ry++) {

                double ky = ry*recipBoxSize[1];

                if (ry >= 0)
                    for (int n = 0; n < numberOfAtoms; n++) {
                        int subset = atomSubsets[n];
                        tab_xy[n] = EIR(subset, rx, n, 0)*EIR(subset, ry, n, 1);
                    }
                else
                    for (int n = 0; n < numberOfAtoms; n++) {
                        int subset = atomSubsets[n];
                        tab_xy[n]= EIR(subset, rx, n, 0)*conj(EIR(subset, -ry, n, 1));
                    }

                for (int rz = lowrz; rz < numRz; rz++) {

                    if (rz >= 0)
                        for (int n = 0; n < numberOfAtoms; n++) {
                            int subset = atomSubsets[n];
                            tab_qxyz[n] = atomParameters[n][QIndex]*(tab_xy[n]*EIR(subset, rz, n, 2));
                        }
                    else
                        for (int n = 0; n < numberOfAtoms; n++) {
                            int subset = atomSubsets[n];
                            tab_qxyz[n] = atomParameters[n][QIndex]*(tab_xy[n]*conj(EIR(subset, -rz, n, 2)));
                        }

                    vector<double> cs(numberOfSubsets, 0);
                    vector<double> ss(numberOfSubsets, 0);

                    for (int n = 0; n < numberOfAtoms; n++) {
                        int subset = atomSubsets[n];
                        cs[subset] += tab_qxyz[n].real();
                        ss[subset] += tab_qxyz[n].imag();
                    }

                    double kz = rz*recipBoxSize[2];
                    double k2 = kx*kx + ky*ky + kz*kz;
                    double ak = exp(k2*factorEwald)/k2;

                    for (int n = 0; n < numberOfAtoms; n++) {
                        int i = atomSubsets[n];
                        for (int j = 0; j < numberOfSubsets; j++) {
                            int slice = i > j ? i*(i+1)/2+j : j*(j+1)/2+i;
                            double force = 2*recipCoeff*sliceLambdas[slice][Coul]*ak*(cs[j]*tab_qxyz[n].imag() - ss[j]*tab_qxyz[n].real());
                            forces[n][0] += force*kx;
                            forces[n][1] += force*ky;
                            forces[n][2] += force*kz;
                        }
                    }

                    for (int j = 0; j < numberOfSubsets; j++) {
                        for (int i = 0; i < j; i++)
                            sliceEnergies[j*(j+1)/2+i][Coul] += 2*recipCoeff*ak*(cs[i]*cs[j] + ss[i]*ss[j]);
                        sliceEnergies[j*(j+3)/2][Coul] += recipCoeff*ak*(cs[j]*cs[j] + ss[j]*ss[j]);
                    }

                    lowrz = 1 - numRz;
                }
                lowry = 1 - numRy;
            }
        }
    }

    // **************************************************************************************
    // SHORT-RANGE ENERGY AND FORCES
    // **************************************************************************************

    if (!includeDirect)
        return;

    for (auto& pair : *neighborList) {
        int ii = pair.first;
        int jj = pair.second;

        int si = atomSubsets[ii];
        int sj = atomSubsets[jj];
        int slice = si > sj ? si*(si+1)/2+sj : sj*(sj+1)/2+si;

        double deltaR[2][ReferenceForce::LastDeltaRIndex];
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
        double r         = deltaR[0][ReferenceForce::RIndex];
        double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
        double switchValue = 1, switchDeriv = 0;
        if (useSwitch && r > switchingDistance) {
            double t = (r-switchingDistance)/(cutoffDistance-switchingDistance);
            switchValue = 1+t*t*t*(-10+t*(15-t*6));
            switchDeriv = t*t*(-30+t*(60-t*30))/(cutoffDistance-switchingDistance);
        }
        double alphaR = alphaEwald*r;

        double dEdRCoul = ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR*inverseR*inverseR;
        dEdRCoul *= erfc(alphaR) + 2*alphaR*exp(-alphaR*alphaR)/SQRT_PI;

        double sig = atomParameters[ii][SigIndex] +  atomParameters[jj][SigIndex];
        double sig2 = inverseR*sig;
        sig2 *= sig2;
        double sig6 = sig2*sig2*sig2;
        double eps = atomParameters[ii][EpsIndex]*atomParameters[jj][EpsIndex];
        double dEdRvdW = switchValue*eps*(12.0*sig6 - 6.0)*sig6*inverseR*inverseR;
        double vdwEnergy = eps*(sig6-1.0)*sig6;

        if (ljpme) {
            double dalphaR   = alphaDispersionEwald*r;
            double dar2 = dalphaR*dalphaR;
            double dar4 = dar2*dar2;
            double dar6 = dar4*dar2;
            double inverseR2 = inverseR*inverseR;
            double c6i = 8.0*pow(atomParameters[ii][SigIndex], 3.0)*atomParameters[ii][EpsIndex];
            double c6j = 8.0*pow(atomParameters[jj][SigIndex], 3.0)*atomParameters[jj][EpsIndex];
            // For the energies and forces, we first add the regular Lorentz−Berthelot terms.  The C12 term is treated as usual
            // but we then subtract out (remembering that the C6 term is negative) the multiplicative C6 term that has been
            // computed in real space.  Finally, we add a potential shift term to account for the difference between the LB
            // and multiplicative functional forms at the cutoff.
            double emult = c6i*c6j*inverseR2*inverseR2*inverseR2*(1.0 - EXP(-dar2)*(1.0 + dar2 + 0.5*dar4));
            dEdRvdW += 6.0*c6i*c6j*inverseR2*inverseR2*inverseR2*inverseR2*(1.0 - EXP(-dar2)*(1.0 + dar2 + 0.5*dar4 + dar6/6.0));

            double inverseCut2 = 1.0/(cutoffDistance*cutoffDistance);
            double inverseCut6 = inverseCut2*inverseCut2*inverseCut2;
            sig2 = atomParameters[ii][SigIndex] +  atomParameters[jj][SigIndex];
            sig2 *= sig2;
            sig6 = sig2*sig2*sig2;
            // The additive part of the potential shift
            double potentialshift = eps*(1.0-sig6*inverseCut6)*sig6*inverseCut6;
            dalphaR   = alphaDispersionEwald*cutoffDistance;
            dar2 = dalphaR*dalphaR;
            dar4 = dar2*dar2;
            // The multiplicative part of the potential shift
            potentialshift -= c6i*c6j*inverseCut6*(1.0 - EXP(-dar2)*(1.0 + dar2 + 0.5*dar4));
            vdwEnergy += emult + potentialshift;
        }

        if (useSwitch) {
            dEdRvdW -= vdwEnergy*switchDeriv*inverseR;
            vdwEnergy *= switchValue;
        }

        // accumulate forces

        double factor = sliceLambdas[slice][vdW]*dEdRvdW+sliceLambdas[slice][Coul]*dEdRCoul;
        for (int kk = 0; kk < 3; kk++) {
            double force = factor*deltaR[0][kk];
            forces[ii][kk] += force;
            forces[jj][kk] -= force;
        }

        // accumulate energies
        sliceEnergies[slice][vdW] += vdwEnergy;
        sliceEnergies[slice][Coul] += ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR*erfc(alphaR);
    }

    // Now subtract off the exclusions, since they were implicitly included in the reciprocal space sum.

    const double TWO_OVER_SQRT_PI = 2/sqrt(PI_M);
    for (int i = 0; i < numberOfAtoms; i++)
        for (int exclusion : exclusions[i]) {
            if (exclusion > i) {
                int ii = i;
                int jj = exclusion;

                int si = atomSubsets[ii];
                int sj = atomSubsets[jj];
                int slice = si > sj ? si*(si+1)/2+sj : sj*(sj+1)/2+si;

                double deltaR[2][ReferenceForce::LastDeltaRIndex];
                if (periodicExceptions)
                    ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
                else
                    ReferenceForce::getDeltaR(atomCoordinates[jj], atomCoordinates[ii], deltaR[0]);
                double r        = deltaR[0][ReferenceForce::RIndex];
                double inverseR = 1.0/(deltaR[0][ReferenceForce::RIndex]);
                double alphaR   = alphaEwald*r;
                if (erf(alphaR) > 1e-6) {
                    double dEdR = ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR*inverseR*inverseR;
                    dEdR = dEdR*(erf(alphaR) - 2*alphaR*exp(-alphaR*alphaR)/SQRT_PI);

                    // accumulate forces
                    double factor = sliceLambdas[slice][Coul]*dEdR;
                    for (int kk = 0; kk < 3; kk++) {
                        double force = factor*deltaR[0][kk];
                        forces[ii][kk] -= force;
                        forces[jj][kk] += force;
                    }

                    // accumulate energies

                    sliceEnergies[slice][Coul] -= ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR*erf(alphaR);
                }
                else
                    sliceEnergies[slice][Coul] -= alphaEwald*TWO_OVER_SQRT_PI*ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex];

                if (ljpme) {
                    // Dispersion terms.  Here we just back out the reciprocal space terms, and don't add any extra real space terms.
                    double dalphaR   = alphaDispersionEwald*r;
                    double inverseR2 = inverseR*inverseR;
                    double dar2 = dalphaR*dalphaR;
                    double dar4 = dar2*dar2;
                    double dar6 = dar4*dar2;
                    double c6i = 8.0*pow(atomParameters[ii][SigIndex], 3.0)*atomParameters[ii][EpsIndex];
                    double c6j = 8.0*pow(atomParameters[jj][SigIndex], 3.0)*atomParameters[jj][EpsIndex];
                    sliceEnergies[slice][vdW] += c6i*c6j*inverseR2*inverseR2*inverseR2*(1.0 - EXP(-dar2)*(1.0 + dar2 + 0.5*dar4));
                    double dEdR = -6.0*c6i*c6j*inverseR2*inverseR2*inverseR2*inverseR2*(1.0 - EXP(-dar2)*(1.0 + dar2 + 0.5*dar4 + dar6/6.0));
                    double factor = sliceLambdas[slice][vdW]*dEdR;
                    for (int kk = 0; kk < 3; kk++) {
                        double force = factor*deltaR[0][kk];
                        forces[ii][kk] -= force;
                        forces[jj][kk] += force;
                    }
                }
            }
        }
}


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

void ReferenceSlicedLJCoulombIxn::calculatePairIxn(int numberOfAtoms, vector<Vec3>& atomCoordinates, int numberOfSubsets, const vector<int>& atomSubsets,
                const vector<vector<double>>& atomParameters, const vector<vector<double>>& sliceLambdas, const vector<set<int>>& exclusions,
                vector<Vec3>& forces, vector<vector<double>>& sliceEnergies, bool includeDirect, bool includeReciprocal) const {

    if (ewald || pme || ljpme) {
        calculateEwaldIxn(numberOfAtoms, atomCoordinates, numberOfSubsets, atomSubsets, atomParameters, sliceLambdas, exclusions, forces,
                          sliceEnergies, includeDirect, includeReciprocal);
        return;
    }
    if (!includeDirect)
        return;
    if (cutoff) {
        for (auto& pair : *neighborList)
            calculateOneIxn(pair.first, pair.second, atomCoordinates, atomSubsets, atomParameters, sliceLambdas, forces, sliceEnergies);
    }
    else {
        for (int ii = 0; ii < numberOfAtoms; ii++) {
            // loop over atom pairs

            for (int jj = ii+1; jj < numberOfAtoms; jj++)
                if (exclusions[jj].find(ii) == exclusions[jj].end())
                    calculateOneIxn(ii, jj, atomCoordinates, atomSubsets, atomParameters, sliceLambdas, forces, sliceEnergies);
        }
    }
}

/**---------------------------------------------------------------------------------------

     Calculate LJ Coulomb pair ixn between two atoms

     @param ii               the index of the first atom
     @param jj               the index of the second atom
     @param atomCoordinates  atom coordinates
     @param atomSubsets      atom subsets
     @param atomParameters   atom parameters (charges, c6, c12, ...)     atomParameters[atomIndex][paramterIndex]
     @param sliceLambda      Coulomb and LJ scaling parameters for each slice
     @param forces           force array (forces added)
     @param sliceEnergies    the energy of each slice

     --------------------------------------------------------------------------------------- */

void ReferenceSlicedLJCoulombIxn::calculateOneIxn(int ii, int jj, vector<Vec3>& atomCoordinates, const vector<int>& atomSubsets,
                                            const vector<vector<double>>& atomParameters, const vector<vector<double>>& sliceLambdas, vector<Vec3>& forces,
                                            vector<vector<double>>& sliceEnergies) const {
    double deltaR[2][ReferenceForce::LastDeltaRIndex];

    int si = atomSubsets[ii];
    int sj = atomSubsets[jj];
    int slice = si > sj ? si*(si+1)/2+sj : sj*(sj+1)/2+si;

    // get deltaR, R2, and R between 2 atoms

    if (periodic)
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
    else
        ReferenceForce::getDeltaR(atomCoordinates[jj], atomCoordinates[ii], deltaR[0]);

    double r2        = deltaR[0][ReferenceForce::R2Index];
    double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
    double switchValue = 1, switchDeriv = 0;
    if (useSwitch) {
        double r = deltaR[0][ReferenceForce::RIndex];
        if (r > switchingDistance) {
            double t = (r-switchingDistance)/(cutoffDistance-switchingDistance);
            switchValue = 1+t*t*t*(-10+t*(15-t*6));
            switchDeriv = t*t*(-30+t*(60-t*30))/(cutoffDistance-switchingDistance);
        }
    }
    double sig = atomParameters[ii][SigIndex] +  atomParameters[jj][SigIndex];
    double sig2 = inverseR*sig;
    sig2 *= sig2;
    double sig6 = sig2*sig2*sig2;

    double eps = atomParameters[ii][EpsIndex]*atomParameters[jj][EpsIndex];
    double dEdRvdW = switchValue*eps*(12.0*sig6 - 6.0)*sig6*inverseR*inverseR;
    double dEdRCoul = inverseR*inverseR;
    if (cutoff)
        dEdRCoul *= ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*(inverseR-2.0f*krf*r2);
    else
        dEdRCoul *= ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR;
    double energy = eps*(sig6-1.0)*sig6;
    if (useSwitch) {
        dEdRvdW -= energy*switchDeriv*inverseR;
        energy *= switchValue;
    }
    sliceEnergies[slice][vdW] += energy;
    if (cutoff)
        sliceEnergies[slice][Coul] += ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*(inverseR+krf*r2-crf);
    else
        sliceEnergies[slice][Coul] += ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR;


    // accumulate forces
    double factor = sliceLambdas[slice][vdW]*dEdRvdW+sliceLambdas[slice][Coul]*dEdRCoul;
    for (int kk = 0; kk < 3; kk++) {
        double force  = factor*deltaR[0][kk];
        forces[ii][kk] += force;
        forces[jj][kk] -= force;
    }
}
