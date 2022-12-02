
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

#include <string.h>
#include <sstream>

#include "internal/ReferenceSlicedLJCoulomb14.h"
#include "openmm/reference/ReferenceForce.h"
#include "openmm/reference/SimTKOpenMMUtilities.h"

using namespace std;
using namespace PmeSlicing;
using namespace OpenMM;

/**---------------------------------------------------------------------------------------

   ReferenceSlicedLJCoulomb14 constructor

   --------------------------------------------------------------------------------------- */

ReferenceSlicedLJCoulomb14::ReferenceSlicedLJCoulomb14() : periodic(false) {
}

/**---------------------------------------------------------------------------------------

   ReferenceSlicedLJCoulomb14 destructor

   --------------------------------------------------------------------------------------- */

ReferenceSlicedLJCoulomb14::~ReferenceSlicedLJCoulomb14() {
}

void ReferenceSlicedLJCoulomb14::setPeriodic(OpenMM::Vec3* vectors) {
    periodic = true;
    periodicBoxVectors[0] = vectors[0];
    periodicBoxVectors[1] = vectors[1];
    periodicBoxVectors[2] = vectors[2];
}

/**---------------------------------------------------------------------------------------

   Calculate LJ 1-4 ixn

   @param atomIndices      atom indices of 4 atoms in bond
   @param atomCoordinates  atom coordinates
   @param parameters       three parameters:
                                        parameters[0]= (c12/c6)**1/6  (sigma)
                                        parameters[1]= c6*c6/c12      (4*epsilon)
                                        parameters[2]= epsfac*q1*q2
   @param forces           force array (forces added to current values)
   @param totalEnergy      if not null, the energy will be added to this

   --------------------------------------------------------------------------------------- */

void ReferenceSlicedLJCoulomb14::calculateBondIxn(vector<int>& atomIndices, vector<Vec3>& atomCoordinates,
                                     vector<double>& parameters, vector<Vec3>& forces,
                                     double* totalEnergy, double* energyParamDerivs) {
    double deltaR[2][ReferenceForce::LastDeltaRIndex];

    // get deltaR, R2, and R between 2 atoms

    int atomAIndex = atomIndices[0];
    int atomBIndex = atomIndices[1];
    if (periodic)
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[atomBIndex], atomCoordinates[atomAIndex], periodicBoxVectors, deltaR[0]);
    else
        ReferenceForce::getDeltaR(atomCoordinates[atomBIndex], atomCoordinates[atomAIndex], deltaR[0]);

    double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
    double sig2      = inverseR*parameters[0];
           sig2     *= sig2;
    double sig6      = sig2*sig2*sig2;

    double dEdR      = parameters[1]*(12.0*sig6 - 6.0)*sig6;
           dEdR     += ONE_4PI_EPS0*parameters[2]*inverseR;
           dEdR     *= inverseR*inverseR;

    // accumulate forces

    for (int ii = 0; ii < 3; ii++) {
        double force        = dEdR*deltaR[0][ii];
        forces[atomAIndex][ii] += force;
        forces[atomBIndex][ii] -= force;
    }

    // accumulate energies

    if (totalEnergy != NULL)
        *totalEnergy += parameters[1]*(sig6 - 1.0)*sig6 + (ONE_4PI_EPS0*parameters[2]*inverseR);
}
