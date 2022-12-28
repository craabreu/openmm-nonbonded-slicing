
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

#include "internal/ReferenceSlicedLJCoulomb14.h"
#include "openmm/reference/ReferenceForce.h"
#include "openmm/reference/SimTKOpenMMUtilities.h"

using namespace std;
using namespace NonbondedSlicing;
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

       @param atomIndices      atom indices of the atoms in each pair
       @param atomCoordinates  atom coordinates
       @param parameters       (sigma, 4*epsilon, charge product) for each pair
       @param forces           force array (forces added to current values)
       @param sliceLambdas     the scaling parameters of the slice
       @param sliceEnergies    the energies of the slice

   --------------------------------------------------------------------------------------- */

void ReferenceSlicedLJCoulomb14::calculateBondIxn(vector<int>& atomIndices, vector<Vec3>& atomCoordinates,
                                     vector<double>& parameters, vector<Vec3>& forces,
                                     vector<double>& sliceLambdas, vector<double>& sliceEnergies) {
    double deltaR[2][ReferenceForce::LastDeltaRIndex];

    // get deltaR, R2, and R between 2 atoms

    int i = atomIndices[0];
    int j = atomIndices[1];
    if (periodic)
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[j], atomCoordinates[i], periodicBoxVectors, deltaR[0]);
    else
        ReferenceForce::getDeltaR(atomCoordinates[j], atomCoordinates[i], deltaR[0]);

    double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
    double sig2      = inverseR*parameters[0];
           sig2     *= sig2;
    double sig6      = sig2*sig2*sig2;

    double dEdR      = sliceLambdas[vdW]*parameters[1]*(12.0*sig6 - 6.0)*sig6;
           dEdR     += sliceLambdas[Coul]*ONE_4PI_EPS0*parameters[2]*inverseR;
           dEdR     *= inverseR*inverseR;

    // accumulate forces

    for (int k = 0; k < 3; k++) {
        double force = dEdR*deltaR[0][k];
        forces[i][k] += force;
        forces[j][k] -= force;
    }

    // accumulate energies
    sliceEnergies[vdW] += parameters[1]*(sig6 - 1.0)*sig6;
    sliceEnergies[Coul] += ONE_4PI_EPS0*parameters[2]*inverseR;
}
