
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

ReferenceSlicedLJCoulomb14::ReferenceSlicedLJCoulomb14() : periodic(false), coulombDecapping(false) {
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
       @param decapped         whether the slice is decapped
       @param mode             which part of the interactions to compute

   --------------------------------------------------------------------------------------- */

void ReferenceSlicedLJCoulomb14::setCappedLJ(const CappedLJ& capped) {
    cappedLJ = capped;
}

void ReferenceSlicedLJCoulomb14::setCoulombDecapping(bool decapping, const CappedCoulomb& capped) {
    coulombDecapping = decapping;
    cappedCoulomb = capped;
}

void ReferenceSlicedLJCoulomb14::setInnerSplit(const InnerSplit& split) {
    innerSplit = split;
}

void ReferenceSlicedLJCoulomb14::calculateBondIxn(vector<int>& atomIndices, vector<Vec3>& atomCoordinates,
                                     vector<double>& parameters, vector<Vec3>& forces,
                                     vector<double>& sliceLambdas, vector<double>& sliceEnergies,
                                     bool decapped, Mode mode) {
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

    // The full interaction. Its inner part is that of the bare potential.
    double vdwEnergy = parameters[1]*(sig6-1.0)*sig6;
    double rdUdr = parameters[1]*(12.0*sig6-6.0)*sig6;
    double r = deltaR[0][ReferenceForce::RIndex];
    double innerVdwForce = 0.0, innerCoulForce = 0.0, innerVdwEnergy = 0.0, innerCoulEnergy = 0.0;
    if (mode != Full && r < innerSplit.rc) {
        double switchValue = innerSplit.switchValue(r);
        innerVdwForce = switchValue*parameters[1]*(12.0*sig6 - 6.0)*sig6*inverseR*inverseR;
        innerCoulForce = switchValue*ONE_4PI_EPS0*parameters[2]*inverseR*inverseR*inverseR;
        double sigma6 = pow(parameters[0], 6);
        double e[3];
        innerSplit.energyTerms(r, e);
        innerVdwEnergy = parameters[1]*(sigma6*sigma6*e[0] - sigma6*e[1]);
        innerCoulEnergy = ONE_4PI_EPS0*parameters[2]*e[2];
    }
    // Below sigma, a decapped slice takes the capped potential and the decapping term. With an inner force group,
    // sigma cannot exceed the inner switching distance, so both belong to the inner part and the remainder is unchanged.
    double decapEnergy = 0.0, decapRdUdr = 0.0;
    if (mode != Remainder && decapped && sig6 > 1.0) {
        double capEnergy = vdwEnergy, capRdUdr = rdUdr;
        cappedLJ.split(1.0/(inverseR*parameters[0]), parameters[1], capEnergy, capRdUdr, decapEnergy, decapRdUdr);
        if (mode == Full) {
            vdwEnergy = capEnergy;
            rdUdr = capRdUdr;
        }
        else {
            double sigma6 = pow(parameters[0], 6);
            innerVdwForce = capRdUdr*inverseR*inverseR;
            innerVdwEnergy = capEnergy - parameters[1]*sigma6*(sigma6*innerSplit.K[0] - innerSplit.K[1]);
        }
    }
    double vdwForce = rdUdr*inverseR*inverseR;
    double qq = ONE_4PI_EPS0*parameters[2];
    double coulEnergy = qq*inverseR, coulRdUdr = qq*inverseR;
    // With Coulomb decapping (never combined with an inner force group), the 1/r term is capped below sigma too
    if (coulombDecapping && mode == Full && decapped && sig6 > 1.0) {
        double decapQEnergy, decapQRdUdr;
        cappedCoulomb.split(1.0/(inverseR*parameters[0]), qq/parameters[0], coulEnergy, coulRdUdr, decapQEnergy, decapQRdUdr);
        decapEnergy += decapQEnergy;
        decapRdUdr += decapQRdUdr;
    }
    double coulForce = coulRdUdr*inverseR*inverseR;
    if (mode == Inner) {
        vdwForce = innerVdwForce;
        coulForce = innerCoulForce;
        vdwEnergy = innerVdwEnergy;
        coulEnergy = innerCoulEnergy;
    }
    else if (mode == Remainder) {
        vdwForce -= innerVdwForce;
        coulForce -= innerCoulForce;
        vdwEnergy -= innerVdwEnergy;
        coulEnergy -= innerCoulEnergy;
        // The remainder of a decapped pair below sigma is the constant of the inner split, set directly rather than as
        // the difference of two terms of order x^-12
        if (decapped && sig6 > 1.0) {
            double sigma6 = pow(parameters[0], 6);
            vdwForce = 0.0;
            vdwEnergy = parameters[1]*sigma6*(sigma6*innerSplit.K[0] - innerSplit.K[1]);
        }
    }
    double dEdR = sliceLambdas[vdW]*vdwForce + sliceLambdas[Coul]*coulForce + sliceLambdas[decap]*decapRdUdr*inverseR*inverseR;

    // accumulate forces

    for (int k = 0; k < 3; k++) {
        double force = dEdR*deltaR[0][k];
        forces[i][k] += force;
        forces[j][k] -= force;
    }

    // accumulate energies
    sliceEnergies[vdW] += vdwEnergy;
    sliceEnergies[decap] += decapEnergy;
    sliceEnergies[Coul] += coulEnergy;
}
