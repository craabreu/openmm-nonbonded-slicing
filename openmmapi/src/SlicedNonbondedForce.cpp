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

#include "SlicedNonbondedForce.h"
#include "internal/SlicedNonbondedForceImpl.h"
#include "openmm/internal/AssertionUtilities.h"
#include <string.h>

using namespace std;
using namespace OpenMM;
using namespace PmeSlicing;

#define ASSERT_VALID(name, value, number) {if (value < 0 || value >= number) throwException(__FILE__, __LINE__, name " out of range");};

SlicedNonbondedForce::SlicedNonbondedForce(int numSubsets) : NonbondedForce(), numSubsets(numSubsets) {
}

SlicedNonbondedForce::SlicedNonbondedForce(const NonbondedForce& force, int numSubsets) {
    setForceGroup(force.getForceGroup());
    setName(force.getName());
    setNonbondedMethod((SlicedNonbondedForce::NonbondedMethod) force.getNonbondedMethod());
    setCutoffDistance(force.getCutoffDistance());
    setUseSwitchingFunction(force.getUseSwitchingFunction());
    setSwitchingDistance(force.getSwitchingDistance());
    setEwaldErrorTolerance(force.getEwaldErrorTolerance());
    setReactionFieldDielectric(force.getReactionFieldDielectric());
    setUseDispersionCorrection(force.getUseDispersionCorrection());
    setIncludeDirectSpace(force.getIncludeDirectSpace());
    double alpha;
    int nx, ny, nz;
    force.getPMEParameters(alpha, nx, ny, nz);
    setPMEParameters(alpha, nx, ny, nz);
    force.getLJPMEParameters(alpha, nx, ny, nz);
    setReciprocalSpaceForceGroup(force.getReciprocalSpaceForceGroup());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        addParticle(charge, sigma, epsilon);
    }
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        addException(particle1, particle2, chargeProd, sigma, epsilon);
    }
    setExceptionsUsePeriodicBoundaryConditions(force.getExceptionsUsePeriodicBoundaryConditions());
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        addGlobalParameter(force.getGlobalParameterName(i), force.getGlobalParameterDefaultValue(i));
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string parameter;
        int index;
        double chargeScale, sigmaScale, epsilonScale;
        force.getParticleParameterOffset(i, parameter, index, chargeScale, sigmaScale, epsilonScale);
        addParticleParameterOffset(parameter, index, chargeScale, sigmaScale, epsilonScale);
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string parameter;
        int index;
        double chargeProdScale, sigmaScale, epsilonScale;
        force.getExceptionParameterOffset(i, parameter, index, chargeProdScale, sigmaScale, epsilonScale);
        addExceptionParameterOffset(parameter, index, chargeProdScale, sigmaScale, epsilonScale);
    }
}

void SlicedNonbondedForce::setParticleSubset(int index, int subset) {
    ASSERT_VALID("Index", index, getNumParticles());
    ASSERT_VALID("Subset", subset, getNumSubsets());
    subsets[index] = subset;
}

int SlicedNonbondedForce::getParticleSubset(int index) const {
    ASSERT_VALID("Index", index, getNumParticles());
    auto element = subsets.find(index);
    return element == subsets.end() ? 0 : element->second;
}

ForceImpl* SlicedNonbondedForce::createImpl() const {
    return new SlicedNonbondedForceImpl(*this);
}