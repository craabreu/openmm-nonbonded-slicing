/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "SlicedNonbondedForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

void testSerialization() {
    // Create a Force.

    SlicedNonbondedForce force(3);
    force.setForceGroup(3);
    force.setName("custom name");
    force.setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    force.setSwitchingDistance(1.5);
    force.setUseSwitchingFunction(true);
    force.setCutoffDistance(2.0);
    force.setEwaldErrorTolerance(1e-3);
    force.setReactionFieldDielectric(50.0);
    force.setUseDispersionCorrection(false);
    force.setExceptionsUsePeriodicBoundaryConditions(true);
    force.setIncludeDirectSpace(false);
    double alpha = 0.5;
    int nx = 3, ny = 5, nz = 7;
    force.setPMEParameters(alpha, nx, ny, nz);
    double dalpha = 0.8;
    int dnx = 4, dny = 6, dnz = 7;
    force.setLJPMEParameters(dalpha, dnx, dny, dnz);
    force.addParticle(1, 0.1, 0.01);
    force.addParticle(0.5, 0.2, 0.02);
    force.addParticle(-0.5, 0.3, 0.03);
    force.setParticleSubset(0, 1);
    force.setParticleSubset(1, 2);
    force.addException(0, 1, 2, 0.5, 0.1);
    force.addException(1, 2, 0.2, 0.4, 0.2);
    force.addGlobalParameter("scale1", 1.0);
    force.addGlobalParameter("scale2", 2.0);
    force.addParticleParameterOffset("scale1", 2, 1.5, 2.0, 2.5);
    force.addExceptionParameterOffset("scale2", 1, -0.1, -0.2, -0.3);
    force.addGlobalParameter("lambda", 0.5);
    force.addScalingParameter("lambda", 0, 1, true, true);
    force.addScalingParameter("lambda", 1, 1, false, true);
    force.addScalingParameterDerivative("lambda");

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<SlicedNonbondedForce>(&force, "Force", buffer);
    SlicedNonbondedForce* copy = XmlSerializer::deserialize<SlicedNonbondedForce>(buffer);

    // Compare the two forces to see if they are identical.

    SlicedNonbondedForce& force2 = *copy;
    ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
    ASSERT_EQUAL(force.getName(), force2.getName());
    ASSERT_EQUAL(force.getNonbondedMethod(), force2.getNonbondedMethod());
    ASSERT_EQUAL(force.getSwitchingDistance(), force2.getSwitchingDistance());
    ASSERT_EQUAL(force.getUseSwitchingFunction(), force2.getUseSwitchingFunction());
    ASSERT_EQUAL(force.getCutoffDistance(), force2.getCutoffDistance());
    ASSERT_EQUAL(force.getEwaldErrorTolerance(), force2.getEwaldErrorTolerance());
    ASSERT_EQUAL(force.getReactionFieldDielectric(), force2.getReactionFieldDielectric());
    ASSERT_EQUAL(force.getUseDispersionCorrection(), force2.getUseDispersionCorrection());
    ASSERT_EQUAL(force.getExceptionsUsePeriodicBoundaryConditions(), force2.getExceptionsUsePeriodicBoundaryConditions());
    ASSERT_EQUAL(force.getNumParticles(), force2.getNumParticles());
    ASSERT_EQUAL(force.getNumExceptions(), force2.getNumExceptions());
    ASSERT_EQUAL(force.getNumGlobalParameters(), force2.getNumGlobalParameters());
    ASSERT_EQUAL(force.getNumParticleParameterOffsets(), force2.getNumParticleParameterOffsets());
    ASSERT_EQUAL(force.getNumExceptionParameterOffsets(), force2.getNumExceptionParameterOffsets());
    ASSERT_EQUAL(force.getIncludeDirectSpace(), force2.getIncludeDirectSpace());
    double alpha2;
    int nx2, ny2, nz2;
    force2.getPMEParameters(alpha2, nx2, ny2, nz2);
    ASSERT_EQUAL(alpha, alpha2);
    ASSERT_EQUAL(nx, nx2);
    ASSERT_EQUAL(ny, ny2);
    ASSERT_EQUAL(nz, nz2);
    double dalpha2;
    int dnx2, dny2, dnz2;
    force2.getLJPMEParameters(dalpha2, dnx2, dny2, dnz2);
    ASSERT_EQUAL(dalpha, dalpha2);
    ASSERT_EQUAL(dnx, dnx2);
    ASSERT_EQUAL(dny, dny2);
    ASSERT_EQUAL(dnz, dnz2);
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        ASSERT_EQUAL(force.getGlobalParameterName(i), force2.getGlobalParameterName(i));
        ASSERT_EQUAL(force.getGlobalParameterDefaultValue(i), force2.getGlobalParameterDefaultValue(i));
    }
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        int index1, index2;
        string param1, param2;
        double charge1, sigma1, epsilon1;
        double charge2, sigma2, epsilon2;
        force.getParticleParameterOffset(i, param1, index1, charge1, sigma1, epsilon1);
        force2.getParticleParameterOffset(i, param2, index2, charge2, sigma2, epsilon2);
        ASSERT_EQUAL(index1, index1);
        ASSERT_EQUAL(param1, param2);
        ASSERT_EQUAL(charge1, charge2);
        ASSERT_EQUAL(sigma1, sigma2);
        ASSERT_EQUAL(epsilon1, epsilon2);
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        int index1, index2;
        string param1, param2;
        double charge1, sigma1, epsilon1;
        double charge2, sigma2, epsilon2;
        force.getExceptionParameterOffset(i, param1, index1, charge1, sigma1, epsilon1);
        force2.getExceptionParameterOffset(i, param2, index2, charge2, sigma2, epsilon2);
        ASSERT_EQUAL(index1, index1);
        ASSERT_EQUAL(param1, param2);
        ASSERT_EQUAL(charge1, charge2);
        ASSERT_EQUAL(sigma1, sigma2);
        ASSERT_EQUAL(epsilon1, epsilon2);
    }
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge1, sigma1, epsilon1;
        double charge2, sigma2, epsilon2;
        force.getParticleParameters(i, charge1, sigma1, epsilon1);
        force2.getParticleParameters(i, charge2, sigma2, epsilon2);
        ASSERT_EQUAL(charge1, charge2);
        ASSERT_EQUAL(sigma1, sigma2);
        ASSERT_EQUAL(epsilon1, epsilon2);
        int subset1 = force.getParticleSubset(i);
        int subset2 = force2.getParticleSubset(i);
        ASSERT_EQUAL(subset1, subset2);
    }
    ASSERT_EQUAL(force.getNumExceptions(), force2.getNumExceptions());
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int a1, a2, b1, b2;
        double charge1, sigma1, epsilon1;
        double charge2, sigma2, epsilon2;
        force.getExceptionParameters(i, a1, b1, charge1, sigma1, epsilon1);
        force2.getExceptionParameters(i, a2, b2, charge2, sigma2, epsilon2);
        ASSERT_EQUAL(a1, a2);
        ASSERT_EQUAL(b1, b2);
        ASSERT_EQUAL(charge1, charge2);
        ASSERT_EQUAL(sigma1, sigma2);
        ASSERT_EQUAL(epsilon1, epsilon2);
    }
    ASSERT_EQUAL(force.getNumScalingParameters(), force2.getNumScalingParameters());
    for (int i = 0; i < force.getNumScalingParameters(); i++) {
        string parameter1, parameter2;
        int subset11, subset21, subset12, subset22;
        bool includeCoulomb1, includeLJ1, includeCoulomb2, includeLJ2;
        force.getScalingParameter(i, parameter1, subset11, subset21, includeCoulomb1, includeLJ1);
        force2.getScalingParameter(i, parameter2, subset12, subset22, includeCoulomb2, includeLJ2);
        ASSERT_EQUAL(parameter1, parameter2);
        ASSERT_EQUAL(subset11, subset12);
        ASSERT_EQUAL(subset21, subset22);
        ASSERT_EQUAL(includeLJ1, includeLJ2);
        ASSERT_EQUAL(includeCoulomb1, includeCoulomb2);
    }
    ASSERT_EQUAL(force.getNumScalingParameterDerivatives(), force2.getNumScalingParameterDerivatives());
    for (int i = 0; i < force.getNumScalingParameterDerivatives(); i++)
        ASSERT_EQUAL(force.getScalingParameterDerivativeName(i), force2.getScalingParameterDerivativeName(i))
}

int main() {
    try {
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
