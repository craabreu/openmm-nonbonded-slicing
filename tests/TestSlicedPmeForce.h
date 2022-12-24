/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential calculations on the basis *
 * of atom pairs and for applying scaling parameters to selected slices.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "SlicedPmeForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/NonbondedForce.h"
#include "sfmt/SFMT.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

const double TOL = 1e-3;

#define assertEnergy(state0, state1) { \
    ASSERT_EQUAL_TOL(state0.getPotentialEnergy(), state1.getPotentialEnergy(), TOL); \
}

#define assertForces(state0, state1) { \
    const vector<Vec3>& forces0 = state0.getForces(); \
    const vector<Vec3>& forces1 = state1.getForces(); \
    for (int i = 0; i < forces0.size(); i++) \
        ASSERT_EQUAL_VEC(forces0[i], forces1[i], TOL); \
}

#define assertForcesAndEnergy(context) { \
    State state0 = context.getState(State::Forces | State::Energy, false, 1<<0); \
    State state1 = context.getState(State::Forces | State::Energy, false, 1<<1); \
    assertEnergy(state0, state1); \
    assertForces(state0, state1); \
}

void testInstantiateFromNonbondedForce() {
    NonbondedForce* force = new NonbondedForce();
    force->setNonbondedMethod(NonbondedForce::PME);
    force->addParticle(0.0, 1.0, 0.5);
    force->addParticle(1.0, 0.5, 0.6);
    force->addParticle(-1.0, 2.0, 0.7);
    force->addParticle(0.5, 2.0, 0.8);
    force->addException(0, 3, 0.0, 1.0, 0.0);
    force->addException(2, 3, 0.5, 1.0, 1.5);
    force->addException(0, 1, 1.0, 1.5, 1.0);
    force->addGlobalParameter("p1", 0.5);
    force->addGlobalParameter("p2", 1.0);
    force->addParticleParameterOffset("p1", 0, 3.0, 0.5, 0.5);
    force->addParticleParameterOffset("p2", 1, 1.0, 1.0, 2.0);
    force->addExceptionParameterOffset("p1", 1, 0.5, 0.5, 1.5);

    NonbondedForce* lennardJones = new NonbondedForce();
    lennardJones->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    lennardJones->addParticle(0.0, 1.0, 0.5);
    lennardJones->addParticle(0.0, 0.5, 0.6);
    lennardJones->addParticle(0.0, 2.0, 0.7);
    lennardJones->addParticle(0.0, 2.0, 0.8);
    lennardJones->addException(0, 3, 0.0, 1.0, 0.0);
    lennardJones->addException(2, 3, 0.0, 1.0, 1.5);
    lennardJones->addException(0, 1, 0.0, 1.5, 1.0);
    lennardJones->addGlobalParameter("p1", 0.5);
    lennardJones->addGlobalParameter("p2", 1.0);
    lennardJones->addParticleParameterOffset("p1", 0, 0.0, 0.5, 0.5);
    lennardJones->addParticleParameterOffset("p2", 1, 0.0, 1.0, 2.0);
    lennardJones->addExceptionParameterOffset("p1", 1, 0.0, 0.5, 1.5);
    lennardJones->setForceGroup(1);

    SlicedPmeForce* coulomb = new SlicedPmeForce(*force, 1);
    coulomb->setForceGroup(1);

    System system;
    double L = 4.0;
    system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);

    system.addForce(force);
    system.addForce(lennardJones);
    system.addForce(coulomb);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);

    vector<Vec3> positions(4);
    for (int i = 0; i < 4; i++)
        positions[i] = Vec3(i, 0, 0);

    context.setPositions(positions);

    assertForcesAndEnergy(context);
}

void testCoulomb() {
    double L = 5.0;

    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));

    SlicedPmeForce* force = new SlicedPmeForce(1);
    force->setForceGroup(1);
    force->addParticle(1.5);
    force->addParticle(-1.5);
    system.addForce(force);
    ASSERT(system.usesPeriodicBoundaryConditions());

    NonbondedForce* forceRef = new NonbondedForce();
    forceRef->setNonbondedMethod(NonbondedForce::PME);
    forceRef->addParticle(1.5, 1.0, 0.0);
    forceRef->addParticle(-1.5, 1.0, 0.0);
    system.addForce(forceRef);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(2, 0, 0);
    context.setPositions(positions);

    assertForcesAndEnergy(context);
}

void testExclusionsAnd14() {
    double L = 5.0;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));

    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    for (int i = 0; i < 5; ++i) {
        system.addParticle(1.0);
        nonbonded->addParticle(0, 1, 0);
    }

    vector<pair<int, int>> bonds;
    bonds.push_back(pair<int, int>(0, 1));
    bonds.push_back(pair<int, int>(1, 2));
    bonds.push_back(pair<int, int>(2, 3));
    bonds.push_back(pair<int, int>(3, 4));
    nonbonded->createExceptionsFromBonds(bonds, 0.0, 0.0);
    int first14, second14;
    for (int i = 0; i < nonbonded->getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        nonbonded->getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if ((particle1 == 0 && particle2 == 3) || (particle1 == 3 && particle2 == 0))
            first14 = i;
        if ((particle1 == 1 && particle2 == 4) || (particle1 == 4 && particle2 == 1))
            second14 = i;
    }
    system.addForce(nonbonded);

    SlicedPmeForce* slicedNonbonded = new SlicedPmeForce(*nonbonded, 1);
    slicedNonbonded->setForceGroup(1);
    system.addForce(slicedNonbonded);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    for (int i = 1; i < 5; ++i) {

        vector<Vec3> positions(5);
        const double r = 1.0;
        for (int j = 0; j < 5; ++j) {
            nonbonded->setParticleParameters(j, 0, 1, 0);
            slicedNonbonded->setParticleCharge(j, 0);
            positions[j] = Vec3(0, j, 0);
        }
        positions[i] = Vec3(r, 0, 0);

        // Test Coulomb forces

        nonbonded->setParticleParameters(0, 2, 1, 0);
        nonbonded->setExceptionParameters(first14, 0, 3, i == 3 ? 4/1.2 : 0, 1, 0);
        nonbonded->setExceptionParameters(second14, 1, 4, 0, 1, 0);
        nonbonded->setParticleParameters(i, 2, 1, 0);

        slicedNonbonded->setParticleCharge(0, 2);
        slicedNonbonded->setParticleCharge(i, 2);
        slicedNonbonded->setExceptionParameters(first14, 0, 3, i == 3 ? 4/1.2 : 0);
        slicedNonbonded->setExceptionParameters(second14, 1, 4, 0);

        context.reinitialize();
        context.setPositions(positions);
        assertForcesAndEnergy(context);
    }
}

void testPeriodic() {
    const double L = 4.0;
    const double cutoff = 2.0;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    for (int i = 0; i < 3; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(1.0, 0.0, 1.0);
    }
    nonbonded->addException(0, 1, 0.0, 1.0, 0.0);
    nonbonded->setCutoffDistance(cutoff);
    system.addForce(nonbonded);
    ASSERT(system.usesPeriodicBoundaryConditions());

    SlicedPmeForce* slicedNonbonded = new SlicedPmeForce(*nonbonded, 1);
    slicedNonbonded->setForceGroup(1);
    system.addForce(slicedNonbonded);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    vector<Vec3> positions(3);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(2, 0, 0);
    positions[2] = Vec3(3, 0, 0);
    context.setPositions(positions);
    assertForcesAndEnergy(context);
}

void testPeriodicExceptions() {
    const double L = 4.0;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    system.addParticle(1.0);
    system.addParticle(1.0);
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->addParticle(1.0, 1.0, 0.0);
    nonbonded->addParticle(-1.0, 1.0, 0.0);
    nonbonded->addException(0, 1, 1.0, 1.0, 0.0);
    const double cutoff = 2.0;
    nonbonded->setCutoffDistance(cutoff);
    system.addForce(nonbonded);

    SlicedPmeForce* slicedNonbonded = new SlicedPmeForce(*nonbonded, 1);
    slicedNonbonded->setForceGroup(1);
    system.addForce(slicedNonbonded);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(3, 0, 0);
    context.setPositions(positions);
    assertForcesAndEnergy(context);

    // Now make exceptions periodic and see if it changes correctly.

    nonbonded->setExceptionsUsePeriodicBoundaryConditions(true);
    slicedNonbonded->setExceptionsUsePeriodicBoundaryConditions(true);
    context.reinitialize(true);
    assertForcesAndEnergy(context);
}

void testTriclinic() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    Vec3 a(3.1, 0, 0);
    Vec3 b(0.4, 3.5, 0);
    Vec3 c(-0.1, -0.5, 4.0);
    system.setDefaultPeriodicBoxVectors(a, b, c);

    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->addParticle(1.0, 1.0, 0.0);
    nonbonded->addParticle(-1.0, 1.0, 0.0);
    const double cutoff = 1.5;
    nonbonded->setCutoffDistance(cutoff);
    system.addForce(nonbonded);

    SlicedPmeForce* slicedNonbonded = new SlicedPmeForce(*nonbonded, 1);
    slicedNonbonded->setForceGroup(1);
    system.addForce(slicedNonbonded);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);

    for (int iteration = 0; iteration < 10; iteration++) {
        // Generate random positions for the two particles.

        positions[0] = a*genrand_real2(sfmt) + b*genrand_real2(sfmt) + c*genrand_real2(sfmt);
        positions[1] = a*genrand_real2(sfmt) + b*genrand_real2(sfmt) + c*genrand_real2(sfmt);
        context.setPositions(positions);

        assertForcesAndEnergy(context);
    }
}

void testLargeSystem() {
    const int numMolecules = 600;
    const int numParticles = numMolecules*2;
    const double cutoff = 3.5;
    const double L = 20.0;
    const double tol = 2e-3;
    System system;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));

    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->setCutoffDistance(cutoff);
    vector<Vec3> positions(numParticles);

    int M = static_cast<int>(std::pow(numMolecules, 1.0/3.0));
    if (M*M*M < numMolecules) M++;
    double sqrt3 = std::sqrt(3);
    for (int k = 0; k < numMolecules; k++) {
        int iz = k/(M*M);
        int iy = (k - iz*M*M)/M;
        int ix = k - M*(iy + iz*M);
        double x = (ix + 0.5)*L/M;
        double y = (iy + 0.5)*L/M;
        double z = (iz + 0.5)*L/M;
        double dx = (0.5 - ix%2)/2;
        double dy = (0.5 - iy%2)/2;
        double dz = (0.5 - iz%2)/2;
        nonbonded->addParticle(1.0, 1.0, 0.0);
        nonbonded->addParticle(-1.0, 1.0, 0.0);
        nonbonded->addException(2*k, 2*k+1, 0.0, 1.0, 0.0);
        positions[2*k] = Vec3(x+dx, y+dy, z+dz);
        positions[2*k+1] = Vec3(x-dx, y-dy, z-dz);
    }
    SlicedPmeForce* slicedNonbonded = new SlicedPmeForce(*nonbonded, 2);
    for (int i = 0; i < numParticles/2; i++)
        slicedNonbonded->setParticleSubset(i, 1);

    slicedNonbonded->setForceGroup(1);

    system.addForce(nonbonded);
    system.addForce(slicedNonbonded);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    assertForcesAndEnergy(context);
}

void testChangingParameters() {
    const int numMolecules = 600;
    const int numParticles = numMolecules*2;
    const double cutoff = 2.0;
    const double L = 20.0;
    const double tol = 2e-3;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    SlicedPmeForce* nonbonded0 = new SlicedPmeForce(1);
    SlicedPmeForce* nonbonded1 = new SlicedPmeForce(1);
    nonbonded1->setForceGroup(1);
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);

    for (int i = 0; i < numMolecules; i++) {
        if (i < numMolecules/2) {
            nonbonded0->addParticle(-1.0);
            nonbonded0->addParticle(1.0);
            nonbonded1->addParticle(-1.5);
            nonbonded1->addParticle(1.5);
        }
        else {
            nonbonded0->addParticle(2.0);
            nonbonded0->addParticle(-2.0);
            nonbonded1->addParticle(3.0);
            nonbonded1->addParticle(-3.0);
        }
        positions[2*i] = L*Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt));
        positions[2*i+1] = Vec3(positions[2*i][0]+1.0, positions[2*i][1], positions[2*i][2]);
        system.addConstraint(2*i, 2*i+1, 1.0);
        nonbonded0->addException(2*i, 2*i+1, 0.0);
        nonbonded1->addException(2*i, 2*i+1, 0.0);
    }
    nonbonded0->setCutoffDistance(cutoff);
    nonbonded1->setCutoffDistance(cutoff);
    system.addForce(nonbonded0);
    system.addForce(nonbonded1);

    // // See if the forces and energies match the Reference platform.
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);

    // // Now modify parameters and see if they agree.

    for (int i = 0; i < numParticles; i++) {
        double charge = nonbonded0->getParticleCharge(i);
        nonbonded0->setParticleCharge(i, 1.5*charge);
    }
    nonbonded0->updateParametersInContext(context);
    assertForcesAndEnergy(context);
}

void testChargeOffsets() {
    const double L = 20.0;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    for (int i = 0; i < 5; i++)
        system.addParticle(1.0);
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->addParticle(0.0, 1.0, 0.0);
    nonbonded->addParticle(1.0, 1.0, 0.0);
    nonbonded->addParticle(-1.0, 1.0, 0.0);
    nonbonded->addParticle(0.5, 1.0, 0.0);
    nonbonded->addParticle(-0.5, 1.0, 0.0);
    nonbonded->addException(0, 3, 0.0, 1.0, 0.0);
    nonbonded->addException(2, 3, 0.5, 1.0, 0.0);
    nonbonded->addException(0, 1, 1.0, 1.0, 0.0);
    nonbonded->addGlobalParameter("p1", 0.0);
    nonbonded->addGlobalParameter("p2", 1.0);
    nonbonded->addParticleParameterOffset("p1", 0, 3.0, 0.0, 0.0);
    nonbonded->addParticleParameterOffset("p2", 1, 1.0, 0.0, 0.0);
    nonbonded->addExceptionParameterOffset("p1", 1, 0.5, 0.0, 0.0);
    system.addForce(nonbonded);

    SlicedPmeForce* force = new SlicedPmeForce(1);
    force->setForceGroup(1);
    force->addParticle(0.0);
    force->addParticle(1.0);
    force->addParticle(-1.0);
    force->addParticle(0.5);
    force->addParticle(-0.5);
    force->addException(0, 3, 0.0);
    force->addException(2, 3, 0.5);
    force->addException(0, 1, 1.0);
    force->addGlobalParameter("p1", 0.0);
    force->addGlobalParameter("p2", 1.0);
    force->addParticleChargeOffset("p1", 0, 3.0);
    force->addParticleChargeOffset("p2", 1, 1.0);
    force->addExceptionChargeOffset("p1", 1, 0.5);
    system.addForce(force);

    vector<Vec3> positions(5);
    for (int i = 0; i < 5; i++)
        positions[i] = Vec3(i, 0, 0);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);

    assertForcesAndEnergy(context);

    context.setParameter("p1", 0.5);
    context.setParameter("p2", 1.5);

    assertForcesAndEnergy(context);
}

void testEwaldExceptions() {
    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    SlicedPmeForce* force = new SlicedPmeForce(1);
    system.addForce(force);
    force->setCutoffDistance(1.0);
    force->addParticle(1.0);
    force->addParticle(1.0);
    force->addParticle(-1.0);
    force->addParticle(-1.0);
    vector<Vec3> positions = {
        Vec3(0, 0, 0),
        Vec3(1.5, 0, 0),
        Vec3(0, 0.5, 0.5),
        Vec3(0.2, 1.3, 0)
    };
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);

    // Compute the energy.

    double e1 = context.getState(State::Energy).getPotentialEnergy();

    // Add a periodic exception and see if the energy changes by the correct amount.

    force->addException(0, 1, 0.2);
    force->setExceptionsUsePeriodicBoundaryConditions(true);
    context.reinitialize(true);
    double e2 = context.getState(State::Energy).getPotentialEnergy();
    double r = 0.5;
    double expectedChange = ONE_4PI_EPS0*(0.2-1.0)/r;
    ASSERT_EQUAL_TOL(expectedChange, e2-e1, 1e-5);
}

void testDirectAndReciprocal() {
    // Create a minimal system with direct space and reciprocal space in different force groups.

    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    SlicedPmeForce* force = new SlicedPmeForce(1);
    system.addForce(force);
    force->setCutoffDistance(1.0);
    force->setReciprocalSpaceForceGroup(1);
    force->addParticle(1.0);
    force->addParticle(1.0);
    force->addParticle(-1.0);
    force->addParticle(-1.0);
    force->addException(0, 2, -2.0);
    vector<Vec3> positions = {
        Vec3(0, 0, 0),
        Vec3(1.5, 0, 0),
        Vec3(0, 0.5, 0.5),
        Vec3(0.2, 1.3, 0)
    };
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);

    // Compute the direct and reciprocal space energies together and separately.

    double e1 = context.getState(State::Energy).getPotentialEnergy();
    double e2 = context.getState(State::Energy, true, 1<<0).getPotentialEnergy();
    double e3 = context.getState(State::Energy, true, 1<<1).getPotentialEnergy();
    ASSERT_EQUAL_TOL(e1, e2+e3, 1e-5);
    ASSERT(e2 != 0);
    ASSERT(e3 != 0);

    // Completely disable the direct space calculation.

    force->setIncludeDirectSpace(false);
    context.reinitialize(true);
    double e4 = context.getState(State::Energy).getPotentialEnergy();
    ASSERT_EQUAL_TOL(e3, e4, 1e-5);
}

void testNonbondedSwitchingParameters(bool exceptions) {
    const int numMolecules = 600;
    const int numParticles = numMolecules*2;
    const double cutoff = 3.5;
    const double L = 20.0;
    const double tol = 2e-3;
    System system1, system2;
    for (int i = 0; i < numParticles; i++) {
        system1.addParticle(1.0);
        system2.addParticle(1.0);
    }
    system1.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    system2.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));

    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->setCutoffDistance(cutoff);
    vector<Vec3> positions(numParticles);

    double intraChargeProd = -0.5;

    int M = static_cast<int>(std::pow(numMolecules, 1.0/3.0));
    if (M*M*M < numMolecules) M++;
    double sqrt3 = std::sqrt(3);
    for (int k = 0; k < numMolecules; k++) {
        int iz = k/(M*M);
        int iy = (k - iz*M*M)/M;
        int ix = k - M*(iy + iz*M);
        double x = (ix + 0.5)*L/M;
        double y = (iy + 0.5)*L/M;
        double z = (iz + 0.5)*L/M;
        double dx = (0.5 - ix%2)/2;
        double dy = (0.5 - iy%2)/2;
        double dz = (0.5 - iz%2)/2;
        nonbonded->addParticle(1.0, 1.0, 0.0);
        nonbonded->addParticle(-1.0, 1.0, 0.0);
        if (exceptions)
            nonbonded->addException(2*k, 2*k+1, intraChargeProd, 1.0, 0.0);
        positions[2*k] = Vec3(x+dx, y+dy, z+dz);
        positions[2*k+1] = Vec3(x-dx, y-dy, z-dz);
    }

    double lambda = 0.5;

    SlicedPmeForce* slicedNonbonded = new SlicedPmeForce(*nonbonded, 2);
    for (int k = 0; k < numMolecules; k++)
        slicedNonbonded->setParticleSubset(2*k, 1);

    nonbonded->addGlobalParameter("lambda", lambda);
    for (int k = 0; k < numMolecules; k++) {
        double charge, sigma, epsilon;
        nonbonded->getParticleParameters(2*k, charge, sigma, epsilon);
        nonbonded->setParticleParameters(2*k, 0.0, sigma, epsilon);
        nonbonded->addParticleParameterOffset("lambda", 2*k, charge, 0.0, 0.0);
    }

    slicedNonbonded->addGlobalParameter("lambda", lambda);
    slicedNonbonded->addGlobalParameter("lambdaSq", lambda*lambda);
    slicedNonbonded->addSwitchingParameter("lambda", 0, 1);
    slicedNonbonded->addSwitchingParameter("lambdaSq", 1, 1);

    if (exceptions)
        for (int i = 0; i < nonbonded->getNumExceptions(); i++) {
            int p1, p2;
            double chargeProd, sigma, epsilon;
            nonbonded->getExceptionParameters(i, p1, p2, chargeProd, sigma, epsilon);
            nonbonded->setExceptionParameters(i, p1, p2, 0.0, sigma, epsilon);
            nonbonded->addExceptionParameterOffset("lambda", i, chargeProd, 0.0, 0.0);
        }

    system1.addForce(nonbonded);
    system2.addForce(slicedNonbonded);

    nonbonded->setReciprocalSpaceForceGroup(1);
    VerletIntegrator integrator1(0.01);
    Context context1(system1, integrator1, platform);
    context1.setPositions(positions);

    slicedNonbonded->setReciprocalSpaceForceGroup(1);
    VerletIntegrator integrator2(0.01);
    Context context2(system2, integrator2, platform);
    context2.setPositions(positions);

    // Direct space:
    State state1 = context1.getState(State::Energy | State::Forces, false, 1<<0);
    State state2 = context2.getState(State::Energy | State::Forces, false, 1<<0);
    assertEnergy(state1, state2);
    assertForces(state1, state2);

    // Reciprocal space:
    state1 = context1.getState(State::Energy | State::Forces, false, 1<<1);
    state2 = context2.getState(State::Energy | State::Forces, false, 1<<1);
    assertEnergy(state1, state2);
    assertForces(state1, state2);

    // Change of switching parameter value:
    lambda = 0.8;

    context1.setParameter("lambda", lambda);
    context2.setParameter("lambda", lambda);
    context2.setParameter("lambdaSq", lambda*lambda);

    state1 = context1.getState(State::Energy | State::Forces);
    state2 = context2.getState(State::Energy | State::Forces);
    assertEnergy(state1, state2);
    assertForces(state1, state2);

    // Derivatives:
    context1.setParameter("lambda", 0);
    double energy0 = context1.getState(State::Energy).getPotentialEnergy();
    context1.setParameter("lambda", 1);
    double energy1 = context1.getState(State::Energy).getPotentialEnergy();

    slicedNonbonded->addSwitchingParameterDerivative("lambda");
    slicedNonbonded->addSwitchingParameterDerivative("lambdaSq");
    context2.reinitialize(true);
    state2 = context2.getState(State::ParameterDerivatives);
    auto derivatives = state2.getEnergyParameterDerivatives();
    ASSERT_EQUAL_TOL(energy1-energy0, derivatives["lambda"]+derivatives["lambdaSq"], TOL);

    slicedNonbonded->addGlobalParameter("remainder", 1.0);
    slicedNonbonded->addSwitchingParameter("remainder", 0, 0);
    slicedNonbonded->addSwitchingParameterDerivative("remainder");
    context2.reinitialize(true);
    context2.setParameter("lambda", 1.0);
    context2.setParameter("lambdaSq", 1.0);
    state2 = context2.getState(State::Energy | State::ParameterDerivatives);
    double energy = state2.getPotentialEnergy();
    derivatives = state2.getEnergyParameterDerivatives();
    double sum = derivatives["lambda"]+derivatives["lambdaSq"]+derivatives["remainder"];
    ASSERT_EQUAL_TOL(energy, sum, TOL);
}

void runPlatformTests();

extern "C" OPENMM_EXPORT void registerNonbondedSlicingReferenceKernelFactories();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testInstantiateFromNonbondedForce();
        testCoulomb();
        testExclusionsAnd14();
        testPeriodic();
        testPeriodicExceptions();
        testTriclinic();
        testLargeSystem();
        testChangingParameters();
        testChargeOffsets();
        testEwaldExceptions();
        testDirectAndReciprocal();
        testNonbondedSwitchingParameters(false);
        testNonbondedSwitchingParameters(true);
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
