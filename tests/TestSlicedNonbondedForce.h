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
#include "internal/AssertionUtilities.h"
#include "openmm/NonbondedForce.h"
#include "openmm/Context.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "sfmt/SFMT.h"
#include <iomanip>
#include <vector>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;

void testInstantiateFromNonbondedForce(NonbondedForce::NonbondedMethod method) {
    NonbondedForce* force = new NonbondedForce();
    force->setNonbondedMethod(method);
    force->addParticle(0.0, 1.0, 0.5);
    force->addParticle(1.0, 0.5, 0.6);
    force->addParticle(-1.0, 2.0, 0.7);
    force->addParticle(0.5, 2.0, 0.8);
    force->addParticle(-0.5, 2.0, 0.8);
    force->addException(0, 3, 0.0, 1.0, 0.0);
    force->addException(2, 3, 0.5, 1.0, 1.5);
    force->addException(0, 1, 1.0, 1.5, 1.0);
    force->addGlobalParameter("p1", 0.5);
    force->addGlobalParameter("p2", 1.0);
    force->addParticleParameterOffset("p1", 0, 3.0, 0.5, 0.5);
    force->addParticleParameterOffset("p2", 1, 1.0, 1.0, 2.0);
    force->addExceptionParameterOffset("p1", 1, 0.5, 0.5, 1.5);

    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(*force, 3);
    sliced->setForceGroup(1);

    int N = force->getNumParticles();

    System system;
    double L = (double) N;
    system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    for (int i = 0; i < N; i++)
        system.addParticle(1.0);

    system.addForce(force);
    system.addForce(sliced);

    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);

    vector<Vec3> positions(N);
    for (int i = 0; i < N; i++)
        positions[i] = Vec3(i, 0, 0);

    context.setPositions(positions);

    assertForcesAndEnergy(context, TOL);
}

void testCoulomb() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* forceField = new SlicedNonbondedForce(1);
    forceField->addParticle(0.5, 1, 0);
    forceField->addParticle(-1.5, 1, 0);
    system.addForce(forceField);
    ASSERT(!forceField->usesPeriodicBoundaryConditions());
    ASSERT(!system.usesPeriodicBoundaryConditions());
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(2, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    double force = ONE_4PI_EPS0*(-0.75)/4.0;
    assertEqualVec(Vec3(-force, 0, 0), forces[0], TOL);
    assertEqualVec(Vec3(force, 0, 0), forces[1], TOL);
    assertEqualTo(ONE_4PI_EPS0*(-0.75)/2.0, state.getPotentialEnergy(), TOL);
}

void testLJ() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* forceField = new SlicedNonbondedForce(1);
    forceField->addParticle(0, 1.2, 1);
    forceField->addParticle(0, 1.4, 2);
    system.addForce(forceField);
    ASSERT(!forceField->usesPeriodicBoundaryConditions());
    ASSERT(!system.usesPeriodicBoundaryConditions());
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(2, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    double x = 1.3/2.0;
    double eps = SQRT_TWO;
    double force = 4.0*eps*(12*pow(x, 12.0)-6*pow(x, 6.0))/2.0;
    assertEqualVec(Vec3(-force, 0, 0), forces[0], TOL);
    assertEqualVec(Vec3(force, 0, 0), forces[1], TOL);
    assertEqualTo(4.0*eps*(pow(x, 12.0)-pow(x, 6.0)), state.getPotentialEnergy(), TOL);
}

void testExclusionsAnd14() {
    System system;
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(1);
    for (int i = 0; i < 5; ++i) {
        system.addParticle(1.0);
        sliced->addParticle(0, 1.5, 0);
    }
    vector<pair<int, int> > bonds;
    bonds.push_back(pair<int, int>(0, 1));
    bonds.push_back(pair<int, int>(1, 2));
    bonds.push_back(pair<int, int>(2, 3));
    bonds.push_back(pair<int, int>(3, 4));
    sliced->createExceptionsFromBonds(bonds, 0.0, 0.0);
    int first14, second14;
    for (int i = 0; i < sliced->getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        sliced->getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if ((particle1 == 0 && particle2 == 3) || (particle1 == 3 && particle2 == 0))
            first14 = i;
        if ((particle1 == 1 && particle2 == 4) || (particle1 == 4 && particle2 == 1))
            second14 = i;
    }
    system.addForce(sliced);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    for (int i = 1; i < 5; ++i) {

        // Test LJ forces

        vector<Vec3> positions(5);
        const double r = 1.0;
        for (int j = 0; j < 5; ++j) {
            sliced->setParticleParameters(j, 0, 1.5, 0);
            positions[j] = Vec3(0, j, 0);
        }
        sliced->setParticleParameters(0, 0, 1.5, 1);
        sliced->setParticleParameters(i, 0, 1.5, 1);
        sliced->setExceptionParameters(first14, 0, 3, 0, 1.5, i == 3 ? 0.5 : 0.0);
        sliced->setExceptionParameters(second14, 1, 4, 0, 1.5, 0.0);
        positions[i] = Vec3(r, 0, 0);
        context.reinitialize();
        context.setPositions(positions);
        State state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces = state.getForces();
        double x = 1.5/r;
        double eps = 1.0;
        double force = 4.0*eps*(12*pow(x, 12.0)-6*pow(x, 6.0))/r;
        double energy = 4.0*eps*(pow(x, 12.0)-pow(x, 6.0));
        if (i == 3) {
            force *= 0.5;
            energy *= 0.5;
        }
        if (i < 3) {
            force = 0;
            energy = 0;
        }
        assertEqualVec(Vec3(-force, 0, 0), forces[0], TOL);
        assertEqualVec(Vec3(force, 0, 0), forces[i], TOL);
        assertEqualTo(energy, state.getPotentialEnergy(), TOL);

        // Test Coulomb forces

        sliced->setParticleParameters(0, 2, 1.5, 0);
        sliced->setParticleParameters(i, 2, 1.5, 0);
        sliced->setExceptionParameters(first14, 0, 3, i == 3 ? 4/1.2 : 0, 1.5, 0);
        sliced->setExceptionParameters(second14, 1, 4, 0, 1.5, 0);
        context.reinitialize();
        context.setPositions(positions);
        state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces2 = state.getForces();
        force = ONE_4PI_EPS0*4/(r*r);
        energy = ONE_4PI_EPS0*4/r;
        if (i == 3) {
            force /= 1.2;
            energy /= 1.2;
        }
        if (i < 3) {
            force = 0;
            energy = 0;
        }
        assertEqualVec(Vec3(-force, 0, 0), forces2[0], TOL);
        assertEqualVec(Vec3(force, 0, 0), forces2[i], TOL);
        assertEqualTo(energy, state.getPotentialEnergy(), TOL);
    }
}

void testCutoff() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* forceField = new SlicedNonbondedForce(1);
    forceField->addParticle(1.0, 1, 0);
    forceField->addParticle(1.0, 1, 0);
    forceField->addParticle(1.0, 1, 0);
    forceField->setNonbondedMethod(SlicedNonbondedForce::CutoffNonPeriodic);
    const double cutoff = 2.9;
    forceField->setCutoffDistance(cutoff);
    const double eps = 50.0;
    forceField->setReactionFieldDielectric(eps);
    system.addForce(forceField);
    ASSERT(!forceField->usesPeriodicBoundaryConditions());
    ASSERT(!system.usesPeriodicBoundaryConditions());
    Context context(system, integrator, platform);
    vector<Vec3> positions(3);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 2, 0);
    positions[2] = Vec3(0, 3, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    const double krf = (1.0/(cutoff*cutoff*cutoff))*(eps-1.0)/(2.0*eps+1.0);
    const double crf = (1.0/cutoff)*(3.0*eps)/(2.0*eps+1.0);
    const double force1 = ONE_4PI_EPS0*(1.0)*(0.25-2.0*krf*2.0);
    const double force2 = ONE_4PI_EPS0*(1.0)*(1.0-2.0*krf*1.0);
    assertEqualVec(Vec3(0, -force1, 0), forces[0], TOL);
    assertEqualVec(Vec3(0, force1-force2, 0), forces[1], TOL);
    assertEqualVec(Vec3(0, force2, 0), forces[2], TOL);
    const double energy1 = ONE_4PI_EPS0*(1.0)*(0.5+krf*4.0-crf);
    const double energy2 = ONE_4PI_EPS0*(1.0)*(1.0+krf*1.0-crf);
    assertEqualTo(energy1+energy2, state.getPotentialEnergy(), TOL);
}

void testCutoff14() {
    System system;
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(1);
    sliced->setNonbondedMethod(SlicedNonbondedForce::CutoffNonPeriodic);
    for (int i = 0; i < 5; i++) {
        system.addParticle(1.0);
        sliced->addParticle(0, 1.5, 0);
    }
    const double cutoff = 3.5;
    sliced->setCutoffDistance(cutoff);
    const double eps = 30.0;
    sliced->setReactionFieldDielectric(eps);
    vector<pair<int, int> > bonds;
    bonds.push_back(pair<int, int>(0, 1));
    bonds.push_back(pair<int, int>(1, 2));
    bonds.push_back(pair<int, int>(2, 3));
    bonds.push_back(pair<int, int>(3, 4));
    sliced->createExceptionsFromBonds(bonds, 0.0, 0.0);
    int first14, second14;
    for (int i = 0; i < sliced->getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        sliced->getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if ((particle1 == 0 && particle2 == 3) || (particle1 == 3 && particle2 == 0))
            first14 = i;
        if ((particle1 == 1 && particle2 == 4) || (particle1 == 4 && particle2 == 1))
            second14 = i;
    }
    system.addForce(sliced);
    ASSERT(!sliced->usesPeriodicBoundaryConditions());
    ASSERT(!system.usesPeriodicBoundaryConditions());
    Context context(system, integrator, platform);
    vector<Vec3> positions(5);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(1, 0, 0);
    positions[2] = Vec3(2, 0, 0);
    positions[3] = Vec3(3, 0, 0);
    positions[4] = Vec3(4, 0, 0);
    context.setPositions(positions);
    for (int i = 1; i < 5; ++i) {

        // Test LJ forces

        sliced->setParticleParameters(0, 0, 1.5, 1);
        for (int j = 1; j < 5; ++j)
            sliced->setParticleParameters(j, 0, 1.5, 0);
        sliced->setParticleParameters(i, 0, 1.5, 1);
        sliced->setExceptionParameters(first14, 0, 3, 0, 1.5, i == 3 ? 0.5 : 0.0);
        sliced->setExceptionParameters(second14, 1, 4, 0, 1.5, 0.0);
        context.reinitialize(true);
        State state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces = state.getForces();
        double r = positions[i][0];
        double x = 1.5/r;
        double e = 1.0;
        double force = 4.0*e*(12*pow(x, 12.0)-6*pow(x, 6.0))/r;
        double energy = 4.0*e*(pow(x, 12.0)-pow(x, 6.0));
        if (i == 3) {
            force *= 0.5;
            energy *= 0.5;
        }
        if (i < 3 || r > cutoff) {
            force = 0;
            energy = 0;
        }
        assertEqualVec(Vec3(-force, 0, 0), forces[0], TOL);
        assertEqualVec(Vec3(force, 0, 0), forces[i], TOL);
        assertEqualTo(energy, state.getPotentialEnergy(), TOL);

        // Test Coulomb forces

        const double q = 0.7;
        sliced->setParticleParameters(0, q, 1.5, 0);
        sliced->setParticleParameters(i, q, 1.5, 0);
        sliced->setExceptionParameters(first14, 0, 3, i == 3 ? q*q/1.2 : 0, 1.5, 0);
        sliced->setExceptionParameters(second14, 1, 4, 0, 1.5, 0);
        context.reinitialize(true);
        state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces2 = state.getForces();
        force = ONE_4PI_EPS0*q*q/(r*r);
        energy = ONE_4PI_EPS0*q*q/r;
        if (i == 3) {
            force /= 1.2;
            energy /= 1.2;
        }
        if (i < 3 || r > cutoff) {
            force = 0;
            energy = 0;
        }
        assertEqualVec(Vec3(-force, 0, 0), forces2[0], TOL);
        assertEqualVec(Vec3(force, 0, 0), forces2[i], TOL);
        assertEqualTo(energy, state.getPotentialEnergy(), TOL);
    }
}

void testPeriodic() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(1);
    sliced->addParticle(1.0, 1, 0);
    sliced->addParticle(1.0, 1, 0);
    sliced->addParticle(1.0, 1, 0);
    sliced->addException(0, 1, 0.0, 1.0, 0.0);
    sliced->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    const double cutoff = 2.0;
    sliced->setCutoffDistance(cutoff);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    system.addForce(sliced);
    ASSERT(sliced->usesPeriodicBoundaryConditions());
    ASSERT(system.usesPeriodicBoundaryConditions());
    Context context(system, integrator, platform);
    vector<Vec3> positions(3);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(2, 0, 0);
    positions[2] = Vec3(3, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    const vector<Vec3>& forces = state.getForces();
    const double eps = 78.3;
    const double krf = (1.0/(cutoff*cutoff*cutoff))*(eps-1.0)/(2.0*eps+1.0);
    const double crf = (1.0/cutoff)*(3.0*eps)/(2.0*eps+1.0);
    const double force = ONE_4PI_EPS0*(1.0)*(1.0-2.0*krf*1.0);
    assertEqualVec(Vec3(force, 0, 0), forces[0], TOL);
    assertEqualVec(Vec3(-force, 0, 0), forces[1], TOL);
    assertEqualVec(Vec3(0, 0, 0), forces[2], TOL);
    assertEqualTo(2*ONE_4PI_EPS0*(1.0)*(1.0+krf*1.0-crf), state.getPotentialEnergy(), TOL);
}

void testPeriodicExceptions() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(1);
    sliced->addParticle(1.0, 1, 0);
    sliced->addParticle(1.0, 1, 0);
    sliced->addException(0, 1, 1.0, 1.0, 0.0);
    sliced->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    const double cutoff = 2.0;
    sliced->setCutoffDistance(cutoff);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    system.addForce(sliced);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(3, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    vector<Vec3> forces = state.getForces();
    double force = ONE_4PI_EPS0/(3*3);
    assertEqualVec(Vec3(-force, 0, 0), forces[0], TOL);
    assertEqualVec(Vec3(force, 0, 0), forces[1], TOL);
    assertEqualTo(ONE_4PI_EPS0/3, state.getPotentialEnergy(), TOL);

    // Now make exceptions periodic and see if it changes correctly.

    sliced->setExceptionsUsePeriodicBoundaryConditions(true);
    context.reinitialize(true);
    state = context.getState(State::Forces | State::Energy);
    forces = state.getForces();
    force = ONE_4PI_EPS0/(1*1);
    assertEqualVec(Vec3(force, 0, 0), forces[0], TOL);
    assertEqualVec(Vec3(-force, 0, 0), forces[1], TOL);
    assertEqualTo(ONE_4PI_EPS0/1, state.getPotentialEnergy(), TOL);
}

void testTriclinic() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    Vec3 a(3.1, 0, 0);
    Vec3 b(0.4, 3.5, 0);
    Vec3 c(-0.1, -0.5, 4.0);
    system.setDefaultPeriodicBoxVectors(a, b, c);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(1);
    sliced->addParticle(1.0, 1, 0);
    sliced->addParticle(1.0, 1, 0);
    sliced->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    const double cutoff = 1.5;
    sliced->setCutoffDistance(cutoff);
    system.addForce(sliced);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    const double eps = 78.3;
    const double krf = (1.0/(cutoff*cutoff*cutoff))*(eps-1.0)/(2.0*eps+1.0);
    const double crf = (1.0/cutoff)*(3.0*eps)/(2.0*eps+1.0);
    for (int iteration = 0; iteration < 50; iteration++) {
        // Generate random positions for the two particles.

        positions[0] = a*genrand_real2(sfmt) + b*genrand_real2(sfmt) + c*genrand_real2(sfmt);
        positions[1] = a*genrand_real2(sfmt) + b*genrand_real2(sfmt) + c*genrand_real2(sfmt);
        context.setPositions(positions);

        // Loop over all possible periodic copies and find the nearest one.

        Vec3 delta;
        double distance2 = 100.0;
        for (int i = -1; i < 2; i++)
            for (int j = -1; j < 2; j++)
                for (int k = -1; k < 2; k++) {
                    Vec3 d = positions[1]-positions[0]+a*i+b*j+c*k;
                    if (d.dot(d) < distance2) {
                        delta = d;
                        distance2 = d.dot(d);
                    }
                }
        double distance = sqrt(distance2);

        // See if the force and energy are correct.

        State state = context.getState(State::Forces | State::Energy);
        if (distance >= cutoff) {
            ASSERT_EQUAL(0.0, state.getPotentialEnergy());
            assertEqualVec(Vec3(0, 0, 0), state.getForces()[0], 0);
            assertEqualVec(Vec3(0, 0, 0), state.getForces()[1], 0);
        }
        else {
            const Vec3 force = delta*ONE_4PI_EPS0*(-1.0/(distance*distance*distance)+2.0*krf);
            assertEqualTo(ONE_4PI_EPS0*(1.0/distance+krf*distance*distance-crf), state.getPotentialEnergy(), 1e-4);
            assertEqualVec(force, state.getForces()[0], 1e-4);
            assertEqualVec(-force, state.getForces()[1], 1e-4);
        }
    }
}

void testLargeSystem() {
    const int numMolecules = 600;
    const int numParticles = numMolecules*2;
    const double cutoff = 2.0;
    const double boxSize = 20.0;
    const double tol = 2e-3;
    System system;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    NonbondedForce* nonbonded = new NonbondedForce();
    HarmonicBondForce* bonds = new HarmonicBondForce();
    vector<Vec3> positions(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);

    for (int i = 0; i < numMolecules; i++) {
        if (i < numMolecules/2) {
            nonbonded->addParticle(-1.0, 0.2, 0.1);
            nonbonded->addParticle(1.0, 0.1, 0.1);
        }
        else {
            nonbonded->addParticle(-1.0, 0.2, 0.2);
            nonbonded->addParticle(1.0, 0.1, 0.2);
        }
        positions[2*i] = Vec3(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
        positions[2*i+1] = Vec3(positions[2*i][0]+1.0, positions[2*i][1], positions[2*i][2]);
        bonds->addBond(2*i, 2*i+1, 1.0, 0.1);
        nonbonded->addException(2*i, 2*i+1, 0.0, 0.15, 0.0);
    }

    // Try with no cutoffs and make sure it agrees with the Reference platform.

    nonbonded->setNonbondedMethod(SlicedNonbondedForce::NoCutoff);
    nonbonded->setForceGroup(0);
    system.addForce(nonbonded);
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(*nonbonded, 1);
    sliced->setForceGroup(1);
    system.addForce(sliced);
    bonds->setForceGroup(2);
    system.addForce(bonds);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    assertForcesAndEnergy(context, TOL);

    // Now try cutoffs but not periodic boundary conditions.

    nonbonded->setNonbondedMethod(NonbondedForce::CutoffNonPeriodic);
    nonbonded->setCutoffDistance(cutoff);
    sliced->setNonbondedMethod(SlicedNonbondedForce::CutoffNonPeriodic);
    sliced->setCutoffDistance(cutoff);
    context.reinitialize(true);
    assertForcesAndEnergy(context, TOL);

    // Now do the same thing with periodic boundary conditions.

    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    sliced->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    context.reinitialize(true);
    assertForcesAndEnergy(context, TOL);
}

void testHugeSystem(double tol=1e-5) {
    // Create a system with over 3 million particles.

    const int gridSize = 150;
    const int numParticles = gridSize*gridSize*gridSize;
    const double spacing = 0.3;
    const double boxSize = gridSize*spacing;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    SlicedNonbondedForce* force = new SlicedNonbondedForce(1);
    system.addForce(force);
    force->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    force->setCutoffDistance(1.0);
    force->setUseSwitchingFunction(true);
    force->setSwitchingDistance(0.9);
    vector<Vec3> positions;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            for (int k = 0; k < gridSize; k++) {
                system.addParticle(1.0);
                force->addParticle(0.0, 0.1, 1.0);
                positions.push_back(Vec3(i*spacing+genrand_real2(sfmt)*0.1, j*spacing+genrand_real2(sfmt)*0.1, k*spacing+genrand_real2(sfmt)*0.1));
            }
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);

    // Compute the norm of the force.

    State state = context.getState(State::Forces);
    double norm = 0.0;
    for (int i = 0; i < numParticles; ++i) {
        Vec3 f = state.getForces()[i];
        norm += f[0]*f[0] + f[1]*f[1] + f[2]*f[2];
    }
    norm = sqrt(norm);

    // Take a small step in the direction of the energy gradient and see whether the potential energy changes by the expected amount.

    const double delta = 0.3;
    double step = 0.5*delta/norm;
    vector<Vec3> positions2(numParticles), positions3(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        Vec3 p = positions[i];
        Vec3 f = state.getForces()[i];
        positions2[i] = Vec3(p[0]-f[0]*step, p[1]-f[1]*step, p[2]-f[2]*step);
        positions3[i] = Vec3(p[0]+f[0]*step, p[1]+f[1]*step, p[2]+f[2]*step);
    }
    context.setPositions(positions2);
    State state2 = context.getState(State::Energy);
    context.setPositions(positions3);
    State state3 = context.getState(State::Energy);
    assertEqualTo(state2.getPotentialEnergy(), state3.getPotentialEnergy()+norm*delta, tol)
}

void testDispersionCorrection() {
    // Create a box full of identical particles.
    int gridSize = 5;
    int numParticles = gridSize*gridSize*gridSize;
    double boxSize = gridSize*0.7;
    double cutoff = boxSize/3;
    double tol = (platform.getName() == "Reference" || platform.getPropertyDefaultValue("Precision") == "double") ? 1e-5 : 1e-3;
    System system;
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(1);
    vector<Vec3> positions(numParticles);
    int index = 0;
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            for (int k = 0; k < gridSize; k++) {
                system.addParticle(1.0);
                sliced->addParticle(0, 1.1, 0.5);
                positions[index] = Vec3(i*boxSize/gridSize, j*boxSize/gridSize, k*boxSize/gridSize);
                index++;
            }
    sliced->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    sliced->setCutoffDistance(cutoff);
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    system.addForce(sliced);

    // See if the correction has the correct value.

    Context context(system, integrator, platform);
    context.setPositions(positions);
    double energy1 = context.getState(State::Energy).getPotentialEnergy();
    sliced->setUseDispersionCorrection(false);
    context.reinitialize();
    context.setPositions(positions);
    double energy2 = context.getState(State::Energy).getPotentialEnergy();
    double term1 = (0.5*pow(1.1, 12)/pow(cutoff, 9))/9;
    double term2 = (0.5*pow(1.1, 6)/pow(cutoff, 3))/3;
    double expected = 8*M_PI*numParticles*numParticles*(term1-term2)/(boxSize*boxSize*boxSize);
    assertEqualTo(expected, energy1-energy2, tol);

    // Now modify half the particles to be different, and see if it is still correct.

    int numType2 = 0;
    for (int i = 0; i < numParticles; i += 2) {
        sliced->setParticleParameters(i, 0, 1, 1);
        numType2++;
    }
    int numType1 = numParticles-numType2;
    sliced->updateParametersInContext(context);
    energy2 = context.getState(State::Energy).getPotentialEnergy();
    sliced->setUseDispersionCorrection(true);
    context.reinitialize();
    context.setPositions(positions);
    energy1 = context.getState(State::Energy).getPotentialEnergy();
    term1 = ((numType1*(numType1+1))/2)*(0.5*pow(1.1, 12)/pow(cutoff, 9))/9;
    term2 = ((numType1*(numType1+1))/2)*(0.5*pow(1.1, 6)/pow(cutoff, 3))/3;
    term1 += ((numType2*(numType2+1))/2)*(1*pow(1.0, 12)/pow(cutoff, 9))/9;
    term2 += ((numType2*(numType2+1))/2)*(1*pow(1.0, 6)/pow(cutoff, 3))/3;
    double combinedSigma = 0.5*(1+1.1);
    double combinedEpsilon = sqrt(1*0.5);
    term1 += (numType1*numType2)*(combinedEpsilon*pow(combinedSigma, 12)/pow(cutoff, 9))/9;
    term2 += (numType1*numType2)*(combinedEpsilon*pow(combinedSigma, 6)/pow(cutoff, 3))/3;
    term1 /= (numParticles*(numParticles+1))/2;
    term2 /= (numParticles*(numParticles+1))/2;
    expected = 8*M_PI*numParticles*numParticles*(term1-term2)/(boxSize*boxSize*boxSize);
    assertEqualTo(expected, energy1-energy2, tol);
}

void testChangingParameters() {
    const int numMolecules = 600;
    const int numParticles = numMolecules*2;
    const double cutoff = 2.0;
    const double boxSize = 20.0;
    const double tol = 2e-3;
    System system;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    NonbondedForce* nonbonded = new NonbondedForce();
    vector<Vec3> positions(numParticles);
    int M = (int) pow(numMolecules, 1.0/3.0);
    if (M*M*M < numMolecules) M++;
    const double sqrt3 = sqrt(3);
    for (int k = 0; k < numMolecules; k++) {
        int iz = k/(M*M);
        int iy = (k - iz*M*M)/M;
        int ix = k - M*(iy + iz*M);
        double x = (ix + 0.5)*boxSize/M;
        double y = (iy + 0.5)*boxSize/M;
        double z = (iz + 0.5)*boxSize/M;
        double dx = (0.5 - ix%2)/2;
        double dy = (0.5 - iy%2)/2;
        double dz = (0.5 - iz%2)/2;
        if (k < numMolecules/2) {
            nonbonded->addParticle(-1.0, 0.2, 0.1);
            nonbonded->addParticle(1.0, 0.1, 0.1);
        }
        else {
            nonbonded->addParticle(-1.0, 0.2, 0.2);
            nonbonded->addParticle(1.0, 0.1, 0.2);
        }
        positions[2*k] = Vec3(x+dx, y+dy, z+dz);
        positions[2*k+1] = Vec3(x-dx, y-dy, z-dz);
        system.addConstraint(2*k, 2*k+1, 1.0);
        nonbonded->addException(2*k, 2*k+1, 0.0, 0.15, 0.0);
    }
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->setCutoffDistance(cutoff);
    nonbonded->setForceGroup(0);
    system.addForce(nonbonded);
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(*nonbonded, 1);
    sliced->setForceGroup(1);
    system.addForce(sliced);
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));

    // See if the forces and energies match the Reference platform.

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    assertForcesAndEnergy(context, TOL);

    // Now modify parameters and see if they still agree.

    for (int i = 0; i < numParticles; i += 5) {
        double charge, sigma, epsilon;
        nonbonded->getParticleParameters(i, charge, sigma, epsilon);
        nonbonded->setParticleParameters(i, 1.5*charge, 1.1*sigma, 1.7*epsilon);
        sliced->getParticleParameters(i, charge, sigma, epsilon);
        sliced->setParticleParameters(i, 1.5*charge, 1.1*sigma, 1.7*epsilon);
    }
    nonbonded->updateParametersInContext(context);
    sliced->updateParametersInContext(context);
    assertForcesAndEnergy(context, TOL);
}

void testSwitchingFunction(SlicedNonbondedForce::NonbondedMethod method) {
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(6, 0, 0), Vec3(0, 6, 0), Vec3(0, 0, 6));
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(1);
    sliced->addParticle(0, 1.2, 1);
    sliced->addParticle(0, 1.4, 2);
    sliced->setNonbondedMethod(method);
    sliced->setCutoffDistance(2.0);
    sliced->setUseSwitchingFunction(true);
    sliced->setSwitchingDistance(1.5);
    sliced->setUseDispersionCorrection(false);
    system.addForce(sliced);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    double eps = SQRT_TWO;

    // Compute the interaction at various distances.

    for (double r = 1.0; r < 2.5; r += 0.1) {
        positions[1] = Vec3(r, 0, 0);
        context.setPositions(positions);
        State state = context.getState(State::Forces | State::Energy);

        // See if the energy is correct.

        double x = 1.3/r;
        double expectedEnergy = 4.0*eps*(pow(x, 12.0)-pow(x, 6.0));
        double switchValue;
        if (r <= 1.5)
            switchValue = 1;
        else if (r >= 2.0)
            switchValue = 0;
        else {
            double t = (r-1.5)/0.5;
            switchValue = 1+t*t*t*(-10+t*(15-t*6));
        }
        assertEqualTo(switchValue*expectedEnergy, state.getPotentialEnergy(), TOL);

        // See if the force is the gradient of the energy.

        double delta = 1e-3;
        positions[1] = Vec3(r-delta, 0, 0);
        context.setPositions(positions);
        double e1 = context.getState(State::Energy).getPotentialEnergy();
        positions[1] = Vec3(r+delta, 0, 0);
        context.setPositions(positions);
        double e2 = context.getState(State::Energy).getPotentialEnergy();
        assertEqualTo((e2-e1)/(2*delta), state.getForces()[0][0], 1e-3);
    }
}

void testTwoForces() {
    // Create a system with two SlicedNonbondedForces.

    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* nb1 = new SlicedNonbondedForce(1);
    nb1->addParticle(-1.5, 1, 1.2);
    nb1->addParticle(0.5, 1, 1.0);
    system.addForce(nb1);
    SlicedNonbondedForce* nb2 = new SlicedNonbondedForce(1);
    nb2->addParticle(0.4, 1.4, 0.5);
    nb2->addParticle(0.3, 1.8, 1.0);
    nb2->setForceGroup(1);
    system.addForce(nb2);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(1.5, 0, 0);
    context.setPositions(positions);
    State state1 = context.getState(State::Energy, false, 1<<0);
    assertEqualTo(ONE_4PI_EPS0*(-1.5*0.5)/1.5 + 4.0*sqrt(1.2*1.0)*(pow(1.0/1.5, 12.0)-pow(1.0/1.5, 6.0)), state1.getPotentialEnergy(), TOL);
    State state2 = context.getState(State::Energy, false, 1<<1);
    assertEqualTo(ONE_4PI_EPS0*(0.4*0.3)/1.5 + 4.0*sqrt(0.5*1.0)*(pow(1.6/1.5, 12.0)-pow(1.6/1.5, 6.0)), state2.getPotentialEnergy(), TOL);
    State state = context.getState(State::Energy);
    assertEqualTo(state1.getPotentialEnergy()+state2.getPotentialEnergy(), state.getPotentialEnergy(), TOL);

    // Try modifying them and see if they're still correct.

    nb1->setParticleParameters(0, -1.2, 1.1, 1.4);
    nb1->updateParametersInContext(context);
    nb2->setParticleParameters(0, 0.5, 1.6, 0.6);
    nb2->updateParametersInContext(context);
    state1 = context.getState(State::Energy, false, 1<<0);
    assertEqualTo(ONE_4PI_EPS0*(-1.2*0.5)/1.5 + 4.0*sqrt(1.4*1.0)*(pow(1.05/1.5, 12.0)-pow(1.05/1.5, 6.0)), state1.getPotentialEnergy(), TOL);
    state2 = context.getState(State::Energy, false, 1<<1);
    assertEqualTo(ONE_4PI_EPS0*(0.5*0.3)/1.5 + 4.0*sqrt(0.6*1.0)*(pow(1.7/1.5, 12.0)-pow(1.7/1.5, 6.0)), state2.getPotentialEnergy(), TOL);

    // Make sure it also works with PME.

    nb1->setNonbondedMethod(SlicedNonbondedForce::PME);
    nb2->setNonbondedMethod(SlicedNonbondedForce::PME);
    context.reinitialize(true);
    state1 = context.getState(State::Energy, false, 1<<0);
    state2 = context.getState(State::Energy, false, 1<<1);
    state = context.getState(State::Energy);
    assertEqualTo(state1.getPotentialEnergy()+state2.getPotentialEnergy(), state.getPotentialEnergy(), TOL);
}

void testParameterOffsets() {
    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    SlicedNonbondedForce* force = new SlicedNonbondedForce(1);
    force->addParticle(0.0, 1.0, 0.5);
    force->addParticle(1.0, 0.5, 0.6);
    force->addParticle(-1.0, 2.0, 0.7);
    force->addParticle(0.5, 2.0, 0.8);
    force->addException(0, 3, 0.0, 1.0, 0.0);
    force->addException(2, 3, 0.5, 1.0, 1.5);
    force->addException(0, 1, 1.0, 1.5, 1.0);
    force->addGlobalParameter("p1", 0.0);
    force->addGlobalParameter("p2", 1.0);
    force->addParticleParameterOffset("p1", 0, 3.0, 0.5, 0.5);
    force->addParticleParameterOffset("p2", 1, 1.0, 1.0, 2.0);
    force->addExceptionParameterOffset("p1", 1, 0.5, 0.5, 1.5);
    system.addForce(force);
    vector<Vec3> positions(4);
    for (int i = 0; i < 4; i++)
        positions[i] = Vec3(i, 0, 0);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    ASSERT_EQUAL(2, context.getParameters().size());
    ASSERT_EQUAL(0.0, context.getParameter("p1"));
    ASSERT_EQUAL(1.0, context.getParameter("p2"));
    context.setParameter("p1", 0.5);
    context.setParameter("p2", 1.5);

    // Compute the expected parameters for the six interactions.

    vector<double> particleCharge = {0.0+3.0*0.5, 1.0+1.0*1.5, -1.0, 0.5};
    vector<double> particleSigma = {1.0+0.5*0.5, 0.5+1.0*1.5, 2.0, 2.0};
    vector<double> particleEpsilon = {0.5+0.5*0.5, 0.6+2.0*1.5, 0.7, 0.8};
    double pairChargeProd[4][4], pairSigma[4][4], pairEpsilon[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = i+1; j < 4; j++) {
            pairChargeProd[i][j] = particleCharge[i]*particleCharge[j];
            pairSigma[i][j] = 0.5*(particleSigma[i]+particleSigma[j]);
            pairEpsilon[i][j] = sqrt(particleEpsilon[i]*particleEpsilon[j]);
        }
    pairChargeProd[0][3] = 0.0;
    pairSigma[0][3] = 1.0;
    pairEpsilon[0][3] = 0.0;
    pairChargeProd[2][3] = 0.5+0.5*0.5;
    pairSigma[2][3] = 1.0+0.5*0.5;
    pairEpsilon[2][3] = 1.5+1.5*0.5;
    pairChargeProd[0][1] = 1.0;
    pairSigma[0][1] = 1.5;
    pairEpsilon[0][1] = 1.0;

    // Compute the expected energy.

    double energy = 0.0;
    for (int i = 0; i < 4; i++)
        for (int j = i+1; j < 4; j++) {
            double dist = j-i;
            double x = pairSigma[i][j]/dist;
            energy += ONE_4PI_EPS0*pairChargeProd[i][j]/dist + 4.0*pairEpsilon[i][j]*(pow(x, 12.0)-pow(x, 6.0));
        }
    assertEqualTo(energy, context.getState(State::Energy).getPotentialEnergy(), 1e-5);
}

void testEwaldExceptions() {
    // Create a minimal system using LJPME.

    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    SlicedNonbondedForce* force = new SlicedNonbondedForce(1);
    system.addForce(force);
    force->setNonbondedMethod(SlicedNonbondedForce::LJPME);
    force->setCutoffDistance(1.0);
    force->addParticle(1.0, 0.5, 1.0);
    force->addParticle(1.0, 0.5, 1.0);
    force->addParticle(-1.0, 0.5, 1.0);
    force->addParticle(-1.0, 0.5, 1.0);
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

    force->addException(0, 1, 0.2, 0.8, 2.0);
    force->setExceptionsUsePeriodicBoundaryConditions(true);
    context.reinitialize(true);
    double e2 = context.getState(State::Energy).getPotentialEnergy();
    double r = 0.5;
    double expectedChange = ONE_4PI_EPS0*(0.2-1.0)/r + 4*2.0*(pow(0.8/r, 12)-pow(0.8/r, 6)) - 4*1.0*(pow(0.5/r, 12)-pow(0.5/r, 6));
    assertEqualTo(expectedChange, e2-e1, 1e-5);
}

void testDirectAndReciprocal() {
    // Create a minimal system with direct space and reciprocal space in different force groups.

    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    SlicedNonbondedForce* force = new SlicedNonbondedForce(1);
    system.addForce(force);
    force->setNonbondedMethod(SlicedNonbondedForce::PME);
    force->setCutoffDistance(1.0);
    force->setReciprocalSpaceForceGroup(1);
    force->addParticle(1.0, 0.5, 1.0);
    force->addParticle(1.0, 0.5, 1.0);
    force->addParticle(-1.0, 0.5, 1.0);
    force->addParticle(-1.0, 0.5, 1.0);
    force->addException(0, 2, -2.0, 0.5, 3.0);
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
    assertEqualTo(e1, e2+e3, 1e-5);
    ASSERT(e2 != 0);
    ASSERT(e3 != 0);

    // Completely disable the direct space calculation.

    force->setIncludeDirectSpace(false);
    context.reinitialize(true);
    double e4 = context.getState(State::Energy).getPotentialEnergy();
    assertEqualTo(e3, e4, 1e-5);
}

void testNonbondedSlicing(OpenMM_SFMT::SFMT& sfmt, NonbondedForce::NonbondedMethod method, bool exceptions, bool lj) {
    bool includeLJ = lj;
    bool includeCoulomb = !lj;

    const int numMolecules = 100;
    const int numParticles = numMolecules*2;
    const double cutoff = 3.5;
    const double L = exceptions ? 7.0 : 10.0;
    double tol = (platform.getName() == "Reference" || platform.getPropertyDefaultValue("Precision") == "double") ? 1e-5 : 1e-3;

    System system1, system2;
    for (int i = 0; i < numParticles; i++) {
        system1.addParticle(1.0);
        system2.addParticle(1.0);
    }
    system1.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    system2.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));

    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(method);
    nonbonded->setCutoffDistance(cutoff);
    nonbonded->setUseDispersionCorrection(true);
    nonbonded->setReciprocalSpaceForceGroup(1);
    nonbonded->setEwaldErrorTolerance(1e-4);
    vector<Vec3> positions(numParticles);

    auto q = [](int k) { return 1-2*(k%2); };
    double qiqj = -0.5;
    double eps = 1;
    double epsij = 2;

    int M = (int) pow(numMolecules, 1.0/3.0);
    if (M*M*M < numMolecules)
        M++;
    for (int k = 0; k < numMolecules; k++) {
        int iz = k/(M*M);
        int iy = (k - iz*M*M)/M;
        int ix = k - M*(iy + iz*M);
        Vec3 center = Vec3(ix+0.5, iy+0.5, iz+0.5)*L/M;
        Vec3 delta = Vec3(0.5-ix%2, 0.5-iy%2, 0.5-iz%2)/2;
        int i = 2*k, j = i+1;
        positions[i] = center + delta;
        positions[j] = center - delta;
        nonbonded->addParticle(q(i), 1, eps);
        nonbonded->addParticle(q(j), 1, eps);
        if (exceptions)
            nonbonded->addException(i, j, qiqj, 1, epsij);
    }

    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(*nonbonded, 2);

    for (int k = 0; k < numParticles; k++)
        if (genrand_real2(sfmt) < 0.5)
            sliced->setParticleSubset(k, 1);

    string param1 = includeCoulomb ? "lambda" : "sqrtLambda";
    sliced->addGlobalParameter(param1, 1);
    sliced->addScalingParameter(param1, 0, 1, includeCoulomb, includeLJ);

    string param2 = includeCoulomb ? "lambdaSq" : "lambda";
    sliced->addGlobalParameter(param2, 1);
    sliced->addScalingParameter(param2, 1, 1, includeCoulomb, includeLJ);

    system1.addForce(nonbonded);
    system2.addForce(sliced);

    vector<pair<string, string>> particleScale(numParticles, make_pair("one", "one"));
    for (int k = 0; k < numParticles; k++)
        if (sliced->getParticleSubset(k) == 1)
            particleScale[k] = make_pair(includeCoulomb ? "lambda" : "one", includeLJ ? "lambda" : "one");

    int numExceptions = nonbonded->getNumExceptions();
    vector<pair<string, string>> exceptionScale(numExceptions, make_pair("one", "one"));
    for (int k = 0; k < numExceptions; k++) {
        int i, j;
        double chargeProd, sigma, epsilon;
        nonbonded->getExceptionParameters(k, i, j, chargeProd, sigma, epsilon);
        int si = sliced->getParticleSubset(i);
        int sj = sliced->getParticleSubset(j);
        if (si != sj || si == 1) {
            string parameter = si != sj ? param1 : param2;
            exceptionScale[k] = make_pair(includeCoulomb ? parameter : "one", includeLJ ? parameter : "one");
        }
    }

    VerletIntegrator integrator1(0.01);
    Context context1(system1, integrator1, platform);
    context1.setPositions(positions);

    VerletIntegrator integrator2(0.01);
    Context context2(system2, integrator2, platform);
    context2.setPositions(positions);

    // Direct space

    State state1 = context1.getState(State::Energy | State::Forces, false, 1<<0);
    State state2 = context2.getState(State::Energy | State::Forces, false, 1<<0);
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);

    // Reciprocal space

    state1 = context1.getState(State::Energy | State::Forces, false, 1<<1);
    state2 = context2.getState(State::Energy | State::Forces, false, 1<<1);
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);

    // Overall

    state1 = context1.getState(State::Energy | State::Forces);
    state2 = context2.getState(State::Energy | State::Forces);
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);

    double energy_lambda_one = state1.getPotentialEnergy();

    // Change of scaling parameter value

    map<string, double> value;
    value["one"] = 1;
    value["lambda"] = value["sqrtLambda"] = value["lambdaSq"] = 0;
    for (int k = 0; k < numParticles; k++)
        nonbonded->setParticleParameters(k, q(k)*value[particleScale[k].first], 1, eps*value[particleScale[k].second]);
    for (int k = 0; k < numExceptions; k++)
        nonbonded->setExceptionParameters(k, 2*k, 2*k+1, qiqj*value[exceptionScale[k].first], 1, epsij*value[exceptionScale[k].second]);
    nonbonded->updateParametersInContext(context1);
    context2.setParameter(param1, value[param1]);
    context2.setParameter(param2, value[param2]);

    state1 = context1.getState(State::Energy | State::Forces);
    state2 = context2.getState(State::Energy | State::Forces);
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);

    double energy_lambda_zero = state1.getPotentialEnergy();

    value["lambda"] = 0.5;
    value["sqrtLambda"] = sqrt(value["lambda"]);
    value["lambdaSq"] = value["lambda"]*value["lambda"];
    for (int k = 0; k < numParticles; k++)
        nonbonded->setParticleParameters(k, q(k)*value[particleScale[k].first], 1, eps*value[particleScale[k].second]);
    for (int k = 0; k < numExceptions; k++)
        nonbonded->setExceptionParameters(k, 2*k, 2*k+1, qiqj*value[exceptionScale[k].first], 1, epsij*value[exceptionScale[k].second]);
    nonbonded->updateParametersInContext(context1);
    context2.setParameter(param1, value[param1]);
    context2.setParameter(param2, value[param2]);

    state1 = context1.getState(State::Energy | State::Forces);
    state2 = context2.getState(State::Energy | State::Forces);
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);

    // Derivatives

    sliced->addScalingParameterDerivative(param1);
    sliced->addScalingParameterDerivative(param2);
    context2.reinitialize(true);
    state2 = context2.getState(State::ParameterDerivatives);
    map<string, double> derivatives = state2.getEnergyParameterDerivatives();
    assertEqualTo(energy_lambda_one - energy_lambda_zero, derivatives[param1]+derivatives[param2], tol);

    // Sum of derivatives

    for (int k = 0; k < nonbonded->getNumParticles(); k++)
        nonbonded->setParticleParameters(k, includeCoulomb ? q(k) : 0, 1, includeLJ ? eps : 0);
    for (int k = 0; k < nonbonded->getNumExceptions(); k++)
        nonbonded->setExceptionParameters(k, 2*k, 2*k+1, includeCoulomb ? qiqj : 0, 1, includeLJ ? epsij : 0);
    nonbonded->updateParametersInContext(context1);
    state1 = context1.getState(State::Energy | State::Forces);
    double energy = state1.getPotentialEnergy();

    sliced->addGlobalParameter("remainder", 1.0);
    sliced->addScalingParameter("remainder", 0, 0, includeCoulomb, includeLJ);
    sliced->addScalingParameterDerivative("remainder");
    context2.reinitialize(true);
    state2 = context2.getState(State::Energy | State::ParameterDerivatives);
    derivatives = state2.getEnergyParameterDerivatives();
    double sum = derivatives[param1]+derivatives[param2]+derivatives["remainder"];
    assertEqualTo(energy, sum, tol);
}

void testScalingParameterSeparation(OpenMM_SFMT::SFMT& sfmt, NonbondedForce::NonbondedMethod method, bool exceptions) {
    const int numMolecules = 100;
    const int numParticles = numMolecules*2;
    const double cutoff = 3.5;
    const double L = exceptions ? 7.0 : 10.0;
    double tol = (platform.getName() == "Reference" || platform.getPropertyDefaultValue("Precision") == string("double")) ? 1e-5 : 1e-3;

    System system1, system2;
    for (int i = 0; i < numParticles; i++) {
        system1.addParticle(1.0);
        system2.addParticle(1.0);
    }
    system1.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    system2.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));

    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->setNonbondedMethod(method);
    nonbonded->setCutoffDistance(cutoff);
    nonbonded->setUseDispersionCorrection(true);
    nonbonded->setReciprocalSpaceForceGroup(1);
    nonbonded->setEwaldErrorTolerance(1e-4);
    vector<Vec3> positions(numParticles);

    auto q = [](int k) { return 1-2*(k%2); };
    double qiqj = -0.5;
    double eps = 1;
    double epsij = 2;

    int M = (int) pow(numMolecules, 1.0/3.0);
    if (M*M*M < numMolecules)
        M++;
    for (int k = 0; k < numMolecules; k++) {
        int iz = k/(M*M);
        int iy = (k - iz*M*M)/M;
        int ix = k - M*(iy + iz*M);
        Vec3 center = Vec3(ix+0.5, iy+0.5, iz+0.5)*L/M;
        Vec3 delta = Vec3(0.5-ix%2, 0.5-iy%2, 0.5-iz%2)/2;
        int i = 2*k, j = i+1;
        positions[i] = center + delta;
        positions[j] = center - delta;
        nonbonded->addParticle(q(i), 1, eps);
        nonbonded->addParticle(q(j), 1, eps);
        if (exceptions)
            nonbonded->addException(i, j, qiqj, 1, epsij);
    }

    SlicedNonbondedForce* sliced1 = new SlicedNonbondedForce(*nonbonded, 2);
    SlicedNonbondedForce* sliced2 = new SlicedNonbondedForce(*nonbonded, 2);

    for (int k = 0; k < numParticles; k++)
        if (genrand_real2(sfmt) < 0.5) {
            sliced1->setParticleSubset(k, 1);
            sliced2->setParticleSubset(k, 1);
        }

    double lambda = 0.5;
    sliced1->addGlobalParameter("lambda", lambda);
    sliced1->addScalingParameter("lambda", 0, 1, true, true);
    sliced1->addScalingParameterDerivative("lambda");
    sliced2->addGlobalParameter("lambdaCoulomb", lambda);
    sliced2->addGlobalParameter("lambdaLJ", lambda);
    sliced2->addScalingParameter("lambdaCoulomb", 0, 1, true, false);
    sliced2->addScalingParameter("lambdaLJ", 0, 1, false, true);
    sliced2->addScalingParameterDerivative("lambdaCoulomb");
    sliced2->addScalingParameterDerivative("lambdaLJ");

    double value = 0.3;
    sliced1->addGlobalParameter("alpha", value);
    sliced1->addScalingParameter("alpha", 0, 0, true, true);
    sliced1->addScalingParameterDerivative("alpha");

    sliced1->addGlobalParameter("beta", value);
    sliced1->addScalingParameter("beta", 1, 1, true, true);
    sliced1->addScalingParameterDerivative("beta");

    sliced2->addGlobalParameter("gamma", value);
    sliced2->addScalingParameter("gamma", 0, 0, true, true);
    sliced2->addScalingParameter("gamma", 1, 1, true, true);
    sliced2->addScalingParameterDerivative("gamma");

    system1.addForce(sliced1);
    system2.addForce(sliced2);

    VerletIntegrator integrator1(0.01);
    Context context1(system1, integrator1, platform);
    context1.setPositions(positions);

    VerletIntegrator integrator2(0.01);
    Context context2(system2, integrator2, platform);
    context2.setPositions(positions);

    // Overall

    State state1 = context1.getState(State::Energy | State::Forces | State::ParameterDerivatives);
    State state2 = context2.getState(State::Energy | State::Forces | State::ParameterDerivatives);
    map<string, double> derivatives1 = state1.getEnergyParameterDerivatives();
    map<string, double> derivatives2 = state2.getEnergyParameterDerivatives();
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);
    assertEqualTo(derivatives1["lambda"], derivatives2["lambdaCoulomb"]+derivatives2["lambdaLJ"], tol);
    assertEqualTo(state1.getPotentialEnergy(), lambda*derivatives1["lambda"]+value*(derivatives1["alpha"]+derivatives1["beta"]), tol);
    assertEqualTo(derivatives1["alpha"]+derivatives1["beta"], derivatives2["gamma"], tol);

    // Direct space

    state1 = context1.getState(State::Energy | State::Forces | State::ParameterDerivatives, false, 1<<0);
    state2 = context2.getState(State::Energy | State::Forces | State::ParameterDerivatives, false, 1<<0);
    derivatives1 = state1.getEnergyParameterDerivatives();
    derivatives2 = state2.getEnergyParameterDerivatives();
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);
    assertEqualTo(derivatives1["lambda"], derivatives2["lambdaCoulomb"]+derivatives2["lambdaLJ"], tol);
    assertEqualTo(state1.getPotentialEnergy(), lambda*derivatives1["lambda"]+value*(derivatives1["alpha"]+derivatives1["beta"]), tol);
    assertEqualTo(derivatives1["alpha"]+derivatives1["beta"], derivatives2["gamma"], tol);

    // // Reciprocal space

    state1 = context1.getState(State::Energy | State::Forces | State::ParameterDerivatives, false, 1<<1);
    state2 = context2.getState(State::Energy | State::Forces | State::ParameterDerivatives, false, 1<<1);
    derivatives1 = state1.getEnergyParameterDerivatives();
    derivatives2 = state2.getEnergyParameterDerivatives();
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);
    assertEqualTo(derivatives1["lambda"], derivatives2["lambdaCoulomb"]+derivatives2["lambdaLJ"], tol);
    assertEqualTo(state1.getPotentialEnergy(), lambda*derivatives1["lambda"]+value*(derivatives1["alpha"]+derivatives1["beta"]), tol);
    assertEqualTo(derivatives1["alpha"]+derivatives1["beta"], derivatives2["gamma"], tol);
}

void runPlatformTests();

int main(int argc, char* argv[]) {
    vector<NonbondedForce::NonbondedMethod> nonbondedMethods = {
        NonbondedForce::NoCutoff,
        NonbondedForce::CutoffNonPeriodic,
        NonbondedForce::CutoffPeriodic,
        NonbondedForce::Ewald,
        NonbondedForce::PME,
        NonbondedForce::LJPME
    };
    vector <bool> booleanValues = {false, true};
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    try {
        initializeTests(argc, argv);
        for (auto method : nonbondedMethods)
            testInstantiateFromNonbondedForce(method);
        testCoulomb();
        testLJ();
        testExclusionsAnd14();
        testCutoff();
        testCutoff14();
        testPeriodic();
        testPeriodicExceptions();
        testTriclinic();
        testLargeSystem();
        testDispersionCorrection();
        testChangingParameters();
        testSwitchingFunction(SlicedNonbondedForce::CutoffNonPeriodic);
        testSwitchingFunction(SlicedNonbondedForce::PME);
        testTwoForces();
        testParameterOffsets();
        testEwaldExceptions();
        testDirectAndReciprocal();
        for (auto method : nonbondedMethods)
            for (auto exceptions : booleanValues) {
                for (auto lj : booleanValues)
                    testNonbondedSlicing(sfmt, method, exceptions, lj);
                testScalingParameterSeparation(sfmt, method, exceptions);
            }
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
