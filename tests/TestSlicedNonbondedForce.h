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
#include "openmm/NonbondedForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "sfmt/SFMT.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace PmeSlicing;
using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;

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

    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(*force, 1);
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

    assertForcesAndEnergy(context);
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
    ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(-0.75)/2.0, state.getPotentialEnergy(), TOL);
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
    double force = 4.0*eps*(12*std::pow(x, 12.0)-6*std::pow(x, 6.0))/2.0;
    ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_TOL(4.0*eps*(std::pow(x, 12.0)-std::pow(x, 6.0)), state.getPotentialEnergy(), TOL);
}

void testExclusionsAnd14() {
    System system;
    SlicedNonbondedForce* slicedNonbonded = new SlicedNonbondedForce(1);
    for (int i = 0; i < 5; ++i) {
        system.addParticle(1.0);
        slicedNonbonded->addParticle(0, 1.5, 0);
    }
    vector<pair<int, int> > bonds;
    bonds.push_back(pair<int, int>(0, 1));
    bonds.push_back(pair<int, int>(1, 2));
    bonds.push_back(pair<int, int>(2, 3));
    bonds.push_back(pair<int, int>(3, 4));
    slicedNonbonded->createExceptionsFromBonds(bonds, 0.0, 0.0);
    int first14, second14;
    for (int i = 0; i < slicedNonbonded->getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        slicedNonbonded->getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if ((particle1 == 0 && particle2 == 3) || (particle1 == 3 && particle2 == 0))
            first14 = i;
        if ((particle1 == 1 && particle2 == 4) || (particle1 == 4 && particle2 == 1))
            second14 = i;
    }
    system.addForce(slicedNonbonded);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    for (int i = 1; i < 5; ++i) {

        // Test LJ forces

        vector<Vec3> positions(5);
        const double r = 1.0;
        for (int j = 0; j < 5; ++j) {
            slicedNonbonded->setParticleParameters(j, 0, 1.5, 0);
            positions[j] = Vec3(0, j, 0);
        }
        slicedNonbonded->setParticleParameters(0, 0, 1.5, 1);
        slicedNonbonded->setParticleParameters(i, 0, 1.5, 1);
        slicedNonbonded->setExceptionParameters(first14, 0, 3, 0, 1.5, i == 3 ? 0.5 : 0.0);
        slicedNonbonded->setExceptionParameters(second14, 1, 4, 0, 1.5, 0.0);
        positions[i] = Vec3(r, 0, 0);
        context.reinitialize();
        context.setPositions(positions);
        State state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces = state.getForces();
        double x = 1.5/r;
        double eps = 1.0;
        double force = 4.0*eps*(12*std::pow(x, 12.0)-6*std::pow(x, 6.0))/r;
        double energy = 4.0*eps*(std::pow(x, 12.0)-std::pow(x, 6.0));
        if (i == 3) {
            force *= 0.5;
            energy *= 0.5;
        }
        if (i < 3) {
            force = 0;
            energy = 0;
        }
        ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[0], TOL);
        ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[i], TOL);
        ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), TOL);

        // Test Coulomb forces

        slicedNonbonded->setParticleParameters(0, 2, 1.5, 0);
        slicedNonbonded->setParticleParameters(i, 2, 1.5, 0);
        slicedNonbonded->setExceptionParameters(first14, 0, 3, i == 3 ? 4/1.2 : 0, 1.5, 0);
        slicedNonbonded->setExceptionParameters(second14, 1, 4, 0, 1.5, 0);
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
        ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces2[0], TOL);
        ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces2[i], TOL);
        ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), TOL);
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
    ASSERT_EQUAL_VEC(Vec3(0, -force1, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(0, force1-force2, 0), forces[1], TOL);
    ASSERT_EQUAL_VEC(Vec3(0, force2, 0), forces[2], TOL);
    const double energy1 = ONE_4PI_EPS0*(1.0)*(0.5+krf*4.0-crf);
    const double energy2 = ONE_4PI_EPS0*(1.0)*(1.0+krf*1.0-crf);
    ASSERT_EQUAL_TOL(energy1+energy2, state.getPotentialEnergy(), TOL);
}

void testCutoff14() {
    System system;
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* slicedNonbonded = new SlicedNonbondedForce(1);
    slicedNonbonded->setNonbondedMethod(SlicedNonbondedForce::CutoffNonPeriodic);
    for (int i = 0; i < 5; i++) {
        system.addParticle(1.0);
        slicedNonbonded->addParticle(0, 1.5, 0);
    }
    const double cutoff = 3.5;
    slicedNonbonded->setCutoffDistance(cutoff);
    const double eps = 30.0;
    slicedNonbonded->setReactionFieldDielectric(eps);
    vector<pair<int, int> > bonds;
    bonds.push_back(pair<int, int>(0, 1));
    bonds.push_back(pair<int, int>(1, 2));
    bonds.push_back(pair<int, int>(2, 3));
    bonds.push_back(pair<int, int>(3, 4));
    slicedNonbonded->createExceptionsFromBonds(bonds, 0.0, 0.0);
    int first14, second14;
    for (int i = 0; i < slicedNonbonded->getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        slicedNonbonded->getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if ((particle1 == 0 && particle2 == 3) || (particle1 == 3 && particle2 == 0))
            first14 = i;
        if ((particle1 == 1 && particle2 == 4) || (particle1 == 4 && particle2 == 1))
            second14 = i;
    }
    system.addForce(slicedNonbonded);
    ASSERT(!slicedNonbonded->usesPeriodicBoundaryConditions());
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

        slicedNonbonded->setParticleParameters(0, 0, 1.5, 1);
        for (int j = 1; j < 5; ++j)
            slicedNonbonded->setParticleParameters(j, 0, 1.5, 0);
        slicedNonbonded->setParticleParameters(i, 0, 1.5, 1);
        slicedNonbonded->setExceptionParameters(first14, 0, 3, 0, 1.5, i == 3 ? 0.5 : 0.0);
        slicedNonbonded->setExceptionParameters(second14, 1, 4, 0, 1.5, 0.0);
        context.reinitialize(true);
        State state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces = state.getForces();
        double r = positions[i][0];
        double x = 1.5/r;
        double e = 1.0;
        double force = 4.0*e*(12*std::pow(x, 12.0)-6*std::pow(x, 6.0))/r;
        double energy = 4.0*e*(std::pow(x, 12.0)-std::pow(x, 6.0));
        if (i == 3) {
            force *= 0.5;
            energy *= 0.5;
        }
        if (i < 3 || r > cutoff) {
            force = 0;
            energy = 0;
        }
        ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[0], TOL);
        ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[i], TOL);
        ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), TOL);

        // Test Coulomb forces

        const double q = 0.7;
        slicedNonbonded->setParticleParameters(0, q, 1.5, 0);
        slicedNonbonded->setParticleParameters(i, q, 1.5, 0);
        slicedNonbonded->setExceptionParameters(first14, 0, 3, i == 3 ? q*q/1.2 : 0, 1.5, 0);
        slicedNonbonded->setExceptionParameters(second14, 1, 4, 0, 1.5, 0);
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
        ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces2[0], TOL);
        ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces2[i], TOL);
        ASSERT_EQUAL_TOL(energy, state.getPotentialEnergy(), TOL);
    }
}

void testPeriodic() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* slicedNonbonded = new SlicedNonbondedForce(1);
    slicedNonbonded->addParticle(1.0, 1, 0);
    slicedNonbonded->addParticle(1.0, 1, 0);
    slicedNonbonded->addParticle(1.0, 1, 0);
    slicedNonbonded->addException(0, 1, 0.0, 1.0, 0.0);
    slicedNonbonded->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    const double cutoff = 2.0;
    slicedNonbonded->setCutoffDistance(cutoff);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    system.addForce(slicedNonbonded);
    ASSERT(slicedNonbonded->usesPeriodicBoundaryConditions());
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
    ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_VEC(Vec3(0, 0, 0), forces[2], TOL);
    ASSERT_EQUAL_TOL(2*ONE_4PI_EPS0*(1.0)*(1.0+krf*1.0-crf), state.getPotentialEnergy(), TOL);
}

void testPeriodicExceptions() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* slicedNonbonded = new SlicedNonbondedForce(1);
    slicedNonbonded->addParticle(1.0, 1, 0);
    slicedNonbonded->addParticle(1.0, 1, 0);
    slicedNonbonded->addException(0, 1, 1.0, 1.0, 0.0);
    slicedNonbonded->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    const double cutoff = 2.0;
    slicedNonbonded->setCutoffDistance(cutoff);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    system.addForce(slicedNonbonded);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(3, 0, 0);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    vector<Vec3> forces = state.getForces();
    double force = ONE_4PI_EPS0/(3*3);
    ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0/3, state.getPotentialEnergy(), TOL);

    // Now make exceptions periodic and see if it changes correctly.

    slicedNonbonded->setExceptionsUsePeriodicBoundaryConditions(true);
    context.reinitialize(true);
    state = context.getState(State::Forces | State::Energy);
    forces = state.getForces();
    force = ONE_4PI_EPS0/(1*1);
    ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0/1, state.getPotentialEnergy(), TOL);
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
    SlicedNonbondedForce* slicedNonbonded = new SlicedNonbondedForce(1);
    slicedNonbonded->addParticle(1.0, 1, 0);
    slicedNonbonded->addParticle(1.0, 1, 0);
    slicedNonbonded->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    const double cutoff = 1.5;
    slicedNonbonded->setCutoffDistance(cutoff);
    system.addForce(slicedNonbonded);
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
            ASSERT_EQUAL_VEC(Vec3(0, 0, 0), state.getForces()[0], 0);
            ASSERT_EQUAL_VEC(Vec3(0, 0, 0), state.getForces()[1], 0);
        }
        else {
            const Vec3 force = delta*ONE_4PI_EPS0*(-1.0/(distance*distance*distance)+2.0*krf);
            ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(1.0/distance+krf*distance*distance-crf), state.getPotentialEnergy(), 1e-4);
            ASSERT_EQUAL_VEC(force, state.getForces()[0], 1e-4);
            ASSERT_EQUAL_VEC(-force, state.getForces()[1], 1e-4);
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
    SlicedNonbondedForce* slicedNonbonded = new SlicedNonbondedForce(*nonbonded, 1);
    slicedNonbonded->setForceGroup(1);
    system.addForce(slicedNonbonded);
    bonds->setForceGroup(2);
    system.addForce(bonds);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    assertForcesAndEnergy(context);

    // Now try cutoffs but not periodic boundary conditions.

    nonbonded->setNonbondedMethod(NonbondedForce::CutoffNonPeriodic);
    nonbonded->setCutoffDistance(cutoff);
    slicedNonbonded->setNonbondedMethod(SlicedNonbondedForce::CutoffNonPeriodic);
    slicedNonbonded->setCutoffDistance(cutoff);
    context.reinitialize(true);
    assertForcesAndEnergy(context);

    // Now do the same thing with periodic boundary conditions.

    nonbonded->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    slicedNonbonded->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    context.reinitialize(true);
    assertForcesAndEnergy(context);
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
    norm = std::sqrt(norm);

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
    ASSERT_EQUAL_TOL(state2.getPotentialEnergy(), state3.getPotentialEnergy()+norm*delta, tol)
}

void testDispersionCorrection() {
    // Create a box full of identical particles.
    int gridSize = 5;
    int numParticles = gridSize*gridSize*gridSize;
    double boxSize = gridSize*0.7;
    double cutoff = boxSize/3;
    System system;
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* slicedNonbonded = new SlicedNonbondedForce(1);
    vector<Vec3> positions(numParticles);
    int index = 0;
    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridSize; j++)
            for (int k = 0; k < gridSize; k++) {
                system.addParticle(1.0);
                slicedNonbonded->addParticle(0, 1.1, 0.5);
                positions[index] = Vec3(i*boxSize/gridSize, j*boxSize/gridSize, k*boxSize/gridSize);
                index++;
            }
    slicedNonbonded->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    slicedNonbonded->setCutoffDistance(cutoff);
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    system.addForce(slicedNonbonded);

    // See if the correction has the correct value.

    Context context(system, integrator, platform);
    context.setPositions(positions);
    double energy1 = context.getState(State::Energy).getPotentialEnergy();
    slicedNonbonded->setUseDispersionCorrection(false);
    context.reinitialize();
    context.setPositions(positions);
    double energy2 = context.getState(State::Energy).getPotentialEnergy();
    double term1 = (0.5*pow(1.1, 12)/pow(cutoff, 9))/9;
    double term2 = (0.5*pow(1.1, 6)/pow(cutoff, 3))/3;
    double expected = 8*M_PI*numParticles*numParticles*(term1-term2)/(boxSize*boxSize*boxSize);
    ASSERT_EQUAL_TOL(expected, energy1-energy2, 1e-4);

    // Now modify half the particles to be different, and see if it is still correct.

    int numType2 = 0;
    for (int i = 0; i < numParticles; i += 2) {
        slicedNonbonded->setParticleParameters(i, 0, 1, 1);
        numType2++;
    }
    int numType1 = numParticles-numType2;
    slicedNonbonded->updateParametersInContext(context);
    energy2 = context.getState(State::Energy).getPotentialEnergy();
    slicedNonbonded->setUseDispersionCorrection(true);
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
    ASSERT_EQUAL_TOL(expected, energy1-energy2, 1e-4);
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
        system.addConstraint(2*i, 2*i+1, 1.0);
        nonbonded->addException(2*i, 2*i+1, 0.0, 0.15, 0.0);
    }
    nonbonded->setNonbondedMethod(NonbondedForce::PME);
    nonbonded->setCutoffDistance(cutoff);
    nonbonded->setForceGroup(0);
    system.addForce(nonbonded);
    SlicedNonbondedForce* slicedNonbonded = new SlicedNonbondedForce(*nonbonded, 1);
    slicedNonbonded->setForceGroup(1);
    system.addForce(slicedNonbonded);
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));

    // See if the forces and energies match the Reference platform.

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    assertForcesAndEnergy(context);

    // Now modify parameters and see if they still agree.

    for (int i = 0; i < numParticles; i += 5) {
        double charge, sigma, epsilon;
        nonbonded->getParticleParameters(i, charge, sigma, epsilon);
        nonbonded->setParticleParameters(i, 1.5*charge, 1.1*sigma, 1.7*epsilon);
        slicedNonbonded->getParticleParameters(i, charge, sigma, epsilon);
        slicedNonbonded->setParticleParameters(i, 1.5*charge, 1.1*sigma, 1.7*epsilon);
    }
    nonbonded->updateParametersInContext(context);
    slicedNonbonded->updateParametersInContext(context);
    assertForcesAndEnergy(context);
}

void testSwitchingFunction(SlicedNonbondedForce::NonbondedMethod method) {
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(6, 0, 0), Vec3(0, 6, 0), Vec3(0, 0, 6));
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* slicedNonbonded = new SlicedNonbondedForce(1);
    slicedNonbonded->addParticle(0, 1.2, 1);
    slicedNonbonded->addParticle(0, 1.4, 2);
    slicedNonbonded->setNonbondedMethod(method);
    slicedNonbonded->setCutoffDistance(2.0);
    slicedNonbonded->setUseSwitchingFunction(true);
    slicedNonbonded->setSwitchingDistance(1.5);
    slicedNonbonded->setUseDispersionCorrection(false);
    system.addForce(slicedNonbonded);
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
        double expectedEnergy = 4.0*eps*(std::pow(x, 12.0)-std::pow(x, 6.0));
        double switchValue;
        if (r <= 1.5)
            switchValue = 1;
        else if (r >= 2.0)
            switchValue = 0;
        else {
            double t = (r-1.5)/0.5;
            switchValue = 1+t*t*t*(-10+t*(15-t*6));
        }
        ASSERT_EQUAL_TOL(switchValue*expectedEnergy, state.getPotentialEnergy(), TOL);

        // See if the force is the gradient of the energy.

        double delta = 1e-3;
        positions[1] = Vec3(r-delta, 0, 0);
        context.setPositions(positions);
        double e1 = context.getState(State::Energy).getPotentialEnergy();
        positions[1] = Vec3(r+delta, 0, 0);
        context.setPositions(positions);
        double e2 = context.getState(State::Energy).getPotentialEnergy();
        ASSERT_EQUAL_TOL((e2-e1)/(2*delta), state.getForces()[0][0], 1e-3);
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
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(-1.5*0.5)/1.5 + 4.0*sqrt(1.2*1.0)*(pow(1.0/1.5, 12.0)-pow(1.0/1.5, 6.0)), state1.getPotentialEnergy(), TOL);
    State state2 = context.getState(State::Energy, false, 1<<1);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(0.4*0.3)/1.5 + 4.0*sqrt(0.5*1.0)*(pow(1.6/1.5, 12.0)-pow(1.6/1.5, 6.0)), state2.getPotentialEnergy(), TOL);
    State state = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(state1.getPotentialEnergy()+state2.getPotentialEnergy(), state.getPotentialEnergy(), TOL);

    // Try modifying them and see if they're still correct.

    nb1->setParticleParameters(0, -1.2, 1.1, 1.4);
    nb1->updateParametersInContext(context);
    nb2->setParticleParameters(0, 0.5, 1.6, 0.6);
    nb2->updateParametersInContext(context);
    state1 = context.getState(State::Energy, false, 1<<0);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(-1.2*0.5)/1.5 + 4.0*sqrt(1.4*1.0)*(pow(1.05/1.5, 12.0)-pow(1.05/1.5, 6.0)), state1.getPotentialEnergy(), TOL);
    state2 = context.getState(State::Energy, false, 1<<1);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(0.5*0.3)/1.5 + 4.0*sqrt(0.6*1.0)*(pow(1.7/1.5, 12.0)-pow(1.7/1.5, 6.0)), state2.getPotentialEnergy(), TOL);

    // Make sure it also works with PME.

    nb1->setNonbondedMethod(SlicedNonbondedForce::PME);
    nb2->setNonbondedMethod(SlicedNonbondedForce::PME);
    context.reinitialize(true);
    state1 = context.getState(State::Energy, false, 1<<0);
    state2 = context.getState(State::Energy, false, 1<<1);
    state = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(state1.getPotentialEnergy()+state2.getPotentialEnergy(), state.getPotentialEnergy(), TOL);
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
    ASSERT_EQUAL_TOL(energy, context.getState(State::Energy).getPotentialEnergy(), 1e-5);
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
    ASSERT_EQUAL_TOL(expectedChange, e2-e1, 1e-5);
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
    ASSERT_EQUAL_TOL(e1, e2+e3, 1e-5);
    ASSERT(e2 != 0);
    ASSERT(e3 != 0);

    // Completely disable the direct space calculation.

    force->setIncludeDirectSpace(false);
    context.reinitialize(true);
    double e4 = context.getState(State::Energy).getPotentialEnergy();
    ASSERT_EQUAL_TOL(e3, e4, 1e-5);
}

void runPlatformTests();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testInstantiateFromNonbondedForce(NonbondedForce::NoCutoff);
        testInstantiateFromNonbondedForce(NonbondedForce::CutoffNonPeriodic);
        testInstantiateFromNonbondedForce(NonbondedForce::CutoffPeriodic);
        testInstantiateFromNonbondedForce(NonbondedForce::Ewald);
        testInstantiateFromNonbondedForce(NonbondedForce::PME);
        testInstantiateFromNonbondedForce(NonbondedForce::LJPME);
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
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
