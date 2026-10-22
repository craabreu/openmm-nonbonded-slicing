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
#include "AssertionUtilities.h"
#include "openmm/NonbondedForce.h"
#include "openmm/Context.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/CustomBondForce.h"
#include "openmm/CustomNonbondedForce.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "sfmt/SFMT.h"
#include <array>
#include <cmath>
#include <functional>
#include <limits>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

const double TOL = 1e-4;

void testInstantiateFromNonbondedForce(NonbondedForce::NonbondedMethod method) {
    NonbondedForce* force = new NonbondedForce();
    force->setCutoffDistance(2.0);
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
    force->addParticleParameterOffset("p1", 0, -2.0, 0.5, 0.5);
    force->addParticleParameterOffset("p2", 1, 1.0, 1.0, 2.0);
    force->addExceptionParameterOffset("p1", 1, 0.5, 0.5, 1.5);
    force->setReciprocalSpaceForceGroup(2);

    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(*force, 1);
    sliced->setForceGroup(1);
    sliced->setReciprocalSpaceForceGroup(3);

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

    State state = context.getState(State::Forces | State::Energy, true);

    State state1 = context.getState(State::Forces | State::Energy, true, 1<<0);
    State state2 = context.getState(State::Forces | State::Energy, true, 1<<1);
    assertEnergy(state1, state2, TOL);
    assertForces(state1, state2, TOL);

    context.setParameter("p1", 1);

    State state3 = context.getState(State::Forces | State::Energy, true, 1<<2);
    State state4 = context.getState(State::Forces | State::Energy, true, 1<<3);
    assertEnergy(state3, state4, TOL);
    assertForces(state3, state4, TOL);
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

    // Try with no cutoffs and make sure it agrees with NonbondedForce.

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

void testDispersionCorrection() {
    // Create a box full of identical particles.
    int gridSize = 5;
    int numParticles = gridSize*gridSize*gridSize;
    double boxSize = gridSize*0.7;
    double cutoff = boxSize/3;
    double tol = (
        platform.getName() == "Reference" || platform.getPropertyDefaultValue("Precision") == "double"
    ) ? 1e-4 : 1e-3;
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
    nonbonded->setReciprocalSpaceForceGroup(2);
    system.addForce(nonbonded);
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(*nonbonded, 1);
    sliced->setForceGroup(1);
    sliced->setReciprocalSpaceForceGroup(3);
    system.addForce(sliced);
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));

    // See if the forces and energies match the Reference platform.

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    State state1 = context.getState(State::Forces | State::Energy, true, 1<<0);
    State state2 = context.getState(State::Forces | State::Energy, true, 1<<1);
    assertEnergy(state1, state2, TOL);
    assertForces(state1, state2, TOL);

    State state3 = context.getState(State::Forces | State::Energy, true, 1<<2);
    State state4 = context.getState(State::Forces | State::Energy, true, 1<<3);
    assertEnergy(state3, state4, TOL);
    assertForces(state3, state4, TOL);

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

void testExceptionSlicesAfterUpdate(SlicedNonbondedForce::NonbondedMethod method) {
    const int numParticles = 12;
    const double chargeProdChange = 1.7;
    const double epsilonChange = 0.6;

    // Particles 0-5 belong to subset 0 and 6-11 to subset 1. Exceptions (0,1), (6,7) and (8,9) have
    // slices (0,0), (1,1) and (1,1), whereas (0,6) and (1,7) have slice (0,1).

    const vector<array<int, 2>> exceptions = {{0, 1}, {6, 7}, {8, 9}, {0, 6}, {1, 7}};

    auto createSystem = [&](bool modified, System& system, SlicedNonbondedForce*& sliced) {
        for (int i = 0; i < numParticles; i++)
            system.addParticle(1.0);
        sliced = new SlicedNonbondedForce(2);
        for (int i = 0; i < numParticles; i++) {
            sliced->addParticle(i%2 == 0 ? 0.5 : -0.5, 0.25 + 0.01*i, 0.4 + 0.05*(i%3));
            sliced->setParticleSubset(i, i < 6 ? 0 : 1);
        }
        for (int k = 0; k < (int) exceptions.size(); k++) {
            double chargeProd = 0.3 + 0.1*k;
            double epsilon = 0.5 + 0.1*k;
            if (modified && k == 1) {
                chargeProd *= chargeProdChange;
                epsilon *= epsilonChange;
            }
            sliced->addException(exceptions[k][0], exceptions[k][1], chargeProd, 0.2 + 0.01*k, epsilon);
        }
        sliced->setNonbondedMethod(method);
        sliced->setCutoffDistance(1.0);
        sliced->addGlobalParameter("lambdaCross", 0.3);
        sliced->addGlobalParameter("lambdaOne", 0.6);
        sliced->addScalingParameter("lambdaCross", 0, 1, true, true);
        sliced->addScalingParameter("lambdaOne", 1, 1, true, true);
        sliced->addEnergyParameterDerivative("lambdaCross");
        sliced->addEnergyParameterDerivative("lambdaOne");
        system.addForce(sliced);
    };

    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++)
        positions[i] = Vec3(0.3*(i%3) + 0.05*i, 0.3*((i/3)%2) + 0.02*i, 0.3*(i/6) + 0.03*i);

    auto evaluate = [&](Context& context, State& state, map<string, double>& derivatives) {
        state = context.getState(State::Energy | State::Forces | State::ParameterDerivatives);
        derivatives = state.getEnergyParameterDerivatives();
    };

    System system, modifiedSystem;
    SlicedNonbondedForce *sliced, *modifiedSliced;
    createSystem(false, system, sliced);
    createSystem(true, modifiedSystem, modifiedSliced);
    VerletIntegrator integrator1(0.001), integrator2(0.001);
    Context context(system, integrator1, platform);
    Context modifiedContext(modifiedSystem, integrator2, platform);
    context.setPositions(positions);
    modifiedContext.setPositions(positions);

    State state1, state2;
    map<string, double> derivatives1, derivatives2;
    evaluate(context, state1, derivatives1);

    // Updating the parameters without changing them must not alter anything.

    sliced->updateParametersInContext(context);
    evaluate(context, state2, derivatives2);
    assertEnergy(state1, state2, TOL);
    assertForces(state1, state2, TOL);
    for (auto& entry : derivatives1)
        ASSERT_EQUAL_TOL(entry.second, derivatives2[entry.first], TOL);

    // Changing one exception must agree with a context created from scratch with that change.

    int particle1, particle2;
    double chargeProd, sigma, epsilon;
    sliced->getExceptionParameters(1, particle1, particle2, chargeProd, sigma, epsilon);
    sliced->setExceptionParameters(1, particle1, particle2, chargeProd*chargeProdChange, sigma, epsilon*epsilonChange);
    sliced->updateParametersInContext(context);
    evaluate(context, state1, derivatives1);
    evaluate(modifiedContext, state2, derivatives2);
    assertEnergy(state1, state2, TOL);
    assertForces(state1, state2, TOL);
    for (auto& entry : derivatives1)
        ASSERT_EQUAL_TOL(entry.second, derivatives2[entry.first], TOL);
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
    assertEqualTo(
        ONE_4PI_EPS0*(-1.5*0.5)/1.5 + 4.0*sqrt(1.2*1.0)*(pow(1.0/1.5, 12.0)-pow(1.0/1.5, 6.0)),
        state1.getPotentialEnergy(),
        TOL
    );
    State state2 = context.getState(State::Energy, false, 1<<1);
    assertEqualTo(
        ONE_4PI_EPS0*(0.4*0.3)/1.5 + 4.0*sqrt(0.5*1.0)*(pow(1.6/1.5, 12.0)-pow(1.6/1.5, 6.0)),
        state2.getPotentialEnergy(),
        TOL
    );
    State state = context.getState(State::Energy);
    assertEqualTo(
        state1.getPotentialEnergy()+state2.getPotentialEnergy(), state.getPotentialEnergy(), TOL
    );

    // Try modifying them and see if they're still correct.

    nb1->setParticleParameters(0, -1.2, 1.1, 1.4);
    nb1->updateParametersInContext(context);
    nb2->setParticleParameters(0, 0.5, 1.6, 0.6);
    nb2->updateParametersInContext(context);
    state1 = context.getState(State::Energy, false, 1<<0);
    assertEqualTo(
        ONE_4PI_EPS0*(-1.2*0.5)/1.5 + 4.0*sqrt(1.4*1.0)*(pow(1.05/1.5, 12.0)-pow(1.05/1.5, 6.0)),
        state1.getPotentialEnergy(),
        TOL
    );
    state2 = context.getState(State::Energy, false, 1<<1);
    assertEqualTo(
        ONE_4PI_EPS0*(0.5*0.3)/1.5 + 4.0*sqrt(0.6*1.0)*(pow(1.7/1.5, 12.0)-pow(1.7/1.5, 6.0)),
        state2.getPotentialEnergy(),
        TOL
    );

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
    assertEqualTo(energy, context.getState(State::Energy).getPotentialEnergy(), 1e-4);
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
    assertEqualTo(expectedChange, e2-e1, 1e-4);
}

void testLJPMEDerivativesWithoutCharges() {
    // LJPME with no charges and a requested derivative. The Coulomb PME kernels never run, so the
    // buffers they would have filled are cleared instead; on OpenCL devices without a PME queue the
    // derivative buffer belongs to the context and must not be touched. The energy is linear in the
    // scaling parameter, so its derivative is the difference between the energies at 1 and 0.

    const double lambda = 0.4;
    string precision = platform.getName() == "Reference" ? "double" : platform.getPropertyDefaultValue("Precision");
    double tol = precision == "double" ? 1e-4 : (precision == "mixed" ? 1e-3 : 5e-3);
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(10, 0, 0), Vec3(0, 10, 0), Vec3(0, 0, 10));
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(2);
    sliced->setNonbondedMethod(SlicedNonbondedForce::LJPME);
    sliced->setCutoffDistance(3.5);
    sliced->addParticle(0.0, 3.0, 10.0);
    sliced->addParticle(0.0, 3.0, 10.0);
    sliced->setParticleSubset(1, 1);
    sliced->addGlobalParameter("lambda", lambda);
    sliced->addScalingParameter("lambda", 0, 1, false, true);
    sliced->addEnergyParameterDerivative("lambda");
    system.addForce(sliced);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(3.2, 0, 0)});
    double derivative = context.getState(State::Energy | State::ParameterDerivatives).getEnergyParameterDerivatives().at("lambda");
    context.setParameter("lambda", 1.0);
    double energyOne = context.getState(State::Energy).getPotentialEnergy();
    context.setParameter("lambda", 0.0);
    double energyZero = context.getState(State::Energy).getPotentialEnergy();
    assertEqualTo(energyOne-energyZero, derivative, tol);
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
    assertEqualTo(e1, e2+e3, 1e-4);
    ASSERT(e2 != 0);
    ASSERT(e3 != 0);

    // Completely disable the direct space calculation.

    force->setIncludeDirectSpace(false);
    context.reinitialize(true);
    double e4 = context.getState(State::Energy).getPotentialEnergy();
    assertEqualTo(e3, e4, 1e-4);
}

void testExclusionsInReciprocalGroup(NonbondedForce::NonbondedMethod method) {
    // Create a box of rigid three-site molecules whose intramolecular pairs are excluded.

    const int numMolecules = 64;
    const int numParticles = 3*numMolecules;
    const double cutoff = 1.0;
    const double L = 2.5;
    double tol = (platform.getName() == "Reference" || platform.getPropertyDefaultValue("Precision") == "double") ? 1e-4 : 1e-3;

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
    nonbonded->setUseDispersionCorrection(false);
    if (method == NonbondedForce::PME)
        nonbonded->setPMEParameters(3.0, 24, 24, 24);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions(numParticles);
    int M = (int) round(pow(numMolecules, 1.0/3.0));
    for (int k = 0; k < numMolecules; k++) {
        int iz = k/(M*M);
        int iy = (k - iz*M*M)/M;
        int ix = k - M*(iy + iz*M);
        Vec3 jitter = Vec3(genrand_real2(sfmt), genrand_real2(sfmt), genrand_real2(sfmt))*0.1;
        Vec3 center = Vec3(ix+0.5, iy+0.5, iz+0.5)*L/M + jitter;
        Vec3 d1(0.1, 0, 0), d2(-0.0333, 0.0943, 0);
        for (int rotation = 0; rotation < k%3; rotation++) {
            d1 = Vec3(d1[2], d1[0], d1[1]);
            d2 = Vec3(d2[2], d2[0], d2[1]);
        }
        int i = 3*k;
        positions[i] = center;
        positions[i+1] = center + d1;
        positions[i+2] = center + d2;
        nonbonded->addParticle(-0.834, 0.315, 0.636);
        nonbonded->addParticle(0.417, 1, 0);
        nonbonded->addParticle(0.417, 1, 0);
        nonbonded->addException(i, i+1, 0, 1, 0);
        nonbonded->addException(i, i+2, 0, 1, 0);
        nonbonded->addException(i+1, i+2, 0, 1, 0);
    }

    // Put direct and reciprocal space in different force groups, with the exclusion corrections in the
    // reciprocal space one, and scale two slices.

    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(*nonbonded, 2);
    for (int i = 0; i < numParticles; i++)
        sliced->setParticleSubset(i, (i/3)%2);
    sliced->addGlobalParameter("lambdaCross", 1);
    sliced->addScalingParameter("lambdaCross", 0, 1, true, true);
    sliced->addEnergyParameterDerivative("lambdaCross");
    sliced->addGlobalParameter("lambdaCoul11", 1);
    sliced->addScalingParameter("lambdaCoul11", 1, 1, true, false);
    sliced->addEnergyParameterDerivative("lambdaCoul11");
    sliced->setForceGroup(0);
    sliced->setReciprocalSpaceForceGroup(1);
    sliced->setReciprocalExclusions(true);
    system2.addForce(sliced);

    // The reference direct space is a CustomNonbondedForce with the same exclusions, which
    // means no correction for the reciprocal space interactions between excluded pairs, and
    // with the same scaling of slices.

    VerletIntegrator integrator2(0.01);
    Context context2(system2, integrator2, platform);
    context2.setPositions(positions);
    double alpha;
    if (method == NonbondedForce::PME) {
        int nx, ny, nz;
        sliced->getPMEParametersInContext(context2, alpha, nx, ny, nz);
    }
    else {
        int kmaxx, kmaxy, kmaxz;
        NonbondedForceImpl::calcEwaldParameters(system2, *sliced, alpha, kmaxx, kmaxy, kmaxz);
    }
    CustomNonbondedForce* direct = new CustomNonbondedForce(
        "scaleCoul*ONE_4PI_EPS0*q1*q2*erfc(alpha*r)/r + scaleLJ*4*eps*((sig/r)^12-(sig/r)^6);"
        "scaleCoul=select(subset1+subset2-1, select(subset1+subset2-2, 1, lambdaCoul11), lambdaCross);"
        "scaleLJ=select(subset1+subset2-1, 1, lambdaCross);"
        "sig=0.5*(sigma1+sigma2); eps=sqrt(epsilon1*epsilon2)"
    );
    direct->setNonbondedMethod(CustomNonbondedForce::CutoffPeriodic);
    direct->setCutoffDistance(cutoff);
    direct->addGlobalParameter("ONE_4PI_EPS0", ONE_4PI_EPS0);
    direct->addGlobalParameter("alpha", alpha);
    direct->addGlobalParameter("lambdaCross", 1);
    direct->addGlobalParameter("lambdaCoul11", 1);
    direct->addEnergyParameterDerivative("lambdaCross");
    direct->addEnergyParameterDerivative("lambdaCoul11");
    direct->addPerParticleParameter("q");
    direct->addPerParticleParameter("sigma");
    direct->addPerParticleParameter("epsilon");
    direct->addPerParticleParameter("subset");
    for (int i = 0; i < numParticles; i++) {
        double charge, sigma, epsilon;
        nonbonded->getParticleParameters(i, charge, sigma, epsilon);
        direct->addParticle({charge, sigma, epsilon, (double) sliced->getParticleSubset(i)});
    }
    for (int i = 0; i < nonbonded->getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        nonbonded->getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        direct->addExclusion(particle1, particle2);
    }
    direct->setForceGroup(0);
    system1.addForce(direct);
    nonbonded->setForceGroup(1);
    system1.addForce(nonbonded);
    VerletIntegrator integrator1(0.01);
    Context context1(system1, integrator1, platform);
    context1.setPositions(positions);

    // Direct space

    State state1 = context1.getState(State::Energy | State::Forces, false, 1<<0);
    State state2 = context2.getState(State::Energy | State::Forces, false, 1<<0);
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);

    // Direct and reciprocal space together

    state1 = context1.getState(State::Energy | State::Forces, false, 1<<1);
    state2 = context2.getState(State::Energy | State::Forces, false, (1<<0) + (1<<1));
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);

    // Overall

    State state3 = context2.getState(State::Energy | State::Forces);
    assertEnergy(state2, state3, tol);
    assertForces(state2, state3, tol);

    // Scaled direct space and its derivatives

    for (Context* context : {&context1, &context2}) {
        context->setParameter("lambdaCross", 0.7);
        context->setParameter("lambdaCoul11", 0.4);
    }
    state1 = context1.getState(State::Energy | State::Forces | State::ParameterDerivatives, false, 1<<0);
    state2 = context2.getState(State::Energy | State::Forces | State::ParameterDerivatives, false, 1<<0);
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);
    map<string, double> derivatives1 = state1.getEnergyParameterDerivatives();
    map<string, double> derivatives2 = state2.getEnergyParameterDerivatives();
    for (string name : {"lambdaCross", "lambdaCoul11"}) {
        ASSERT(derivatives1[name] != 0);
        assertEqualTo(derivatives1[name], derivatives2[name], tol);
    }
}

void testNonbondedSlicing(OpenMM_SFMT::SFMT& sfmt, NonbondedForce::NonbondedMethod method, int numSubsets, bool offsets, bool exceptions, bool lj) {
    bool includeLJ = lj;
    bool includeCoulomb = !lj;

    const int numMolecules = 100;
    const int numParticles = numMolecules*2;
    const double cutoff = 3.5;
    const double L = exceptions ? 7.0 : 10.0;
    double tol = (platform.getName() == "Reference" || platform.getPropertyDefaultValue("Precision") == "double") ? 1e-4 : 1e-3;

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

    int M = (int) pow(numMolecules, 1.0/3.0);
    double eps = 1.0;
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
            nonbonded->addException(i, j, q(i)*q(j), 1, eps);
    }

    typedef struct {
        int particle;
        string parameter;
        double charge, sigma, epsilon;
    } ParticleOffset;
    vector<ParticleOffset> particleOffsets;

    typedef struct {
        int exception;
        string parameter;
        double chargeProd, sigma, epsilon;
    } ExceptionOffset;
    vector<ExceptionOffset> exceptionOffsets;

    double offsetParamValue = 0.4;
    if (offsets) {
        string offsetParam = "offsetLambda";
        particleOffsets.push_back({0, offsetParam, 1.0, 0.0, 0.0});
        particleOffsets.push_back({1, offsetParam, 0.0, 0.0, 1.0});
        if (exceptions) {
            exceptionOffsets.push_back({0, offsetParam, 1.0, 0.0, 0.0});
            exceptionOffsets.push_back({1, offsetParam, 0.0, 0.0, 1.0});
        }
        nonbonded->addGlobalParameter(offsetParam, offsetParamValue);
        for (auto& po : particleOffsets)
            nonbonded->addParticleParameterOffset(po.parameter, po.particle, po.charge, po.sigma, po.epsilon);
        for (auto& eo : exceptionOffsets)
            nonbonded->addExceptionParameterOffset(eo.parameter, eo.exception, eo.chargeProd, eo.sigma, eo.epsilon);
    }

    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(*nonbonded, numSubsets);

    for (int k = 0; k < numParticles; k++)
        sliced->setParticleSubset(k, (int) (genrand_real2(sfmt)*numSubsets));

    string param01 = includeCoulomb ? "lambda" : "sqrtLambda";
    sliced->addGlobalParameter(param01, 1);
    for (int s = 0; s < numSubsets; s++)
        if (s != 1 && !(numSubsets == 3 && s == 2))
            sliced->addScalingParameter(param01, s, 1, includeCoulomb, includeLJ);

    string param11 = includeCoulomb ? "lambdaSq" : "lambda";
    sliced->addGlobalParameter(param11, 1);
    sliced->addScalingParameter(param11, 1, 1, includeCoulomb, includeLJ);

    string param21 = param01 + "2";
    if (numSubsets == 3) {
        sliced->addGlobalParameter(param21, 1);
        sliced->addScalingParameter(param21, 2, 1, includeCoulomb, includeLJ);
    }

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
        if (si == 1 || sj == 1) {
            string parameter;
            if (si == sj)
                parameter = param11;
            else if (numSubsets == 3 && (si == 2 || sj == 2))
                parameter = param21;
            else
                parameter = param01;
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

    State state1 = context1.getState(State::Energy | State::Forces | State::Positions, false, 1<<0);
    State state2 = context2.getState(State::Energy | State::Forces | State::Positions, false, 1<<0);
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

    // Change of scaling parameter value to zero

    map<string, double> value;
    value["one"] = 1;
    value["lambda"] = value["sqrtLambda"] = value["lambdaSq"] = 0;
    if (numSubsets == 3)
        value[param21] = value[param01];
    for (int k = 0; k < numParticles; k++)
        nonbonded->setParticleParameters(
            k, q(k)*value[particleScale[k].first], 1, eps*value[particleScale[k].second]
        );
    for (int k = 0; k < numExceptions; k++)
        nonbonded->setExceptionParameters(
            k, 2*k, 2*k+1, q(2*k)*q(2*k+1)*value[exceptionScale[k].first], 1, eps*value[exceptionScale[k].second]
        );
    for (int j = 0; j < particleOffsets.size(); j++) {
        const auto& po = particleOffsets[j];
        int k = po.particle;
        nonbonded->setParticleParameterOffset(
            j, po.parameter, k, po.charge*value[particleScale[k].first], po.sigma, po.epsilon*value[particleScale[k].second]
        );
    }
    for (int j = 0; j < exceptionOffsets.size(); j++) {
        const auto& eo = exceptionOffsets[j];
        int k = eo.exception;
        nonbonded->setExceptionParameterOffset(
            j, eo.parameter, k, eo.chargeProd*value[exceptionScale[k].first], eo.sigma, eo.epsilon*value[exceptionScale[k].second]
        );
    }
    nonbonded->updateParametersInContext(context1);
    context2.setParameter(param01, value[param01]);
    context2.setParameter(param11, value[param11]);
    if (numSubsets == 3)
        context2.setParameter(param21, value[param21]);

    // Direct space

    state1 = context1.getState(State::Energy | State::Forces, false, 1<<0);
    state2 = context2.getState(State::Energy | State::Forces, false, 1<<0);
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

    double energy_lambda_zero = state1.getPotentialEnergy();

    value["lambda"] = 0.5;
    value["sqrtLambda"] = sqrt(value["lambda"]);
    value["lambdaSq"] = value["lambda"]*value["lambda"];
    if (numSubsets == 3)
        value[param21] = value[param01];
    for (int k = 0; k < numParticles; k++)
        nonbonded->setParticleParameters(
            k, q(k)*value[particleScale[k].first], 1, eps*value[particleScale[k].second]
        );
    for (int k = 0; k < numExceptions; k++)
        nonbonded->setExceptionParameters(
            k, 2*k, 2*k+1, q(2*k)*q(2*k+1)*value[exceptionScale[k].first], 1, eps*value[exceptionScale[k].second]
        );
    for (int j = 0; j < particleOffsets.size(); j++) {
        const auto& po = particleOffsets[j];
        int k = po.particle;
        nonbonded->setParticleParameterOffset(
            j, po.parameter, k, po.charge*value[particleScale[k].first], po.sigma, po.epsilon*value[particleScale[k].second]
        );
    }
    for (int j = 0; j < exceptionOffsets.size(); j++) {
        const auto& eo = exceptionOffsets[j];
        int k = eo.exception;
        nonbonded->setExceptionParameterOffset(
            j, eo.parameter, k, eo.chargeProd*value[exceptionScale[k].first], eo.sigma, eo.epsilon*value[exceptionScale[k].second]
        );
    }
    nonbonded->updateParametersInContext(context1);
    context2.setParameter(param01, value[param01]);
    context2.setParameter(param11, value[param11]);
    if (numSubsets == 3)
        context2.setParameter(param21, value[param21]);

    // Direct space

    state1 = context1.getState(State::Energy | State::Forces, false, 1<<0);
    state2 = context2.getState(State::Energy | State::Forces, false, 1<<0);
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

    // Derivatives

    sliced->addEnergyParameterDerivative(param01);
    sliced->addEnergyParameterDerivative(param11);
    if (numSubsets == 3)
        sliced->addEnergyParameterDerivative(param21);
    context2.reinitialize(true);
    state2 = context2.getState(State::Energy | State::ParameterDerivatives);
    map<string, double> derivatives = state2.getEnergyParameterDerivatives();
    assertEqualTo(
        energy_lambda_one - energy_lambda_zero,
        derivatives[param01]+derivatives[param11]+derivatives[param21],
        tol
    );

    // Derivatives are only computed along with the energy

    derivatives = context2.getState(State::ParameterDerivatives).getEnergyParameterDerivatives();
    ASSERT(isnan(derivatives[param01]));

    // Sum of derivatives

    for (int k = 0; k < nonbonded->getNumParticles(); k++)
        nonbonded->setParticleParameters(k, includeCoulomb ? q(k) : 0, 1, includeLJ ? eps : 0);
    for (int k = 0; k < nonbonded->getNumExceptions(); k++)
        nonbonded->setExceptionParameters(k, 2*k, 2*k+1, includeCoulomb ? q(2*k)*q(2*k+1) : 0, 1, includeLJ ? eps : 0);
    for (int j = 0; j < particleOffsets.size(); j++) {
        const auto& po = particleOffsets[j];
        nonbonded->setParticleParameterOffset(
            j, po.parameter, po.particle, includeCoulomb ? po.charge : 0, po.sigma, includeLJ ? po.epsilon : 0
        );
    }
    for (int j = 0; j < exceptionOffsets.size(); j++) {
        const auto& eo = exceptionOffsets[j];
        nonbonded->setExceptionParameterOffset(
            j, eo.parameter, eo.exception, includeCoulomb ? eo.chargeProd : 0, eo.sigma, includeLJ ? eo.epsilon : 0
        );
    }
    nonbonded->updateParametersInContext(context1);
    state1 = context1.getState(State::Energy | State::Forces);
    double energy = state1.getPotentialEnergy();

    sliced->addGlobalParameter("remainder", 1.0);
    for (int s1 = 0; s1 < numSubsets; s1++)
        for (int s2 = s1; s2 < numSubsets; s2++)
            if (s1 != 1 && s2 != 1)
                sliced->addScalingParameter("remainder", s1, s2, includeCoulomb, includeLJ);
    sliced->addEnergyParameterDerivative("remainder");
    context2.reinitialize(true);
    state2 = context2.getState(State::Energy | State::ParameterDerivatives);
    derivatives = state2.getEnergyParameterDerivatives();
    double sum = derivatives[param01]+derivatives[param11]+derivatives[param21]+derivatives["remainder"];
    assertEqualTo(energy, sum, tol);
}

void testScalingParameterSeparation(OpenMM_SFMT::SFMT& sfmt, NonbondedForce::NonbondedMethod method, int numSubsets, bool exceptions) {
    const int numMolecules = 100;
    const int numParticles = numMolecules*2;
    const double cutoff = 3.5;
    const double L = exceptions ? 7.0 : 10.0;
    double tol = (
        platform.getName() == "Reference" || platform.getPropertyDefaultValue("Precision") == string("double")
    ) ? 1e-4 : 1e-3;

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
        nonbonded->addParticle(q(i), 1, 1);
        nonbonded->addParticle(q(j), 1, 1);
        if (exceptions)
            nonbonded->addException(i, j, q(i)*q(j), 1, 1);
    }

    SlicedNonbondedForce* sliced1 = new SlicedNonbondedForce(*nonbonded, numSubsets);
    SlicedNonbondedForce* sliced2 = new SlicedNonbondedForce(*nonbonded, numSubsets);

    for (int k = 0; k < numParticles; k++) {
        int subset = (int) (genrand_real2(sfmt)*numSubsets);
        sliced1->setParticleSubset(k, subset);
        sliced2->setParticleSubset(k, subset);
    }

    double lambda = 0.5;
    sliced1->addGlobalParameter("lambda", lambda);
    sliced1->addScalingParameter("lambda", 0, 1, true, true);
    sliced1->addEnergyParameterDerivative("lambda");
    sliced2->addGlobalParameter("lambdaCoulomb", lambda);
    sliced2->addGlobalParameter("lambdaLJ", lambda);
    sliced2->addScalingParameter("lambdaCoulomb", 0, 1, true, false);
    sliced2->addScalingParameter("lambdaLJ", 0, 1, false, true);
    sliced2->addEnergyParameterDerivative("lambdaCoulomb");
    sliced2->addEnergyParameterDerivative("lambdaLJ");

    double value = 0.3;
    sliced1->addGlobalParameter("alpha", value);
    sliced1->addScalingParameter("alpha", 0, 0, true, true);
    sliced1->addEnergyParameterDerivative("alpha");

    sliced1->addGlobalParameter("beta", value);
    sliced1->addScalingParameter("beta", 1, 1, true, true);
    sliced1->addEnergyParameterDerivative("beta");

    sliced2->addGlobalParameter("gamma", value);
    sliced2->addScalingParameter("gamma", 0, 0, true, true);
    sliced2->addScalingParameter("gamma", 1, 1, true, true);
    sliced2->addEnergyParameterDerivative("gamma");

    if (numSubsets > 2) {
        sliced1->addGlobalParameter("rest", value);
        sliced2->addGlobalParameter("rest", value);
        for (int s1 = 0; s1 < numSubsets; s1++)
            for (int s2 = max(s1, 2); s2 < numSubsets; s2++) {
                sliced1->addScalingParameter("rest", s1, s2, true, true);
                sliced2->addScalingParameter("rest", s1, s2, true, true);
            }
        sliced1->addEnergyParameterDerivative("rest");
        sliced2->addEnergyParameterDerivative("rest");
    }

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
    assertEqualTo(
        state1.getPotentialEnergy(),
        lambda*derivatives1["lambda"]+value*(derivatives1["alpha"]+derivatives1["beta"]+derivatives1["rest"]),
        tol
    );
    assertEqualTo(derivatives1["alpha"]+derivatives1["beta"], derivatives2["gamma"], tol);

    // Direct space

    state1 = context1.getState(State::Energy | State::Forces | State::ParameterDerivatives, false, 1<<0);
    state2 = context2.getState(State::Energy | State::Forces | State::ParameterDerivatives, false, 1<<0);
    derivatives1 = state1.getEnergyParameterDerivatives();
    derivatives2 = state2.getEnergyParameterDerivatives();
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);
    assertEqualTo(derivatives1["lambda"], derivatives2["lambdaCoulomb"]+derivatives2["lambdaLJ"], tol);
    assertEqualTo(
        state1.getPotentialEnergy(),
        lambda*derivatives1["lambda"]+value*(derivatives1["alpha"]+derivatives1["beta"]+derivatives1["rest"]),
        tol
    );
    assertEqualTo(derivatives1["alpha"]+derivatives1["beta"], derivatives2["gamma"], tol);

    // // Reciprocal space

    state1 = context1.getState(State::Energy | State::Forces | State::ParameterDerivatives, false, 1<<1);
    state2 = context2.getState(State::Energy | State::Forces | State::ParameterDerivatives, false, 1<<1);
    derivatives1 = state1.getEnergyParameterDerivatives();
    derivatives2 = state2.getEnergyParameterDerivatives();
    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);
    assertEqualTo(derivatives1["lambda"], derivatives2["lambdaCoulomb"]+derivatives2["lambdaLJ"], tol);
    assertEqualTo(
        state1.getPotentialEnergy(),
        lambda*derivatives1["lambda"]+value*(derivatives1["alpha"]+derivatives1["beta"]+derivatives1["rest"]),
        tol
    );
    assertEqualTo(derivatives1["alpha"]+derivatives1["beta"], derivatives2["gamma"], tol);
}

bool throwsException(const function<void()>& call) {
    try {
        call();
    }
    catch (const OpenMMException&) {
        return true;
    }
    return false;
}

void testSoftcore(OpenMM_SFMT::SFMT& sfmt, NonbondedForce::NonbondedMethod method, bool exceptions, bool useSwitch) {
    // Two subsets. Slice (0, 1) is scaled by "lambda" (Coulomb and LJ) and uses the soft-core
    // potential. The difference between the soft-core and the hard-core sliced forces is compared
    // with a CustomNonbondedForce (and a CustomBondForce for the 1-4 exceptions) that computes
    // lambda*(U_softcore - U_LJ) for the pairs in that slice. Everything else cancels in the
    // difference, including the dispersion correction and the LJPME grid terms.

    const int numMolecules = 100;
    const int numParticles = numMolecules*2;
    const double cutoff = 3.5;
    const double L = exceptions ? 7.0 : 10.0;
    const double alpha = 0.5;
    const double lambda = 0.4;
    const bool ljpme = (method == NonbondedForce::LJPME);
    const bool useCutoff = (method != NonbondedForce::NoCutoff);
    const bool periodic = (useCutoff && method != NonbondedForce::CutoffNonPeriodic);
    string precision = platform.getName() == "Reference" ? "double" : platform.getPropertyDefaultValue("Precision");
    double tol = precision == "double" ? 1e-4 : (precision == "mixed" ? 1e-3 : 5e-3);

    System systemSoft, systemHard, systemDiff;
    for (int i = 0; i < numParticles; i++) {
        systemSoft.addParticle(1.0);
        systemHard.addParticle(1.0);
        systemDiff.addParticle(1.0);
    }
    systemSoft.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    systemHard.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    systemDiff.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));

    NonbondedForce nonbonded;
    nonbonded.setNonbondedMethod(method);
    nonbonded.setCutoffDistance(cutoff);
    nonbonded.setUseDispersionCorrection(true);
    nonbonded.setEwaldErrorTolerance(1e-4);
    nonbonded.setUseSwitchingFunction(useSwitch);
    nonbonded.setSwitchingDistance(0.8*cutoff);
    vector<Vec3> positions(numParticles);

    // Small charges keep the Coulomb energy from dominating the differences compared below.
    // The first two molecules have no Lennard-Jones parameters at all.

    auto q = [](int k) { return 0.2*(1-2*(k%2)); };
    auto lj = [](int k) { return k < 4 ? 0.0 : 1.0; };

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
        nonbonded.addParticle(q(i), lj(i), lj(i));
        nonbonded.addParticle(q(j), lj(j), lj(j));
        if (exceptions)
            nonbonded.addException(i, j, q(i)*q(j), 1.0, lj(i)*lj(j));
    }

    vector<int> subset(numParticles);
    for (int k = 0; k < numParticles; k++)
        subset[k] = (int) (genrand_real2(sfmt)*2);

    auto makeSliced = [&](bool softcore) {
        SlicedNonbondedForce* sliced = new SlicedNonbondedForce(nonbonded, 2);
        for (int k = 0; k < numParticles; k++)
            sliced->setParticleSubset(k, subset[k]);
        sliced->addGlobalParameter("lambda", lambda);
        sliced->addScalingParameter("lambda", 0, 1, true, true, softcore ? alpha : 0.0);
        sliced->addEnergyParameterDerivative("lambda");
        return sliced;
    };
    SlicedNonbondedForce* slicedSoft = makeSliced(true);
    SlicedNonbondedForce* slicedHard = makeSliced(false);
    systemSoft.addForce(slicedSoft);
    systemHard.addForce(slicedHard);

    // lambda*(U_softcore - U_LJ). With LJPME, the sliced force also shifts the additive LJ term
    // to zero at the cutoff, and the hard-core and soft-core shifts differ.

    stringstream cutoffStr;
    cutoffStr<<cutoff;
    string pairExpression = "lambda*4*eps*(ssc*(ssc-1) - sig6*(sig6-1)";
    if (ljpme)
        pairExpression += " - scc*(scc-1) + sig6c*(sig6c-1)";
    pairExpression += ");"
        "ssc=sig6/(1+alpha*(1-lambda)*sig6); scc=sig6c/(1+alpha*(1-lambda)*sig6c);"
        "sig6=(sigma/r)^6; sig6c=(sigma/"+cutoffStr.str()+")^6;";
    string bondExpression = "lambda*4*eps*(ssc*(ssc-1) - sig6*(sig6-1));"
        "ssc=sig6/(1+alpha*(1-lambda)*sig6); sig6=(sigma/r)^6";

    CustomNonbondedForce* custom = new CustomNonbondedForce(pairExpression+"sigma=0.5*(sigma1+sigma2); eps=sqrt(eps1*eps2)");
    custom->setNonbondedMethod(periodic ? CustomNonbondedForce::CutoffPeriodic : useCutoff ? CustomNonbondedForce::CutoffNonPeriodic : CustomNonbondedForce::NoCutoff);
    custom->setCutoffDistance(cutoff);
    custom->setUseSwitchingFunction(useSwitch);
    custom->setSwitchingDistance(0.8*cutoff);
    custom->addPerParticleParameter("sigma");
    custom->addPerParticleParameter("eps");
    custom->addGlobalParameter("lambda", lambda);
    custom->addGlobalParameter("alpha", alpha);
    custom->addEnergyParameterDerivative("lambda");
    set<int> subset0, subset1;
    for (int k = 0; k < numParticles; k++) {
        custom->addParticle({lj(k), lj(k)});
        (subset[k] == 0 ? subset0 : subset1).insert(k);
    }
    custom->addInteractionGroup(subset0, subset1);
    CustomBondForce* bonds = new CustomBondForce(bondExpression);
    bonds->addPerBondParameter("sigma");
    bonds->addPerBondParameter("eps");
    bonds->addGlobalParameter("lambda", lambda);
    bonds->addGlobalParameter("alpha", alpha);
    bonds->addEnergyParameterDerivative("lambda");
    if (exceptions)
        for (int k = 0; k < numMolecules; k++) {
            int i = 2*k, j = i+1;
            custom->addExclusion(i, j);
            if (subset[i] != subset[j])
                bonds->addBond(i, j, {1.0, lj(i)*lj(j)});
        }
    systemDiff.addForce(custom);
    systemDiff.addForce(bonds);

    VerletIntegrator integratorSoft(0.01), integratorHard(0.01), integratorDiff(0.01);
    Context contextSoft(systemSoft, integratorSoft, platform);
    Context contextHard(systemHard, integratorHard, platform);
    Context contextDiff(systemDiff, integratorDiff, platform);
    contextSoft.setPositions(positions);
    contextHard.setPositions(positions);
    contextDiff.setPositions(positions);

    auto check = [&](double lambdaValue) {
        contextSoft.setParameter("lambda", lambdaValue);
        contextHard.setParameter("lambda", lambdaValue);
        contextDiff.setParameter("lambda", lambdaValue);
        int options = State::Energy | State::Forces | State::ParameterDerivatives;
        State soft = contextSoft.getState(options);
        State hard = contextHard.getState(options);
        State diff = contextDiff.getState(options);
        assertEqualTo(diff.getPotentialEnergy(), soft.getPotentialEnergy()-hard.getPotentialEnergy(), tol);
        const vector<Vec3>& forcesSoft = soft.getForces();
        const vector<Vec3>& forcesHard = hard.getForces();
        const vector<Vec3>& forcesDiff = diff.getForces();
        for (int k = 0; k < numParticles; k++)
            assertEqualVec(forcesDiff[k], forcesSoft[k]-forcesHard[k], tol);
        double derivSoft = soft.getEnergyParameterDerivatives().at("lambda");
        double derivHard = hard.getEnergyParameterDerivatives().at("lambda");
        double derivDiff = diff.getEnergyParameterDerivatives().at("lambda");
        assertEqualTo(derivDiff, derivSoft-derivHard, tol);
    };
    check(lambda);
    check(0.0);
    check(1.0);
    check(0.75);
}

void testSoftcoreOverlap() {
    // Two nearly overlapping particles: the soft-core energy is finite and matches the formula.

    const double sigma = 1.0, epsilon = 1.0, alpha = 0.5, lambda = 0.3, r = 0.01;
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(2);
    sliced->setNonbondedMethod(SlicedNonbondedForce::NoCutoff);
    sliced->addParticle(0.0, sigma, epsilon);
    sliced->addParticle(0.0, sigma, epsilon);
    sliced->setParticleSubset(1, 1);
    sliced->addGlobalParameter("lambda", lambda);
    sliced->addScalingParameter("lambda", 0, 1, false, true, alpha);
    sliced->addEnergyParameterDerivative("lambda");
    system.addForce(sliced);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(r, 0, 0)});
    State state = context.getState(State::Energy | State::Forces | State::ParameterDerivatives);

    double sig6 = pow(sigma/r, 6);
    double s = sig6/(1 + alpha*(1-lambda)*sig6);
    double energy = lambda*4*epsilon*(s*s - s);
    double force = lambda*4*epsilon*(2*s - 1)*s*s*6*pow(r, 5)/pow(sigma, 6);
    double derivative = 4*epsilon*(s*s - s) + lambda*4*epsilon*alpha*s*s*(2*s - 1);
    assertEqualTo(energy, state.getPotentialEnergy(), TOL);
    assertEqualVec(Vec3(-force, 0, 0), state.getForces()[0], TOL);
    assertEqualVec(Vec3(force, 0, 0), state.getForces()[1], TOL);
    assertEqualTo(derivative, state.getEnergyParameterDerivatives().at("lambda"), TOL);
}

void testSoftcoreSinglePair(NonbondedForce::NonbondedMethod method, bool useSwitch, double r) {
    // A single pair with large sigma, so that the switching function and the LJPME potential shift
    // at the cutoff have a large effect on the soft-core minus hard-core differences, which are
    // compared with closed forms (the force through a finite difference of the energy).

    const double sigma = 3.0, epsilon = 10.0, alpha = 0.5, lambda = 0.4;
    const double cutoff = 3.5, switchDist = 2.8;
    const bool ljpme = (method == NonbondedForce::LJPME);
    const bool useCutoff = (method != NonbondedForce::NoCutoff);
    string precision = platform.getName() == "Reference" ? "double" : platform.getPropertyDefaultValue("Precision");
    double tol = precision == "double" ? 1e-4 : (precision == "mixed" ? 1e-3 : 5e-3);

    NonbondedForce nonbonded;
    nonbonded.setNonbondedMethod(method);
    nonbonded.setCutoffDistance(cutoff);
    nonbonded.setUseDispersionCorrection(false);
    nonbonded.setEwaldErrorTolerance(1e-4);
    nonbonded.setUseSwitchingFunction(useSwitch);
    nonbonded.setSwitchingDistance(switchDist);
    nonbonded.addParticle(0.0, sigma, epsilon);
    nonbonded.addParticle(0.0, sigma, epsilon);

    auto makeContext = [&](bool softcore, System& system, VerletIntegrator& integrator) {
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.setDefaultPeriodicBoxVectors(Vec3(10, 0, 0), Vec3(0, 10, 0), Vec3(0, 0, 10));
        SlicedNonbondedForce* sliced = new SlicedNonbondedForce(nonbonded, 2);
        sliced->setParticleSubset(1, 1);
        sliced->addGlobalParameter("lambda", lambda);
        sliced->addScalingParameter("lambda", 0, 1, false, true, softcore ? alpha : 0.0);
        sliced->addEnergyParameterDerivative("lambda");
        system.addForce(sliced);
        return new Context(system, integrator, platform);
    };
    System systemSoft, systemHard;
    VerletIntegrator integratorSoft(0.01), integratorHard(0.01);
    unique_ptr<Context> contextSoft(makeContext(true, systemSoft, integratorSoft));
    unique_ptr<Context> contextHard(makeContext(false, systemHard, integratorHard));
    contextSoft->setPositions({Vec3(0, 0, 0), Vec3(r, 0, 0)});
    contextHard->setPositions({Vec3(0, 0, 0), Vec3(r, 0, 0)});
    int options = State::Energy | State::Forces | State::ParameterDerivatives;
    State soft = contextSoft->getState(options);
    State hard = contextHard->getState(options);

    // U_softcore - U_LJ and its derivative with respect to lambda (lambda*U_softcore is the energy)

    auto unscaled = [&](double x, double& diff, double& derivCorrection) {
        double sig6 = pow(sigma/x, 6);
        double s = sig6/(1 + alpha*(1-lambda)*sig6);
        diff = 4*epsilon*((s*s - s) - (sig6*sig6 - sig6));
        derivCorrection = lambda*4*epsilon*alpha*s*s*(2*s - 1);
    };
    double diffCut, corrCut;
    unscaled(cutoff, diffCut, corrCut);
    auto switchFunction = [&](double x) {
        if (!useSwitch || x <= switchDist)
            return 1.0;
        double t = (x-switchDist)/(cutoff-switchDist);
        return 1 - 10*t*t*t + 15*t*t*t*t - 6*t*t*t*t*t;
    };
    auto energyDifference = [&](double x) {
        double diff, corr;
        unscaled(x, diff, corr);
        if (ljpme)
            diff -= diffCut;
        return lambda*switchFunction(x)*diff;
    };
    double diff, corr;
    unscaled(r, diff, corr);
    double derivative = diff + corr;
    if (ljpme)
        derivative -= diffCut + corrCut;
    derivative *= switchFunction(r);
    const double h = 1e-5;
    double force = -(energyDifference(r+h) - energyDifference(r-h))/(2*h);

    assertEqualTo(energyDifference(r), soft.getPotentialEnergy()-hard.getPotentialEnergy(), tol);
    assertEqualVec(Vec3(-force, 0, 0), soft.getForces()[0]-hard.getForces()[0], tol);
    assertEqualVec(Vec3(force, 0, 0), soft.getForces()[1]-hard.getForces()[1], tol);
    assertEqualTo(derivative, soft.getEnergyParameterDerivatives().at("lambda")-hard.getEnergyParameterDerivatives().at("lambda"), tol);
}

void testSoftcoreSinglePairSeparate(NonbondedForce::NonbondedMethod method, bool useSwitch, double r) {
    // Like testSoftcoreSinglePair, but with separate energy (lambdaE) and softness (lambdaS) parameters.

    const double sigma = 3.0, epsilon = 10.0, alpha = 0.5, lambdaE = 0.4, lambdaS = 0.7;
    const double cutoff = 3.5, switchDist = 2.8;
    const bool ljpme = (method == NonbondedForce::LJPME);
    string precision = platform.getName() == "Reference" ? "double" : platform.getPropertyDefaultValue("Precision");
    double tol = precision == "double" ? 1e-4 : (precision == "mixed" ? 1e-3 : 5e-3);

    NonbondedForce nonbonded;
    nonbonded.setNonbondedMethod(method);
    nonbonded.setCutoffDistance(cutoff);
    nonbonded.setUseDispersionCorrection(false);
    nonbonded.setEwaldErrorTolerance(1e-4);
    nonbonded.setUseSwitchingFunction(useSwitch);
    nonbonded.setSwitchingDistance(switchDist);
    nonbonded.addParticle(0.0, sigma, epsilon);
    nonbonded.addParticle(0.0, sigma, epsilon);

    auto makeContext = [&](bool softcore, System& system, VerletIntegrator& integrator) {
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.setDefaultPeriodicBoxVectors(Vec3(10, 0, 0), Vec3(0, 10, 0), Vec3(0, 0, 10));
        SlicedNonbondedForce* sliced = new SlicedNonbondedForce(nonbonded, 2);
        sliced->setParticleSubset(1, 1);
        sliced->addGlobalParameter("lambdaE", lambdaE);
        sliced->addScalingParameter("lambdaE", 0, 1, false, true);
        sliced->addEnergyParameterDerivative("lambdaE");
        if (softcore) {
            sliced->addGlobalParameter("lambdaS", lambdaS);
            sliced->addScalingParameter("lambdaS", 0, 1, false, false, alpha);
            sliced->addEnergyParameterDerivative("lambdaS");
        }
        system.addForce(sliced);
        return new Context(system, integrator, platform);
    };
    System systemSoft, systemHard;
    VerletIntegrator integratorSoft(0.01), integratorHard(0.01);
    unique_ptr<Context> contextSoft(makeContext(true, systemSoft, integratorSoft));
    unique_ptr<Context> contextHard(makeContext(false, systemHard, integratorHard));
    contextSoft->setPositions({Vec3(0, 0, 0), Vec3(r, 0, 0)});
    contextHard->setPositions({Vec3(0, 0, 0), Vec3(r, 0, 0)});
    int options = State::Energy | State::Forces | State::ParameterDerivatives;
    State soft = contextSoft->getState(options);
    State hard = contextHard->getState(options);

    // U_softcore - U_LJ and the derivative of U_softcore with respect to lambdaS (divided by lambdaE)

    auto unscaled = [&](double x, double& diff, double& softDeriv) {
        double sig6 = pow(sigma/x, 6);
        double s = sig6/(1 + alpha*(1-lambdaS)*sig6);
        diff = 4*epsilon*((s*s - s) - (sig6*sig6 - sig6));
        softDeriv = 4*epsilon*alpha*s*s*(2*s - 1);
    };
    double diffCut, softCut;
    unscaled(cutoff, diffCut, softCut);
    auto switchFunction = [&](double x) {
        if (!useSwitch || x <= switchDist)
            return 1.0;
        double t = (x-switchDist)/(cutoff-switchDist);
        return 1 - 10*t*t*t + 15*t*t*t*t - 6*t*t*t*t*t;
    };
    auto energyDifference = [&](double x) {
        double diff, deriv;
        unscaled(x, diff, deriv);
        if (ljpme)
            diff -= diffCut;
        return lambdaE*switchFunction(x)*diff;
    };
    double diff, deriv;
    unscaled(r, diff, deriv);
    if (ljpme)
        deriv -= softCut;
    double softDerivative = lambdaE*switchFunction(r)*deriv;
    const double h = 1e-5;
    double force = -(energyDifference(r+h) - energyDifference(r-h))/(2*h);

    assertEqualTo(energyDifference(r), soft.getPotentialEnergy()-hard.getPotentialEnergy(), tol);
    assertEqualVec(Vec3(-force, 0, 0), soft.getForces()[0]-hard.getForces()[0], tol);
    assertEqualVec(Vec3(force, 0, 0), soft.getForces()[1]-hard.getForces()[1], tol);
    assertEqualTo(energyDifference(r)/lambdaE, soft.getEnergyParameterDerivatives().at("lambdaE")-hard.getEnergyParameterDerivatives().at("lambdaE"), tol);
    assertEqualTo(softDerivative, soft.getEnergyParameterDerivatives().at("lambdaS"), tol);
}

void testSoftcoreSeparateParameters(OpenMM_SFMT::SFMT& sfmt, NonbondedForce::NonbondedMethod method, bool exceptions) {
    // Slice (0, 1): "lambdaE" scales Coulomb and LJ, "lambdaS" softens. Soft-core minus hard-core
    // sliced forces are compared with Custom* forces computing lambdaE*(U_softcore - U_LJ), with
    // derivatives with respect to both parameters.

    const int numMolecules = 100;
    const int numParticles = numMolecules*2;
    const double cutoff = 3.5;
    const double L = exceptions ? 7.0 : 10.0;
    const double alpha = 0.5;
    const bool ljpme = (method == NonbondedForce::LJPME);
    const bool useCutoff = (method != NonbondedForce::NoCutoff);
    const bool periodic = (useCutoff && method != NonbondedForce::CutoffNonPeriodic);
    string precision = platform.getName() == "Reference" ? "double" : platform.getPropertyDefaultValue("Precision");
    double tol = precision == "double" ? 1e-4 : (precision == "mixed" ? 1e-3 : 5e-3);

    System systemSoft, systemHard, systemDiff;
    for (int i = 0; i < numParticles; i++) {
        systemSoft.addParticle(1.0);
        systemHard.addParticle(1.0);
        systemDiff.addParticle(1.0);
    }
    systemSoft.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    systemHard.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    systemDiff.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));

    NonbondedForce nonbonded;
    nonbonded.setNonbondedMethod(method);
    nonbonded.setCutoffDistance(cutoff);
    nonbonded.setUseDispersionCorrection(true);
    nonbonded.setEwaldErrorTolerance(1e-4);
    vector<Vec3> positions(numParticles);
    auto q = [](int k) { return 0.2*(1-2*(k%2)); };
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
        nonbonded.addParticle(q(i), 1.0, 1.0);
        nonbonded.addParticle(q(j), 1.0, 1.0);
        if (exceptions)
            nonbonded.addException(i, j, q(i)*q(j), 1.0, 1.0);
    }
    vector<int> subset(numParticles);
    for (int k = 0; k < numParticles; k++)
        subset[k] = (int) (genrand_real2(sfmt)*2);

    auto makeSliced = [&](bool softcore) {
        SlicedNonbondedForce* sliced = new SlicedNonbondedForce(nonbonded, 2);
        for (int k = 0; k < numParticles; k++)
            sliced->setParticleSubset(k, subset[k]);
        sliced->addGlobalParameter("lambdaE", 1.0);
        sliced->addGlobalParameter("lambdaS", 1.0);
        sliced->addScalingParameter("lambdaE", 0, 1, true, true);
        sliced->addEnergyParameterDerivative("lambdaE");
        if (softcore) {
            sliced->addScalingParameter("lambdaS", 0, 1, false, false, alpha);
            sliced->addEnergyParameterDerivative("lambdaS");
        }
        return sliced;
    };
    systemSoft.addForce(makeSliced(true));
    systemHard.addForce(makeSliced(false));

    stringstream cutoffStr;
    cutoffStr<<cutoff;
    string pairExpression = "lambdaE*4*eps*(ssc*(ssc-1) - sig6*(sig6-1)";
    if (ljpme)
        pairExpression += " - scc*(scc-1) + sig6c*(sig6c-1)";
    pairExpression += ");"
        "ssc=sig6/(1+alpha*(1-lambdaS)*sig6); scc=sig6c/(1+alpha*(1-lambdaS)*sig6c);"
        "sig6=(sigma/r)^6; sig6c=(sigma/"+cutoffStr.str()+")^6;";
    string bondExpression = "lambdaE*4*eps*(ssc*(ssc-1) - sig6*(sig6-1));"
        "ssc=sig6/(1+alpha*(1-lambdaS)*sig6); sig6=(sigma/r)^6";
    CustomNonbondedForce* custom = new CustomNonbondedForce(pairExpression+"sigma=0.5*(sigma1+sigma2); eps=sqrt(eps1*eps2)");
    custom->setNonbondedMethod(periodic ? CustomNonbondedForce::CutoffPeriodic : useCutoff ? CustomNonbondedForce::CutoffNonPeriodic : CustomNonbondedForce::NoCutoff);
    custom->setCutoffDistance(cutoff);
    custom->addPerParticleParameter("sigma");
    custom->addPerParticleParameter("eps");
    custom->addGlobalParameter("lambdaE", 1.0);
    custom->addGlobalParameter("lambdaS", 1.0);
    custom->addGlobalParameter("alpha", alpha);
    custom->addEnergyParameterDerivative("lambdaE");
    custom->addEnergyParameterDerivative("lambdaS");
    set<int> subset0, subset1;
    for (int k = 0; k < numParticles; k++) {
        custom->addParticle({1.0, 1.0});
        (subset[k] == 0 ? subset0 : subset1).insert(k);
    }
    custom->addInteractionGroup(subset0, subset1);
    CustomBondForce* bonds = new CustomBondForce(bondExpression);
    bonds->addPerBondParameter("sigma");
    bonds->addPerBondParameter("eps");
    bonds->addGlobalParameter("lambdaE", 1.0);
    bonds->addGlobalParameter("lambdaS", 1.0);
    bonds->addGlobalParameter("alpha", alpha);
    bonds->addEnergyParameterDerivative("lambdaE");
    bonds->addEnergyParameterDerivative("lambdaS");
    if (exceptions)
        for (int k = 0; k < numMolecules; k++) {
            int i = 2*k, j = i+1;
            custom->addExclusion(i, j);
            if (subset[i] != subset[j])
                bonds->addBond(i, j, {1.0, 1.0});
        }
    systemDiff.addForce(custom);
    systemDiff.addForce(bonds);

    VerletIntegrator integratorSoft(0.01), integratorHard(0.01), integratorDiff(0.01);
    Context contextSoft(systemSoft, integratorSoft, platform);
    Context contextHard(systemHard, integratorHard, platform);
    Context contextDiff(systemDiff, integratorDiff, platform);
    contextSoft.setPositions(positions);
    contextHard.setPositions(positions);
    contextDiff.setPositions(positions);

    auto check = [&](double lambdaE, double lambdaS) {
        for (Context* context : {&contextSoft, &contextHard, &contextDiff}) {
            context->setParameter("lambdaE", lambdaE);
            if (context != &contextHard)
                context->setParameter("lambdaS", lambdaS);
        }
        int options = State::Energy | State::Forces | State::ParameterDerivatives;
        State soft = contextSoft.getState(options);
        State hard = contextHard.getState(options);
        State diff = contextDiff.getState(options);
        assertEqualTo(diff.getPotentialEnergy(), soft.getPotentialEnergy()-hard.getPotentialEnergy(), tol);
        const vector<Vec3>& forcesSoft = soft.getForces();
        const vector<Vec3>& forcesHard = hard.getForces();
        const vector<Vec3>& forcesDiff = diff.getForces();
        for (int k = 0; k < numParticles; k++)
            assertEqualVec(forcesDiff[k], forcesSoft[k]-forcesHard[k], tol);
        map<string, double> dSoft = soft.getEnergyParameterDerivatives();
        map<string, double> dHard = hard.getEnergyParameterDerivatives();
        map<string, double> dDiff = diff.getEnergyParameterDerivatives();
        assertEqualTo(dDiff.at("lambdaE"), dSoft.at("lambdaE")-dHard.at("lambdaE"), tol);
        assertEqualTo(dDiff.at("lambdaS"), dSoft.at("lambdaS"), tol);
    };
    check(0.4, 0.7);
    check(1.0, 0.3);
    check(0.5, 1.0);
    check(0.0, 0.5);
}

void testSoftcoreOverlapSeparate() {
    // Two nearly overlapping particles with different energy and softness parameters.

    const double sigma = 1.0, epsilon = 1.0, alpha = 0.5, lambdaE = 0.3, lambdaS = 0.6, r = 0.01;
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(2);
    sliced->setNonbondedMethod(SlicedNonbondedForce::NoCutoff);
    sliced->addParticle(0.0, sigma, epsilon);
    sliced->addParticle(0.0, sigma, epsilon);
    sliced->setParticleSubset(1, 1);
    sliced->addGlobalParameter("lambdaE", lambdaE);
    sliced->addGlobalParameter("lambdaS", lambdaS);
    sliced->addScalingParameter("lambdaE", 0, 1, false, true);
    sliced->addScalingParameter("lambdaS", 0, 1, false, false, alpha);
    sliced->addEnergyParameterDerivative("lambdaE");
    sliced->addEnergyParameterDerivative("lambdaS");
    system.addForce(sliced);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(r, 0, 0)});
    State state = context.getState(State::Energy | State::Forces | State::ParameterDerivatives);

    double sig6 = pow(sigma/r, 6);
    double s = sig6/(1 + alpha*(1-lambdaS)*sig6);
    double energy = lambdaE*4*epsilon*(s*s - s);
    double force = lambdaE*4*epsilon*(2*s - 1)*s*s*6*pow(r, 5)/pow(sigma, 6);
    double dE = 4*epsilon*(s*s - s);
    double dS = lambdaE*4*epsilon*alpha*s*s*(2*s - 1);
    assertEqualTo(energy, state.getPotentialEnergy(), TOL);
    assertEqualVec(Vec3(-force, 0, 0), state.getForces()[0], TOL);
    assertEqualVec(Vec3(force, 0, 0), state.getForces()[1], TOL);
    assertEqualTo(dE, state.getEnergyParameterDerivatives().at("lambdaE"), TOL);
    assertEqualTo(dS, state.getEnergyParameterDerivatives().at("lambdaS"), TOL);
}

void testSoftnessOnly() {
    // A softness parameter on a slice with no Lennard-Jones scaling parameter: lambdaE is 1.

    const double sigma = 1.0, epsilon = 1.0, alpha = 0.5, lambdaS = 0.4, r = 0.9;
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    SlicedNonbondedForce* sliced = new SlicedNonbondedForce(2);
    sliced->setNonbondedMethod(SlicedNonbondedForce::NoCutoff);
    sliced->addParticle(0.0, sigma, epsilon);
    sliced->addParticle(0.0, sigma, epsilon);
    sliced->setParticleSubset(1, 1);
    sliced->addGlobalParameter("lambdaS", lambdaS);
    sliced->addScalingParameter("lambdaS", 0, 1, false, false, alpha);
    sliced->addEnergyParameterDerivative("lambdaS");
    system.addForce(sliced);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions({Vec3(0, 0, 0), Vec3(r, 0, 0)});
    State state = context.getState(State::Energy | State::Forces | State::ParameterDerivatives);

    double sig6 = pow(sigma/r, 6);
    double s = sig6/(1 + alpha*(1-lambdaS)*sig6);
    double energy = 4*epsilon*(s*s - s);
    double force = 4*epsilon*(2*s - 1)*s*s*6*pow(r, 5)/pow(sigma, 6);
    double dS = 4*epsilon*alpha*s*s*(2*s - 1);
    assertEqualTo(energy, state.getPotentialEnergy(), TOL);
    assertEqualVec(Vec3(-force, 0, 0), state.getForces()[0], TOL);
    assertEqualVec(Vec3(force, 0, 0), state.getForces()[1], TOL);
    assertEqualTo(dS, state.getEnergyParameterDerivatives().at("lambdaS"), TOL);
}

void testErrors() {
    SlicedNonbondedForce force(3);
    force.addParticle(0.0, 1.0, 0.0);
    force.addGlobalParameter("a", 1.0);
    force.addGlobalParameter("b", 1.0);
    force.addGlobalParameter("c", 1.0);

    ASSERT(throwsException([&]() { force.setParticleSubset(0, 3); }));
    ASSERT(throwsException([&]() { force.setParticleSubset(1, 0); }));
    ASSERT(throwsException([&]() { force.addScalingParameter("a", 0, 3, true, false); }));
    ASSERT(throwsException([&]() { force.addScalingParameter("a", 0, 1, false, false); }));
    ASSERT(throwsException([&]() { force.addScalingParameter("unknown", 0, 1, true, false); }));

    force.addScalingParameter("a", 0, 1, true, false);
    force.addScalingParameter("b", 0, 1, false, true);
    ASSERT(throwsException([&]() { force.addScalingParameter("c", 1, 0, true, false); }));
    ASSERT(throwsException([&]() { force.setScalingParameter(0, "a", 0, 1, true, true); }));
    ASSERT(throwsException([&]() { force.setScalingParameter(1, "b", 1, 0, true, true); }));
    ASSERT(throwsException([&]() { force.setScalingParameter(2, "a", 0, 1, true, false); }));
    force.setScalingParameter(0, "a", 1, 0, true, false);
    force.setScalingParameter(0, "a", 2, 2, true, true);
    force.addScalingParameter("c", 0, 1, true, false);

    // Soft-core entries: alpha must be finite and non-negative, an entry needs a role, and a slice
    // has at most one softness parameter.

    ASSERT(throwsException([&]() { force.addScalingParameter("c", 0, 2, false, false); }));
    ASSERT(throwsException([&]() { force.addScalingParameter("c", 0, 2, false, false, -0.5); }));
    ASSERT(throwsException([&]() { force.addScalingParameter("c", 0, 2, false, false, numeric_limits<double>::quiet_NaN()); }));
    ASSERT(throwsException([&]() { force.addScalingParameter("c", 0, 2, false, false, numeric_limits<double>::infinity()); }));
    int soft = force.addScalingParameter("c", 0, 2, false, false, 0.5);
    assertEqualTo(0.5, force.getScalingParameterSoftcoreAlpha(soft), 0.0);
    assertEqualTo(0.0, force.getScalingParameterSoftcoreAlpha(0), 0.0);
    ASSERT(throwsException([&]() { force.getScalingParameterSoftcoreAlpha(force.getNumScalingParameters()); }));
    ASSERT(throwsException([&]() { force.addScalingParameter("b", 2, 0, true, false, 0.3); }));
    force.addScalingParameter("b", 2, 0, true, false);
    force.setScalingParameter(soft, "c", 0, 2, false, true, 0.0);
    assertEqualTo(0.0, force.getScalingParameterSoftcoreAlpha(soft), 0.0);
    force.setScalingParameter(soft, "c", 0, 2, false, true, 0.5);
    force.addEnergyParameterDerivative("c");

    ASSERT(throwsException([&]() { force.addEnergyParameterDerivative("unknown"); }));
    force.addGlobalParameter("d", 1.0);
    ASSERT(throwsException([&]() { force.addEnergyParameterDerivative("d"); }));
    force.addEnergyParameterDerivative("a");
    ASSERT(throwsException([&]() { force.addEnergyParameterDerivative("a"); }));

    System system;
    system.addParticle(1.0);
    SlicedNonbondedForce* clash = new SlicedNonbondedForce(1);
    clash->addParticle(1.0, 1.0, 0.0);
    clash->addGlobalParameter("p", 1.0);
    clash->addScalingParameter("p", 0, 0, true, true);
    clash->addParticleParameterOffset("p", 0, 1.0, 0.0, 0.0);
    system.addForce(clash);
    VerletIntegrator integrator(0.01);
    ASSERT(throwsException([&]() { Context context(system, integrator, platform); }));

    // A softness-only scaling parameter is valid on its own: the slice is softened at full strength.

    System system2;
    system2.addParticle(1.0);
    SlicedNonbondedForce* softOnly = new SlicedNonbondedForce(2);
    softOnly->addParticle(1.0, 1.0, 1.0);
    softOnly->addGlobalParameter("q", 1.0);
    softOnly->addScalingParameter("q", 0, 1, false, false, 0.5);
    softOnly->addEnergyParameterDerivative("q");
    system2.addForce(softOnly);
    VerletIntegrator integrator2(0.01);
    Context context2(system2, integrator2, platform);
}

void testSoftcoreRoleCombinations(NonbondedForce::NonbondedMethod method) {
    // Three subsets and scaling parameters that cover every role at once: "a" scales the LJ term of
    // slice (0, 1) and softens slice (0, 2); "b" scales the Coulomb term of slice (0, 1) and softens
    // slices (0, 1) and (1, 2); "d" scales and softens slice (2, 2). Each derivative must match a
    // central finite difference of the energy. The same system with one parameter per role must
    // give the same energy and forces, and derivatives whose sums match those of the shared ones.
    // The GPU test binaries do not register the plugin's Reference kernels, so both checks use the
    // tested platform.

    const int numParticles = 48;
    const int M = 4;
    const double spacing = 1.3;
    const double L = M*spacing;
    const double cutoff = 2.5;
    string precision = platform.getName() == "Reference" ? "double" : platform.getPropertyDefaultValue("Precision");
    double tol = precision == "double" ? 1e-4 : (precision == "mixed" ? 1e-3 : 5e-3);

    NonbondedForce nonbonded;
    nonbonded.setNonbondedMethod(method);
    nonbonded.setCutoffDistance(cutoff);
    nonbonded.setUseDispersionCorrection(true);
    nonbonded.setEwaldErrorTolerance(1e-5);
    vector<Vec3> positions(numParticles);
    auto q = [](int k) { return 0.2*(1-2*(k%2)); };
    for (int k = 0; k < numParticles; k++) {
        int iz = k/(M*M);
        int iy = (k - iz*M*M)/M;
        int ix = k - M*(iy + iz*M);
        positions[k] = Vec3(ix+0.5, iy+0.5, iz+0.5)*spacing + Vec3(sin(1.7*k), sin(2.3*k+1), sin(3.1*k+2))*0.05;
        nonbonded.addParticle(q(k), 1.0+0.05*(k%3), 0.8+0.1*(k%4));
    }
    // Exceptions between particles of different subsets (k, k+1) and of the same subset (k, k+3).
    for (int k = 0; k < numParticles; k += 2)
        nonbonded.addException(k, k+1, 0.5*q(k)*q(k+1), 1.0, 0.5);
    for (int k = 1; k+3 < numParticles; k += 6)
        nonbonded.addException(k, k+3, 0.5*q(k)*q(k+3), 1.0, 0.5);

    struct Role {
        string parameter;
        int subset1, subset2;
        bool includeCoulomb, includeLJ;
        double alpha;
    };
    vector<Role> sharedRoles = {
        {"a", 0, 1, false, true, 0.0},
        {"a", 0, 2, false, false, 0.5},
        {"b", 0, 1, true, false, 0.4},
        {"b", 1, 2, false, false, 0.6},
        {"d", 2, 2, true, true, 0.5}
    };
    vector<Role> splitRoles = {
        {"aE", 0, 1, false, true, 0.0},
        {"aS", 0, 2, false, false, 0.5},
        {"bC", 0, 1, true, false, 0.0},
        {"bS1", 0, 1, false, false, 0.4},
        {"bS2", 1, 2, false, false, 0.6},
        {"dE", 2, 2, true, true, 0.0},
        {"dS", 2, 2, false, false, 0.5}
    };
    // Split parameters take the value of the shared parameter named by their first letter.
    map<string, double> values = {{"a", 0.6}, {"b", 0.7}, {"d", 0.5}};
    auto buildSystem = [&](System& system, const vector<Role>& roles) {
        system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
        for (int k = 0; k < numParticles; k++)
            system.addParticle(1.0);
        SlicedNonbondedForce* sliced = new SlicedNonbondedForce(nonbonded, 3);
        for (int k = 0; k < numParticles; k++)
            sliced->setParticleSubset(k, k%3);
        set<string> added;
        for (auto& role : roles) {
            if (added.insert(role.parameter).second)
                sliced->addGlobalParameter(role.parameter, values[role.parameter.substr(0, 1)]);
            sliced->addScalingParameter(role.parameter, role.subset1, role.subset2, role.includeCoulomb, role.includeLJ, role.alpha);
        }
        for (const string& parameter : added)
            sliced->addEnergyParameterDerivative(parameter);
        system.addForce(sliced);
    };
    System system, splitSystem;
    buildSystem(system, sharedRoles);
    buildSystem(splitSystem, splitRoles);

    VerletIntegrator integrator(0.01), splitIntegrator(0.01);
    Context context(system, integrator, platform);
    Context splitContext(splitSystem, splitIntegrator, platform);
    context.setPositions(positions);
    splitContext.setPositions(positions);
    int options = State::Energy | State::Forces | State::ParameterDerivatives;
    State state = context.getState(options);
    State splitState = splitContext.getState(options);
    map<string, double> derivs = state.getEnergyParameterDerivatives();
    map<string, double> splitDerivs = splitState.getEnergyParameterDerivatives();
    assertEnergy(state, splitState, tol);
    assertForces(state, splitState, tol);

    // Finite differences: a small step on the Reference platform, a larger one elsewhere to keep
    // the rounding errors of the energy small.
    bool reference = platform.getName() == "Reference";
    double h = reference ? 1e-4 : 1e-2;
    double fdTol = reference ? 1e-5 : tol;
    for (auto& entry : values) {
        context.setParameter(entry.first, entry.second+h);
        double energyPlus = context.getState(State::Energy).getPotentialEnergy();
        context.setParameter(entry.first, entry.second-h);
        double energyMinus = context.getState(State::Energy).getPotentialEnergy();
        context.setParameter(entry.first, entry.second);
        assertEqualTo((energyPlus-energyMinus)/(2*h), derivs[entry.first], fdTol);
        double sum = 0.0;
        for (auto& split : splitDerivs)
            if (split.first[0] == entry.first[0])
                sum += split.second;
        assertEqualTo(derivs[entry.first], sum, tol);
    }
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
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    try {
        initializeTests(argc, argv);
        testErrors();
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
        testExceptionSlicesAfterUpdate(SlicedNonbondedForce::NoCutoff);
        testExceptionSlicesAfterUpdate(SlicedNonbondedForce::CutoffNonPeriodic);
        testSwitchingFunction(SlicedNonbondedForce::CutoffNonPeriodic);
        testSwitchingFunction(SlicedNonbondedForce::PME);
        testTwoForces();
        testParameterOffsets();
        testEwaldExceptions();
        testDirectAndReciprocal();
        testLJPMEDerivativesWithoutCharges();
        testExclusionsInReciprocalGroup(NonbondedForce::Ewald);
        testExclusionsInReciprocalGroup(NonbondedForce::PME);
        for (auto method : nonbondedMethods)
            for (int numSubsets : {2, 3})
                for (auto offsets : {false, true})
                    for (auto exceptions : {false, true})
                        for (auto lj : {false, true})
                            testNonbondedSlicing(sfmt, method, numSubsets, offsets, exceptions, lj);
        for (auto method : nonbondedMethods)
            for (int numSubsets : {2, 3})
                for (auto exceptions : {false, true})
                    testScalingParameterSeparation(sfmt, method, numSubsets, exceptions);
        testSoftcoreOverlap();
        for (auto method : nonbondedMethods)
            for (auto exceptions : {false, true})
                testSoftcore(sfmt, method, exceptions, false);
        testSoftcore(sfmt, NonbondedForce::CutoffPeriodic, true, true);
        testSoftcore(sfmt, NonbondedForce::PME, true, true);
        testSoftcoreSinglePair(NonbondedForce::CutoffPeriodic, true, 3.2);
        testSoftcoreSinglePair(NonbondedForce::PME, true, 3.2);
        testSoftcoreSinglePair(NonbondedForce::LJPME, false, 3.3);
        testSoftcoreSinglePair(NonbondedForce::NoCutoff, false, 3.2);
        testSoftcoreSinglePairSeparate(NonbondedForce::CutoffPeriodic, true, 3.2);
        testSoftcoreSinglePairSeparate(NonbondedForce::PME, true, 3.2);
        testSoftcoreSinglePairSeparate(NonbondedForce::LJPME, false, 3.3);
        testSoftcoreSinglePairSeparate(NonbondedForce::NoCutoff, false, 3.2);
        testSoftcoreOverlapSeparate();
        testSoftnessOnly();
        for (auto method : nonbondedMethods)
            for (auto exceptions : {false, true})
                testSoftcoreSeparateParameters(sfmt, method, exceptions);
        for (auto method : {NonbondedForce::CutoffPeriodic, NonbondedForce::PME, NonbondedForce::LJPME})
            testSoftcoreRoleCombinations(method);
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
