/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2021 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "SlicedNonbondedForce.h"
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

const double TOL = 1e-5;

void testCoulomb(Platform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* forceField = new SlicedNonbondedForce();
    forceField->addParticle(0.5);
    forceField->addParticle(-1.5);
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

void testExclusionsAnd14(Platform& platform) {
    System system;
    SlicedNonbondedForce* nonbonded = new SlicedNonbondedForce();
    for (int i = 0; i < 5; ++i) {
        system.addParticle(1.0);
        nonbonded->addParticle(0);
    }
    vector<pair<int, int> > bonds;
    bonds.push_back(pair<int, int>(0, 1));
    bonds.push_back(pair<int, int>(1, 2));
    bonds.push_back(pair<int, int>(2, 3));
    bonds.push_back(pair<int, int>(3, 4));
    nonbonded->createExceptionsFromBonds(bonds, 0.0, 0.0);
    int first14, second14;
    for (int i = 0; i < nonbonded->getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd;
        nonbonded->getExceptionParameters(i, particle1, particle2, chargeProd);
        if ((particle1 == 0 && particle2 == 3) || (particle1 == 3 && particle2 == 0))
            first14 = i;
        if ((particle1 == 1 && particle2 == 4) || (particle1 == 4 && particle2 == 1))
            second14 = i;
    }
    system.addForce(nonbonded);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    for (int i = 1; i < 5; ++i) {

        vector<Vec3> positions(5);
        const double r = 1.0;
        for (int j = 0; j < 5; ++j) {
            nonbonded->setParticleCharge(j, 0);
            positions[j] = Vec3(0, j, 0);
        }
        positions[i] = Vec3(r, 0, 0);

        // Test Coulomb forces

        nonbonded->setParticleCharge(0, 2);
        nonbonded->setParticleCharge(i, 2);
        nonbonded->setExceptionParameters(first14, 0, 3, i == 3 ? 4/1.2 : 0);
        nonbonded->setExceptionParameters(second14, 1, 4, 0);
        context.reinitialize();
        context.setPositions(positions);
        State state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces2 = state.getForces();
        double force = ONE_4PI_EPS0*4/(r*r);
        double energy = ONE_4PI_EPS0*4/r;
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

void testCutoff14(Platform& platform) {
    System system;
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* nonbonded = new SlicedNonbondedForce();
    nonbonded->setNonbondedMethod(SlicedNonbondedForce::CutoffNonPeriodic);
    for (int i = 0; i < 5; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(0);
    }
    const double cutoff = 3.5;
    nonbonded->setCutoffDistance(cutoff);
    const double eps = 30.0;
    nonbonded->setReactionFieldDielectric(eps);
    vector<pair<int, int> > bonds;
    bonds.push_back(pair<int, int>(0, 1));
    bonds.push_back(pair<int, int>(1, 2));
    bonds.push_back(pair<int, int>(2, 3));
    bonds.push_back(pair<int, int>(3, 4));
    nonbonded->createExceptionsFromBonds(bonds, 0.0, 0.0);
    int first14, second14;
    for (int i = 0; i < nonbonded->getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd;
        nonbonded->getExceptionParameters(i, particle1, particle2, chargeProd);
        if ((particle1 == 0 && particle2 == 3) || (particle1 == 3 && particle2 == 0))
            first14 = i;
        if ((particle1 == 1 && particle2 == 4) || (particle1 == 4 && particle2 == 1))
            second14 = i;
    }
    system.addForce(nonbonded);
    ASSERT(!nonbonded->usesPeriodicBoundaryConditions());
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

        double r = positions[i][0];

        // Test Coulomb forces

        const double q = 0.7;
        nonbonded->setParticleCharge(0, q);
        nonbonded->setParticleCharge(i, q);
        nonbonded->setExceptionParameters(first14, 0, 3, i == 3 ? q*q/1.2 : 0);
        nonbonded->setExceptionParameters(second14, 1, 4, 0);
        context.reinitialize(true);
        State state = context.getState(State::Forces | State::Energy);
        const vector<Vec3>& forces2 = state.getForces();
        double force = ONE_4PI_EPS0*q*q/(r*r);
        double energy = ONE_4PI_EPS0*q*q/r;
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

void testPeriodic(Platform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* nonbonded = new SlicedNonbondedForce();
    nonbonded->addParticle(1.0);
    nonbonded->addParticle(1.0);
    nonbonded->addParticle(1.0);
    nonbonded->addException(0, 1, 0.0);
    nonbonded->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    const double cutoff = 2.0;
    nonbonded->setCutoffDistance(cutoff);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    system.addForce(nonbonded);
    ASSERT(nonbonded->usesPeriodicBoundaryConditions());
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

void testPeriodicExceptions(Platform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* nonbonded = new SlicedNonbondedForce();
    nonbonded->addParticle(1.0);
    nonbonded->addParticle(1.0);
    nonbonded->addException(0, 1, 1.0);
    nonbonded->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    const double cutoff = 2.0;
    nonbonded->setCutoffDistance(cutoff);
    system.setDefaultPeriodicBoxVectors(Vec3(4, 0, 0), Vec3(0, 4, 0), Vec3(0, 0, 4));
    system.addForce(nonbonded);
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
    
    nonbonded->setExceptionsUsePeriodicBoundaryConditions(true);
    context.reinitialize(true);
    state = context.getState(State::Forces | State::Energy);
    forces = state.getForces();
    force = ONE_4PI_EPS0/(1*1);
    ASSERT_EQUAL_VEC(Vec3(force, 0, 0), forces[0], TOL);
    ASSERT_EQUAL_VEC(Vec3(-force, 0, 0), forces[1], TOL);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0/1, state.getPotentialEnergy(), TOL);
}

void testTriclinic(Platform& platform) {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    Vec3 a(3.1, 0, 0);
    Vec3 b(0.4, 3.5, 0);
    Vec3 c(-0.1, -0.5, 4.0);
    system.setDefaultPeriodicBoxVectors(a, b, c);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* nonbonded = new SlicedNonbondedForce();
    nonbonded->addParticle(1.0);
    nonbonded->addParticle(1.0);
    nonbonded->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    const double cutoff = 1.5;
    nonbonded->setCutoffDistance(cutoff);
    system.addForce(nonbonded);
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

void testLargeSystem(Platform& platform) {
    const int numMolecules = 600;
    const int numParticles = numMolecules*2;
    const double cutoff = 3.5;
    const double boxSize = 20.0;
    const double tol = 2e-3;
    System system;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    SlicedNonbondedForce* coulomb = new SlicedNonbondedForce();
    NonbondedForce* lennard_jones = new NonbondedForce();
    vector<Vec3> positions(numParticles);

    int M = static_cast<int>(std::pow(numMolecules, 1.0/3.0));
    if (M*M*M < numMolecules) M++;
    double sqrt3 = std::sqrt(3);
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
        lennard_jones->addParticle(0.0, 1.0, 1.0);
        lennard_jones->addParticle(0.0, 1.0, 1.0);
        lennard_jones->addException(2*k, 2*k+1, 0.0, 1.0, 0.0);
        coulomb->addParticle(-1.0);
        coulomb->addParticle(1.0);
        coulomb->addException(2*k, 2*k+1, 0.0);
        positions[2*k] = Vec3(x+dx, y+dy, z+dz);
        positions[2*k+1] = Vec3(x-dx, y-dy, z-dz);
    }
    // Try with no cutoffs and make sure it agrees with the Reference platform.

    lennard_jones->setNonbondedMethod(NonbondedForce::NoCutoff);
    coulomb->setNonbondedMethod(SlicedNonbondedForce::NoCutoff);
    system.addForce(lennard_jones);
    system.addForce(coulomb);
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_VEC(state.getForces()[0], Vec3(-54.127623, -53.981191, -54.49897), tol);
    ASSERT_EQUAL_VEC(state.getForces()[numMolecules], Vec3(66.552935, -58.243278, 67.180363), tol);
    ASSERT_EQUAL_VEC(state.getForces()[numParticles-1], Vec3(-19.198413, -19.527801, -20.850797), tol);
    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), 24330.897, tol);

    // Now try cutoffs but not periodic boundary conditions.

    lennard_jones->setNonbondedMethod(NonbondedForce::CutoffNonPeriodic);
    lennard_jones->setCutoffDistance(cutoff);
    coulomb->setNonbondedMethod(SlicedNonbondedForce::CutoffNonPeriodic);
    coulomb->setCutoffDistance(cutoff);
    context.reinitialize(true);
    state = context.getState(State::Positions | State::Velocities | State::Forces | State::Energy);
    ASSERT_EQUAL_VEC(state.getForces()[0], Vec3(-52.500464, -52.500464, -52.500464), tol);
    ASSERT_EQUAL_VEC(state.getForces()[numMolecules], Vec3(72.366705, -72.366705, 72.366705), tol);
    ASSERT_EQUAL_VEC(state.getForces()[numParticles-1], Vec3(-19.566868, -20.274485, -20.982101), tol);
    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), 27032.639, tol);

    // Now do the same thing with periodic boundary conditions.

    lennard_jones->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    coulomb->setNonbondedMethod(SlicedNonbondedForce::CutoffPeriodic);
    context.reinitialize(true);
    state = context.getState(State::Positions | State::Velocities | State::Forces | State::Energy);
    ASSERT_EQUAL_VEC(state.getForces()[0], Vec3(-29.842085, -29.842085, -68.534195), tol);
    ASSERT_EQUAL_VEC(state.getForces()[numMolecules], Vec3(72.366705, -72.366705, 72.366705), tol);
    ASSERT_EQUAL_VEC(state.getForces()[numParticles-1], Vec3(-19.566868, -20.274485, -20.982101), tol);
    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), 26978.984, tol);
}

void testChangingParameters(Platform& platform) {
    const int numMolecules = 600;
    const int numParticles = numMolecules*2;
    const double cutoff = 2.0;
    const double boxSize = 20.0;
    const double tol = 2e-3;
    System system0, system1;
    for (int i = 0; i < numParticles; i++) {
        system0.addParticle(1.0);
        system1.addParticle(1.0);
    }
    SlicedNonbondedForce* nonbonded0 = new SlicedNonbondedForce();
    SlicedNonbondedForce* nonbonded1 = new SlicedNonbondedForce();
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
        positions[2*i] = Vec3(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
        positions[2*i+1] = Vec3(positions[2*i][0]+1.0, positions[2*i][1], positions[2*i][2]);
        system0.addConstraint(2*i, 2*i+1, 1.0);
        nonbonded0->addException(2*i, 2*i+1, 0.0);
        system1.addConstraint(2*i, 2*i+1, 1.0);
        nonbonded1->addException(2*i, 2*i+1, 0.0);
    }
    nonbonded0->setNonbondedMethod(SlicedNonbondedForce::PME);
    nonbonded0->setCutoffDistance(cutoff);
    system0.addForce(nonbonded0);
    system0.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
    nonbonded1->setNonbondedMethod(SlicedNonbondedForce::PME);
    nonbonded1->setCutoffDistance(cutoff);
    system1.addForce(nonbonded1);
    system1.setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));

    // // See if the forces and energies match the Reference platform.
    VerletIntegrator integrator0(0.01);
    VerletIntegrator integrator1(0.01);
    Context context0(system0, integrator0, platform);
    Context context1(system1, integrator1, platform);
    context0.setPositions(positions);
    context1.setPositions(positions);
    
    // // Now modify parameters and see if they agree.

    for (int i = 0; i < numParticles; i++) {
        double charge = nonbonded0->getParticleCharge(i);
        nonbonded0->setParticleCharge(i, 1.5*charge);
    }
    nonbonded0->updateParametersInContext(context0);
    State state0 = context0.getState(State::Forces | State::Energy);
    State state1 = context1.getState(State::Forces | State::Energy);
    for (int i = 0; i < numParticles; i++)
        ASSERT_EQUAL_VEC(state0.getForces()[i], state1.getForces()[i], tol);
    ASSERT_EQUAL_TOL(state0.getPotentialEnergy(), state1.getPotentialEnergy(), tol);
}

void testTwoForces(Platform& platform) {
    // Create a system with two SlicedNonbondedForces.
    
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    VerletIntegrator integrator(0.01);
    SlicedNonbondedForce* nb1 = new SlicedNonbondedForce();
    nb1->addParticle(-1.5);
    nb1->addParticle(0.5);
    system.addForce(nb1);
    SlicedNonbondedForce* nb2 = new SlicedNonbondedForce();
    nb2->addParticle(0.4);
    nb2->addParticle(0.3);
    nb2->setForceGroup(1);
    system.addForce(nb2);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(1.5, 0, 0);
    context.setPositions(positions);
    State state1 = context.getState(State::Energy, false, 1<<0);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(-1.5*0.5)/1.5, state1.getPotentialEnergy(), TOL);
    State state2 = context.getState(State::Energy, false, 1<<1);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(0.4*0.3)/1.5, state2.getPotentialEnergy(), TOL);
    State state = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(state1.getPotentialEnergy()+state2.getPotentialEnergy(), state.getPotentialEnergy(), TOL);
    
    // Try modifying them and see if they're still correct.
    
    nb1->setParticleCharge(0, -1.2);
    nb1->updateParametersInContext(context);
    nb2->setParticleCharge(0, 0.5);
    nb2->updateParametersInContext(context);
    state1 = context.getState(State::Energy, false, 1<<0);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(-1.2*0.5)/1.5, state1.getPotentialEnergy(), TOL);
    state2 = context.getState(State::Energy, false, 1<<1);
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(0.5*0.3)/1.5, state2.getPotentialEnergy(), TOL);
    
    // Make sure it also works with PME.
    
    nb1->setNonbondedMethod(SlicedNonbondedForce::PME);
    nb2->setNonbondedMethod(SlicedNonbondedForce::PME);
    context.reinitialize(true);
    state1 = context.getState(State::Energy, false, 1<<0);
    state2 = context.getState(State::Energy, false, 1<<1);
    state = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(state1.getPotentialEnergy()+state2.getPotentialEnergy(), state.getPotentialEnergy(), TOL);
}

void testParameterOffsets(Platform& platform) {
    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    SlicedNonbondedForce* force = new SlicedNonbondedForce();
    force->addParticle(0.0);
    force->addParticle(1.0);
    force->addParticle(-1.0);
    force->addParticle(0.5);
    force->addException(0, 3, 0.0);
    force->addException(2, 3, 0.5);
    force->addException(0, 1, 1.0);
    force->addGlobalParameter("p1", 0.0);
    force->addGlobalParameter("p2", 1.0);
    force->addParticleParameterOffset("p1", 0, 3.0);
    force->addParticleParameterOffset("p2", 1, 1.0);
    force->addExceptionParameterOffset("p1", 1, 0.5);
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
    double pairChargeProd[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = i+1; j < 4; j++)
            pairChargeProd[i][j] = particleCharge[i]*particleCharge[j];
    pairChargeProd[0][3] = 0.0;
    pairChargeProd[2][3] = 0.5+0.5*0.5;
    pairChargeProd[0][1] = 1.0;
    
    // Compute the expected energy.

    double energy = 0.0;
    for (int i = 0; i < 4; i++)
        for (int j = i+1; j < 4; j++) {
            double dist = j-i;
            energy += ONE_4PI_EPS0*pairChargeProd[i][j]/dist;
        }
    ASSERT_EQUAL_TOL(energy, context.getState(State::Energy).getPotentialEnergy(), 1e-5);
}

void testEwaldExceptions(Platform& platform) {
    // Create a minimal system using LJPME.

    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    SlicedNonbondedForce* force = new SlicedNonbondedForce();
    system.addForce(force);
    force->setNonbondedMethod(SlicedNonbondedForce::LJPME);
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

void testDirectAndReciprocal(Platform& platform) {
    // Create a minimal system with direct space and reciprocal space in different force groups.

    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    SlicedNonbondedForce* force = new SlicedNonbondedForce();
    system.addForce(force);
    force->setNonbondedMethod(SlicedNonbondedForce::PME);
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

void testInstantiateFromNonbondedForce(Platform& platform) {
    OpenMM::NonbondedForce* force = new OpenMM::NonbondedForce();
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
    vector<Vec3> positions(4);
    for (int i = 0; i < 4; i++)
        positions[i] = Vec3(i, 0, 0);

    System system1;
    for (int i = 0; i < 4; i++)
        system1.addParticle(1.0);
    system1.addForce(force);
    VerletIntegrator integrator1(0.001);
    Context context1(system1, integrator1, platform);
    context1.setPositions(positions);
    State state1 = context1.getState(State::Forces | State::Energy);

    SlicedNonbondedForce* coulomb = new SlicedNonbondedForce(*force);

    OpenMM::NonbondedForce* lennard_jones = new OpenMM::NonbondedForce();
    lennard_jones->addParticle(0.0, 1.0, 0.5);
    lennard_jones->addParticle(0.0, 0.5, 0.6);
    lennard_jones->addParticle(0.0, 2.0, 0.7);
    lennard_jones->addParticle(0.0, 2.0, 0.8);
    lennard_jones->addException(0, 3, 0.0, 1.0, 0.0);
    lennard_jones->addException(2, 3, 0.0, 1.0, 1.5);
    lennard_jones->addException(0, 1, 0.0, 1.5, 1.0);
    lennard_jones->addGlobalParameter("p1", 0.5);
    lennard_jones->addGlobalParameter("p2", 1.0);
    lennard_jones->addParticleParameterOffset("p1", 0, 0.0, 0.5, 0.5);
    lennard_jones->addParticleParameterOffset("p2", 1, 0.0, 1.0, 2.0);
    lennard_jones->addExceptionParameterOffset("p1", 1, 0.0, 0.5, 1.5);

    System system2;
    system2.addForce(coulomb);
    system2.addForce(lennard_jones);
    for (int i = 0; i < 4; i++)
        system2.addParticle(1.0);
    VerletIntegrator integrator2(0.001);
    Context context2(system2, integrator2, platform);
    context2.setPositions(positions);
    State state2 = context2.getState(State::Forces | State::Energy);

    ASSERT_EQUAL_TOL(state1.getPotentialEnergy(), state2.getPotentialEnergy(), 1e-5);
    const vector<Vec3>& forces1 = state1.getForces();
    const vector<Vec3>& forces2 = state2.getForces();
    for (int i = 0; i < 4; i++)
        ASSERT_EQUAL_VEC(forces1[i], forces2[i], 1e-5);
}

void runPlatformTests();

extern "C" OPENMM_EXPORT void registerNonbondedSlicingReferenceKernelFactories();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testCoulomb(platform);
        testExclusionsAnd14(platform);
        testCutoff14(platform);
        testPeriodic(platform);
        testPeriodicExceptions(platform);
        testTriclinic(platform);
        testLargeSystem(platform);
        testChangingParameters(platform);
        testTwoForces(platform);
        testParameterOffsets(platform);
        testEwaldExceptions(platform);
        testDirectAndReciprocal(platform);
        testInstantiateFromNonbondedForce(platform);
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
