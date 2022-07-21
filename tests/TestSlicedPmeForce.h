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

using namespace PmeSlicing;
using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;

#define assertForcesAndEnergy(context) { \
    State state0 = context.getState(State::Forces | State::Energy, false, 1<<0); \
    State state1 = context.getState(State::Forces | State::Energy, false, 1<<1); \
    ASSERT_EQUAL_TOL(state0.getPotentialEnergy(), state1.getPotentialEnergy(), TOL); \
    const vector<Vec3>& forces0 = state0.getForces(); \
    const vector<Vec3>& forces1 = state1.getForces(); \
    for (int i = 0; i < context.getSystem().getNumParticles(); i++) \
        ASSERT_EQUAL_VEC(forces0[i], forces1[i], TOL); \
}

void testInstantiateFromNonbondedForce(Platform& platform) {
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

    SlicedPmeForce* coulomb = new SlicedPmeForce(*force);
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

void testCoulomb(Platform& platform) {
    double L = 5.0;

    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));

    SlicedPmeForce* force = new SlicedPmeForce();
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

void testExclusionsAnd14(Platform& platform) {
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

    SlicedPmeForce* slicedNonbonded = new SlicedPmeForce(*nonbonded);
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

void testPeriodic(Platform& platform) {
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

    SlicedPmeForce* slicedNonbonded = new SlicedPmeForce(*nonbonded);
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

void testPeriodicExceptions(Platform& platform) {
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

    SlicedPmeForce* slicedNonbonded = new SlicedPmeForce(*nonbonded);
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

void testTriclinic(Platform& platform) {
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

    SlicedPmeForce* slicedNonbonded = new SlicedPmeForce(*nonbonded);
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

void testLargeSystem(Platform& platform) {
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
    SlicedPmeForce* slicedNonbonded = new SlicedPmeForce(*nonbonded);
    slicedNonbonded->setForceGroup(1);

    system.addForce(nonbonded);
    system.addForce(slicedNonbonded);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    assertForcesAndEnergy(context);
}

void testChangingParameters(Platform& platform) {
    const int numMolecules = 600;
    const int numParticles = numMolecules*2;
    const double cutoff = 2.0;
    const double L = 20.0;
    const double tol = 2e-3;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    SlicedPmeForce* nonbonded0 = new SlicedPmeForce();
    SlicedPmeForce* nonbonded1 = new SlicedPmeForce();
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

void testParameterOffsets(Platform& platform) {
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

    SlicedPmeForce* force = new SlicedPmeForce();
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
    force->addParticleParameterOffset("p1", 0, 3.0);
    force->addParticleParameterOffset("p2", 1, 1.0);
    force->addExceptionParameterOffset("p1", 1, 0.5);
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

void testEwaldExceptions(Platform& platform) {
    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    SlicedPmeForce* force = new SlicedPmeForce();
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

void testDirectAndReciprocal(Platform& platform) {
    // Create a minimal system with direct space and reciprocal space in different force groups.

    System system;
    for (int i = 0; i < 4; i++)
        system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(2, 0, 0), Vec3(0, 2, 0), Vec3(0, 0, 2));
    SlicedPmeForce* force = new SlicedPmeForce();
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

void runPlatformTests();

extern "C" OPENMM_EXPORT void registerPmeSlicingReferenceKernelFactories();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testInstantiateFromNonbondedForce(platform);
        testCoulomb(platform);
        testExclusionsAnd14(platform);
        testPeriodic(platform);
        testPeriodicExceptions(platform);
        testTriclinic(platform);
        testLargeSystem(platform);
        testChangingParameters(platform);
        testParameterOffsets(platform);
        testEwaldExceptions(platform);
        testDirectAndReciprocal(platform);
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
