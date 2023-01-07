/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "CudaNonbondedSlicingTests.h"
#include "TestSlicedNonbondedForce.h"
#include "openmm/NonbondedForce.h"
// #include <cuda.h>
#include <string>

void testParallelComputation(SlicedNonbondedForce::NonbondedMethod method) {
    System system;
    const int numParticles = 200;
    for (int i = 0; i < numParticles; i++)
        system.addParticle(1.0);
    SlicedNonbondedForce* force = new SlicedNonbondedForce(1);
    for (int i = 0; i < numParticles; i++)
        force->addParticle(i%2-0.5, 0.5, 1.0);
    force->setNonbondedMethod(method);
    system.addForce(force);
    system.setDefaultPeriodicBoxVectors(Vec3(5,0,0), Vec3(0,5,0), Vec3(0,0,5));
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++)
        positions[i] = Vec3(5*genrand_real2(sfmt), 5*genrand_real2(sfmt), 5*genrand_real2(sfmt));
    force->addGlobalParameter("scale", 0.5);
    for (int i = 0; i < numParticles; ++i)
        for (int j = 0; j < i; ++j) {
            Vec3 delta = positions[i]-positions[j];
            if (delta.dot(delta) < 0.1) {
                force->addException(i, j, 0, 1, 0);
            }
            else if (delta.dot(delta) < 0.2) {
                int index = force->addException(i, j, 0.5, 1, 1.0);
                force->addExceptionParameterOffset("scale", index, 0.5, 0.4, 0.3);
            }
        }

    // Create two contexts, one with a single device and one with two devices.

    VerletIntegrator integrator1(0.01);
    Context context1(system, integrator1, platform);
    context1.setPositions(positions);
    State state1 = context1.getState(State::Forces | State::Energy);
    VerletIntegrator integrator2(0.01);
    string deviceIndex = platform.getPropertyValue(context1, CudaPlatform::CudaDeviceIndex());
    map<string, string> props;
    props[CudaPlatform::CudaDeviceIndex()] = deviceIndex+","+deviceIndex;
    Context context2(system, integrator2, platform, props);
    context2.setPositions(positions);
    State state2 = context2.getState(State::Forces | State::Energy);

    // See if they agree.

    ASSERT_EQUAL_TOL(state1.getPotentialEnergy(), state2.getPotentialEnergy(), 1e-5);
    for (int i = 0; i < numParticles; i++)
        ASSERT_EQUAL_VEC(state1.getForces()[i], state2.getForces()[i], 1e-5);

    // Modify some particle parameters and see if they still agree.

    for (int i = 0; i < numParticles; i += 5) {
        double charge, sigma, epsilon;
        force->getParticleParameters(i, charge, sigma, epsilon);
        force->setParticleParameters(i, 0.9*charge, sigma, epsilon);
    }
    force->updateParametersInContext(context1);
    force->updateParametersInContext(context2);
    state1 = context1.getState(State::Forces | State::Energy);
    state2 = context2.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(state1.getPotentialEnergy(), state2.getPotentialEnergy(), 1e-5);
    for (int i = 0; i < numParticles; i++)
        ASSERT_EQUAL_VEC(state1.getForces()[i], state2.getForces()[i], 1e-5);
}

void testReordering() {
    // Check that reordering of atoms doesn't alter their positions.

    const int numParticles = 200;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(6, 0, 0), Vec3(2.1, 6, 0), Vec3(-1.5, -0.5, 6));
    SlicedNonbondedForce *nonbonded = new SlicedNonbondedForce(1);
    nonbonded->setNonbondedMethod(SlicedNonbondedForce::PME);
    system.addForce(nonbonded);
    vector<Vec3> positions;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(0.0, 0.0, 0.0);
        positions.push_back(Vec3(genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5)*20);
    }
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    integrator.step(1);
    State state = context.getState(State::Positions | State::Velocities);
    for (int i = 0; i < numParticles; i++) {
        ASSERT_EQUAL_VEC(positions[i], state.getPositions()[i], 1e-6);
    }
}

void testDeterministicForces() {
    // Check that the CudaDeterministicForces property works correctly.

    const int numParticles = 1000;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(6, 0, 0), Vec3(2.1, 6, 0), Vec3(-1.5, -0.5, 6));
    SlicedNonbondedForce *nonbonded = new SlicedNonbondedForce(1);
    nonbonded->setNonbondedMethod(SlicedNonbondedForce::PME);
    system.addForce(nonbonded);
    vector<Vec3> positions;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        nonbonded->addParticle(i%2 == 0 ? 1 : -1, 1, 0);
        positions.push_back(Vec3(genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5)*6);
    }
    VerletIntegrator integrator(0.001);
    map<string, string> properties;
    properties[CudaPlatform::CudaDeterministicForces()] = "true";
    Context context(system, integrator, platform, properties);
    context.setPositions(positions);
    State state1 = context.getState(State::Forces);
    State state2 = context.getState(State::Forces);

    // All forces should be *exactly* equal.

    for (int i = 0; i < numParticles; i++) {
        ASSERT_EQUAL(state1.getForces()[i][0], state2.getForces()[i][0]);
        ASSERT_EQUAL(state1.getForces()[i][1], state2.getForces()[i][1]);
        ASSERT_EQUAL(state1.getForces()[i][2], state2.getForces()[i][2]);
    }
}

// bool canRunHugeTest() {
//     // Create a minimal context just to see which device is being used.

//     System system;
//     system.addParticle(1.0);
//     VerletIntegrator integrator(1.0);
//     Context context(system, integrator, platform);
//     int deviceIndex = stoi(platform.getPropertyValue(context, CudaPlatform::CudaDeviceIndex()));

//     // Find out how much memory the device has.

//     CUdevice device;
//     cuDeviceGet(&device, deviceIndex);
//     size_t memory;
//     cuDeviceTotalMem(&memory, device);

//     // Only run the huge test if the device has at least 4 GB of memory.

//     return (memory >= 4L*(1<<30));
// }

void testUseCuFFT() {
    const int numMolecules = 100;
    const int numParticles = numMolecules*2;
    const double cutoff = 3.5;
    const double L = 10.0;
    double tol = platform.getPropertyDefaultValue("Precision") == "double" ? 1e-5 : 1e-3;

    System system1, system2;
    for (int i = 0; i < numParticles; i++) {
        system1.addParticle(1.0);
        system2.addParticle(1.0);
    }
    system1.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));
    system2.setDefaultPeriodicBoxVectors(Vec3(L, 0, 0), Vec3(0, L, 0), Vec3(0, 0, L));

    SlicedNonbondedForce* nonbonded1 = new SlicedNonbondedForce(2);
    nonbonded1->setNonbondedMethod(nonbonded1->PME);
    nonbonded1->setCutoffDistance(cutoff);
    nonbonded1->setUseDispersionCorrection(true);
    nonbonded1->setReciprocalSpaceForceGroup(1);
    nonbonded1->setEwaldErrorTolerance(1e-4);

    vector<Vec3> positions(numParticles);

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
        nonbonded1->addParticle(1, 1, 1);
        nonbonded1->addParticle(-1, 1, 1);
    }

    SlicedNonbondedForce* nonbonded2 = new SlicedNonbondedForce(*nonbonded1, 2);
    nonbonded2->setUseCuFFT(!nonbonded1->getUseCudaFFT());
    assertEqualTo(nonbonded1->getUseCudaFFT(), !nonbonded2->getUseCudaFFT(), tol);

    system1.addForce(nonbonded1);
    system2.addForce(nonbonded2);

    VerletIntegrator integrator1(0.01);
    Context context1(system1, integrator1, platform);
    context1.setPositions(positions);

    VerletIntegrator integrator2(0.01);
    Context context2(system2, integrator2, platform);
    context2.setPositions(positions);

    State state1 = context1.getState(State::Energy | State::Forces);
    State state2 = context2.getState(State::Energy | State::Forces);

    assertEnergy(state1, state2, tol);
    assertForces(state1, state2, tol);
}

void runPlatformTests() {
    testParallelComputation(SlicedNonbondedForce::NoCutoff);
    testParallelComputation(SlicedNonbondedForce::Ewald);
    testParallelComputation(SlicedNonbondedForce::PME);
    testParallelComputation(SlicedNonbondedForce::LJPME);
    testReordering();
    testDeterministicForces();
    testUseCuFFT();
    // if (canRunHugeTest())
    //     testHugeSystem();
}
