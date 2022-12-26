/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "OpenCLNonbondedSlicingTests.h"
#include "TestSlicedNonbondedForce.h"
// #include <openmm/opencl/opencl.hpp>
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
    string deviceIndex = platform.getPropertyValue(context1, OpenCLPlatform::OpenCLDeviceIndex());
    map<string, string> props;
    props[OpenCLPlatform::OpenCLDeviceIndex()] = deviceIndex+","+deviceIndex;
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

// bool canRunHugeTest() {
//     // Create a minimal context just to see which platform and device are being used.

//     System system;
//     system.addParticle(1.0);
//     VerletIntegrator integrator(1.0);
//     Context context(system, integrator, platform);
//     int platformIndex = stoi(platform.getPropertyValue(context, OpenCLPlatform::OpenCLPlatformIndex()));
//     int deviceIndex = stoi(platform.getPropertyValue(context, OpenCLPlatform::OpenCLDeviceIndex()));

//     // Find out how much memory the device has.

//     vector<cl::Platform> platforms;
//     cl::Platform::get(&platforms);
//     vector<cl::Device> devices;
//     platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);
//     long long memory = devices[deviceIndex].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

//     // Only run the huge test if the device has at least 4 GB of memory.

//     return (memory >= 4*(long long)(1<<30));
// }

void runPlatformTests() {
    testParallelComputation(SlicedNonbondedForce::NoCutoff);
    testParallelComputation(SlicedNonbondedForce::Ewald);
    testParallelComputation(SlicedNonbondedForce::PME);
    testParallelComputation(SlicedNonbondedForce::LJPME);
    testReordering();
    // if (canRunHugeTest()) {
    //     double tol = (platform.getPropertyDefaultValue("Precision") == "single" ? 1e-4 : 1e-5);
    //     testHugeSystem(platform, tol);
    // }
}
