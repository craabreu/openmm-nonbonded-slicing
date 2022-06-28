import math
import pytest

import openmm as mm
import nativenonbondedplugin as plugin

from openmm import unit

ONE_4PI_EPS0 = 138.935456
TOL = 1E-5

def ASSERT(a):
    assert a

def ASSERT_EQUAL_TOL(a, b, tol):
    if unit.is_quantity(b):
        assert a == pytest.approx(b/b.unit, tol)
    else:
        assert a == pytest.approx(b, tol)

def ASSERT_EQUAL_VEC(a, b, tol):
    ASSERT_EQUAL_TOL(a.x, b.x, tol)
    ASSERT_EQUAL_TOL(a.y, b.y, tol)
    ASSERT_EQUAL_TOL(a.z, b.z, tol)

def executeCoulombTest(platformName):
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    integrator = mm.VerletIntegrator(0.01)
    forceField = plugin.NativeNonbondedForce()
    forceField.addParticle(0.5, 1, 0)
    forceField.addParticle(-1.5, 1, 0)
    system.addForce(forceField)
    ASSERT(not forceField.usesPeriodicBoundaryConditions())
    ASSERT(not system.usesPeriodicBoundaryConditions())
    platform = mm.Platform.getPlatformByName(platformName)
    context = mm.Context(system, integrator, platform)
    positions = [mm.Vec3(0, 0, 0), mm.Vec3(2, 0, 0)]
    context.setPositions(positions)
    state = context.getState(getForces=True, getEnergy=True)
    forces = state.getForces()
    force = ONE_4PI_EPS0*(-0.75)/4.0
    ASSERT_EQUAL_VEC(mm.Vec3(-force, 0, 0), forces[0], TOL)
    ASSERT_EQUAL_VEC(mm.Vec3(force, 0, 0), forces[1], TOL)
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(-0.75)/2.0, state.getPotentialEnergy(), TOL)

def executeLJTest(platformName):
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    integrator = mm.VerletIntegrator(0.01)
    forceField = plugin.NativeNonbondedForce()
    forceField.addParticle(0, 1.2, 1)
    forceField.addParticle(0, 1.4, 2)
    system.addForce(forceField)
    ASSERT(not forceField.usesPeriodicBoundaryConditions())
    ASSERT(not system.usesPeriodicBoundaryConditions())
    platform = mm.Platform.getPlatformByName(platformName)
    context = mm.Context(system, integrator, platform)
    positions = [mm.Vec3(0, 0, 0), mm.Vec3(2, 0, 0)]
    context.setPositions(positions)
    state = context.getState(getForces=True, getEnergy=True)
    forces = state.getForces()
    x = 1.3/2.0
    eps = math.sqrt(2)
    force = 4.0*eps*(12*x**12-6*x**6)/2.0
    ASSERT_EQUAL_VEC(mm.Vec3(-force, 0, 0), forces[0], TOL)
    ASSERT_EQUAL_VEC(mm.Vec3(force, 0, 0), forces[1], TOL)
    ASSERT_EQUAL_TOL(4.0*eps*(x**12-x**6), state.getPotentialEnergy(), TOL)


def testCoulombReference():
    executeCoulombTest("Reference")

def testCoulombCuda():
    executeCoulombTest("CUDA")

def testCoulombOpenCL():
    executeCoulombTest("OpenCL")

def testLJReference():
    executeCoulombTest("Reference")

def testLJCuda():
    executeCoulombTest("CUDA")

def testLJOpenCL():
    executeCoulombTest("OpenCL")
