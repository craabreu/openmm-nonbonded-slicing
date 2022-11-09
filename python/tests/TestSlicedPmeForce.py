import math

import pmeslicing as plugin
import numpy as np
import openmm as mm
import pytest
from openmm import unit

ONE_4PI_EPS0 = 138.935456

TOL = 1.0E-3

cases = [
    ('Reference', ''),
    ('CUDA', 'single'),
    ('CUDA', 'mixed'),
    ('CUDA', 'double'),
    ('OpenCL', 'single'),
    ('OpenCL', 'mixed'),
    ('OpenCL', 'double'),
]

ids = [''.join(case) for case in cases]


def value(x):
    return x/x.unit if unit.is_quantity(x) else x


def ASSERT(cond):
    assert cond


def ASSERT_EQUAL_TOL(expected, found, tol):
    exp = value(expected)
    assert abs(exp - value(found))/max(abs(exp), 1.0) <= tol


def ASSERT_EQUAL_VEC(expected, found, tol):
    ASSERT_EQUAL_TOL(expected.x, found.x, tol)
    ASSERT_EQUAL_TOL(expected.y, found.y, tol)
    ASSERT_EQUAL_TOL(expected.z, found.z, tol)


def assert_forces_and_energy(context):
    state0 = context.getState(getForces=True, getEnergy=True, groups={0})
    state1 = context.getState(getForces=True, getEnergy=True, groups={1})
    for force0, force1 in zip(state0.getForces(), state1.getForces()):
        ASSERT_EQUAL_VEC(force0, force1, TOL)
    ASSERT_EQUAL_TOL(state0.getPotentialEnergy(), state1.getPotentialEnergy(), TOL)


@pytest.mark.parametrize('platformName, precision', cases, ids=ids)
def testCoulomb(platformName, precision):
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    system.addParticle(1.0)
    system.addParticle(1.0)
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.addParticle(1.5, 1.0, 0.0)
    nonbonded.addParticle(-1.5, 1.0, 0.0)
    system.addForce(nonbonded)
    assert system.usesPeriodicBoundaryConditions()

    slicedNonbonded = plugin.SlicedPmeForce(nonbonded)
    slicedNonbonded.setForceGroup(1)
    system.addForce(slicedNonbonded)

    charge, sigma, epsilon = nonbonded.getParticleParameters(0)
    assert slicedNonbonded.getParticleCharge(0) == charge

    integrator = mm.VerletIntegrator(0.01)
    platform = mm.Platform.getPlatformByName(platformName)
    properties = {} if platformName == 'Reference' else {'Precision': precision}
    context = mm.Context(system, integrator, platform, properties)

    assert nonbonded.getPMEParameters() == slicedNonbonded.getPMEParameters()   
    assert nonbonded.getPMEParametersInContext(context) == slicedNonbonded.getPMEParametersInContext(context)
 
    positions = [mm.Vec3(0, 0, 0), mm.Vec3(2, 0, 0)]
    context.setPositions(positions)
    assert_forces_and_energy(context)


@pytest.mark.parametrize('platformName, precision', cases, ids=ids)
def testLargeSystem(platformName, precision):
    numMolecules = 600
    numParticles = numMolecules*2
    cutoff = 2.0
    boxSize = 20.0
    tol = 2e-3
    reference = mm.Platform.getPlatformByName("Reference")
    platform = mm.Platform.getPlatformByName(platformName)
    system = mm.System()
    for i in range(numParticles):
        system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(boxSize, 0, 0),
        mm.Vec3(0, boxSize, 0),
        mm.Vec3(0, 0, boxSize),
    )

    nonbonded = plugin.SlicedPmeForce(2)
    nonbonded.setCutoffDistance(cutoff)
    positions = np.empty(numParticles, mm.Vec3)

    M = int(numMolecules**(1.0/3.0))
    if (M*M*M < numMolecules):
        M += 1
    sqrt3 = math.sqrt(3)
    for k in range(numMolecules):
        iz = k//(M*M)
        iy = (k - iz*M*M)//M
        ix = k - M*(iy + iz*M)
        x = (ix + 0.5)*boxSize/M
        y = (iy + 0.5)*boxSize/M
        z = (iz + 0.5)*boxSize/M
        dx = (0.5 - ix%2)/2
        dy = (0.5 - iy%2)/2
        dz = (0.5 - iz%2)/2
        nonbonded.addParticle(1.0, 1)
        nonbonded.addParticle(-1.0, 1)
        nonbonded.addException(2*k, 2*k+1, 0.0)
        positions[2*k] = mm.Vec3(x+dx, y+dy, z+dz)
        positions[2*k+1] = mm.Vec3(x-dx, y-dy, z-dz)

    nonbonded.setParticleSubset(0, 0)
    nonbonded.setParticleSubset(1, 0)

    assert nonbonded.getParticleSubset(0) == 0
    assert nonbonded.getParticleSubset(2) == 1

    system.addForce(nonbonded)
    integrator1 = mm.VerletIntegrator(0.01)
    integrator2 = mm.VerletIntegrator(0.01)
    properties = {} if platformName == 'Reference' else {'Precision': precision}
    context = mm.Context(system, integrator1, platform, properties)
    referenceContext = mm.Context(system, integrator2, reference)
    context.setPositions(positions)
    referenceContext.setPositions(positions)
    kwargs = dict(
        getPositions=True,
        getVelocities=True,
        getForces=True,
        getEnergy=True,
    )
    state = context.getState(**kwargs)
    referenceState = referenceContext.getState(**kwargs)
    for i in range(numParticles):
        ASSERT_EQUAL_VEC(state.getPositions()[i], referenceState.getPositions()[i], tol)
        ASSERT_EQUAL_VEC(state.getVelocities()[i], referenceState.getVelocities()[i], tol)
        ASSERT_EQUAL_VEC(state.getForces()[i], referenceState.getForces()[i], tol)
    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), referenceState.getPotentialEnergy(), tol)
