import nonbondedslicing as plugin
import numpy as np
import openmm as mm
import pytest
from openmm import unit

ONE_4PI_EPS0 = 138.935456

TOL = 1.0E-5

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


@pytest.mark.parametrize('platformName, precision', cases, ids=ids)
def testCoulomb(platformName, precision):
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    integrator = mm.VerletIntegrator(0.01)
    forceField = plugin.SlicedNonbondedForce()
    forceField.addParticle(0.5, 1, 0)
    forceField.addParticle(-1.5, 1, 0)
    system.addForce(forceField)
    ASSERT(not forceField.usesPeriodicBoundaryConditions())
    ASSERT(not system.usesPeriodicBoundaryConditions())
    platform = mm.Platform.getPlatformByName(platformName)
    properties = {} if platformName == 'Reference' else {'Precision': precision}
    context = mm.Context(system, integrator, platform, properties)
    positions = [mm.Vec3(0, 0, 0), mm.Vec3(2, 0, 0)]
    context.setPositions(positions)
    state = context.getState(getForces=True, getEnergy=True)
    forces = state.getForces()
    force = ONE_4PI_EPS0*(-0.75)/4.0
    ASSERT_EQUAL_VEC(mm.Vec3(-force, 0, 0), forces[0], TOL)
    ASSERT_EQUAL_VEC(mm.Vec3(force, 0, 0), forces[1], TOL)
    ASSERT_EQUAL_TOL(ONE_4PI_EPS0*(-0.75)/2.0, state.getPotentialEnergy(), TOL)


@pytest.mark.parametrize('platformName, precision', cases, ids=ids)
def testLJ(platformName, precision):
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    integrator = mm.VerletIntegrator(0.01)
    forceField = plugin.SlicedNonbondedForce()
    forceField.addParticle(0, 1.2, 1)
    forceField.addParticle(0, 1.4, 2)
    system.addForce(forceField)
    ASSERT(not forceField.usesPeriodicBoundaryConditions())
    ASSERT(not system.usesPeriodicBoundaryConditions())
    platform = mm.Platform.getPlatformByName(platformName)
    properties = {} if platformName == 'Reference' else {'Precision': precision}
    context = mm.Context(system, integrator, platform, properties)
    positions = [mm.Vec3(0, 0, 0), mm.Vec3(2, 0, 0)]
    context.setPositions(positions)
    state = context.getState(getForces=True, getEnergy=True)
    forces = state.getForces()
    x = 1.3/2.0
    eps = np.sqrt(2.0)
    force = 4.0*eps*(12*x**12-6*x**6)/2.0
    ASSERT_EQUAL_VEC(mm.Vec3(-force, 0, 0), forces[0], TOL)
    ASSERT_EQUAL_VEC(mm.Vec3(force, 0, 0), forces[1], TOL)
    ASSERT_EQUAL_TOL(4.0*eps*(x**12-x**6), state.getPotentialEnergy(), TOL)


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

    nonbonded = plugin.SlicedNonbondedForce()
    bonds = mm.HarmonicBondForce()
    positions = np.empty(numParticles, mm.Vec3)
    velocities = np.empty(numParticles, mm.Vec3)
    rng = np.random.default_rng(19283)

    def random_vec():
        return mm.Vec3(rng.random(), rng.random(), rng.random())

    for i in range(numMolecules):
        if (i < numMolecules/2):
            nonbonded.addParticle(-1.0, 0.2, 0.1)
            nonbonded.addParticle(1.0, 0.1, 0.1)
        else:
            nonbonded.addParticle(-1.0, 0.2, 0.2)
            nonbonded.addParticle(1.0, 0.1, 0.2)

        positions[2*i] = boxSize*random_vec()
        positions[2*i+1] = positions[2*i] + mm.Vec3(1.0, 0.0, 0.0)
        velocities[2*i] = random_vec()
        velocities[2*i+1] = random_vec()
        bonds.addBond(2*i, 2*i+1, 1.0, 0.1)
        nonbonded.addException(2*i, 2*i+1, 0.0, 0.15, 0.0)

    # Try with no cutoffs and make sure it agrees with the Reference platform.

    nonbonded.setNonbondedMethod(plugin.SlicedNonbondedForce.NoCutoff)
    system.addForce(nonbonded)
    system.addForce(bonds)
    integrator1 = mm.VerletIntegrator(0.01)
    integrator2 = mm.VerletIntegrator(0.01)
    properties = {} if platformName == 'Reference' else {'Precision': precision}
    context = mm.Context(system, integrator1, platform, properties)
    referenceContext = mm.Context(system, integrator2, reference)
    context.setPositions(positions)
    context.setVelocities(velocities)
    referenceContext.setPositions(positions)
    referenceContext.setVelocities(velocities)
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

    # Now try cutoffs but not periodic boundary conditions.

    nonbonded.setNonbondedMethod(plugin.SlicedNonbondedForce.CutoffNonPeriodic)
    nonbonded.setCutoffDistance(cutoff)
    context.reinitialize(True)
    referenceContext.reinitialize(True)
    state = context.getState(**kwargs)
    referenceState = referenceContext.getState(**kwargs)
    for i in range(numParticles):
        ASSERT_EQUAL_VEC(state.getPositions()[i], referenceState.getPositions()[i], tol)
        ASSERT_EQUAL_VEC(state.getVelocities()[i], referenceState.getVelocities()[i], tol)
        ASSERT_EQUAL_VEC(state.getForces()[i], referenceState.getForces()[i], tol)
    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), referenceState.getPotentialEnergy(), tol)

    # Now do the same thing with periodic boundary conditions.

    nonbonded.setNonbondedMethod(plugin.SlicedNonbondedForce.CutoffPeriodic)
    context.reinitialize(True)
    referenceContext.reinitialize(True)
    state = context.getState(**kwargs)
    referenceState = referenceContext.getState(**kwargs)
    for i in range(numParticles):
        dx = state.getPositions()[i][0]-referenceState.getPositions()[i][0]
        dy = state.getPositions()[i][1]-referenceState.getPositions()[i][1]
        dz = state.getPositions()[i][2]-referenceState.getPositions()[i][2]
        dx /= dx.unit
        dy /= dy.unit
        dz /= dz.unit
        ASSERT_EQUAL_TOL(dx-np.floor(dx/boxSize+0.5)*boxSize, 0, tol)
        ASSERT_EQUAL_TOL(dy-np.floor(dy/boxSize+0.5)*boxSize, 0, tol)
        ASSERT_EQUAL_TOL(dz-np.floor(dz/boxSize+0.5)*boxSize, 0, tol)
        ASSERT_EQUAL_VEC(state.getVelocities()[i], referenceState.getVelocities()[i], tol)
        ASSERT_EQUAL_VEC(state.getForces()[i], referenceState.getForces()[i], tol)
    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), referenceState.getPotentialEnergy(), tol)
