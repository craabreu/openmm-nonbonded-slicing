import math

import nonbondedslicing as plugin
import numpy as np
import openmm as mm
import pytest
from openmm import unit

ONE_4PI_EPS0 = 138.935456

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


def assert_forces_and_energy(context, tol):
    state0 = context.getState(getForces=True, getEnergy=True, groups={0})
    state1 = context.getState(getForces=True, getEnergy=True, groups={1})
    for force0, force1 in zip(state0.getForces(), state1.getForces()):
        ASSERT_EQUAL_VEC(force0, force1, tol)
    ASSERT_EQUAL_TOL(state0.getPotentialEnergy(), state1.getPotentialEnergy(), tol)


@pytest.mark.parametrize('platformName', ['Reference', 'CUDA', 'OpenCL'])
def testParameterClash(platformName):
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = plugin.SlicedNonbondedForce(1)
    force.addParticle(1.5, 1, 0)
    force.addParticle(-1.5, 1, 0)
    force.addGlobalParameter("param", 1)
    force.addScalingParameter("param", 0, 0, True, True)
    force.addParticleParameterOffset("param", 0, 1, 1, 0)
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.01)
    platform = mm.Platform.getPlatformByName(platformName)
    with pytest.raises(Exception):
        context = mm.Context(system, integrator, platform)


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

    slicedNonbonded = plugin.SlicedNonbondedForce(nonbonded, 1)
    slicedNonbonded.setForceGroup(1)
    system.addForce(slicedNonbonded)

    charge1, sigma1, epsilon1 = nonbonded.getParticleParameters(0)
    charge2, sigma2, epsilon2 = slicedNonbonded.getParticleParameters(0)
    assert charge1 == charge2 and sigma1 == sigma2 and epsilon1 == epsilon2

    integrator = mm.VerletIntegrator(0.01)
    platform = mm.Platform.getPlatformByName(platformName)
    properties = {} if platformName == 'Reference' else {'Precision': precision}
    context = mm.Context(system, integrator, platform, properties)

    alpha1, nx1, ny1, nz1 = nonbonded.getPMEParameters()
    alpha2, nx2, ny2, nz2 = slicedNonbonded.getPMEParameters()
    assert alpha1 == alpha2 and nx1 == nx2 and ny1 == ny2 and nz1 == nz2


    alpha1, nx1, ny1, nz1 = nonbonded.getPMEParametersInContext(context)
    alpha2, nx2, ny2, nz2 = slicedNonbonded.getPMEParametersInContext(context)
    assert alpha1 == alpha2 and nx1 == nx2 and ny1 == ny2 and nz1 == nz2

    positions = [mm.Vec3(0, 0, 0), mm.Vec3(2, 0, 0)]
    context.setPositions(positions)
    tol = 1E-5 if platformName == 'Reference' or precision == 'double' else 1E-3
    assert_forces_and_energy(context, tol)


@pytest.mark.parametrize('platformName, precision', cases, ids=ids)
def testLargeSystem(platformName, precision):
    numMolecules = 600
    numParticles = numMolecules*2
    cutoff = 2.0
    boxSize = 20.0
    tol = 1E-5 if platformName == 'Reference' or precision == 'double' else 1E-3
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

    nonbonded = plugin.SlicedNonbondedForce(2)
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
        nonbonded.addParticle(1.0, 1, 0)
        nonbonded.addParticle(-1.0, 1, 0)
        nonbonded.addException(2*k, 2*k+1, 0, 1, 0)
        positions[2*k] = mm.Vec3(x+dx, y+dy, z+dz)
        positions[2*k+1] = mm.Vec3(x-dx, y-dy, z-dz)

    nonbonded.setParticleSubset(0, 0)
    nonbonded.setParticleSubset(1, 1)

    assert nonbonded.getParticleSubset(0) == 0
    assert nonbonded.getParticleSubset(1) == 1

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
