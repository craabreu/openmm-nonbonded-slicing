import copy
import pickle

import nonbondedslicing as plugin
import openmm as mm
import pytest
from openmm import unit

ONE_4PI_EPS0 = 138.935456


def _isAvailable(platformName):
    try:
        platform = mm.Platform.getPlatformByName(platformName)
        system = mm.System()
        system.addParticle(1.0)
        mm.Context(system, mm.VerletIntegrator(1.0), platform)
        return True
    except Exception:
        return False


available = {name: _isAvailable(name) for name in ["Reference", "CUDA", "OpenCL"]}

cases = [
    pytest.param(name, precision, id=name+precision,
                 marks=pytest.mark.skipif(not available[name], reason=f"{name} platform not available"))
    for name, precision in [
        ("Reference", ""),
        ("CUDA", "single"), ("CUDA", "mixed"), ("CUDA", "double"),
        ("OpenCL", "single"), ("OpenCL", "mixed"), ("OpenCL", "double"),
    ]
]


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


def testErrors():
    force = plugin.SlicedNonbondedForce(3)
    force.addParticle(0.0, 1.0, 0.0)
    for name in "abcd":
        force.addGlobalParameter(name, 1.0)
    with pytest.raises(Exception, match="out of range"):
        force.setParticleSubset(0, 3)
    with pytest.raises(Exception, match="cannot be both false"):
        force.addScalingParameter("a", 0, 1, False, False)
    with pytest.raises(Exception, match="There is no global parameter called"):
        force.addScalingParameter("unknown", 0, 1, True, False)
    force.addScalingParameter("a", 0, 1, True, False)
    force.addScalingParameter("b", 0, 1, False, True)
    with pytest.raises(Exception, match="Clash detected between scaling parameters"):
        force.addScalingParameter("c", 1, 0, True, False)
    with pytest.raises(Exception, match="has already been defined for this slice"):
        force.setScalingParameter(0, "a", 0, 1, True, True)
    with pytest.raises(Exception, match="There is no scaling parameter called"):
        force.addEnergyParameterDerivative("d")
    force.addEnergyParameterDerivative("a")
    with pytest.raises(Exception, match="has already been requested"):
        force.addEnergyParameterDerivative("a")


def testParameterClash():
    system = mm.System()
    system.addParticle(1.0)
    force = plugin.SlicedNonbondedForce(1)
    force.addParticle(1.5, 1, 0)
    force.addGlobalParameter("param", 1)
    force.addScalingParameter("param", 0, 0, True, True)
    force.addParticleParameterOffset("param", 0, 1, 1, 0)
    system.addForce(force)
    platform = mm.Platform.getPlatformByName("Reference")
    with pytest.raises(Exception, match="Cannot use a global parameter for both"):
        mm.Context(system, mm.VerletIntegrator(0.01), platform)


@pytest.mark.parametrize("platformName, precision", cases)
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
    assert alpha1 == alpha2
    assert nx1 == nx2 and ny1 == ny2 and nz1 == nz2

    positions = [mm.Vec3(0, 0, 0), mm.Vec3(2, 0, 0)]
    context.setPositions(positions)
    tol = 1e-4 if platformName == 'Reference' or precision == 'double' else 1e-3
    assert_forces_and_energy(context, tol)


@pytest.mark.parametrize("platformName, precision", cases)
def testLargeSystem(platformName, precision):
    numMolecules = 600
    numParticles = numMolecules*2
    boxSize = 20.0
    tol = 1e-4 if platformName == "Reference" or precision == "double" else 1e-3
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(2.0)
    positions = []
    M = int(numMolecules**(1.0/3.0))
    if M*M*M < numMolecules:
        M += 1
    for k in range(numMolecules):
        iz = k//(M*M)
        iy = (k - iz*M*M)//M
        ix = k - M*(iy + iz*M)
        center = mm.Vec3(ix + 0.5, iy + 0.5, iz + 0.5)*(boxSize/M)
        delta = mm.Vec3(0.5 - ix%2, 0.5 - iy%2, 0.5 - iz%2)/2
        positions += [center + delta, center - delta]
        nonbonded.addParticle(1.0, 0.5, 0.2)
        nonbonded.addParticle(-1.0, 0.5, 0.2)
        nonbonded.addException(2*k, 2*k+1, 0.0, 0.5, 0.0)

    force = plugin.SlicedNonbondedForce(nonbonded, 3)
    for i in range(numParticles):
        force.setParticleSubset(i, i % 3)
    force.addGlobalParameter("lambda", 0.5)
    force.addScalingParameter("lambda", 0, 1, True, True)
    force.addScalingParameter("lambda", 1, 2, True, False)
    force.addEnergyParameterDerivative("lambda")

    system = mm.System()
    for i in range(numParticles):
        system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(boxSize, 0, 0), mm.Vec3(0, boxSize, 0), mm.Vec3(0, 0, boxSize))
    system.addForce(force)

    def compute(platform, properties):
        context = mm.Context(system, mm.VerletIntegrator(0.01), platform, properties)
        context.setPositions(positions)
        return context.getState(getEnergy=True, getForces=True, getParameterDerivatives=True)

    properties = {} if platformName == "Reference" else {"Precision": precision}
    state = compute(mm.Platform.getPlatformByName(platformName), properties)
    reference = compute(mm.Platform.getPlatformByName("Reference"), {})
    ASSERT_EQUAL_TOL(reference.getPotentialEnergy(), state.getPotentialEnergy(), tol)
    ASSERT_EQUAL_TOL(
        reference.getEnergyParameterDerivatives()["lambda"],
        state.getEnergyParameterDerivatives()["lambda"],
        tol,
    )
    for f0, f1 in zip(reference.getForces(), state.getForces()):
        ASSERT_EQUAL_VEC(f0, f1, tol)


def testScalingParameterAccessors():
    force = plugin.SlicedNonbondedForce(3)
    force.addGlobalParameter("a", 1.0)
    force.addGlobalParameter("b", 1.0)
    index = force.addScalingParameter("a", 2, 1, True, False)
    assert force.getNumScalingParameters() == 1
    assert list(force.getScalingParameter(index)) == ["a", 2, 1, True, False]
    force.setScalingParameter(index, "b", 0, 2, False, True)
    assert list(force.getScalingParameter(index)) == ["b", 0, 2, False, True]


def _sampleForce():
    force = plugin.SlicedNonbondedForce(3)
    force.addParticle(1.0, 0.3, 0.5)
    force.addParticle(-1.0, 0.3, 0.5)
    force.setParticleSubset(1, 2)
    force.addGlobalParameter("a", 0.5)
    force.addScalingParameter("a", 2, 1, True, False)
    force.addEnergyParameterDerivative("a")
    force.setUseCuFFT(False)
    return force


@pytest.mark.parametrize(
    "duplicate",
    [copy.copy, copy.deepcopy, lambda force: pickle.loads(pickle.dumps(force))],
    ids=["copy", "deepcopy", "pickle"],
)
def testCopyAndPickle(duplicate):
    force = _sampleForce()
    other = duplicate(force)
    assert other is not force
    assert isinstance(other, plugin.SlicedNonbondedForce)
    assert other.getNumSubsets() == 3
    assert other.getParticleSubset(1) == 2
    assert list(other.getScalingParameter(0)) == ["a", 2, 1, True, False]
    assert other.getEnergyParameterDerivativeName(0) == "a"
    assert other.getUseCuFFT() is False
    other.setParticleSubset(0, 1)
    assert force.getParticleSubset(0) == 0


def testSystemSerialization():
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addForce(_sampleForce())
    clone = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    force = plugin.SlicedNonbondedForce.cast(clone.getForce(0))
    assert force.getNumSubsets() == 3
    assert force.getParticleSubset(1) == 2
    assert force.getUseCuFFT() is False


def testDerivatives():
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(1.0)
    positions = []
    for i in range(4):
        for j in range(4):
            for k in range(2):
                positions.append(mm.Vec3(0.8*i + 0.1*k, 0.8*j + 0.05*k, 0.4*k))
                nonbonded.addParticle(0.5 if (i + j + k) % 2 else -0.5, 0.3, 0.5)
    force = plugin.SlicedNonbondedForce(nonbonded, 2)
    for i in range(len(positions)):
        force.setParticleSubset(i, i % 2)
    force.addGlobalParameter("lambda", 0.5)
    force.addScalingParameter("lambda", 0, 1, True, True)
    force.addEnergyParameterDerivative("lambda")
    system = mm.System()
    for _ in positions:
        system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3.2, 0, 0), mm.Vec3(0, 3.2, 0), mm.Vec3(0, 0, 3.2))
    system.addForce(force)
    context = mm.Context(system, mm.VerletIntegrator(0.01), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    derivative = context.getState(getParameterDerivatives=True).getEnergyParameterDerivatives()["lambda"]
    energy = {}
    for scale in (0.0, 1.0):
        context.setParameter("lambda", scale)
        energy[scale] = value(context.getState(getEnergy=True).getPotentialEnergy())
    ASSERT_EQUAL_TOL(energy[1.0] - energy[0.0], derivative, 1e-6)
    assert abs(derivative) > 1.0


def testCastAndIsinstance():
    system = mm.System()
    system.addParticle(1.0)
    system.addForce(mm.HarmonicBondForce())
    force = plugin.SlicedNonbondedForce(2)
    force.addParticle(0.0, 1.0, 0.0)
    system.addForce(force)
    assert not plugin.SlicedNonbondedForce.isinstance(system.getForce(0))
    assert plugin.SlicedNonbondedForce.isinstance(system.getForce(1))
    assert plugin.SlicedNonbondedForce.cast(system.getForce(1)).getNumSubsets() == 2


def testVersion():
    from importlib.metadata import version
    assert plugin.__version__ == version("nonbondedslicing")
