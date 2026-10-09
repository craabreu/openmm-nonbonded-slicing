import copy
import pickle
import random

import nonbondedslicing as plugin
import openmm as mm
import pytest
from openmm import unit


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


TOL_SLICE_DOUBLE = 1e-5
TOL_SLICE_SINGLE = 1e-4


def value(x):
    return x/x.unit if unit.is_quantity(x) else x


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


@pytest.mark.parametrize("platformName, precision", [case for case in cases if case.id != "Reference"])
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


SLICE_SUBSETS = 3
SLICE_KINDS = ("coulomb", "lj")
SLICE_METHODS = {
    "NoCutoff": mm.NonbondedForce.NoCutoff,
    "CutoffNonPeriodic": mm.NonbondedForce.CutoffNonPeriodic,
    "CutoffPeriodic": mm.NonbondedForce.CutoffPeriodic,
    "Ewald": mm.NonbondedForce.Ewald,
    "PME": mm.NonbondedForce.PME,
    "LJPME": mm.NonbondedForce.LJPME,
}
SLICE_COMBOS = [(name, "plain") for name in SLICE_METHODS] + [
    ("CutoffPeriodic", "switching"),
    ("PME", "switching"),
    ("CutoffPeriodic", "triclinic"),
    ("PME", "triclinic"),
    ("LJPME", "triclinic"),
]


def _sliceData(variant):
    rng = random.Random(7)
    numParticles, length = 48, 3.0
    if variant == "triclinic":
        box = [mm.Vec3(length, 0, 0), mm.Vec3(0.6, length, 0), mm.Vec3(-0.4, 0.5, length)]
    else:
        box = [mm.Vec3(length, 0, 0), mm.Vec3(0, length, 0), mm.Vec3(0, 0, length)]
    data = dict(
        box=box,
        positions=[box[0]*rng.random() + box[1]*rng.random() + box[2]*rng.random() for _ in range(numParticles)],
        subsets=[i % SLICE_SUBSETS for i in range(numParticles)],
        charges=[rng.uniform(-1, 1) for _ in range(numParticles)],
        sigmas=[rng.uniform(0.2, 0.35) for _ in range(numParticles)],
        epsilons=[rng.uniform(0.1, 0.8) for _ in range(numParticles)],
        exceptions=[(i, i+1, rng.uniform(-0.3, 0.3), 0.3, rng.uniform(0.0, 0.5)) for i in range(0, numParticles, 6)],
        particleOffsets=[(0, 0.7, 0.0, 0.2), (7, -0.4, 0.0, 0.1)],
        exceptionOffsets=[(1, 0.2, 0.0, 0.1)],
        offsetValue=0.6,
        scales={},
    )
    for i in range(SLICE_SUBSETS):
        for j in range(i, SLICE_SUBSETS):
            for kind in SLICE_KINDS:
                data["scales"][(i, j, kind)] = rng.uniform(-0.5, 2.0)
    return data


def _sliceNonbonded(data, method, variant, active, kind):
    force = mm.NonbondedForce()
    force.setNonbondedMethod(method)
    force.setCutoffDistance(1.0)
    force.setUseDispersionCorrection(True)
    force.setReciprocalSpaceForceGroup(1)
    if variant == "switching":
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(0.8)
    isActive = lambda i: data["subsets"][i] in active
    keepCoulomb = kind in ("coulomb", "all")
    keepLJ = kind in ("lj", "all")
    for i, (q, sigma, epsilon) in enumerate(zip(data["charges"], data["sigmas"], data["epsilons"])):
        force.addParticle(q if isActive(i) and keepCoulomb else 0.0, sigma, epsilon if isActive(i) and keepLJ else 0.0)
    for i, j, chargeProd, sigma, epsilon in data["exceptions"]:
        both = isActive(i) and isActive(j)
        force.addException(i, j, chargeProd if both and keepCoulomb else 0.0, sigma, epsilon if both and keepLJ else 0.0)
    force.addGlobalParameter("offset", data["offsetValue"])
    for i, dq, dsigma, depsilon in data["particleOffsets"]:
        force.addParticleParameterOffset(
            "offset", i, dq if isActive(i) and keepCoulomb else 0.0, dsigma, depsilon if isActive(i) and keepLJ else 0.0
        )
    for k, dq, dsigma, depsilon in data["exceptionOffsets"]:
        i, j = data["exceptions"][k][:2]
        both = isActive(i) and isActive(j)
        force.addExceptionParameterOffset(
            "offset", k, dq if both and keepCoulomb else 0.0, dsigma, depsilon if both and keepLJ else 0.0
        )
    return force


def _sliceContext(data, force, platform, properties):
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*data["box"])
    for _ in data["positions"]:
        system.addParticle(1.0)
    system.addForce(force)
    context = mm.Context(system, mm.VerletIntegrator(0.001), platform, properties)
    context.setPositions(data["positions"])
    return system, context


def _groupEnergies(data, force, platform, properties):
    system, context = _sliceContext(data, force, platform, properties)
    return {group: value(context.getState(getEnergy=True, groups={group}).getPotentialEnergy()) for group in (0, 1)}


@pytest.mark.parametrize("methodName, variant", SLICE_COMBOS, ids=[f"{m}-{v}" for m, v in SLICE_COMBOS])
@pytest.mark.parametrize("platformName, precision", cases)
def testSliceEnergiesAndDerivatives(platformName, precision, methodName, variant):
    method = SLICE_METHODS[methodName]
    platform = mm.Platform.getPlatformByName(platformName)
    properties = {} if platformName == "Reference" else {"Precision": precision}
    tol = TOL_SLICE_DOUBLE if platformName == "Reference" or precision == "double" else TOL_SLICE_SINGLE
    data = _sliceData(variant)

    oracle = {}
    for kind in SLICE_KINDS:
        single = {i: _groupEnergies(data, _sliceNonbonded(data, method, variant, {i}, kind), platform, properties)
                  for i in range(SLICE_SUBSETS)}
        for i in range(SLICE_SUBSETS):
            for j in range(i, SLICE_SUBSETS):
                if i == j:
                    oracle[(i, j, kind)] = single[i]
                else:
                    pair = _groupEnergies(data, _sliceNonbonded(data, method, variant, {i, j}, kind), platform, properties)
                    oracle[(i, j, kind)] = {g: pair[g] - single[i][g] - single[j][g] for g in (0, 1)}
    assert max(abs(e[g]) for e in oracle.values() for g in (0, 1)) > 1.0

    sliced = plugin.SlicedNonbondedForce(_sliceNonbonded(data, method, variant, set(range(SLICE_SUBSETS)), "all"), SLICE_SUBSETS)
    for i, subset in enumerate(data["subsets"]):
        sliced.setParticleSubset(i, subset)
    names = {}
    for (i, j, kind), scale in data["scales"].items():
        names[(i, j, kind)] = name = f"lambda{i}{j}{kind}"
        sliced.addGlobalParameter(name, scale)
        sliced.addScalingParameter(name, i, j, kind == "coulomb", kind == "lj")
        sliced.addEnergyParameterDerivative(name)
    system, context = _sliceContext(data, sliced, platform, properties)

    for groups in ({0}, {1}, {0, 1}):
        expected = {key: sum(oracle[key][g] for g in groups) for key in oracle}
        withEnergy = context.getState(getEnergy=True, getParameterDerivatives=True, groups=groups)
        derivativesOnly = context.getState(getParameterDerivatives=True, groups=groups)
        for state in (withEnergy, derivativesOnly):
            derivatives = state.getEnergyParameterDerivatives()
            for key, name in names.items():
                ASSERT_EQUAL_TOL(expected[key], derivatives[name], tol)
        total = sum(data["scales"][key]*expected[key] for key in expected)
        ASSERT_EQUAL_TOL(total, withEnergy.getPotentialEnergy(), tol)
