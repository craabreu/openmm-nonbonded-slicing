import copy
import math
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
    with pytest.raises(Exception, match="must apply to Coulomb interactions"):
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


def testDecappingAPI():
    force = plugin.SlicedNonbondedForce(3)
    force.addParticle(0.0, 1.0, 1.0)
    force.addGlobalParameter("lambda", 1.0)
    index = force.addScalingParameter("lambda", 0, 1, False, True)
    assert not force.getScalingParameterIncludeDecapping(index)
    assert force.getScalingParameter(index) == ["lambda", 0, 1, False, True]
    force.setScalingParameter(index, "lambda", 1, 0, False, True, True)
    assert force.getScalingParameterIncludeDecapping(index)
    assert force.getScalingParameter(index) == ["lambda", 1, 0, False, True]
    with pytest.raises(Exception):
        force.addScalingParameter("lambda", 0, 2, False, False)
    with pytest.raises(Exception):
        force.addScalingParameter("lambda", 0, 1, True, False, True)
    decap = force.addScalingParameter("lambda", 0, 2, False, False, True)
    with pytest.raises(Exception):
        force.getScalingParameterIncludeDecapping(force.getNumScalingParameters())
    clone = copy.deepcopy(force)
    assert clone.getScalingParameterIncludeDecapping(index)
    assert clone.getScalingParameterIncludeDecapping(decap)


def testCappingOrdersAPI():
    force = plugin.SlicedNonbondedForce(2)
    assert (force.getCappingOrderAtZero(), force.getCappingOrderAtSigma()) == (3, 3)
    for orders in ((0, 3), (3, 0), (7, 3), (3, 7)):
        with pytest.raises(Exception):
            force.setCappingOrders(*orders)
    force.setCappingOrders(4, 2)
    assert (force.getCappingOrderAtZero(), force.getCappingOrderAtSigma()) == (4, 2)
    clone = copy.deepcopy(force)
    assert (clone.getCappingOrderAtZero(), clone.getCappingOrderAtSigma()) == (4, 2)


@pytest.mark.parametrize("platformName, precision", cases)
def testCappingOrders(platformName, precision):
    """The capped energy with orders (4, 2), 4*epsilon*(5.8 - 28.8*x^5 + 23*x^6), and with the default ones."""
    def energy(orders, x):
        system = mm.System()
        for _ in range(2):
            system.addParticle(1.0)
        force = plugin.SlicedNonbondedForce(2)
        force.setNonbondedMethod(plugin.SlicedNonbondedForce.NoCutoff)
        force.addParticle(0.0, 1.0, 0.5)
        force.addParticle(0.0, 1.0, 0.5)
        force.setParticleSubset(1, 1)
        force.addGlobalParameter("lambdaD", 0.0)
        force.addScalingParameter("lambdaD", 0, 1, False, False, True)
        if orders is not None:
            force.setCappingOrders(*orders)
        system.addForce(force)
        platform = mm.Platform.getPlatformByName(platformName)
        properties = {} if platformName == "Reference" else {"Precision": precision}
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform, properties)
        context.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(x, 0, 0)])
        return value(context.getState(getEnergy=True).getPotentialEnergy())

    tol = 1e-6 if platformName == "Reference" or precision == "double" else 1e-4
    for x in (0.0001, 0.5, 0.9):
        ASSERT_EQUAL_TOL(2*(5.8 - 28.8*x**5 + 23*x**6), energy((4, 2), x), tol)
        ASSERT_EQUAL_TOL(2*(-868*x**6 + 2188.8*x**5 - 1440*x**4 + 119.2)/4, energy(None, x), tol)


@pytest.mark.parametrize("platformName, precision", cases)
def testDecapping(platformName, precision):
    """Decapped minus plain sliced energies match (lambdaD - lambdaLJ)*Delta from Custom forces."""
    random.seed(1)
    numMolecules = 100
    numParticles = 2*numMolecules
    boxSize = 10.0
    cutoff = 3.5
    lamLJ, lamD = 0.4, 0.7

    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(cutoff)
    nonbonded.setEwaldErrorTolerance(1e-4)
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
        nonbonded.addParticle(0.2, 1.0, 1.0)
        nonbonded.addParticle(-0.2, 1.0, 1.0)
        nonbonded.addException(2*k, 2*k+1, -0.04, 1.0, 1.0)
    subsets = [random.randint(0, 1) for _ in range(numParticles)]

    def makeSystem(*forces):
        system = mm.System()
        for _ in range(numParticles):
            system.addParticle(1.0)
        system.setDefaultPeriodicBoxVectors(mm.Vec3(boxSize, 0, 0), mm.Vec3(0, boxSize, 0), mm.Vec3(0, 0, boxSize))
        for force in forces:
            system.addForce(force)
        return system

    def makeSliced(decap):
        sliced = plugin.SlicedNonbondedForce(nonbonded, 2)
        for i, s in enumerate(subsets):
            sliced.setParticleSubset(i, s)
        sliced.addGlobalParameter("lambdaLJ", lamLJ)
        sliced.addGlobalParameter("lambdaD", lamD)
        sliced.addScalingParameter("lambdaLJ", 0, 1, True, True)
        sliced.addEnergyParameterDerivative("lambdaLJ")
        if decap:
            sliced.addScalingParameter("lambdaD", 0, 1, False, False, True)
            sliced.addEnergyParameterDerivative("lambdaD")
        return sliced

    expression = ("(lambdaD-lambdaLJ)*step(1-x)*4*eps*(x^(-12)-x^(-6)-(-868*x^6+2188.8*x^5-1440*x^4+119.2)/4);"
                  "x=r/sigma;")
    custom = mm.CustomNonbondedForce(expression + "sigma=0.5*(sigma1+sigma2); eps=sqrt(eps1*eps2)")
    custom.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    custom.setCutoffDistance(cutoff)
    custom.addPerParticleParameter("sigma")
    custom.addPerParticleParameter("eps")
    bonds = mm.CustomBondForce(expression)
    bonds.addPerBondParameter("sigma")
    bonds.addPerBondParameter("eps")
    for force in (custom, bonds):
        for name, value_ in (("lambdaLJ", lamLJ), ("lambdaD", lamD)):
            force.addGlobalParameter(name, value_)
            force.addEnergyParameterDerivative(name)
    for _ in range(numParticles):
        custom.addParticle([1.0, 1.0])
    custom.addInteractionGroup({i for i, s in enumerate(subsets) if s == 0}, {i for i, s in enumerate(subsets) if s == 1})
    for k in range(numMolecules):
        i, j = 2*k, 2*k+1
        custom.addExclusion(i, j)
        if subsets[i] != subsets[j]:
            bonds.addBond(i, j, [1.0, 1.0])

    platform = mm.Platform.getPlatformByName(platformName)
    properties = {} if platformName == "Reference" else {"Precision": precision}
    contexts = []
    for system in (makeSystem(makeSliced(True)), makeSystem(makeSliced(False)), makeSystem(custom, bonds)):
        context = mm.Context(system, mm.VerletIntegrator(0.01), platform, properties)
        context.setPositions(positions)
        contexts.append(context)
    if platformName == "Reference" or precision == "double":
        tol = 1e-4
    else:
        tol = 1e-3 if precision == "mixed" else 5e-3
    for lj, d in ((lamLJ, lamD), (1.0, 0.0), (0.0, 1.0)):
        states = []
        for context in contexts:
            context.setParameter("lambdaLJ", lj)
            context.setParameter("lambdaD", d)
            states.append(context.getState(getEnergy=True, getForces=True, getParameterDerivatives=True))
        decap, plain, diff = states
        scale = max(1.0, abs(value(plain.getPotentialEnergy())))
        ASSERT_EQUAL_TOL(value(diff.getPotentialEnergy())/scale,
                         (value(decap.getPotentialEnergy()) - value(plain.getPotentialEnergy()))/scale, tol)
        for fd_, fp, fx in zip(decap.getForces(), plain.getForces(), diff.getForces()):
            ASSERT_EQUAL_VEC(fx, fd_ - fp, tol)
        dDecap, dPlain, dDiff = (state.getEnergyParameterDerivatives() for state in states)
        ASSERT_EQUAL_TOL(dDiff["lambdaLJ"], dDecap["lambdaLJ"] - dPlain["lambdaLJ"], tol)
        ASSERT_EQUAL_TOL(dDiff["lambdaD"], dDecap["lambdaD"], tol)


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


def _dipoleSystem():
    """1200 particles in 600 dipoles, sliced into 3 subsets."""
    numMolecules = 600
    numParticles = numMolecules*2
    boxSize = 20.0
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
    return system, positions


@pytest.mark.parametrize("platformName, precision", [case for case in cases if case.id != "Reference"])
def testLargeSystem(platformName, precision):
    tol = 1e-4 if platformName == "Reference" or precision == "double" else 1e-3
    system, positions = _dipoleSystem()

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


@pytest.mark.parametrize("platformName, precision", [case for case in cases if case.id != "Reference"])
def testForcesBetweenSorts(platformName, precision):
    """In small systems, the atoms are sorted for PME only at some steps; the forces
    must stay correct in the steps between."""
    tol = 1e-4 if precision == "double" else 1e-3
    system, positions = _dipoleSystem()
    platform = mm.Platform.getPlatformByName(platformName)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, platform, {"Precision": precision})
    context.setPositions(positions)
    reference = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    for _ in range(6):
        integrator.step(1)
        state = context.getState(getPositions=True, getEnergy=True, getForces=True)
        reference.setPositions(state.getPositions())
        expected = reference.getState(getEnergy=True, getForces=True)
        ASSERT_EQUAL_TOL(expected.getPotentialEnergy(), state.getPotentialEnergy(), tol)
        for f0, f1 in zip(expected.getForces(), state.getForces()):
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
    force.setReciprocalExclusions(True)
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
    assert other.getReciprocalExclusions() is True
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
    assert force.getReciprocalExclusions() is True


def testReciprocalExclusionsDefault():
    assert plugin.SlicedNonbondedForce(2).getReciprocalExclusions() is False
    assert plugin.SlicedNonbondedForce(mm.NonbondedForce(), 2).getReciprocalExclusions() is False


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
    derivative = context.getState(getEnergy=True, getParameterDerivatives=True).getEnergyParameterDerivatives()["lambda"]
    energy = {}
    for scale in (0.0, 1.0):
        context.setParameter("lambda", scale)
        energy[scale] = value(context.getState(getEnergy=True).getPotentialEnergy())
    ASSERT_EQUAL_TOL(energy[1.0] - energy[0.0], derivative, 1e-6)
    assert abs(derivative) > 1.0


@pytest.mark.parametrize("methodName", ["NoCutoff", "CutoffPeriodic", "Ewald", "PME", "LJPME"])
@pytest.mark.parametrize("platformName, precision", cases)
def testNoDerivativesWhileAnotherForceHasThem(platformName, precision, methodName):
    """A force without derivatives must work in a context where another force has them."""
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(SLICE_METHODS[methodName])
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
    bond = mm.CustomBondForce("k*r^2")
    bond.addGlobalParameter("k", 1.0)
    bond.addEnergyParameterDerivative("k")
    bond.addBond(0, 1, [])
    bond.setForceGroup(1)
    system = mm.System()
    for _ in positions:
        system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3.2, 0, 0), mm.Vec3(0, 3.2, 0), mm.Vec3(0, 0, 3.2))
    system.addForce(force)
    system.addForce(bond)
    platform = mm.Platform.getPlatformByName(platformName)
    properties = {"Precision": precision} if precision else {}
    context = mm.Context(system, mm.VerletIntegrator(0.01), platform, properties)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getParameterDerivatives=True)
    derivatives = dict(state.getEnergyParameterDerivatives())
    # As OpenMM does for every global parameter, an unrequested derivative is reported as zero
    assert derivatives.get("lambda", 0.0) == 0.0
    delta = positions[1] - positions[0]
    r2 = delta[0]**2 + delta[1]**2 + delta[2]**2
    ASSERT_EQUAL_TOL(r2, derivatives["k"], 1e-5)
    reference = mm.Context(system, mm.VerletIntegrator(0.01), mm.Platform.getPlatformByName("Reference"))
    reference.setPositions(positions)
    expected = value(reference.getState(getEnergy=True, groups={0}).getPotentialEnergy())
    found = value(context.getState(getEnergy=True, groups={0}).getPotentialEnergy())
    ASSERT_EQUAL_TOL(expected, found, TOL_SLICE_SINGLE if precision == "single" else TOL_SLICE_DOUBLE)


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
    from importlib.metadata import packages_distributions, version
    # conda installs the distribution "nonbondedslicing"; PyPI installs "openmm-nonbonded-slicing",
    # whose versions may carry a ".postN" suffix that selects the OpenMM version
    distribution = packages_distributions()["nonbondedslicing"][0]
    assert version(distribution).split(".post")[0] == plugin.__version__


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
        derivatives = withEnergy.getEnergyParameterDerivatives()
        for key, name in names.items():
            ASSERT_EQUAL_TOL(expected[key], derivatives[name], tol)
        # Derivatives are only computed along with the energy
        derivatives = context.getState(getParameterDerivatives=True, groups=groups).getEnergyParameterDerivatives()
        assert all(math.isnan(derivatives[name]) for name in names.values())
        total = sum(data["scales"][key]*expected[key] for key in expected)
        ASSERT_EQUAL_TOL(total, withEnergy.getPotentialEnergy(), tol)


def _innerGroupSystem(split):
    # Lattice of charged LJ particles with 1-2 exclusions and scaled 1-4 exceptions, with neighbors (0.6 nm)
    # and 1-4 pairs (0.69 nm) inside the switching range; group 0 is direct space, 1 the inner group (when
    # split), 2 reciprocal space.
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(1.0)
    positions = []
    for i in range(4):
        for j in range(4):
            for k in range(2):
                positions.append(mm.Vec3(0.6*i + 0.1*k, 0.6*j + 0.05*k, 0.4*k))
                nonbonded.addParticle(0.5 if (i + j + k) % 2 else -0.5, 0.3, 0.5)
    for i in range(0, len(positions), 2):
        nonbonded.addException(i, i + 1, 0.0, 1.0, 0.0)
    for i in range(1, len(positions) - 1, 4):
        nonbonded.addException(i, i + 1, 0.1, 0.25, 0.2)
    force = plugin.SlicedNonbondedForce(nonbonded, 2)
    for i in range(len(positions)):
        force.setParticleSubset(i, i % 2)
    force.addGlobalParameter("lambda", 0.5)
    force.addScalingParameter("lambda", 0, 1, True, True)
    force.addEnergyParameterDerivative("lambda")
    force.setReciprocalSpaceForceGroup(2)
    if split:
        force.setInnerForceGroup(1)
        force.setInnerSwitchingDistance(0.5)
        force.setInnerCutoffDistance(0.8)
    system = mm.System()
    for _ in positions:
        system.addParticle(10.0)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3.2, 0, 0), mm.Vec3(0, 3.2, 0), mm.Vec3(0, 0, 3.2))
    system.addForce(force)
    return system, force, positions


def testInnerGroupAccessors():
    force = plugin.SlicedNonbondedForce(1)
    assert force.getInnerForceGroup() == -1
    force.setInnerForceGroup(3)
    force.setInnerCutoffDistance(0.8*unit.nanometer)
    force.setInnerSwitchingDistance(0.5*unit.nanometer)
    assert force.getInnerForceGroup() == 3
    # Distances come back as quantities in nanometers, like the inherited getters
    for getter, expected in [(force.getInnerCutoffDistance, 0.8), (force.getInnerSwitchingDistance, 0.5)]:
        distance = getter()
        assert unit.is_quantity(distance)
        ASSERT_EQUAL_TOL(expected, distance.value_in_unit(unit.nanometer), 1e-12)
    force.setCutoffDistance(1.0*unit.nanometer)
    assert force.getInnerSwitchingDistance() < force.getInnerCutoffDistance() < force.getCutoffDistance()
    force.setInnerCutoffDistance(8*unit.angstrom)
    ASSERT_EQUAL_TOL(0.8, force.getInnerCutoffDistance().value_in_unit(unit.nanometer), 1e-12)
    with pytest.raises(Exception):
        force.setInnerForceGroup(32)
    copy_ = plugin.SlicedNonbondedForce._deserialize(mm.XmlSerializer.serialize(force))
    assert copy_.getInnerForceGroup() == 3
    ASSERT_EQUAL_TOL(0.5, copy_.getInnerSwitchingDistance().value_in_unit(unit.nanometer), 1e-12)


@pytest.mark.parametrize("platformName, precision", cases)
def testInnerGroupWithDecapping(platformName, precision):
    # Decapping terms go entirely to the inner group, which needs sigma <= the inner switching distance
    def makeContext(sigma):
        system = mm.System()
        for _ in range(2):
            system.addParticle(1.0)
        force = plugin.SlicedNonbondedForce(2)
        force.setNonbondedMethod(plugin.SlicedNonbondedForce.CutoffNonPeriodic)
        force.setCutoffDistance(1.0)
        force.addParticle(0.0, sigma, 0.5)
        force.addParticle(0.0, sigma, 0.5)
        force.setParticleSubset(1, 1)
        force.setInnerForceGroup(1)
        force.setInnerSwitchingDistance(0.5)
        force.setInnerCutoffDistance(0.8)
        force.addGlobalParameter("lambdaLJ", 0.6)
        force.addGlobalParameter("lambdaD", 0.3)
        force.addScalingParameter("lambdaLJ", 0, 1, False, True)
        force.addScalingParameter("lambdaD", 0, 1, False, False, True)
        system.addForce(force)
        platform = mm.Platform.getPlatformByName(platformName)
        properties = {} if platformName == "Reference" else {"Precision": precision}
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform, properties)
        context.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.9*sigma, 0, 0)])
        return context

    with pytest.raises(mm.OpenMMException, match="cannot exceed the inner switching distance"):
        makeContext(0.6)
    context = makeContext(0.3)
    x = 0.9
    delta = 4*0.5*(x**-12 - x**-6 - (-868*x**6 + 2188.8*x**5 - 1440*x**4 + 119.2)/4)
    inner = value(context.getState(getEnergy=True, groups={1}).getPotentialEnergy())
    context.setParameter("lambdaD", 0.6)
    tol = 1e-5 if platformName == "Reference" or precision == "double" else 1e-3
    ASSERT_EQUAL_TOL(inner + 0.3*delta, value(context.getState(getEnergy=True, groups={1}).getPotentialEnergy()), tol)


@pytest.mark.parametrize("platformName, precision", cases)
def testInnerGroupSplit(platformName, precision):
    platform = mm.Platform.getPlatformByName(platformName)
    properties = {} if platformName == "Reference" else {"Precision": precision}
    tol = 1e-5 if platformName == "Reference" or precision == "double" else 1e-4
    contexts = {}
    for split in (True, False):
        system, force, positions = _innerGroupSystem(split)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform, properties)
        context.setPositions(positions)
        contexts[split] = context
    whole = contexts[False].getState(getEnergy=True, getForces=True, getParameterDerivatives=True)
    parts = [contexts[True].getState(getEnergy=True, getForces=True, getParameterDerivatives=True, groups={g}) for g in (0, 1, 2)]
    ASSERT_EQUAL_TOL(whole.getPotentialEnergy(), sum(value(p.getPotentialEnergy()) for p in parts), tol)
    ASSERT_EQUAL_TOL(whole.getEnergyParameterDerivatives()["lambda"], sum(p.getEnergyParameterDerivatives()["lambda"] for p in parts), tol)
    for i, force in enumerate(whole.getForces()):
        total = parts[0].getForces()[i] + parts[1].getForces()[i] + parts[2].getForces()[i]
        ASSERT_EQUAL_VEC(force, total, tol)
    inner = parts[1]
    assert abs(value(inner.getPotentialEnergy())) > 1.0
    # The inner group is linear in lambda: derivative equals E(1) - E(0).
    contexts[True].setParameter("lambda", 1.0)
    e1 = value(contexts[True].getState(getEnergy=True, groups={1}).getPotentialEnergy())
    contexts[True].setParameter("lambda", 0.0)
    e0 = value(contexts[True].getState(getEnergy=True, groups={1}).getPotentialEnergy())
    ASSERT_EQUAL_TOL(e1 - e0, inner.getEnergyParameterDerivatives()["lambda"], 10*tol)


@pytest.mark.parametrize("platformName, precision", cases)
def testInnerGroupMTSIntegration(platformName, precision):
    # Wiring test: an MTSIntegrator stepping the inner group four times per outer step runs, and the
    # group energies keep adding up to the total afterwards.
    platform = mm.Platform.getPlatformByName(platformName)
    properties = {} if platformName == "Reference" else {"Precision": precision}
    system, force, positions = _innerGroupSystem(True)
    integrator = mm.MTSIntegrator(2*unit.femtoseconds, [(2, 1), (0, 1), (1, 4)])
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(300*unit.kelvin, 1)
    integrator.step(50)
    state = context.getState(getEnergy=True)
    total = value(state.getPotentialEnergy())
    assert math.isfinite(total) and math.isfinite(value(state.getKineticEnergy()))
    parts = sum(value(context.getState(getEnergy=True, groups={g}).getPotentialEnergy()) for g in (0, 1, 2))
    tol = 1e-5 if platformName == "Reference" or precision == "double" else 1e-4
    ASSERT_EQUAL_TOL(total, parts, tol)


@pytest.mark.parametrize("methodName", ["Ewald", "PME"])
@pytest.mark.parametrize("platformName, precision", cases)
def testDirectSpaceWithExclusions(platformName, precision, methodName):
    # With reciprocal exclusions, the Ewald exclusion corrections are in the reciprocal space force
    # group, so the direct space force group and its derivatives match a CustomNonbondedForce with the
    # same exclusions
    platform = mm.Platform.getPlatformByName(platformName)
    properties = {} if platformName == "Reference" else {"Precision": precision}
    tol = TOL_SLICE_DOUBLE if platformName == "Reference" or precision == "double" else TOL_SLICE_SINGLE
    rng = random.Random(11)
    n, length, cutoff = 3, 2.4, 1.0
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(SLICE_METHODS[methodName])
    nonbonded.setCutoffDistance(cutoff)
    nonbonded.setUseDispersionCorrection(False)
    positions = []
    for k in range(n**3):
        center = mm.Vec3(*[(i + 0.5 + 0.2*rng.random())*length/n for i in (k % n, k//n % n, k//n**2)])
        positions += [center, center + mm.Vec3(0.1, 0, 0), center + mm.Vec3(-0.0333, 0.0943, 0)]
        for q, sigma, epsilon in [(-0.834, 0.315, 0.636), (0.417, 1, 0), (0.417, 1, 0)]:
            nonbonded.addParticle(q, sigma, epsilon)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            nonbonded.addException(3*k+i, 3*k+j, 0, 1, 0)
    data = dict(box=[mm.Vec3(length, 0, 0), mm.Vec3(0, length, 0), mm.Vec3(0, 0, length)], positions=positions)

    sliced = plugin.SlicedNonbondedForce(nonbonded, 2)
    for i in range(sliced.getNumParticles()):
        sliced.setParticleSubset(i, (i//3) % 2)
    sliced.setReciprocalSpaceForceGroup(1)
    sliced.setReciprocalExclusions(True)
    scaling = {"lambdaCross": (0, 1, True, True), "lambdaCoul11": (1, 1, True, False)}
    for name, (subset1, subset2, includeCoulomb, includeLJ) in scaling.items():
        sliced.addGlobalParameter(name, 0.5)
        sliced.addScalingParameter(name, subset1, subset2, includeCoulomb, includeLJ)
        sliced.addEnergyParameterDerivative(name)
    _, context2 = _sliceContext(data, sliced, platform, properties)

    if methodName == "PME":
        alpha = value(sliced.getPMEParametersInContext(context2)[0])
    else:
        alpha = math.sqrt(-math.log(2*nonbonded.getEwaldErrorTolerance()))/cutoff
    direct = mm.CustomNonbondedForce(
        "scaleCoul*ONE_4PI_EPS0*q1*q2*erfc(alpha*r)/r + scaleLJ*4*eps*((sig/r)^12-(sig/r)^6);"
        "scaleCoul=select(subset1+subset2-1, select(subset1+subset2-2, 1, lambdaCoul11), lambdaCross);"
        "scaleLJ=select(subset1+subset2-1, 1, lambdaCross);"
        "sig=0.5*(sigma1+sigma2); eps=sqrt(epsilon1*epsilon2)"
    )
    direct.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    direct.setCutoffDistance(cutoff)
    # 1/(4 pi epsilon0) in kJ nm/(mol e^2), as in OpenMM
    direct.addGlobalParameter("ONE_4PI_EPS0", 1/(4*math.pi*8.8541878128e-18/(1.602176634e-19**2*6.02214076e23)))
    direct.addGlobalParameter("alpha", alpha)
    for name in scaling:
        direct.addGlobalParameter(name, 0.5)
        direct.addEnergyParameterDerivative(name)
    for parameter in ("q", "sigma", "epsilon", "subset"):
        direct.addPerParticleParameter(parameter)
    for i in range(nonbonded.getNumParticles()):
        q, sigma, epsilon = map(value, nonbonded.getParticleParameters(i))
        direct.addParticle([q, sigma, epsilon, sliced.getParticleSubset(i)])
    for i in range(nonbonded.getNumExceptions()):
        direct.addExclusion(*nonbonded.getExceptionParameters(i)[:2])
    _, context1 = _sliceContext(data, direct, platform, properties)

    state1 = context1.getState(getEnergy=True, getForces=True, getParameterDerivatives=True)
    state2 = context2.getState(getEnergy=True, getForces=True, getParameterDerivatives=True, groups={0})
    ASSERT_EQUAL_TOL(state1.getPotentialEnergy(), state2.getPotentialEnergy(), tol)
    for force1, force2 in zip(state1.getForces(), state2.getForces()):
        ASSERT_EQUAL_VEC(force1, force2, tol)
    derivatives1 = state1.getEnergyParameterDerivatives()
    derivatives2 = state2.getEnergyParameterDerivatives()
    for name in scaling:
        assert derivatives1[name] != 0
        ASSERT_EQUAL_TOL(derivatives1[name], derivatives2[name], tol)
