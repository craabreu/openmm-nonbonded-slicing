"""Shared pieces of the LBF protocol optimization: switching functions, protocols, the system, and the
replica-exchange move.

An LBF protocol scales three solute-solvent terms of a SlicedNonbondedForce with switching functions of a
single coupling parameter lambda in [0, 1] (Correa et al., J. Chem. Theory Comput. 18, 5876, 2022):

    C: capped Lennard-Jones                                  global parameter lambdaLJ    h_C(lambda)
    D: decapping terms (Lennard-Jones and Coulomb)           global parameter lambdaD     h_D(lambda)
    E: electrostatics (Coulomb, capped below sigma)          global parameter lambdaQ     h_E(lambda)

The energy is linear in the three scaling parameters,

    U(x; lambda) = U_rest(x) + h_C U_C(x) + h_D U_D(x) + h_E U_E(x),

where U_C, U_D and U_E are the energy derivatives that the plugin computes along with the energy. The
reduced potential of any state, and of any protocol, is therefore known from the three slice energies of
each sample, which is what replica exchange and the protocol search both rely on. U_rest and the pV term
are the same in every state and cancel.

Each term follows S_{m0,m1}(x), the polynomial step whose first m0 derivatives vanish at x = 0 and whose
first m1 derivatives vanish at x = 1. A term can have a window: it is zero for lambda <= a and switches on
over [a, 1].
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from math import comb

import numpy as np

KB = 0.0083144626181532  # kJ/(mol K)
TERMS = ("C", "D", "E")
PARAMS = {"C": "lambdaLJ", "D": "lambdaD", "E": "lambdaQ"}
SMOOTH = -1  # m0 of the C-infinity step psi(x) = f(x)/(f(x) + f(1-x)), f(x) = exp(-1/x)


def beta(temperature):
    """1/kT in mol/kJ."""
    return 1.0 / (KB * temperature)


# ----------------------------------------------------------------------------------------------------------
# Switching functions
# ----------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def coefficients(m0, m1):
    """Integer b_j of S_{m0,m1}(x) = x^(m0+1) sum_j b_j x^j, with S(1) = 1 and S^(i)(1) = 0 for i = 1..m1."""
    b = [1] + [0] * m1
    for i in range(1, m1 + 1):
        b[i] = -sum(comb(m0 + 1, i - j) * b[j] for j in range(i))
    for i in range(m1 - 1, -1, -1):
        b[i] -= sum(comb(j, i) * b[j] for j in range(i + 1, m1 + 1))
    return tuple(b)


def switch(m0, m1, x):
    """S_{m0,m1}(x) and its derivative, for x in [0, 1]."""
    x = np.asarray(x, dtype=float)
    s = np.zeros_like(x)
    ds = np.zeros_like(x)
    for j, bj in enumerate(coefficients(m0, m1)):
        p = m0 + 1 + j
        s += bj * x**p
        ds += bj * p * x ** (p - 1)
    return s, ds


def smooth_step(x):
    """C-infinity step on [0, 1], whose derivatives all vanish at both ends, and its derivative."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    inner = (x > 0) & (x < 1)
    xi = np.where(inner, x, 0.5)
    # psi = 1/(1 + exp(g)) with g = 1/x - 1/(1-x), written with tanh to avoid overflow near the ends
    g = 1 / xi - 1 / (1 - xi)
    s = np.where(inner, 0.5 * (1 - np.tanh(0.5 * g)), (x >= 1).astype(float))
    with np.errstate(over="ignore"):  # cosh overflows near the ends, where the derivative is 0
        ds = np.where(inner, 0.25 / np.cosh(0.5 * g) ** 2 * (1 / xi**2 + 1 / (1 - xi) ** 2), 0.0)
    return s, ds


# ----------------------------------------------------------------------------------------------------------
# Protocols
# ----------------------------------------------------------------------------------------------------------

_SPEC = re.compile(r"([CDE])\((inf|\d+)(?:,(\d+))?\)(?:\[([0-9.]+)\])?")


@dataclass(frozen=True)
class Protocol:
    """Switching functions of the three terms, as (m0, m1, a) each; m0 = SMOOTH selects the C-infinity step,
    and a is the start of the term's window (0 for none).

    The text form is, e.g., "C(1,0) D(11,1)[0.25] E(2,0)", with "D(inf)[0.25]" for the C-infinity step.
    """

    specs: tuple

    @classmethod
    def parse(cls, text):
        if text.strip().lower() == "paper":
            return PaperProtocol()
        found = {}
        for term, m0, m1, a in _SPEC.findall(text.replace(" ", "")):
            m0 = SMOOTH if m0 == "inf" else int(m0)
            found[term] = (m0, int(m1 or 0), float(a or 0.0))
        if sorted(found) != list(TERMS) or len(_SPEC.sub("", text.replace(" ", ""))) > 0:
            raise ValueError(f"cannot parse protocol {text!r}; expected e.g. 'C(1,0) D(11,1)[0.25] E(2,0)'")
        return cls(tuple(found[term] for term in TERMS))

    def __str__(self):
        parts = []
        for term, (m0, m1, a) in zip(TERMS, self.specs):
            shape = "inf" if m0 == SMOOTH else f"{m0},{m1}"
            parts.append(f"{term}({shape})" + (f"[{a:g}]" if a > 0 else ""))
        return " ".join(parts)

    def h(self, lam):
        """h (3, n) and dh/dlambda (3, n) at the given lambdas."""
        h, dh = zip(*(_term_h(spec, lam) for spec in self.specs))
        return np.array(h), np.array(dh)


def _term_h(spec, lam):
    m0, m1, a = spec
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    x = np.clip((lam - a) / (1 - a), 0.0, 1.0)
    s, ds = smooth_step(x) if m0 == SMOOTH else switch(m0, m1, x)
    on = lam > a
    return np.where(on, s, 0.0), np.where(on, ds / (1 - a), 0.0)


class PaperProtocol(Protocol):
    """The windowed S_{2,2} protocol of Correa et al. (2022): each term switches on over its own window."""

    WINDOWS = ((0.0, 0.7), (0.4, 0.8), (0.6, 1.0))

    def __init__(self):
        super().__init__(specs=())

    def __str__(self):
        return "paper"

    def h(self, lam):
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        h, dh = [], []
        for start, end in self.WINDOWS:
            x = np.clip((lam - start) / (end - start), 0.0, 1.0)
            s, ds = switch(2, 2, x)
            inside = (lam > start) & (lam < end)
            h.append(s)
            dh.append(np.where(inside, ds / (end - start), 0.0))
        return np.array(h), np.array(dh)


def equal_length_lambdas(lam, sd, num_states):
    """num_states lambdas, from 0 to 1, at equal increments of the thermodynamic length int sd(du/dlambda),
    given sd on an increasing grid lam that spans [0, 1]."""
    lam, sd = np.asarray(lam, dtype=float), np.asarray(sd, dtype=float)
    length = np.concatenate([[0.0], np.cumsum(0.5 * (sd[1:] + sd[:-1]) * np.diff(lam))])
    lambdas = np.interp(np.linspace(0.0, length[-1], num_states), length, lam)
    # where sd vanishes the length is flat and interp may pick any lambda of the stretch, so pin the ends
    lambdas[0], lambdas[-1] = 0.0, 1.0
    return lambdas


# ----------------------------------------------------------------------------------------------------------
# Replica exchange
# ----------------------------------------------------------------------------------------------------------


def mix_replicas(u, state_of_replica, num_attempts, rng, accepted=None, attempted=None):
    """Swap the states of random pairs of replicas with the Metropolis criterion (Chodera and Shirts,
    J. Chem. Phys. 135, 194110, 2011). Every attempt leaves the distribution of state permutations,
    p(sigma) ~ exp(-sum_i u[i, sigma(i)]), invariant, and many attempts per iteration approach drawing a
    new permutation from it.

    u: reduced potentials, u[i, k] for replica i in state k
    state_of_replica: the state of each replica, updated in place
    accepted, attempted: optional (K, K) counters by pair of states, updated in place
    """
    num_replicas = len(state_of_replica)
    pairs = rng.integers(num_replicas, size=(num_attempts, 2))
    log_r = np.log(rng.random(num_attempts))
    s = state_of_replica
    for (i, j), log_ri in zip(pairs, log_r):
        if i == j:
            continue
        si, sj = s[i], s[j]
        delta = u[i, sj] + u[j, si] - u[i, si] - u[j, sj]
        if attempted is not None:
            attempted[si, sj] += 1
            attempted[sj, si] += 1
        if log_ri < -delta:
            s[i], s[j] = sj, si
            if accepted is not None:
                accepted[si, sj] += 1
                accepted[sj, si] += 1
    return s


@dataclass
class Run:
    """The samples of one simulation, by state: series[k] holds the slice energies (n_k, 3) sampled in state k,
    in time order, after equilibration."""

    lambdas: np.ndarray
    H: np.ndarray
    temperature: float
    series: list
    capping_orders: tuple = (3, 3)


def load_run(path, default_temperature=298.15):
    """Load OUTDIR/samples.npz written by simulate.py, or a directory of state_*.npz files with h (3,), lam
    and U (n, 3), one per state, as written by the pilot scripts (already equilibrated, at
    default_temperature, with the default capping orders)."""
    import glob
    import os

    samples = os.path.join(path, "samples.npz")
    if os.path.exists(samples):
        d = np.load(samples)
        history, U = d["state_of_replica_history"], d["U"]
        keep = slice(int(d["equilibration"]), None)
        history, U = history[keep], U[keep]
        replica_in_state = np.argsort(history, axis=1)  # each row of history is a permutation of the states
        rows = np.arange(len(U))[:, None]
        by_state = U[rows, replica_in_state]  # (n, K, 3)
        series = [by_state[:, k] for k in range(by_state.shape[1])]
        orders = tuple(int(m) for m in d["capping_orders"]) if "capping_orders" in d else (3, 3)
        return Run(d["lambdas"], d["H"], float(d["temperature"]), series, orders)
    files = sorted(glob.glob(os.path.join(path, "state_*.npz")))
    if not files:
        raise FileNotFoundError(f"no samples.npz or state_*.npz in {path}")
    states = sorted((np.load(f) for f in files), key=lambda s: float(s["lam"]))
    return Run(
        np.array([float(s["lam"]) for s in states]),
        np.array([s["h"] for s in states]),
        default_temperature,
        [s["U"] for s in states],
    )


# ----------------------------------------------------------------------------------------------------------
# System
# ----------------------------------------------------------------------------------------------------------


def build_system(
    smiles,
    forcefield="openff-2.2.1",
    water="amber/tip3p_standard.xml",
    box_nm=3.0,
    hydrogen_mass=3.0,
    temperature=298.15,
    pressure=1.0,
    cutoff_nm=1.0,
    switch_nm=0.9,
    capping_orders=(3, 3),
):
    """A solute in water with a SlicedNonbondedForce, whose subset 1 is the solute and subset 0 the solvent.

    forcefield is a SMIRNOFF force field (e.g. "openff-2.2.1") or GAFF (e.g. "gaff-2.11") for the solute;
    water is an OpenMM force field file for the solvent. Both decapping terms of the solute-solvent slice
    are multiplied by lambdaD. Particles with zero epsilon (e.g. TIP3P hydrogens) get sigma = 0, so that
    their Coulomb cap radius is half that of their partner instead of an arbitrary placeholder sigma.
    capping_orders are (m0, m1) of the capped potentials, which set the height of the cap: the default
    (3, 3) caps the Lennard-Jones potential at 119 epsilon.

    Returns the system, topology, positions and solute atom indices.
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openff.toolkit import Molecule
    from openmmforcefields.generators import GAFFTemplateGenerator, SMIRNOFFTemplateGenerator

    import nonbondedslicing as nbs

    molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)
    generator_class = GAFFTemplateGenerator if forcefield.startswith("gaff") else SMIRNOFFTemplateGenerator
    generator = generator_class(molecules=molecule, forcefield=forcefield)
    ff = app.ForceField(water)
    ff.registerTemplateGenerator(generator.generator)
    modeller = app.Modeller(molecule.to_topology().to_openmm(), molecule.conformers[0].to_openmm())
    modeller.addSolvent(ff, boxSize=mm.Vec3(box_nm, box_nm, box_nm) * unit.nanometers)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=hydrogen_mass * unit.amu if hydrogen_mass else None,
    )
    index, nonbonded = next((i, f) for i, f in enumerate(system.getForces()) if isinstance(f, mm.NonbondedForce))
    nonbonded.setUseSwitchingFunction(True)
    nonbonded.setSwitchingDistance(switch_nm * unit.nanometers)
    nonbonded.setUseDispersionCorrection(True)
    force = nbs.SlicedNonbondedForce(nonbonded, 2)
    force.setCappingOrders(*capping_orders)
    for i in range(force.getNumParticles()):
        charge, _, epsilon = force.getParticleParameters(i)
        if epsilon.value_in_unit(unit.kilojoule_per_mole) == 0.0:
            force.setParticleParameters(i, charge, 0.0, 0.0)
    solute = list(range(molecule.n_atoms))
    for atom in solute:
        force.setParticleSubset(atom, 1)
    for term in TERMS:
        force.addGlobalParameter(PARAMS[term], 1.0)
    force.addScalingParameter(PARAMS["C"], 0, 1, False, True)
    force.addScalingParameter(PARAMS["D"], 0, 1, False, False, True, True)
    force.addScalingParameter(PARAMS["E"], 0, 1, True, False)
    for term in TERMS:
        force.addEnergyParameterDerivative(PARAMS[term])
    system.removeForce(index)
    system.addForce(force)
    system.addForce(mm.MonteCarloBarostat(pressure * unit.bar, temperature * unit.kelvin, 25))
    return system, modeller.topology, modeller.positions, solute
