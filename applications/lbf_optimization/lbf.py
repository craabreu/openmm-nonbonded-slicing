"""Shared pieces of the LBF protocol optimization: switching functions, protocols, the system, and the
replica-exchange move.

An LBF protocol scales four solute-solvent terms of a SlicedNonbondedForce with switching functions of a
single coupling parameter lambda in [0, 1] (Correa et al., J. Chem. Theory Comput. 18, 5876, 2022):

    C: capped Lennard-Jones                                  global parameter lambdaLJ    h_C(lambda)
    D: Lennard-Jones decapping                               global parameter lambdaD     h_D(lambda)
    Q: Coulomb decapping                                     global parameter lambdaQD    h_Q(lambda)
    E: electrostatics (Coulomb, capped below sigma)          global parameter lambdaQ     h_E(lambda)

The energy is linear in the four scaling parameters,

    U(x; lambda) = U_rest(x) + h_C U_C(x) + h_D U_D(x) + h_Q U_Q(x) + h_E U_E(x),

where U_C, U_D, U_Q and U_E are the energy derivatives that the plugin computes along with the energy. The
reduced potential of any state, and of any protocol, is therefore known from the four slice energies of
each sample, which is what replica exchange and the protocol search both rely on. U_rest and the pV term
are the same in every state and cancel.

The two decapping terms are what distinguishes the electrostatic schemes: h_Q = h_D keeps the Coulomb
potential capped until the core is decapped (the capped-Coulomb scheme), and h_Q = h_E makes h_E switch on
the bare Coulomb potential, as the original plugin and the protocol of Correa et al. do. Both are protocols
of the same basis, so samples of either reweight to the other.

Each term follows S_{m0,m1}(x), the polynomial step whose first m0 derivatives vanish at x = 0 and whose
first m1 derivatives vanish at x = 1, or the exponential (e^(bx) - 1)/(e^b - 1), whose slope is finite at
both ends. A term can have a window: it is zero for lambda <= a and switches on over [a, 1].

The exponential is the shape the capped Lennard-Jones term wants: with it, sd(du/dlambda) is nearly flat
along lambda, where every polynomial step leaves the cavity-growth peak (for phenol, I_var drops from 240
to 204 kT^2 relative to S_{2,1}). The polynomial steps' zero end slopes are what excluded it.

DEFAULT_PROTOCOL is the protocol chosen for phenol, methane, anisole and paracetamol (see RESULTS.md).
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from math import comb

import numpy as np

KB = 0.0083144626181532  # kJ/(mol K)
TERMS = ("C", "D", "Q", "E")
PARAMS = {"C": "lambdaLJ", "D": "lambdaD", "Q": "lambdaQD", "E": "lambdaQ"}
SMOOTH = -1  # m0 of the C-infinity step psi(x) = f(x)/(f(x) + f(1-x)), f(x) = exp(-1/x)
EXP = -2  # m0 of the exponential (e^(bx) - 1)/(e^b - 1); the m1 slot holds the rate b
DEFAULT_PROTOCOL = "C(exp 2.5) D(8,2)[0.4] E(3,1)[0.4] Q=E"
DEFAULT_FORCEFIELD = "openff-2.2.1"
DEFAULT_WATER = "amber/tip3p_standard.xml"


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


def exponential(b, x):
    """(e^(bx) - 1)/(e^b - 1) and its derivative, for x in [0, 1]: a convex rise with finite slope at both
    ends, b/(e^b - 1) at 0 and b e^b/(e^b - 1) at 1."""
    x = np.asarray(x, dtype=float)
    if b <= 0:
        raise ValueError("the rate of the exponential switching function must be positive")
    scale = np.expm1(b)
    return np.expm1(b * x) / scale, b * np.exp(b * x) / scale


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

_SPEC = re.compile(r"([CDQE])\((inf|exp[0-9.]+|\d+)(?:,(\d+))?\)(?:\[([0-9.]+)\])?")
_TIE = re.compile(r"Q=([DE])")


def _shape(m0, m1):
    if m0 == SMOOTH:
        return "inf"
    if m0 == EXP:
        return f"exp {m1:g}"
    return f"{m0},{m1}"


@dataclass(frozen=True)
class Protocol:
    """Switching functions of the four terms, as (m0, m1, a) each; m0 = SMOOTH selects the C-infinity step,
    m0 = EXP the exponential with rate b in the m1 slot, and a is the start of the term's window (0 for none).

    The text form is, e.g., "C(exp 2.5) D(8,2)[0.4] E(3,1)[0.4] Q=E", with "C(2,1)" for a polynomial step and
    "D(inf)[0.4]" for the C-infinity step. The Coulomb decapping Q follows D unless the text ties it to E ("Q=E",
    the bare-Coulomb scheme) or gives it its own switching function ("Q(7,2)[0.4]"). "paper" is the protocol
    of Correa et al. (PaperProtocol) and "default" is DEFAULT_PROTOCOL.
    """

    specs: tuple

    @classmethod
    def parse(cls, text):
        if text.strip().lower() == "paper":
            return PaperProtocol()
        if text.strip().lower() == "default":
            text = DEFAULT_PROTOCOL
        compact = text.replace(" ", "")
        found = {}
        for term, m0, m1, a in _SPEC.findall(compact):
            if m0.startswith("exp"):
                if m1:
                    raise ValueError(f"cannot parse protocol {text!r}: the exponential takes one number, its rate")
                found[term] = (EXP, float(m0[3:]), float(a or 0.0))
            else:
                found[term] = (SMOOTH if m0 == "inf" else int(m0), int(m1 or 0), float(a or 0.0))
        ties = _TIE.findall(compact)
        rest = _TIE.sub("", _SPEC.sub("", compact))
        if set(found) not in ({"C", "D", "E"}, set(TERMS)) or rest or len(ties) + ("Q" in found) > 1:
            raise ValueError(f"cannot parse protocol {text!r}; expected e.g. 'C(exp 2.5) D(7,2)[0.4] E(2,1)[0.4] Q=E'")
        if "Q" not in found:
            found["Q"] = found[ties[0] if ties else "D"]
        return cls(tuple(found[term] for term in TERMS))

    def __str__(self):
        parts = []
        for term, (m0, m1, a) in zip(TERMS, self.specs):
            if term == "Q":
                continue
            parts.append(f"{term}({_shape(m0, m1)})" + (f"[{a:g}]" if a > 0 else ""))
        C, D, Q, E = self.specs
        if Q == E and Q != D:
            parts.append("Q=E")
        elif Q != D:
            m0, m1, a = Q
            parts.append(f"Q({_shape(m0, m1)})" + (f"[{a:g}]" if a > 0 else ""))
        return " ".join(parts)

    def h(self, lam):
        """h (4, n) and dh/dlambda (4, n) at the given lambdas."""
        h, dh = zip(*(_term_h(spec, lam) for spec in self.specs))
        return np.array(h), np.array(dh)

    def charging_hazard(self, weak=1e-3):
        """The largest h_E/h_C where the Lennard-Jones decapping is still weak (h_D < weak). Charging against
        a capped core is a contest between the capped attraction, scaled by h_E, and the capped repulsion,
        scaled by h_C. For phenol in water, a water fuses onto the hydroxyl within nanoseconds above about
        0.5, and reweighting shows the fused state favorable between the sampled states above 0.15, the
        limit search.py applies (see RESULTS.md)."""
        h = self.h(np.linspace(0.0, 1.0, 1001))[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(h[3] > 0, h[3] / h[0], 0.0)
        return float(np.max(ratio[h[1] < weak], initial=0.0))

    def end_orders(self):
        """(p0, p1): the leading powers of dh/dlambda at the ends, lambda^p0 at 0 and (1 - lambda)^p1 at 1,
        from the terms that switch on at lambda = 0 and the smallest end order; the C-infinity step never
        dominates, and the exponential has order 0 at both ends. The thermodynamic length then goes as
        lambda^(p0+1) and (1 - lambda)^(p1+1)."""
        at_zero = [0 if m0 == EXP else m0 for m0, m1, a in self.specs if a == 0 and m0 != SMOOTH]
        at_one = [0 if m0 == EXP else m1 for m0, m1, a in self.specs if m0 != SMOOTH]
        return min(at_zero), min(at_one)


def _term_h(spec, lam):
    m0, m1, a = spec
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    x = np.clip((lam - a) / (1 - a), 0.0, 1.0)
    if m0 == SMOOTH:
        s, ds = smooth_step(x)
    elif m0 == EXP:
        s, ds = exponential(m1, x)
    else:
        s, ds = switch(m0, m1, x)
    on = lam >= a if a == 0 else lam > a  # a term without a window keeps its slope at lambda = 0
    return np.where(on, s, 0.0), np.where(on, ds / (1 - a), 0.0)


class PaperProtocol(Protocol):
    """The windowed S_{2,2} protocol of Correa et al. (2022): each term switches on over its own window, and
    the Coulomb potential is bare (h_Q = h_E), as the plugin of the time had no Coulomb capping."""

    WINDOWS = {"C": (0.0, 0.7), "D": (0.4, 0.8), "E": (0.6, 1.0)}

    def __init__(self):
        super().__init__(specs=())

    def __str__(self):
        return "paper"

    def end_orders(self):
        return 2, 2  # S_{2,2} for C at lambda = 0 and for E at lambda = 1

    def h(self, lam):
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        h, dh = {}, {}
        for term, (start, end) in self.WINDOWS.items():
            x = np.clip((lam - start) / (end - start), 0.0, 1.0)
            s, ds = switch(2, 2, x)
            inside = (lam > start) & (lam < end)
            h[term] = s
            dh[term] = np.where(inside, ds / (end - start), 0.0)
        h["Q"], dh["Q"] = h["E"], dh["E"]
        return np.array([h[t] for t in TERMS]), np.array([dh[t] for t in TERMS])


def replica_flow(history):
    """The flow of replicas through the states of a replica-exchange run (Katzgraber, Trebst, Huse and
    Troyer, J. Stat. Mech. P03018, 2006), from history (n, K), the state of each replica at each iteration.

    A replica is an up-walker from its last visit to state 0 until it reaches state K-1, and a down-walker
    from then until it is back at 0. Returns f (K,), the fraction of up-walkers among the labeled visits of
    each state, which falls from 1 to 0; the visits (K,) counted; and the number of round trips, completed
    when a down-walker returns to state 0."""
    history = np.asarray(history)
    n, K = history.shape
    label = np.zeros(K, int)
    up, down = np.zeros(K), np.zeros(K)
    trips = 0
    for state in history:
        at0, at1 = state == 0, state == K - 1
        trips += int(np.sum(at0 & (label == -1)))
        label = np.where(at0, 1, np.where(at1, -1, label))
        np.add.at(up, state[label == 1], 1)
        np.add.at(down, state[label == -1], 1)
    visits = up + down
    return up / np.maximum(visits, 1), visits, trips


def nonincreasing_fit(y, weights=None):
    """The weighted least-squares fit of y by a nonincreasing sequence (pool adjacent violators)."""
    y = np.asarray(y, dtype=float)
    weights = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    blocks = [[y[i], weights[i], 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            (a, wa, na), (b, wb, nb) = blocks[i], blocks[i + 1]
            blocks[i] = [(a * wa + b * wb) / (wa + wb), wa + wb, na + nb]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[value] * count for value, _, count in blocks])


def feedback_lambdas(lam, f, num_states, weights=None):
    """num_states lambdas, from 0 to 1, by the feedback optimization of Katzgraber et al.: the density of
    states proportional to sqrt(-df/dlambda), with f the fraction of up-walkers at the current states lam,
    fitted monotonically (weighted by weights, e.g. the visits) before differencing. Every state then sees
    the same change of f, which maximizes the rate of round trips."""
    lam = np.asarray(lam, dtype=float)
    f = nonincreasing_fit(f, weights)
    density = np.sqrt(np.maximum(-np.diff(f) / np.diff(lam), 0.0))  # constant on each interval
    if not np.any(density > 0):
        raise ValueError("f does not decrease: no flow to optimize")
    cumulative = np.concatenate([[0.0], np.cumsum(density * np.diff(lam))])
    lambdas = np.interp(np.linspace(0.0, cumulative[-1], num_states), cumulative, lam)
    lambdas[0], lambdas[-1] = 0.0, 1.0
    return lambdas


def bernstein(x, n):
    """The Bernstein basis of degree n at x (m,), as (m, n + 1)."""
    x = np.asarray(x, dtype=float)
    return np.stack([comb(n, j) * x**j * (1 - x) ** (n - j) for j in range(n + 1)], axis=1)


def lambda_of_c(c, p0, p1, w):
    """lambda(c) and dlambda/dc, for c in [0, 1] the thermodynamic length over L.

    The exact map has lambda ~ c^(1/(p0+1)) at 0 and 1 - lambda ~ (1 - c)^(1/(p1+1)) at 1, since sd(du/dlambda)
    vanishes with those powers of dh/dlambda (see Protocol.end_orders). So lambda = lambda0(w(c)) with
    lambda0(w) = (1 - (1 - w)^(1/(p1+1)))^(1/(p0+1)), which has both, and w a nondecreasing Bernstein
    polynomial with coefficients w (w[0] = 0, w[-1] = 1), whose finite slope at the ends keeps them."""
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    n = len(w) - 1
    x = np.clip(bernstein(c, n) @ w, 1e-15, 1 - 1e-15)
    dx = n * (bernstein(c, n - 1) @ np.diff(w))
    q0, q1 = 1 / (p0 + 1), 1 / (p1 + 1)
    inner = 1 - (1 - x) ** q1
    lam = inner**q0
    dlam = q0 * inner ** (q0 - 1) * q1 * (1 - x) ** (q1 - 1) * dx
    lam[c <= 0], lam[c >= 1] = 0.0, 1.0
    return lam, dlam


def fit_lambda_of_c(lam, sd, p0, p1, degree, nodes=4001):
    """The Bernstein coefficients w of lambda_of_c that flatten sd(du/dc) = sd(lambda(c)) dlambda/dc, given
    sd(du/dlambda) on a grid lam spanning [0, 1]: they minimize I_var/L^2 in c, which is exact along the
    path and needs no reweighting. Returns w and the I_var/L^2 reached (1 is flat)."""
    from scipy.optimize import minimize

    lam, sd = np.asarray(lam, dtype=float), np.asarray(sd, dtype=float)
    L = np.trapezoid(sd, lam)
    c = np.linspace(0.0, 1.0, nodes)

    def coefficients(z):
        d = np.exp(z - z.max())
        return np.concatenate([[0.0], np.cumsum(d / d.sum())])

    def objective(z):
        lam_c, dlam = lambda_of_c(c, p0, p1, coefficients(z))
        return np.trapezoid((np.interp(lam_c, lam, sd) * dlam) ** 2, c) / L**2

    result = minimize(objective, np.zeros(degree), method="L-BFGS-B", options=dict(maxiter=5000))
    return coefficients(result.x), float(result.fun)


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
    """The samples of one simulation, by state: series[k] holds the slice energies (n_k, 4) sampled in state
    k, in time order, after equilibration; H (K, 4) holds the scaling parameters of the states; system is
    (smiles, forcefield, water)."""

    lambdas: np.ndarray
    H: np.ndarray
    temperature: float
    series: list
    capping_orders: tuple
    system: tuple


def load_run(path):
    """Load OUTDIR/samples.npz written by simulate.py."""
    import os

    samples = os.path.join(path, "samples.npz")
    if not os.path.exists(samples):
        raise FileNotFoundError(f"no samples.npz in {path}")
    d = np.load(samples)
    if "system" not in d or tuple(str(t) for t in d["terms"]) != TERMS:
        raise SystemExit(f"{samples} was written by an older simulate.py (no solute, or no separate Coulomb "
                         "decapping energy) and cannot be read")
    history, U = d["state_of_replica_history"], d["U"]
    keep = slice(int(d["equilibration"]), None)
    history, U = history[keep], U[keep]
    replica_in_state = np.argsort(history, axis=1)  # each row of history is a permutation of the states
    rows = np.arange(len(U))[:, None]
    by_state = U[rows, replica_in_state]  # (n, K, 4)
    series = [by_state[:, k] for k in range(by_state.shape[1])]
    orders = tuple(int(m) for m in d["capping_orders"])
    system = tuple(str(x) for x in d["system"])
    return Run(d["lambdas"], d["H"], float(d["temperature"]), series, orders, system)


# ----------------------------------------------------------------------------------------------------------
# System
# ----------------------------------------------------------------------------------------------------------


def polar_atoms(topology, solute):
    """The solute atoms that can donate or accept a hydrogen bond (N and O), or every solute heavy atom if
    it has none, and the water oxygens: the pairs whose distance the collapse diagnostic watches."""
    atoms = list(topology.atoms())
    polar = [i for i in solute if atoms[i].element is not None and atoms[i].element.symbol in ("N", "O")]
    if not polar:
        polar = [i for i in solute if atoms[i].element is not None and atoms[i].element.symbol != "H"]
    waters = [a.index for a in atoms if a.residue.name in ("HOH", "WAT", "SOL") and a.element is not None
              and a.element.symbol == "O"]
    return np.array(polar), np.array(waters)


def closest_contact(positions, box, polar, waters):
    """The shortest distance between any of the polar atoms and any of the water atoms, under the minimum
    image convention of a rectangular box (positions (N, 3) and box (3,) in the same unit)."""
    d = positions[polar][:, None, :] - positions[waters][None, :, :]
    d -= box * np.round(d / box)
    return float(np.sqrt(np.min(np.sum(d * d, axis=2))))


def build_system(
    smiles,
    forcefield=DEFAULT_FORCEFIELD,
    water=DEFAULT_WATER,
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
    water is an OpenMM force field file for the solvent. The four terms of the solute-solvent slice get the
    global parameters of PARAMS. Particles with zero epsilon (e.g. TIP3P hydrogens) get sigma = 0, so that
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
    # (includeCoulomb, includeLJ, includeCoulombDecapping, includeLJDecapping)
    force.addScalingParameter(PARAMS["C"], 0, 1, False, True, False, False)
    force.addScalingParameter(PARAMS["D"], 0, 1, False, False, False, True)
    force.addScalingParameter(PARAMS["Q"], 0, 1, False, False, True, False)
    force.addScalingParameter(PARAMS["E"], 0, 1, True, False, False, False)
    for term in TERMS:
        force.addEnergyParameterDerivative(PARAMS[term])
    system.removeForce(index)
    system.addForce(force)
    system.addForce(mm.MonteCarloBarostat(pressure * unit.bar, temperature * unit.kelvin, 25))
    return system, modeller.topology, modeller.positions, solute
