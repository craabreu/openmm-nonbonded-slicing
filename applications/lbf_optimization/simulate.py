"""Sample the states of an LBF protocol by replica exchange of lambda values.

usage: python simulate.py OUTDIR --smiles SMILES [--protocol P] [--nstates 21 | --lambdas ...]
       python simulate.py OUTDIR --plan next.json        (solute, protocol and lambdas from search.py)

The protocol defaults to C(exp 2.5) D(8,2)[0.4] E(3,1)[0.4] Q=E, chosen for four solutes (RESULTS.md).
The default platform is CUDA; --platform OpenCL or CPU (much slower) run elsewhere.

Each replica has its own Context, so an exchange only swaps the global parameters of two Contexts, never
their coordinates. The energy is linear in the scaling parameters, so one energy and parameter-derivative
evaluation per replica gives the reduced potential of every replica in every state, and the exchanges cost
no extra force evaluations (see lbf.py). With --no_exchange, every replica stays in its state.

OUTDIR/samples.npz holds, per iteration, the state of each replica and its slice energies U_C, U_D, U_Q and
U_E (kJ/mol), measured before the exchange, and the collapse diagnostic: the shortest distance (nm) between a
solute N or O (every solute heavy atom if it has none) and a water oxygen. At the end the run reports, per
state, the fraction of frames within 0.2 nm, and flags states where it exceeds 5% while the charges are on
and the Lennard-Jones decapping is weak (h_D < 0.001): there the protocol charges too early and a water
fuses onto the solute (see collapse_report). The first --equilibration iterations are stored too and are
skipped by the other scripts.

OUTDIR/checkpoint.npz is written along with it. Rerunning the same command resumes from it, and rerunning it
with a larger --iterations extends the run; any other change of protocol, states, system or sampling
settings is refused.
"""

import argparse
import json
import os
import time

import numpy as np
import openmm as mm
import openmm.unit as unit

from lbf import (
    DEFAULT_PROTOCOL, PARAMS, SYSTEM_DEFAULTS, TERMS, HelpFormatter, Protocol, beta, build_system, closest_contact,
    mix_replicas, polar_atoms, read_system,
)

COLLAPSE_NM = 0.2  # a polar atom this close to a water oxygen overlaps it; a hydrogen bond is 0.25-0.3 nm
COLLAPSE_FRACTION = 0.05  # of the frames of a charged state with weak decapping
NUM_STATES = 21


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument("outdir", help="output directory; rerun the same command to resume")
    what = parser.add_argument_group("protocol and states (or --plan)")
    what.add_argument("--protocol", help=f'protocol text, "paper" or "default" (default: "{DEFAULT_PROTOCOL}")')
    what.add_argument("--nstates", type=int, help=f"number of states, uniform in lambda (default: {NUM_STATES})")
    what.add_argument("--lambdas", help="comma-separated lambdas in [0, 1], or a file with one per line")
    what.add_argument(
        "--plan", help="JSON file written by search.py --next: the protocol, lambdas and system of the runs it scored"
    )
    run = parser.add_argument_group("sampling")
    run.add_argument("--iterations", type=int, default=2200, help="iterations, including equilibration")
    run.add_argument("--equilibration", type=int, default=200, help="first iterations that the analysis skips")
    run.add_argument("--iteration_ps", type=float, default=1.0, help="dynamics between exchanges, ps")
    run.add_argument("--swap_attempts", type=int, help="pair swaps per iteration (default: K^3 for K states)")
    run.add_argument("--no_exchange", action="store_true", help="keep every replica in its state")
    run.add_argument("--checkpoint_interval", type=int, default=50, help="iterations between checkpoints")
    run.add_argument("--prep_ps", type=float, default=100.0, help="fully coupled equilibration before the run, ps")
    run.add_argument("--dt_fs", type=float, default=4.0, help="time step, fs")
    run.add_argument("--friction", type=float, default=1.0, help="Langevin friction, 1/ps")
    run.add_argument("--seed", type=int, default=1, help="random seed of velocities, dynamics and exchanges")
    run.add_argument("--platform", default="CUDA", help="OpenMM platform: CUDA, OpenCL or CPU")
    run.add_argument("--precision", default="mixed", help="precision on CUDA and OpenCL")
    system = parser.add_argument_group(
        "system (a plan supplies these; an option given with --plan must agree with it)"
    )
    defaults = SYSTEM_DEFAULTS
    system.add_argument("--smiles", help="the solute, as SMILES; required unless --plan gives it")
    system.add_argument(
        "--forcefield", help=f"SMIRNOFF or GAFF force field of the solute (default: {defaults['forcefield']})"
    )
    system.add_argument(
        "--water", help=f"OpenMM force field file of a 3-site water model (default: {defaults['water']})"
    )
    system.add_argument("--box_nm", type=float, help=f"edge of the cubic box, nm (default: {defaults['box_nm']})")
    system.add_argument(
        "--hydrogen_mass", type=float,
        help=f"hydrogen mass, amu; 0 for no repartitioning (default: {defaults['hydrogen_mass']})",
    )
    system.add_argument("--temperature", type=float, help=f"K (default: {defaults['temperature']})")
    system.add_argument("--pressure", type=float, help=f"bar (default: {defaults['pressure']})")
    system.add_argument(
        "--capping_orders", type=int, nargs=2, metavar=("M0", "M1"),
        help="orders of the capped potentials, which set the cap height (default: 3 3, a cap of 119 epsilon)",
    )
    args = parser.parse_args(argv)
    if args.equilibration >= args.iterations:
        parser.error("--equilibration must be smaller than --iterations")
    return args


def resolve_system(args, plan=None):
    """args.system: each setting of SYSTEM_DEFAULTS from its option if given, else from the plan (the runs it
    came from), else the default. An option that disagrees with the plan is refused, since the plan's states
    are placed for its system."""
    planned_system = (plan or {}).get("system", {})
    system = {}
    for name, default in SYSTEM_DEFAULTS.items():
        given, planned = getattr(args, name), planned_system.get(name)
        if name == "capping_orders" and given is not None:
            given = list(given)
        if given is not None and planned is not None and given != planned:
            raise SystemExit(f"--{name} {given} differs from the {name} of {args.plan}: {planned}")
        system[name] = given if given is not None else planned if planned is not None else default
    if system["smiles"] is None:
        raise SystemExit("give the solute with --smiles")
    args.system = system


def checked_lambdas(lambdas, source):
    lambdas = np.sort(np.asarray(lambdas, dtype=float))
    if len(lambdas) == 0 or lambdas[0] < 0 or lambdas[-1] > 1 or np.any(np.diff(lambdas) <= 0):
        raise SystemExit(f"{source}: lambdas must be distinct and within [0, 1], got {lambdas.tolist()}")
    return lambdas


def read_states(args):
    if args.plan:
        given = [f"--{name}" for name in ("protocol", "lambdas", "nstates") if getattr(args, name) is not None]
        if given:
            raise SystemExit(f"{args.plan} sets the protocol and lambdas: drop {', '.join(given)}")
        with open(args.plan) as f:
            plan = json.load(f)
        resolve_system(args, plan)
        return Protocol.parse(plan["protocol"]), checked_lambdas(plan["lambdas"], args.plan)
    resolve_system(args)
    if args.lambdas is not None and args.nstates is not None:
        raise SystemExit("give --nstates or --lambdas, not both")
    if args.lambdas is None:
        lambdas = np.linspace(0.0, 1.0, args.nstates or NUM_STATES)
    elif os.path.exists(args.lambdas):
        lambdas = np.loadtxt(args.lambdas, ndmin=1)
    else:
        lambdas = [float(x) for x in args.lambdas.split(",")]
    return Protocol.parse(args.protocol or DEFAULT_PROTOCOL), checked_lambdas(lambdas, "--lambdas")


def save(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def main():
    args = parse_args()
    protocol, lambdas = read_states(args)
    H = protocol.h(lambdas)[0].T  # (K, 4): the scaling parameters of each state
    num_states = len(lambdas)
    system_settings = args.system
    temperature = system_settings["temperature"]
    beta_ = beta(temperature)
    swap_attempts = num_states**3 if args.swap_attempts is None else args.swap_attempts
    sampling = dict(dt_fs=args.dt_fs, friction=args.friction, iteration_ps=args.iteration_ps,
                    equilibration=args.equilibration, swap_attempts=swap_attempts, exchange=not args.no_exchange)
    os.makedirs(args.outdir, exist_ok=True)
    samples_path = os.path.join(args.outdir, "samples.npz")
    checkpoint_path = os.path.join(args.outdir, "checkpoint.npz")

    system, topology, positions, solute = build_system(
        system_settings["smiles"], system_settings["forcefield"], system_settings["water"], system_settings["box_nm"],
        system_settings["hydrogen_mass"], temperature, system_settings["pressure"],
        capping_orders=tuple(system_settings["capping_orders"]),
    )
    polar, waters = polar_atoms(topology, solute)
    platform = mm.Platform.getPlatformByName(args.platform)
    properties = {"Precision": args.precision} if args.platform in ("CUDA", "OpenCL") else {}
    dt = args.dt_fs * unit.femtosecond
    steps_per_iteration = int(round(args.iteration_ps * 1000 / args.dt_fs))

    start = 0
    if os.path.exists(checkpoint_path):
        cp = np.load(checkpoint_path)
        if str(cp["protocol"]) != str(protocol) or len(cp["lambdas"]) != num_states or not np.allclose(cp["lambdas"], lambdas):
            raise SystemExit(f"{checkpoint_path} is for another protocol or set of states")
        if "sampling" not in cp:
            raise SystemExit(f"{checkpoint_path} was written by an older version of these scripts and cannot be resumed")
        for label, stored, now in (("system", read_system(cp["system"], checkpoint_path), system_settings),
                                   ("sampling", json.loads(str(cp["sampling"])), sampling)):
            changed = [f"{k} {stored.get(k)} -> {now[k]}" for k in now if stored.get(k) != now[k]]
            if changed:
                raise SystemExit(f"{checkpoint_path} was run with other {label} settings: {', '.join(changed)}")
        start = int(cp["iteration"])
        state_of_replica = cp["state_of_replica"].copy()
        U = list(cp["U"])
        contacts = list(cp["contact"])
        history = list(cp["state_of_replica_history"])
        states = list(cp["states"])
        accepted, attempted = cp["accepted"].copy(), cp["attempted"].copy()
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(cp["rng"]))
        print(f"resuming at iteration {start}", flush=True)
    else:
        state_of_replica = np.arange(num_states)
        U, contacts, history, states = [], [], [], None
        accepted, attempted = np.zeros((num_states, num_states), int), np.zeros((num_states, num_states), int)
        rng = np.random.default_rng(args.seed)

    def make_integrator(replica):
        integrator = mm.LangevinMiddleIntegrator(temperature * unit.kelvin, args.friction / unit.picosecond, dt)
        # a different stream per replica and per (re)start
        integrator.setRandomNumberSeed(1 + (args.seed * 1000003 + replica * 1009 + start) % 2147483646)
        return integrator

    def set_state_parameters(context, k):
        for term, value in zip(TERMS, H[k]):
            context.setParameter(PARAMS[term], float(value))

    integrators = [make_integrator(i) for i in range(num_states)]
    contexts = [mm.Context(system, integrators[i], platform, properties) for i in range(num_states)]
    if states is None:
        # equilibrate once at full coupling, then start every replica from there
        context = contexts[0]
        context.setPositions(positions)
        mm.LocalEnergyMinimizer.minimize(context)
        context.setVelocitiesToTemperature(temperature * unit.kelvin, args.seed)
        integrators[0].step(int(round(args.prep_ps * 1000 / args.dt_fs)))
        prepared = context.getState(getPositions=True, getVelocities=True)
        for i, context in enumerate(contexts):
            context.setState(prepared)
            context.setVelocitiesToTemperature(temperature * unit.kelvin, args.seed + i + 1)
    else:
        for context, xml in zip(contexts, states):
            context.setState(mm.XmlSerializer.deserialize(str(xml)))
    for context, k in zip(contexts, state_of_replica):
        set_state_parameters(context, k)

    def checkpoint(iteration):
        common = dict(
            protocol=str(protocol), lambdas=lambdas, H=H, terms=np.array(TERMS), system=json.dumps(system_settings),
            sampling=json.dumps(sampling), iteration_ps=args.iteration_ps, equilibration=args.equilibration,
            exchange=not args.no_exchange,
            state_of_replica_history=np.array(history, dtype=int).reshape(-1, num_states),
            U=np.array(U).reshape(-1, num_states, len(TERMS)), accepted=accepted, attempted=attempted,
            contact=np.array(contacts).reshape(-1, num_states), polar_atoms=polar,
        )
        save(samples_path, **common)
        xml = [mm.XmlSerializer.serialize(c.getState(getPositions=True, getVelocities=True)) for c in contexts]
        save(
            checkpoint_path, **common, iteration=iteration, state_of_replica=state_of_replica, states=np.array(xml),
            rng=json.dumps(rng.bit_generator.state),
        )

    clock = time.time()
    for n in range(start, args.iterations):
        energies = np.empty((num_states, len(TERMS)))
        contact = np.empty(num_states)
        for i, (context, integrator) in enumerate(zip(contexts, integrators)):
            try:
                integrator.step(steps_per_iteration)
                state = context.getState(getEnergy=True, getParameterDerivatives=True, getPositions=True)
            except mm.OpenMMException as error:
                raise RuntimeError(
                    f"replica {i} failed at iteration {n} in state {state_of_replica[i]} "
                    f"(lambda = {lambdas[state_of_replica[i]]:.4f}, h = {np.round(H[state_of_replica[i]], 6)}); "
                    f"rerun to resume from the last checkpoint"
                    + (
                        "; with bare Coulomb, charges on and h_D ~ 0, this is usually a water hydrogen fused onto a "
                        "solute acceptor (or the reverse): the protocol charges too early"
                        if H[state_of_replica[i], 3] > 0 and H[state_of_replica[i], 1] < 1e-3
                        else ""
                    )
                ) from error
            derivatives = state.getEnergyParameterDerivatives()
            energies[i] = [derivatives[PARAMS[term]] for term in TERMS]
            box = np.diag(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
            contact[i] = closest_contact(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), box, polar, waters)
        if not np.all(np.isfinite(energies)):
            bad = int(np.flatnonzero(~np.isfinite(energies).all(axis=1))[0])
            raise RuntimeError(f"replica {bad} has non-finite energies at iteration {n} in state {state_of_replica[bad]}")
        U.append(energies)
        contacts.append(contact)
        history.append(state_of_replica.copy())
        if not args.no_exchange:
            before = state_of_replica.copy()
            mix_replicas(beta_ * energies @ H.T, state_of_replica, swap_attempts, rng, accepted, attempted)
            for i in np.flatnonzero(state_of_replica != before):
                set_state_parameters(contexts[i], state_of_replica[i])
        if (n + 1) % args.checkpoint_interval == 0 or n + 1 == args.iterations:
            checkpoint(n + 1)
            elapsed, clock = time.time() - clock, time.time()
            report = f"iteration {n + 1}/{args.iterations}  {elapsed / args.checkpoint_interval:.2f} s/iteration"
            if not args.no_exchange:
                neighbors = [accepted[k, k + 1] / max(attempted[k, k + 1], 1) for k in range(num_states - 1)]
                report += f"  neighbor acceptance min {min(neighbors):.3f} mean {np.mean(neighbors):.3f}"
            print(report, flush=True)

    print(collapse_report(np.array(contacts), np.array(history), H))


def collapse_report(contact, history, H, threshold=COLLAPSE_NM, fraction=COLLAPSE_FRACTION, weak=1e-3):
    """The fraction of frames of each state in which a solute polar atom is closer than the threshold to a
    water oxygen, marked where it exceeds the given fraction while the charges are on and the Lennard-Jones
    decapping is weak: there the capped core lets the charges pull a water into the solute.

    Brief overlaps are normal wherever the core is capped, with or without charges, so a single short contact
    means nothing; what marks early charging is how often they happen. For phenol, states with the charges
    on and h_D < 1e-3 show 0-1.5% under the protocols of the search and 7-15% under one that charges from
    lambda = 0.2 with capped Coulomb. With bare Coulomb such a protocol crashes instead, and the error says why.
    """
    num_states = H.shape[0]
    short = np.full(num_states, np.nan)
    for k in range(num_states):
        sampled = contact[history == k]
        if len(sampled):
            short[k] = np.mean(sampled < threshold)
    # with the charges off, water passing through a weakly coupled solute is normal, not a collapse
    collapsed = (short > fraction) & (H[:, 1] < weak) & (H[:, 3] > 0)
    lines = [
        f"fraction of frames with a solute polar atom within {threshold:g} nm of a water oxygen, per state "
        f"(! = above {fraction:g} with h_D < {weak:g} and h_E > 0):"
    ]
    lines.append("  " + " ".join(f"{f:.3f}{'!' if c else ' '}" for f, c in zip(short, collapsed)))
    if collapsed.any():
        lines.append(f"  collapse in states {np.flatnonzero(collapsed).tolist()}: the protocol charges too early there")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
