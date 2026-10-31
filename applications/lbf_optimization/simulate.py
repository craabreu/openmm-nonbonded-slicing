"""Sample the states of an LBF protocol by replica exchange of lambda values.

usage: python simulate.py OUTDIR --protocol "C(1,0) D(11,1)[0.25] E(2,0)" [--nstates 21 | --lambdas ...]
       python simulate.py OUTDIR --plan next.json        (protocol and lambdas chosen by search.py)

Each replica has its own Context, so an exchange only swaps the global parameters of two Contexts, never
their coordinates. The energy is linear in the scaling parameters, so one energy and parameter-derivative
evaluation per replica gives the reduced potential of every replica in every state, and the exchanges cost
no extra force evaluations (see lbf.py). With --no-exchange, every replica stays in its state.

OUTDIR/samples.npz holds, per iteration, the state of each replica and its slice energies U_C, U_D, U_Q and
U_E (kJ/mol), measured before the exchange. The first --equilibration iterations are stored too and are skipped
by search.py and validate.py. OUTDIR/checkpoint.npz is written along with it; rerunning the same command
resumes from it.
"""

import argparse
import json
import os
import time

import numpy as np
import openmm as mm
import openmm.unit as unit

from lbf import PARAMS, TERMS, Protocol, beta, build_system, mix_replicas


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outdir")
    what = parser.add_argument_group("protocol and states")
    what.add_argument("--protocol", help='e.g. "C(1,0) D(11,1)[0.25] E(2,0)", or "paper"')
    what.add_argument("--nstates", type=int, default=21, help="uniformly spaced states (default: 21)")
    what.add_argument("--lambdas", help="comma-separated lambdas, or a file with one per line")
    what.add_argument("--plan", help="JSON file with the protocol and lambdas, as written by search.py --next")
    run = parser.add_argument_group("sampling")
    run.add_argument("--iterations", type=int, default=2200, help="including equilibration (default: 2200)")
    run.add_argument("--equilibration", type=int, default=200, help="iterations to discard (default: 200)")
    run.add_argument("--iteration_ps", type=float, default=1.0, help="dynamics between exchanges (default: 1 ps)")
    run.add_argument("--swap_attempts", type=int, help="pair swaps per iteration (default: K^3)")
    run.add_argument("--no_exchange", action="store_true", help="keep every replica in its state")
    run.add_argument("--checkpoint_interval", type=int, default=50, help="iterations (default: 50)")
    run.add_argument("--prep_ps", type=float, default=100.0, help="fully coupled equilibration before (default: 100)")
    run.add_argument("--dt_fs", type=float, default=4.0)
    run.add_argument("--friction", type=float, default=1.0, help="1/ps")
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--platform", default="CUDA")
    run.add_argument("--precision", default="mixed")
    system = parser.add_argument_group("system")
    system.add_argument("--smiles", default="c1ccc(cc1)O", help="solute (default: phenol)")
    system.add_argument("--forcefield", default="openff-2.2.1", help="SMIRNOFF or GAFF force field of the solute")
    system.add_argument("--water", default="amber/tip3p_standard.xml")
    system.add_argument("--box_nm", type=float, default=3.0)
    system.add_argument("--hydrogen_mass", type=float, default=3.0, help="amu; 0 for no mass repartitioning")
    system.add_argument("--temperature", type=float, default=298.15)
    system.add_argument("--pressure", type=float, default=1.0, help="bar")
    system.add_argument(
        "--capping_orders", type=int, nargs=2, default=(3, 3), metavar=("M0", "M1"),
        help="orders of the capped potentials, which set the cap height (default: 3 3, i.e. 119 epsilon)",
    )
    return parser.parse_args()


def read_states(args):
    if args.plan:
        with open(args.plan) as f:
            plan = json.load(f)
        return Protocol.parse(plan["protocol"]), np.array(plan["lambdas"], dtype=float)
    if not args.protocol:
        raise SystemExit("give --protocol or --plan")
    if args.lambdas is None:
        lambdas = np.linspace(0.0, 1.0, args.nstates)
    elif os.path.exists(args.lambdas):
        lambdas = np.loadtxt(args.lambdas, ndmin=1)
    else:
        lambdas = np.array([float(x) for x in args.lambdas.split(",")])
    return Protocol.parse(args.protocol), np.sort(lambdas)


def save(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def main():
    args = parse_args()
    protocol, lambdas = read_states(args)
    H = protocol.h(lambdas)[0].T  # (K, 4): the scaling parameters of each state
    num_states = len(lambdas)
    beta_ = beta(args.temperature)
    swap_attempts = num_states**3 if args.swap_attempts is None else args.swap_attempts
    os.makedirs(args.outdir, exist_ok=True)
    samples_path = os.path.join(args.outdir, "samples.npz")
    checkpoint_path = os.path.join(args.outdir, "checkpoint.npz")

    system, topology, positions, solute = build_system(
        args.smiles, args.forcefield, args.water, args.box_nm, args.hydrogen_mass, args.temperature, args.pressure,
        capping_orders=tuple(args.capping_orders),
    )
    platform = mm.Platform.getPlatformByName(args.platform)
    properties = {"Precision": args.precision} if args.platform in ("CUDA", "OpenCL") else {}
    dt = args.dt_fs * unit.femtosecond
    steps_per_iteration = int(round(args.iteration_ps * 1000 / args.dt_fs))

    start = 0
    if os.path.exists(checkpoint_path):
        cp = np.load(checkpoint_path)
        if str(cp["protocol"]) != str(protocol) or not np.allclose(cp["lambdas"], lambdas):
            raise SystemExit(f"{checkpoint_path} is for another protocol or set of states")
        start = int(cp["iteration"])
        state_of_replica = cp["state_of_replica"].copy()
        U = list(cp["U"])
        history = list(cp["state_of_replica_history"])
        states = list(cp["states"])
        accepted, attempted = cp["accepted"].copy(), cp["attempted"].copy()
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(cp["rng"]))
        print(f"resuming at iteration {start}", flush=True)
    else:
        state_of_replica = np.arange(num_states)
        U, history, states = [], [], None
        accepted, attempted = np.zeros((num_states, num_states), int), np.zeros((num_states, num_states), int)
        rng = np.random.default_rng(args.seed)

    def make_integrator(replica):
        integrator = mm.LangevinMiddleIntegrator(args.temperature * unit.kelvin, args.friction / unit.picosecond, dt)
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
        context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)
        integrators[0].step(int(round(args.prep_ps * 1000 / args.dt_fs)))
        prepared = context.getState(getPositions=True, getVelocities=True)
        for i, context in enumerate(contexts):
            context.setState(prepared)
            context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed + i + 1)
    else:
        for context, xml in zip(contexts, states):
            context.setState(mm.XmlSerializer.deserialize(str(xml)))
    for context, k in zip(contexts, state_of_replica):
        set_state_parameters(context, k)

    def checkpoint(iteration):
        common = dict(
            protocol=str(protocol), lambdas=lambdas, H=H, terms=np.array(TERMS), temperature=args.temperature,
            capping_orders=np.array(args.capping_orders), iteration_ps=args.iteration_ps,
            equilibration=args.equilibration, exchange=not args.no_exchange,
            state_of_replica_history=np.array(history, dtype=int).reshape(-1, num_states),
            U=np.array(U).reshape(-1, num_states, len(TERMS)), accepted=accepted, attempted=attempted,
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
        for i, (context, integrator) in enumerate(zip(contexts, integrators)):
            try:
                integrator.step(steps_per_iteration)
                derivatives = context.getState(getEnergy=True, getParameterDerivatives=True).getEnergyParameterDerivatives()
            except mm.OpenMMException as error:
                raise RuntimeError(
                    f"replica {i} failed at iteration {n} in state {state_of_replica[i]} "
                    f"(lambda = {lambdas[state_of_replica[i]]:.4f}, h = {np.round(H[state_of_replica[i]], 6)}); "
                    f"rerun to resume from the last checkpoint"
                ) from error
            energies[i] = [derivatives[PARAMS[term]] for term in TERMS]
        if not np.all(np.isfinite(energies)):
            bad = int(np.flatnonzero(~np.isfinite(energies).all(axis=1))[0])
            raise RuntimeError(f"replica {bad} has non-finite energies at iteration {n} in state {state_of_replica[bad]}")
        U.append(energies)
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


if __name__ == "__main__":
    main()
