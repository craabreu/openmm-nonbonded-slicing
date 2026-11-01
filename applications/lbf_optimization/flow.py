"""The flow of replicas through the states of a run, and the states that the feedback optimization of
Katzgraber, Trebst, Huse and Troyer (J. Stat. Mech. P03018, 2006) would place, next to the equal-length ones.

usage: python flow.py RUN [--states 21] [--plot flow.png]

Replica exchange labels each replica as an up-walker from its last visit to lambda = 0 until it reaches
lambda = 1, and as a down-walker from then on. The fraction f(lambda) of up-walkers at each state falls
from 1 to 0, and the rate of round trips is largest when every state sees the same change of f, that is,
with the density of states proportional to sqrt(-df/dlambda). That allocation is what the dynamics asks
for, where the equal-length one of search.py --next is what the overlap of neighboring states asks for.
When the two agree, as they do when no state relaxes slowly, the simpler equal-length allocation stands.
"""

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # keep pymbar's JAX off the GPU that simulate.py uses

import numpy as np

from lbf import HelpFormatter, Protocol, equal_length_lambdas, feedback_lambdas, load_run, nonincreasing_fit, replica_flow


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument("run", help="an output directory of simulate.py, with replica exchange")
    parser.add_argument("--states", type=int, default=21, help="states to allocate")
    parser.add_argument("--plot", help="write a figure of f, the two densities and the two allocations")
    args = parser.parse_args()

    from search import Reweighter

    d = np.load(os.path.join(args.run, "samples.npz"))
    if not bool(d["exchange"]):
        raise SystemExit(f"{args.run} ran without replica exchange: no flow to measure")
    history = d["state_of_replica_history"][int(d["equilibration"]):]
    lams, protocol = d["lambdas"], Protocol.parse(str(d["protocol"]))
    f, visits, trips = replica_flow(history)
    print(f"{protocol}: {trips} round trips in {len(history)} iterations"
          + (f" ({len(history) / trips:.0f} iterations per round trip)" if trips else ""))
    print("  lambda " + " ".join(f"{x:5.2f}" for x in lams))
    print("  f      " + " ".join(f"{x:5.2f}" for x in f))
    if trips < 10:
        print("  too few round trips for a reliable allocation")
    try:
        by_flow = feedback_lambdas(lams, f, args.states, visits)
    except ValueError:
        raise SystemExit("  no replica crossed from lambda = 0 to 1: no flow to allocate states by") from None

    run = load_run(args.run)
    reweighter = Reweighter([run])
    grid = np.linspace(0.0, 1.0, 401)
    _, var, _ = reweighter.profile([protocol], grid)
    sd = np.sqrt(var[0])
    by_length = equal_length_lambdas(grid, sd, args.states)
    print(f"\n{args.states} states by the flow:   {np.round(by_flow, 3).tolist()}")
    print(f"{args.states} states by the length: {np.round(by_length, 3).tolist()}")
    print(f"largest difference {np.abs(by_flow - by_length).max():.3f} in lambda")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fit = nonincreasing_fit(f, visits)
        mid = 0.5 * (lams[1:] + lams[:-1])
        density = np.sqrt(np.maximum(-np.diff(fit) / np.diff(lams), 0.0))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
        axes[0].plot(lams, f, "o", ms=4, label=f"{trips} round trips")
        axes[0].plot(lams, fit, lw=1.2)
        axes[0].set_title("fraction of up-walkers f(λ)"); axes[0].legend(fontsize=8)
        axes[1].plot(mid, density / np.trapezoid(density, mid), "o-", ms=4, lw=1.5, label="flow: √(−df/dλ)")
        axes[1].plot(grid, sd / np.trapezoid(sd, grid), "--", lw=1.5, label="length: sd(du/dλ)/L")
        axes[1].set_title("density of states, normalized"); axes[1].legend(fontsize=8)
        axes[2].plot(by_flow, np.ones(args.states), "o", ms=5, label="by the flow")
        axes[2].plot(by_length, np.zeros(args.states), "s", ms=5, mfc="none", label="by the length")
        axes[2].set_ylim(-1, 2); axes[2].set_yticks([]); axes[2].set_title(f"{args.states} states"); axes[2].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("λ"); ax.grid(alpha=0.25)
        fig.suptitle(str(protocol))
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)


if __name__ == "__main__":
    main()
