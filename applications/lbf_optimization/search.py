"""Score LBF protocols by reweighting pooled samples with MBAR, and place the states of the next round.

usage: python search.py RUN [RUN ...] [--a 0.25] [--rank L] [--next next.json [--states 21]]

RUN is an output directory of simulate.py (or a directory of state_*.npz files from the pilot). The reduced
potential of a protocol h(lambda) is u = beta h . U plus terms common to all states (see lbf.py), so any
protocol can be evaluated at any lambda from the stored slice energies. At Gauss-Legendre nodes, MBAR
reweighting gives <du/dlambda> and var(du/dlambda), with du/dlambda = beta dh/dlambda . U, and (in kT):

    L     = int_0^1 sd(du/dlambda) dlambda       thermodynamic length: the cost with optimally spaced states
    I_var = int_0^1 var(du/dlambda) dlambda      the cost with uniformly spaced states (L^2 <= I_var)
    dG    = int_0^1 <du/dlambda> dlambda          must agree across protocols (a check of the quadrature)

I_var/L^2 >= 1 measures how much a protocol depends on the spacing of its states; it is 1 when sd(du/dlambda)
is flat. A protocol counts as explored if the Kish effective sample size is at least --ess at every node.

The Lennard-Jones decapping term D switches on over [a, 1]. The capped Lennard-Jones term C covers [0, 1],
and the electrostatics E cover [0, 1] or, with --aE, a window of its own. The Coulomb decapping Q follows D
(capped Coulomb until the core is decapped) or E (bare Coulomb), as --Q selects; runs that predate the
split of the decapping terms can only weight the former. By default, every switching function has a zero
first derivative at both ends of its interval (m0 >= 1 and m1 >= 1), so that the lambda-force vanishes at
lambda = 0 and 1, as lambda-dynamics needs; widen the ranges to lift this. With --next, the best explored
protocol (or --choose) gets --states lambdas at equal increments of thermodynamic length, written as JSON
for simulate.py --plan.
"""

import argparse
import itertools
import json
import os

# pymbar solves with JAX, which would otherwise claim most of the GPU memory while simulate.py runs there
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.special import logsumexp

from lbf import (
    LEGACY_TERMS, SMOOTH, TERMS, Protocol, beta, energies_in_basis, equal_length_lambdas, in_basis, load_run,
    states_in_basis,
)


def int_range(text):
    lo, hi = (int(x) for x in text.split(":"))
    return range(lo, hi + 1)


def min_h(text):
    term, value = text.split("=")
    if term not in TERMS:
        raise argparse.ArgumentTypeError(f"term must be one of {TERMS}")
    return term, float(value)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--a", type=float, default=0.25, help="start of the decapping window (default: 0.25)")
    families = parser.add_argument_group("protocol families (inclusive ranges lo:hi)")
    families.add_argument("--m0C", type=int_range, default=int_range("1:5"))
    families.add_argument("--m1C", type=int_range, default=int_range("1:2"))
    families.add_argument("--m0D", type=int_range, default=int_range("2:11"))
    families.add_argument("--m1D", type=int_range, default=int_range("1:2"))
    families.add_argument("--m0E", type=int_range, default=int_range("1:11"))
    families.add_argument("--m1E", type=int_range, default=int_range("1:2"))
    families.add_argument(
        "--aE", type=float, nargs="+", metavar="A", help="starts of the window of E (default: 0 and a)",
    )
    families.add_argument(
        "--Q", choices=("D", "E"), nargs="+", default=["D", "E"],
        help="which term the Coulomb decapping follows (default: both, as separate protocols)",
    )
    families.add_argument("--no_smooth_d", action="store_true", help="leave out the C-infinity decapping step")
    families.add_argument(
        "--min_h_at_a", type=min_h, action="append", default=[], metavar="TERM=VALUE",
        help="keep protocols whose term (C or E) is at least VALUE at lambda = a; repeatable",
    )
    scoring = parser.add_argument_group("scoring")
    scoring.add_argument("--rank", choices=("L", "I_var"), default="L")
    scoring.add_argument("--ess", type=float, default=100.0)
    scoring.add_argument("--nodes", type=int, default=64, help="quadrature nodes of the screen")
    scoring.add_argument("--fine_nodes", type=int, default=512, help="quadrature nodes of the rescoring")
    scoring.add_argument("--refine", type=int, default=150, help="rescore this many best by each score")
    scoring.add_argument("--dg_tol", type=float, default=0.05, help="kT")
    scoring.add_argument("--chunk", type=int, default=1500, help="targets reweighted at once (memory)")
    scoring.add_argument("--reference", action="append", default=[], help='protocols to score as references')
    scoring.add_argument(
        "--max_hazard", type=float, help="keep protocols whose h_E/h_C, where h_D < 0.001, stays below this",
    )
    output = parser.add_argument_group("output")
    output.add_argument("--out", default="results.npz")
    output.add_argument("--top", type=int, default=20, help="rows of the ranked table")
    output.add_argument("--next", help="write the protocol and lambdas of the next round to this JSON file")
    output.add_argument("--states", type=int, default=21, help="states of the next round")
    output.add_argument("--choose", help="protocol of the next round (default: the best explored)")
    return parser.parse_args()


def candidates(args):
    decap = [(m0, m1, args.a) for m0, m1 in itertools.product(args.m0D, args.m1D)]
    if not args.no_smooth_d:
        decap.append((SMOOTH, 0, args.a))
    starts = sorted({0.0, args.a}) if args.aE is None else sorted(set(args.aE))
    for C in itertools.product(args.m0C, args.m1C, [0.0]):
        for D in decap:
            for E in itertools.product(args.m0E, args.m1E, starts):
                for tie in args.Q:
                    yield Protocol((C, D, D if tie == "D" else E, E))


class Reweighter:
    """MBAR estimates at arbitrary states h from the pooled samples."""

    def __init__(self, runs, chunk):
        from pymbar import MBAR

        temperatures = {run.temperature for run in runs}
        if len(temperatures) > 1:
            raise SystemExit(f"runs at different temperatures: {sorted(temperatures)}")
        # the capped potentials differ between capping orders, so their samples are of different systems
        orders = {run.capping_orders for run in runs}
        if len(orders) > 1:
            raise SystemExit(f"runs with different capping orders cannot be pooled: {sorted(orders)}")
        self.beta = beta(temperatures.pop())
        # a legacy run pulls the pool down to its basis, where only states and protocols with h_Q = h_D exist
        self.terms = LEGACY_TERMS if any(run.terms == LEGACY_TERMS for run in runs) else TERMS
        try:
            self.h_states = np.concatenate([states_in_basis(self.terms, run.H, run.terms) for run in runs])
        except ValueError as error:
            raise SystemExit(str(error)) from None
        series = [energies_in_basis(self.terms, s, run.terms) for run in runs for s in run.series]
        N_k = np.array([len(s) for s in series])
        self.BU = self.beta * np.concatenate(series)  # (N, len(terms))
        u_kn = self.h_states @ self.BU.T
        self.f_k = MBAR(u_kn, N_k, solver_protocol="robust").f_k
        self.log_denominator = logsumexp(np.log(N_k)[:, None] + self.f_k[:, None] - u_kn, axis=0)
        self.chunk = chunk

    def at(self, H, dH):
        """<du/dlambda>, var(du/dlambda) and Kish ESS at the states H (T, 4), with dh/dlambda dH (T, 4)."""
        H, dH = (x.T for x in in_basis(self.terms, H.T, dH.T))
        mean, var, ess = (np.empty(len(H)) for _ in range(3))
        for start in range(0, len(H), self.chunk):
            sl = slice(start, start + self.chunk)
            log_w = -(H[sl] @ self.BU.T) - self.log_denominator
            log_w -= logsumexp(log_w, axis=1, keepdims=True)
            w = np.exp(log_w)
            g = dH[sl] @ self.BU.T
            mean[sl] = np.sum(w * g, axis=1)
            var[sl] = np.sum(w * (g - mean[sl, None]) ** 2, axis=1)
            ess[sl] = 1.0 / np.sum(w**2, axis=1)
        return mean, var, ess

    def profile(self, protocols, lam):
        """<du/dlambda>, var and ESS of each protocol at the lambdas lam, as (P, n) arrays."""
        H, dH = (np.concatenate(x, axis=1).T for x in zip(*(p.h(lam) for p in protocols)))
        shape = (len(protocols), len(lam))
        return tuple(x.reshape(shape) for x in self.at(H, dH))

    def score(self, protocols, nodes, a):
        """L, I_var, dG and minimum ESS of each protocol by Gauss-Legendre quadrature split at a."""
        lam, weights = [], []
        for lo, hi in ((0.0, a), (a, 1.0)) if 0 < a < 1 else ((0.0, 1.0),):
            x, w = np.polynomial.legendre.leggauss(max(8, int(round(nodes * (hi - lo)))))
            lam.append(lo + 0.5 * (hi - lo) * (x + 1))
            weights.append(0.5 * (hi - lo) * w)
        lam, weights = np.concatenate(lam), np.concatenate(weights)
        mean, var, ess = self.profile(protocols, lam)
        return dict(L=np.sqrt(var) @ weights, I_var=var @ weights, dG=mean @ weights, min_ess=ess.min(axis=1))


def main():
    args = parse_args()
    runs = [load_run(path) for path in args.runs]
    reweighter = Reweighter(runs, args.chunk)
    print(f"{len(reweighter.h_states)} states, {len(reweighter.BU)} samples from {len(runs)} runs")

    protocols = list(candidates(args))
    for term, value in args.min_h_at_a:
        index = TERMS.index(term)
        protocols = [p for p in protocols if p.h(args.a)[0][index, 0] >= value]
    if args.max_hazard is not None:
        protocols = [p for p in protocols if p.charging_hazard() <= args.max_hazard]
    references = [Protocol.parse("paper")] + [Protocol.parse(text) for text in args.reference]
    if reweighter.terms == LEGACY_TERMS:
        protocols = [p for p in protocols if p.representable_in(LEGACY_TERMS)]
        dropped = [str(p) for p in references if not p.representable_in(LEGACY_TERMS)]
        references = [p for p in references if p.representable_in(LEGACY_TERMS)]
        print("a run stored the decapping terms together: only protocols with h_Q = h_D are scored"
              + (f"; references left out: {', '.join(dropped)}" if dropped else ""))
    print(f"{len(protocols)} protocols with the decapping window at a = {args.a:g}")

    # screen everything, then rescore the best by either score with more nodes
    scores = reweighter.score(protocols, args.nodes, args.a)
    top = np.union1d(np.argsort(scores["L"])[: args.refine], np.argsort(scores["I_var"])[: args.refine])
    fine = reweighter.score([protocols[i] for i in top] + references, args.fine_nodes, args.a)
    for key in scores:
        scores[key][top] = fine[key][: len(top)]
    reference_scores = [{key: fine[key][len(top) + j] for key in fine} for j in range(len(references))]
    refined = np.zeros(len(protocols), bool)
    refined[top] = True
    # <du/dlambda> reweighted by MBAR is the gradient of the MBAR free energy surface, so every protocol must
    # integrate to the same dG up to quadrature error: this checks the code and the nodes, not the sampling
    totals = reweighter.h_states.sum(axis=1)
    dG_ref = reweighter.f_k[np.argmax(totals)] - reweighter.f_k[np.argmin(totals)]
    resolved = refined & (np.abs(scores["dG"] - dG_ref) < args.dg_tol)
    explored = resolved & (scores["min_ess"] >= args.ess)
    hazard = np.array([p.charging_hazard() for p in protocols])
    np.savez(
        args.out, protocols=np.array([str(p) for p in protocols]), refined=refined, resolved=resolved,
        explored=explored, hazard=hazard, dG_ref=dG_ref, h_states=reweighter.h_states, f_k=reweighter.f_k,
        a=args.a, **scores,
    )

    def row(label, s, hazard):
        return (
            f"{label:40s} L={s['L']:7.2f}  I_var={s['I_var']:9.2f}  I_var/L^2={s['I_var'] / s['L'] ** 2:5.2f}  "
            f"dG={s['dG']:8.3f}  minESS={s['min_ess']:8.1f}  hazard={hazard:5.2f}"
        )

    print(f"MBAR dG (coupled - decoupled) = {dG_ref:.3f} kT; rescored {refined.sum()}, explored {explored.sum()}; "
          "hazard = max h_E/h_C where h_D < 0.001")
    for protocol, s in zip(references, reference_scores):
        print("reference  " + row(str(protocol), s, protocol.charging_hazard()))
    ranked = np.flatnonzero(explored)
    ranked = ranked[np.argsort(scores[args.rank][ranked])]
    print(f"\nbest explored by {args.rank}:")
    for i in ranked[: args.top]:
        print("  " + row(str(protocols[i]), {key: scores[key][i] for key in scores}, hazard[i]))
    unexplored = np.flatnonzero(resolved & ~explored)
    if len(unexplored):
        j = unexplored[np.argmin(scores[args.rank][unexplored])]
        print("  best unexplored: " + row(str(protocols[j]), {key: scores[key][j] for key in scores}, hazard[j]))

    if args.next:
        if args.choose:
            chosen = Protocol.parse(args.choose)
        elif len(ranked):
            chosen = protocols[ranked[0]]
        else:
            raise SystemExit("no explored protocol to place states for; give --choose")
        grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 401), [args.a]]))
        _, var, ess = reweighter.profile([chosen], grid)
        lambdas = equal_length_lambdas(grid, np.sqrt(var[0]), args.states)
        with open(args.next, "w") as f:
            json.dump({"protocol": str(chosen), "lambdas": lambdas.round(6).tolist()}, f, indent=1)
        print(f"\n{args.next}: {chosen} at {args.states} lambdas of equal thermodynamic length "
              f"(min ESS on the grid {ess.min():.0f}):\n  {np.round(lambdas, 4).tolist()}")


if __name__ == "__main__":
    main()
