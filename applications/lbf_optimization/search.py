"""Score LBF protocols by reweighting pooled samples with MBAR, and place the states of the next round.

usage: python search.py RUN [RUN ...] [--a 0.4] [--rank L] [--next next.json [--states 21]]

RUN is an output directory of simulate.py; runs of different solutes cannot be pooled. The reduced
potential of a protocol h(lambda) is u = beta h . U plus terms common to all states (see lbf.py), so any
protocol can be evaluated at any lambda from the stored slice energies. At Gauss-Legendre nodes, MBAR
reweighting gives <du/dlambda> and var(du/dlambda), with du/dlambda = beta dh/dlambda . U, and (in kT):

    L     = int_0^1 sd(du/dlambda) dlambda       thermodynamic length: the cost with optimally spaced states
    I_var = int_0^1 var(du/dlambda) dlambda      the cost with uniformly spaced states (L^2 <= I_var)
    dG    = int_0^1 <du/dlambda> dlambda          must agree across protocols (a check of the quadrature)

I_var/L^2 >= 1 measures how much a protocol depends on the spacing of its states; it is 1 when sd(du/dlambda)
is flat. A protocol counts as explored if the Kish effective sample size is at least --ess at every node.

The family (see RESULTS.md): the capped Lennard-Jones term C is the exponential (e^(b lambda) - 1)/(e^b - 1)
with b from --bC (polynomial steps for C only with --m0C and --m1C); the Lennard-Jones decapping D is a
polynomial step over [a, 1] with onset power m0 + 1 >= 8, so that var(du/dlambda) is finite where it
starts; the electrostatics E are a polynomial step over [aE, 1] with aE >= a (by default a, a + 0.05 and
a + 0.1: charging after the decapping has started lowers the charging hazard at little cost), and the
Coulomb decapping Q follows E (bare Coulomb; --Q D scores the capped-Coulomb scheme). Protocols whose charging hazard,
max h_E/h_C where h_D < 0.001, exceeds --max_hazard are left out: above it a water can fuse onto a solute
acceptor between the sampled states, which the samples rarely contain and so the scores rarely see. The
ranking is by I_var, the cost of uniform sampling in lambda. With --next, the best explored protocol
(or --choose) gets --states lambdas at equal increments of thermodynamic length, written as JSON for
simulate.py --plan.
"""

import argparse
import itertools
import json
import os

# pymbar solves with JAX, which would otherwise claim most of the GPU memory while simulate.py runs there
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.special import logsumexp

from lbf import DEFAULT_PROTOCOL, EXP, SMOOTH, HelpFormatter, Protocol, beta, equal_length_lambdas, load_run

DEFAULT_CHUNK = 1500  # targets reweighted at once; bounds the memory of Reweighter.at


def int_range(text):
    if text.lower() == "none":
        return range(0)
    lo, hi = (int(x) for x in text.split(":"))
    return range(lo, hi + 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument("runs", nargs="+", help="output directories of simulate.py, all of one system")
    parser.add_argument("--a", type=float, default=0.4, help="start of the decapping window of D")
    families = parser.add_argument_group("protocol family (integer ranges lo:hi, inclusive, or none)")
    families.add_argument(
        "--bC", type=float, nargs="*", metavar="B", default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        help="rates b of the exponential C",
    )
    families.add_argument("--m0C", type=int_range, default="none", help="m0 of polynomial steps for C, with --m1C")
    families.add_argument("--m1C", type=int_range, default="none", help="m1 of polynomial steps for C")
    families.add_argument("--m0D", type=int_range, default="7:11", help="m0 of the decapping step D; keep m0 >= 7")
    families.add_argument("--m1D", type=int_range, default="1:3", help="m1 of the decapping step D")
    families.add_argument("--m0E", type=int_range, default="1:3", help="m0 of the electrostatics step E")
    families.add_argument("--m1E", type=int_range, default="1:3", help="m1 of the electrostatics step E")
    families.add_argument(
        "--aE", type=float, nargs="+", metavar="A",
        help="starts of the window of E, each >= a (default: a, a + 0.05 and a + 0.1)",
    )
    families.add_argument(
        "--Q", choices=("D", "E"), nargs="+", default=["E"],
        help="which term the Coulomb decapping follows: E for bare Coulomb, D for capped Coulomb",
    )
    families.add_argument("--smooth_d", action="store_true", help="also score the C-infinity decapping step")
    families.add_argument(
        "--min_hC_at_a", type=float, default=0.0, metavar="VALUE",
        help="keep only protocols whose capped Lennard-Jones h_C is at least VALUE where decapping starts",
    )
    scoring = parser.add_argument_group("scoring")
    scoring.add_argument("--rank", choices=("L", "I_var"), default="I_var", help="the score that ranks protocols")
    scoring.add_argument(
        "--ess", type=float, default=100.0,
        help="Kish effective sample size a protocol needs at every node to count as explored",
    )
    scoring.add_argument("--nodes", type=int, default=64, help="Gauss-Legendre nodes of the screen")
    scoring.add_argument("--fine_nodes", type=int, default=512, help="Gauss-Legendre nodes of the rescoring")
    scoring.add_argument(
        "--refine", type=int, default=150,
        help="rescore this many best by each score with --fine_nodes; the rest keep the scores of the screen",
    )
    scoring.add_argument("--dg_tol", type=float, default=0.05, help="largest dG deviation (kT) of a resolved protocol")
    scoring.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="targets reweighted at once (memory)")
    scoring.add_argument(
        "--reference", action="append", default=[], help="more protocols to score besides paper and the default",
    )
    scoring.add_argument(
        "--max_hazard", type=float, default=0.15,
        help="keep protocols whose h_E/h_C, where h_D < 0.001, is at most this (inf for all)",
    )
    output = parser.add_argument_group("output")
    output.add_argument("--out", default="results.npz", help="scores of every protocol, for compare_solutes.py")
    output.add_argument("--top", type=int, default=20, help="rows of the ranked table")
    output.add_argument("--next", help="write the protocol, lambdas and system of the next round to this JSON file")
    output.add_argument("--states", type=int, default=21, help="states of the next round")
    output.add_argument("--choose", help='protocol of the next round, e.g. "default" (default: the best explored)')
    args = parser.parse_args()
    if not 0 < args.a < 1:
        parser.error("--a must lie in (0, 1)")
    if args.aE is None:
        args.aE = [round(args.a + step, 6) for step in (0.0, 0.05, 0.1) if args.a + step < 1]
    if min(args.aE) < args.a or max(args.aE) >= 1:
        parser.error("every --aE must lie in [a, 1): the charges follow the decapping")
    args.aE = sorted(set(args.aE))
    return args


def candidates(args):
    capped = [(EXP, b, 0.0) for b in args.bC] + [(m0, m1, 0.0) for m0, m1 in itertools.product(args.m0C, args.m1C)]
    decap = [(m0, m1, args.a) for m0, m1 in itertools.product(args.m0D, args.m1D)]
    if args.smooth_d:
        decap.append((SMOOTH, 0, args.a))
    for C in capped:
        for D in decap:
            for E in itertools.product(args.m0E, args.m1E, args.aE):
                for tie in args.Q:
                    yield Protocol((C, D, D if tie == "D" else E, E))


class Reweighter:
    """MBAR estimates at arbitrary states h from the pooled samples."""

    def __init__(self, runs, chunk=DEFAULT_CHUNK):
        from pymbar import MBAR

        # solute, force fields, box, temperature, pressure and capping orders must agree (lbf.SYSTEM_DEFAULTS)
        first = runs[0].system
        differ = sorted({k for run in runs[1:] for k in first if run.system[k] != first[k]})
        if differ:
            raise SystemExit(f"runs of different systems cannot be pooled: they differ in {', '.join(differ)}")
        self.system = first
        self.beta = beta(first["temperature"])
        self.h_states = np.concatenate([run.H for run in runs])
        series = [s for run in runs for s in run.series]
        N_k = np.array([len(s) for s in series])
        self.BU = self.beta * np.concatenate(series)  # (N, 4)
        u_kn = self.h_states @ self.BU.T
        self.f_k = MBAR(u_kn, N_k, solver_protocol="robust").f_k
        self.log_denominator = logsumexp(np.log(N_k)[:, None] + self.f_k[:, None] - u_kn, axis=0)
        self.chunk = chunk

    def at(self, H, dH):
        """<du/dlambda>, var(du/dlambda) and Kish ESS at the states H (T, 4), with dh/dlambda dH (T, 4)."""
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

    def score(self, protocols, nodes, breaks=()):
        """L, I_var, dG and minimum ESS of each protocol by Gauss-Legendre quadrature split at the breaks (the
        window starts, where the switching functions have kinks)."""
        lam, weights = [], []
        edges = [0.0] + sorted({b for b in breaks if 0 < b < 1}) + [1.0]
        for lo, hi in zip(edges[:-1], edges[1:]):
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
    H = reweighter.h_states
    if not (np.any(np.all(H == 0, axis=1)) and np.any(np.all(np.isclose(H, 1), axis=1))):
        raise SystemExit("the runs must include a decoupled state (lambda = 0) and a coupled one (lambda = 1)")
    print(f"{len(reweighter.h_states)} states, {len(reweighter.BU)} samples from {len(runs)} runs")

    protocols = list(candidates(args))
    protocols = [p for p in protocols if p.h(args.a)[0][0, 0] >= args.min_hC_at_a]
    protocols = [p for p in protocols if p.charging_hazard() <= args.max_hazard]
    if not protocols:
        raise SystemExit("no protocol in the family satisfies the constraints; widen the family or --max_hazard")
    breaks = [args.a] + args.aE
    references = [Protocol.parse(text) for text in ["paper", DEFAULT_PROTOCOL] + args.reference]
    print(f"{len(protocols)} protocols with the decapping window at a = {args.a:g}")

    # screen everything, then rescore the best by either score with more nodes
    scores = reweighter.score(protocols, args.nodes, breaks)
    top = np.union1d(np.argsort(scores["L"])[: args.refine], np.argsort(scores["I_var"])[: args.refine])
    fine = reweighter.score([protocols[i] for i in top] + references, args.fine_nodes, breaks)
    for key in scores:
        scores[key][top] = fine[key][: len(top)]
    reference_scores = [{key: fine[key][len(top) + j] for key in fine} for j in range(len(references))]
    refined = np.zeros(len(protocols), bool)
    refined[top] = True
    # <du/dlambda> reweighted by MBAR is the gradient of the MBAR free energy surface, so every protocol must
    # integrate to the same dG up to quadrature error: this checks the code and the nodes, not the sampling
    totals = reweighter.h_states.sum(axis=1)
    dG_ref = reweighter.f_k[np.argmax(totals)] - reweighter.f_k[np.argmin(totals)]
    resolved = np.abs(scores["dG"] - dG_ref) < args.dg_tol
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
        grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 401), breaks]))
        _, var, ess = reweighter.profile([chosen], grid)
        lambdas = equal_length_lambdas(grid, np.sqrt(var[0]), args.states)
        with open(args.next, "w") as f:
            plan = {"protocol": str(chosen), "lambdas": lambdas.round(6).tolist(), "system": reweighter.system}
            json.dump(plan, f, indent=1)
        print(f"\n{args.next}: {chosen} at {args.states} lambdas of equal thermodynamic length "
              f"(min ESS on the grid {ess.min():.0f}):\n  {np.round(lambdas, 4).tolist()}")


if __name__ == "__main__":
    main()
