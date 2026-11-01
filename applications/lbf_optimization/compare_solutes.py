"""One protocol for several solutes: the protocols that the searches of every solute explored, ranked by how
far their I_var is from each solute's own best.

usage: python compare_solutes.py LABEL=RESULTS[,RESULTS...] [LABEL=RESULTS[,RESULTS...] ...]
                                 [--max_hazard 0.15] [--top 6] [--show PROTOCOL ...]

Each RESULTS is a results.npz written by search.py (--out), for instance one per decapping window start a;
a protocol scored in several files of one solute keeps its lowest I_var. Only protocols explored for every
solute, with charging hazard at most --max_hazard, are compared. For each, the ratio of its I_var to the
solute's own best is the factor of simulation time it costs there for the same error. The table ranks
protocols by the largest ratio (minimax) and by the mean ratio; --show prints the ratios of given protocols
(by default the default protocol of simulate.py).
"""

import argparse

import numpy as np

from lbf import DEFAULT_PROTOCOL, Protocol


def labeled_files(text):
    label, _, paths = text.partition("=")
    if not label or not paths:
        raise argparse.ArgumentTypeError(f"expected LABEL=RESULTS[,RESULTS...], got {text!r}")
    return label, paths.split(",")


def explored(paths, max_hazard):
    """{protocol text: lowest I_var} over the explored protocols of these results files."""
    scores = {}
    for path in paths:
        d = np.load(path)
        for protocol, ok, i_var, hazard in zip(d["protocols"], d["explored"], d["I_var"], d["hazard"]):
            if ok and np.isfinite(i_var) and hazard <= max_hazard:
                key = str(protocol)
                scores[key] = min(float(i_var), scores.get(key, np.inf))
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("solutes", nargs="+", type=labeled_files, metavar="LABEL=RESULTS[,RESULTS...]")
    parser.add_argument("--max_hazard", type=float, default=0.15, help="as in search.py (default: 0.15)")
    parser.add_argument("--top", type=int, default=6, help="rows per ranking (default: 6)")
    parser.add_argument("--show", nargs="+", default=[DEFAULT_PROTOCOL], metavar="PROTOCOL")
    args = parser.parse_args()

    labels = [label for label, _ in args.solutes]
    scores = {label: explored(paths, args.max_hazard) for label, paths in args.solutes}
    for label in labels:
        if not scores[label]:
            raise SystemExit(f"{label}: no explored protocol with hazard <= {args.max_hazard:g}")
    best = {label: min(scores[label].items(), key=lambda item: item[1]) for label in labels}
    print(f"own best I_var (hazard <= {args.max_hazard:g}):")
    for label in labels:
        print(f"  {label:12s} {best[label][1]:8.1f}  {best[label][0]}")

    common = set.intersection(*(set(s) for s in scores.values()))
    ratios = {p: np.array([scores[label][p] / best[label][1] for label in labels]) for p in common}
    width = max(len(p) for p in ratios) if ratios else 40
    header = f"  {'':{width}s}  {'max':>5s}  {'mean':>5s}  " + "  ".join(f"{label:>8s}" for label in labels)

    def row(protocol, r):
        return f"  {protocol:{width}s}  {r.max():5.3f}  {r.mean():5.3f}  " + "  ".join(f"{x:8.3f}" for x in r)

    print(f"\n{len(common)} protocols explored for every solute; I_var relative to each solute's own best")
    for name, key in (("largest ratio", np.max), ("mean ratio", np.mean)):
        print(f"\nbest by {name}:\n{header}")
        for protocol in sorted(ratios, key=lambda p: key(ratios[p]))[: args.top]:
            print(row(protocol, ratios[protocol]))
    print()
    for text in args.show:
        parsed = Protocol.parse(text)
        protocol = str(parsed)
        if protocol in ratios:
            print(row(protocol, ratios[protocol]))
        elif parsed.charging_hazard() > args.max_hazard:
            print(f"  {protocol}: hazard {parsed.charging_hazard():.2f}, above {args.max_hazard:g}")
        else:
            missing = [label for label in labels if protocol not in scores[label]]
            print(f"  {protocol}: not explored, or hazard above {args.max_hazard:g}, for {', '.join(missing)}")


if __name__ == "__main__":
    main()
