"""Free energy of independent runs of one or more protocols, to compare their precision.

usage: python validate.py LABEL=RUN[,RUN...] [LABEL=RUN[,RUN...] ...]

Each RUN is an output directory of simulate.py for one replica of a protocol.
For each run, MBAR gives dG = f(lambda = 1) - f(lambda = 0) in kT, the coupling free energy, from samples
subsampled by the statistical inefficiency of each state's time series. For replica exchange, that series
follows the state, not a replica, so it includes the decorrelation by exchanges. The observable is a proxy
for du/dlambda: the slice energies weighted by the difference of h between the neighboring states.

Per label, it prints dG and the MBAR standard error of each run, and the standard deviation over runs,
which is the honest measure of precision when runs are independent.
"""

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # keep pymbar's JAX off the GPU that simulate.py uses

import numpy as np
from pymbar import MBAR, timeseries

from lbf import beta, load_run


def free_energy(path):
    run = load_run(path)
    b = beta(run.temperature)
    H = run.H
    kept = []
    for k, U in enumerate(run.series):
        neighbors = H[min(k + 1, len(H) - 1)] - H[max(k - 1, 0)]
        observable = b * U @ neighbors
        g = timeseries.statistical_inefficiency(observable)
        kept.append(U[timeseries.subsample_correlated_data(observable, g=g)])
    N_k = np.array([len(U) for U in kept])
    mbar = MBAR(b * H @ np.concatenate(kept).T, N_k, solver_protocol="robust")
    result = mbar.compute_free_energy_differences()
    return result["Delta_f"][0, -1], result["dDelta_f"][0, -1], N_k.sum()


def labeled_runs(text):
    label, _, paths = text.partition("=")
    if not label or not paths:
        raise argparse.ArgumentTypeError(f"expected LABEL=RUN[,RUN...], got {text!r}")
    return label, paths.split(",")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="+", type=labeled_runs, metavar="LABEL=RUN[,RUN...]")
    for label, paths in parser.parse_args().groups:
        dG, error, n = np.array([free_energy(path) for path in paths]).T
        spread = f"{dG.std(ddof=1):.3f}" if len(dG) > 1 else "n/a"
        print(
            f"{label:12s} dG (kT): {np.round(dG, 3).tolist()}  mean {dG.mean():.3f}  sd over runs {spread}  "
            f"MBAR stderr: {np.round(error, 3).tolist()}  uncorrelated samples: {n.astype(int).tolist()}"
        )


if __name__ == "__main__":
    main()
