"""Hydration free energy from runs of simulate.py, and its spread over independent runs.

usage: python free_energy.py LABEL=RUN[,RUN...] [LABEL=RUN[,RUN...] ...]

Each RUN is an output directory of simulate.py with states at lambda = 0 and 1. For each run, MBAR gives
dG = G(lambda = 1) - G(lambda = 0), the free energy of coupling the solute to the water, that is, its
hydration free energy (the solute's internal interactions are on throughout), from samples subsampled by
the statistical inefficiency of each state's time series. For replica exchange, that series follows the
state, not a replica, so it includes the decorrelation by exchanges. The observable is a proxy for
du/dlambda: the slice energies weighted by the difference of h between the neighboring states.

Per label, it prints dG of each run with its MBAR standard error, in kT, kJ/mol and kcal/mol, and, for
several runs, their mean and standard deviation. Only runs that are independent (different seeds) make that
standard deviation a measure of precision.
"""

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # keep pymbar's JAX off the GPU that simulate.py uses

import numpy as np

from lbf import KB, HelpFormatter, beta, load_run

KJ_PER_KCAL = 4.184


def free_energy(path):
    """dG and its MBAR standard error in kT, the number of uncorrelated samples, and the temperature."""
    from pymbar import MBAR, timeseries

    run = load_run(path)
    H = run.H
    if not (np.all(H[0] == 0) and np.allclose(H[-1], 1)):
        raise SystemExit(f"{path} has no state at lambda = 0 or none at lambda = 1: no free energy to compute")
    b = beta(run.temperature)
    kept = []
    for k, U in enumerate(run.series):
        neighbors = H[min(k + 1, len(H) - 1)] - H[max(k - 1, 0)]
        observable = b * U @ neighbors
        g = timeseries.statistical_inefficiency(observable)
        kept.append(U[timeseries.subsample_correlated_data(observable, g=g)])
    N_k = np.array([len(U) for U in kept])
    mbar = MBAR(b * H @ np.concatenate(kept).T, N_k, solver_protocol="robust")
    result = mbar.compute_free_energy_differences()
    return result["Delta_f"][0, -1], result["dDelta_f"][0, -1], N_k.sum(), run.temperature


def labeled_runs(text):
    label, _, paths = text.partition("=")
    if not label or not paths:
        raise argparse.ArgumentTypeError(f"expected LABEL=RUN[,RUN...], got {text!r}")
    return label, paths.split(",")


def in_units(value_kT, temperature):
    kT_kJ = KB * temperature
    return f"{value_kT:8.3f} kT = {value_kT * kT_kJ:8.3f} kJ/mol = {value_kT * kT_kJ / KJ_PER_KCAL:7.3f} kcal/mol"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument(
        "groups", nargs="+", type=labeled_runs, metavar="LABEL=RUN[,RUN...]",
        help="a label and the output directories of simulate.py to average under it",
    )
    for label, paths in parser.parse_args().groups:
        print(label)
        results = [free_energy(path) for path in paths]
        for path, (dG, error, n, temperature) in zip(paths, results):
            print(f"  {path}: dG = {in_units(dG, temperature)}  (MBAR stderr {error:.3f} kT, "
                  f"{n} uncorrelated samples)")
        if len(results) > 1:
            dG = np.array([r[0] for r in results])
            temperature = results[0][3]
            print(f"  mean of {len(dG)} runs: dG = {in_units(dG.mean(), temperature)}  "
                  f"(sd over runs {dG.std(ddof=1):.3f} kT)")


if __name__ == "__main__":
    main()
