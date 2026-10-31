"""Express a protocol in the thermodynamic-length coordinate c, for lambda-dynamics with a flat target.

usage: python reparametrize.py RUN [RUN ...] --protocol "C(2,1) D(7,2)[0.4] E(2,1)[0.4] Q=E" [--degree 8]
                               [--out lambda_of_c.npz] [--plot lambda_of_c.png]

With c(lambda) = int_0^lambda sd(du/dlambda') dlambda' / L, the standard deviation of du/dc is L everywhere,
so sampling c uniformly is as hard at every point of the path: the states that search.py --next places at
equal increments of length are the uniform grid in c. Lambda-dynamics in c needs lambda(c) in closed form.
Its ends are singular, lambda ~ c^(1/(p0+1)) and 1 - lambda ~ (1 - c)^(1/(p1+1)) with (p0, p1) the end
orders of the protocol, so a polynomial in c cannot follow it; lambda(c) = lambda0(w(c)) can, with lambda0
carrying the exponents and w a nondecreasing Bernstein polynomial of modest degree, chosen to flatten
sd(du/dc) (see lbf.lambda_of_c). The switching functions in c are the exact h(lambda(c)), so their bounds,
windows and onset powers are untouched. The flatness I_var/L^2 reached is checked by reweighting.
"""

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # keep pymbar's JAX off the GPU that simulate.py uses

import numpy as np

from lbf import Protocol, fit_lambda_of_c, lambda_of_c, load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="output directories of simulate.py to pool")
    parser.add_argument("--protocol", required=True, help='e.g. "C(2,1) D(7,2)[0.4] E(2,1)[0.4] Q=E"')
    parser.add_argument("--degree", type=int, default=8, help="degree of the Bernstein polynomial w (default: 8)")
    parser.add_argument("--out", default="lambda_of_c.npz")
    parser.add_argument("--plot")
    args = parser.parse_args()

    from search import Reweighter

    protocol = Protocol.parse(args.protocol)
    reweighter = Reweighter([load_run(path) for path in args.runs], 1500)
    lam = np.linspace(0.0, 1.0, 8001)
    _, var, ess = reweighter.profile([protocol], lam)
    sd = np.sqrt(var[0])
    L = np.trapezoid(sd, lam)
    uniform = np.trapezoid(var[0], lam) / L**2
    p0, p1 = protocol.end_orders()
    w, flatness = fit_lambda_of_c(lam, sd, p0, p1, args.degree)

    # the check: reweight at the states lambda(c) with dh/dc = dh/dlambda dlambda/dc
    c = np.linspace(0.0, 1.0, 1001)
    lam_c, dlam = lambda_of_c(c, p0, p1, w)
    H, dH = protocol.h(lam_c)
    _, var_c, _ = reweighter.at(H.T, (dH * dlam).T)
    sd_c = np.sqrt(var_c)
    checked = np.trapezoid(var_c, c) / np.trapezoid(sd_c, c) ** 2
    inner = (c > 0.02) & (c < 0.98)
    print(f"{protocol}: L = {L:.2f} kT, min ESS on the grid {ess.min():.0f}; end orders (p0, p1) = ({p0}, {p1})")
    print(f"I_var/L^2: {uniform:.3f} in lambda, {flatness:.4f} in c (reweighted check {checked:.4f}); "
          f"sd(du/dc) from {sd_c[inner].min():.1f} to {sd_c[inner].max():.1f} kT over (0.02, 0.98)")
    print(f"lambda(c) = (1 - (1 - w)^(1/{p1 + 1}))^(1/{p0 + 1}), w(c) the Bernstein polynomial with coefficients")
    print("  " + ", ".join(f"{x:.6f}" for x in w))
    np.savez(args.out, protocol=str(protocol), L=L, p0=p0, p1=p1, w=w, flatness=flatness, I_var_over_L2_in_lambda=uniform)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        length = np.concatenate([[0.0], np.cumsum(0.5 * (sd[1:] + sd[:-1]) * np.diff(lam))]) / L
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
        axes[0].plot(c, np.interp(c, length, lam), color="0.4", lw=2.5, label="exact")
        axes[0].plot(c, lam_c, "--", lw=1.5, label=f"λ₀(w), degree {args.degree}")
        axes[0].set_title("λ(c)"); axes[0].set_ylabel("λ"); axes[0].legend(fontsize=8)
        for i, term in enumerate("CDQE"):
            if term != "Q" or not np.allclose(H[2], H[1]) and not np.allclose(H[2], H[3]):
                axes[1].plot(c, H[i], lw=2, label=f"h_{term}(λ(c))")
        axes[1].set_title("switching functions in c"); axes[1].legend(fontsize=8)
        axes[2].axhline(L, color="0.4", lw=2, label="flat at L")
        axes[2].plot(c, sd_c, lw=1.5, label=f"λ₀(w): I_var/L² = {checked:.3f}")
        axes[2].plot(lam, sd, lw=1, alpha=0.5, label=f"in λ: I_var/L² = {uniform:.2f}")
        axes[2].set_ylim(0, 1.3 * sd.max()); axes[2].set_title("sd(du/dc) (kT)"); axes[2].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("c"); ax.grid(alpha=0.25)
        fig.suptitle(str(protocol))
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)


if __name__ == "__main__":
    main()
