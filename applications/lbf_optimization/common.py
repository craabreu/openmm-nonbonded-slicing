"""One protocol for all solutes: I_var relative to each solute's own best, over the protocols every search explored.

usage: python common.py [SOLUTE ...] [--max_hazard=0.15]   (run from the directory with the results_*.npz)"""
import sys
import numpy as np
SOLUTES = {"phenol": "results_s4_a0{}.npz", "methane": "results_methane_a0{}.npz",
           "anisole": "results_anisole_a0{}.npz", "paracetamol": "results_paracetamol_a0{}.npz"}
MAX_HAZARD = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--max_hazard=")), 0.25))
names = [a for a in sys.argv[1:] if not a.startswith("--")]
SOLUTES = {k: v for k, v in SOLUTES.items() if k in names or not names}
table = {}
for a in "345":
    for name, pat in SOLUTES.items():
        d = np.load(pat.format(a))
        for p, ok, iv, hz in zip(d["protocols"], d["explored"], d["I_var"], d["hazard"]):
            if ok and np.isfinite(iv) and hz <= MAX_HAZARD:
                table.setdefault(str(p), {})[name] = float(iv)
best = {s: min(v[s] for v in table.values() if s in v) for s in SOLUTES}
rows = []
for p, v in table.items():
    if len(v) == len(SOLUTES):
        r = np.array([v[s] / best[s] for s in SOLUTES])
        rows.append((r.max(), r.mean(), p, r))
print("own best I_var:", {s: round(b, 1) for s, b in best.items()})
print(f"{len(rows)} protocols explored for every solute; ratio to each solute's best ({', '.join(SOLUTES)}):")
for key, label in [(0, "minimax"), (1, "mean")]:
    print(f"best by {label}:")
    for row in sorted(rows, key=lambda r: r[key])[:6]:
        print(f"  {row[2]:42s} max {row[0]:.3f} mean {row[1]:.3f}  " + " ".join(f"{x:.3f}" for x in row[3]))
seed = "C(exp 2.5) D(7,3)[0.4] E(2,1)[0.4] Q=E"
row = next((r for r in rows if r[2] == seed), None)
if row is None:
    print(f"phenol winner {seed}: excluded (hazard above {MAX_HAZARD:g} or not explored for every solute)")
else:
    print(f"phenol winner: {seed} max {row[0]:.3f} mean {row[1]:.3f}  " + " ".join(f"{x:.3f}" for x in row[3]))
