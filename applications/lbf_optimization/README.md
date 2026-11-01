# LBF protocol optimization

Scripts to choose the switching functions of a linear basis function (LBF) alchemical protocol for a
solute in water, using `SlicedNonbondedForce` with Lennard-Jones and Coulomb decapping. They sample a set
of states by replica exchange, score whole families of protocols by MBAR reweighting of those samples,
and place the states of the next round. They are meant to be read and adapted, not used as a library.

## The protocol

Four solute-solvent terms of a `SlicedNonbondedForce` are scaled by functions of one coupling parameter
λ ∈ [0, 1] (Correa et al., *J. Chem. Theory Comput.* 18, 5876, 2022):

| term | scales | global parameter |
|------|--------|------------------|
| C | capped Lennard-Jones | `lambdaLJ` |
| D | Lennard-Jones decapping | `lambdaD` |
| Q | Coulomb decapping | `lambdaQD` |
| E | electrostatics (Coulomb, capped below σ) | `lambdaQ` |

The energy is linear in the four parameters, U = U_rest + h_C U_C + h_D U_D + h_Q U_Q + h_E U_E, and the
plugin computes the four slice energies along with the energy. So the reduced potential of any state is
known from the four slice energies of each sample. Replica exchange gets the energies of every replica in
every state from one evaluation, and the search can reweight the samples to any protocol.

The two decapping terms tell the electrostatic schemes apart. With h_Q = h_D, the Coulomb potential stays
capped until the Lennard-Jones core is decapped, so the charges can switch on while solvent still overlaps
the solute. With h_Q = h_E, the charges switch on the bare Coulomb potential, as in Correa et al., whose
plugin had no Coulomb capping; this is safe only when the charges follow the decapping, since nothing but
the decapped core keeps an attractive pair apart. Both are protocols of the same basis, so samples of either
reweight to the other. Runs made before the split stored the two decapping terms as one energy and can
only weight protocols with h_Q = h_D; `search.py` says so when it pools one.

Each term follows S_{m0,m1}(x) = x^(m0+1) P(x), the polynomial step whose first m0 derivatives vanish at
x = 0 and first m1 derivatives at x = 1, or the exponential (e^(bx) − 1)/(e^b − 1). Written as text, a
protocol is, e.g.,

    C(exp 2.5) D(7,2)[0.4] E(2,1)[0.4] Q=E

that is, h_C the exponential with rate 2.5, h_D zero up to a = 0.4 and S_{7,2} over [0.4, 1], h_E S_{2,1}
over the same window, and the Coulomb decapping tied to E. Q follows D unless the text says `Q=E` or gives
it a switching function of its own, `Q(7,2)[0.4]`. `D(inf)[0.25]` is a C∞ step, and `paper` is the windowed
S_{2,2} protocol of Correa et al., with bare Coulomb.

The exponential is the family for C. Every polynomial step has a zero slope at one end or both, and with
any of them sd(du/dλ) keeps a peak where the cavity grows, at small λ; the exponential, whose slope is finite
at both ends, flattens it. For phenol, replacing S_{2,1} by the exponential with b = 2.5 takes I_var from
240 to 204 kT², and b is set by the charging hazard, not by the score. The pilot found this the long way
round, by mapping λ to the thermodynamic-length coordinate (`reparametrize.py`); the map only rediscovers
the switching function, so the search now scores the family directly. The price is a finite λ-force at
λ = 0 and 1, which is what makes the variance finite there; whether λ-dynamics minds is for the validation
runs to show (see `PLAN.md`).

The window keeps h_D, and the λ-force of decapping, exactly zero while the capped terms are weak. It
also keeps the hazard of a configuration where solvent overlaps the solute's capped core, weighted by a
tiny but nonzero h_D, away from small λ. The window start is a fixed choice, not a searched variable: the
scores measure efficiency, and overlaps are too rare for reweighted variances to see.

The electrostatics have a window too (`--aE`, by default the start of the decapping window). Charging
against a still-capped core is a contest between the attraction, scaled by h_E, and the capped repulsion,
scaled by h_C. A water hydrogen has no Lennard-Jones sphere, so only the capped O–O repulsion holds it off
a solute acceptor. With bare Coulomb (`Q=E`) the attraction is unbounded at contact: a protocol that charges
while h_D is still essentially zero pulls the hydrogen in and the simulation crashes with NaN coordinates.
For phenol this happened within 40 ps at h_E = 0.02 and h_D = 0. With the Coulomb capped until the core is
restored (Q following D), the same protocol does not crash but samples a water fused onto the hydroxyl
(O–Ow 1 Å). Windowing E with D keeps h_E small where h_D is negligible, and the search drops protocols
whose hazard, max h_E/h_C where h_D < 10⁻³, exceeds `--max_hazard` (0.25 by default). The limit was checked
with 2 ns single-state runs of family members, at the λ where h_D reaches 10⁻⁶ and at the peak of
h_E/h_C. Hazards of 0.25 and 0.35 were stable. At 0.5 one state crashed after 900 ps and the other had a
water within 0.2 nm of the hydroxyl in 7% of frames. At 0.7 and 0.9 the water fused within 100 ps, at
h_D = 10⁻³, and the runs crashed. The limit of 0.25 leaves a margin of about 2. The samples never contain a collapse,
so no reweighted score can see it. Instead, `simulate.py` records, per frame, the closest approach of a
solute N or O to a water oxygen. At the end it reports, per state, the fraction of frames within 0.2 nm,
and it flags states with the charges on and h_D < 10⁻³ where that fraction exceeds 5%. Brief overlaps
are normal wherever the core is capped, so it is their frequency that marks early charging: for phenol,
0 to 1.5% under the searched protocols against 7 to 15% under one that charges early. A crash in such a
state is reported with its likely cause.

## Files

| file | what it does |
|------|--------------|
| `lbf.py` | switching functions, `Protocol`, state placement, the replica-exchange move, the system builder, the run loader |
| `simulate.py` | replica exchange of λ among the states of a protocol → `samples.npz` |
| `search.py` | pools runs, solves MBAR, scores the protocol family under the hazard limit → ranked table, `results.npz` and the next round's states |
| `PLAN.md` | the procedure to run from scratch, with what the phenol pilot fixed and what it ruled out |
| `validate.py` | ΔG of independent runs of one or more protocols, and its spread over runs |
| `flow.py` | the replica flow of a run, and the feedback-optimized states next to the equal-length ones |
| `reparametrize.py` | λ(c) in closed form for a protocol, c the thermodynamic length, for λ-dynamics with a flat target |
| `test_lbf.py` | unit tests (`python -m pytest test_lbf.py`) |

Requirements: OpenMM with this plugin (with Coulomb decapping), `openff-toolkit`, `openmmforcefields`,
`pymbar` 4, `scipy`.

## A round

```bash
# 1. sample 21 uniformly spaced states of a seed protocol (one Context per state, on one GPU)
python simulate.py run0 --protocol "C(exp 2) D(7,2)[0.4] E(2,1)[0.4] Q=E"

# 2. score the protocol family against everything sampled so far, and place the next round's states
python search.py run0 --a 0.4 --next next.json

# 3. sample the chosen protocol at those states, and score again with both runs
python simulate.py run1 --plan next.json
python search.py run0 run1 --a 0.4 --next next.json

# 4. compare independent replicas of the final protocol with a reference
python simulate.py final_1 --plan next.json --seed 1   # and so on for more replicas
python simulate.py paper_1 --protocol paper --seed 1
python validate.py final=final_1,final_2,final_3 paper=paper_1,paper_2,paper_3
```

The solute and force fields are options of `simulate.py` (`--smiles`, `--forcefield` with a SMIRNOFF or GAFF
name, `--water`), as are the sampling settings; `python simulate.py --help` lists them. A run stores its
solute and force fields, the plan that `search.py --next` writes carries them, and `simulate.py --plan` uses
them, so a later round cannot silently sample another solute. `--capping_orders M0 M1` sets the orders of the
capped potentials, and with them the height of the cap: (3, 3), the default, caps the Lennard-Jones
potential at 119 ε, (2, 4) at 684 ε. `search.py` does not pool runs of different solutes, force fields or
capping orders, since they sample different systems.

### Replica exchange

Each replica has its own `Context`, so an exchange only swaps the global parameters of two Contexts. After
every iteration (1 ps by default), `--swap_attempts` random pairs of replicas, K³ by default, are swapped
with the Metropolis criterion. Each attempt leaves the distribution of state permutations invariant, and
enough attempts approach drawing a new permutation from it (Chodera and Shirts, *J. Chem. Phys.* 135,
194110, 2011). With only nearby states swapping, the labels move like a shuffle by neighbor transpositions,
which needs about K³ attempts to forget the starting order. They take about 10 ms per iteration for
K = 21. `--no_exchange` keeps every replica in its state.

`samples.npz` stores, per iteration, the state of each replica and its slice energies, including the
`--equilibration` iterations, which the other scripts skip. `checkpoint.npz` is written with it every
`--checkpoint_interval` iterations; rerunning the same command resumes from it. If a replica fails, the
error names its state and λ.

### Scores

At each λ, MBAR reweighting gives the mean and variance of du/dλ, and the search integrates:

- I_var = ∫ var(du/dλ) dλ, the cost with uniformly spaced states, and the relevant measure when λ is
  sampled uniformly, as in λ-dynamics with a flat target. This is the default ranking.
- L = ∫ sd(du/dλ) dλ, the thermodynamic length. It is the cost of a protocol whose states are spaced at
  equal increments of length, which is how `--next` places them (`--rank L`).
- I_var/L² ≥ 1, which is 1 when sd(du/dλ) is flat: how much a protocol depends on the spacing of its states.

Every protocol must integrate to the same ΔG, which checks the quadrature. A protocol counts as explored
when the Kish effective sample size is at least `--ess` at every node; protocols far from the sampled
states are not trusted. `--min_h_at_a C=0.1` keeps only protocols whose capped Lennard-Jones is already
that strong where decapping starts.

### Placing the states

Equal increments of thermodynamic length place the states by the overlap of neighbors, from the
reweighted variance alone. The feedback optimization of Katzgraber, Trebst, Huse and Troyer (*J. Stat.
Mech.* P03018, 2006) places them by the dynamics instead: with each replica labeled by the end state it
visited last, the fraction f(λ) of up-walkers falls from 1 to 0, and the rate of round trips is largest
when the density of states is proportional to √(−df/dλ). `flow.py RUN` measures f from a run's replica
history and prints both allocations. For phenol they agree to within the noise of f (about 100 round trips
in 2 ns), apart from the first interior state, which the flow keeps closer to λ = 0; no state relaxes
slowly, so the simpler equal-length placement is the one used, and the flow is the check.

### The protocol in the thermodynamic-length coordinate

For λ-dynamics with a flat target, the coordinate to sample is c(λ) = ∫₀^λ sd(du/dλ′) dλ′ / L, along which
sd(du/dc) = L everywhere; the equal-length states are its uniform grid. `reparametrize.py RUN --protocol P`
gives λ(c) in closed form: the ends of the exact map are singular, λ ∝ c^(1/(p0+1)) and 1 − λ ∝ (1 − c)^(1/(p1+1))
with (p0, p1) the end orders of the protocol, so λ(c) = (1 − (1 − w(c))^(1/(p1+1)))^(1/(p0+1)) with a
nondecreasing Bernstein polynomial w, whose coefficients are chosen to flatten sd(du/dc). The switching
functions in c are the exact h(λ(c)), so they keep their bounds, windows and onset powers; a degree of 8
leaves I_var/L² within 2% of 1 for phenol. Fitting each h(c) by a polynomial instead can make it negative,
and fitting λ(c) by a plain polynomial cannot follow the ends.
