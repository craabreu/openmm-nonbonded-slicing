# LBF protocol optimization

Scripts to choose the switching functions of a linear basis function (LBF) alchemical protocol for a
solute in water, using `SlicedNonbondedForce` with Lennard-Jones and Coulomb decapping. They sample a set
of states by replica exchange, score whole families of protocols by MBAR reweighting of those samples,
and place the states of the next round. They are meant to be read and adapted, not used as a library.

## The protocol

Three solute-solvent terms of a `SlicedNonbondedForce` are scaled by functions of one coupling parameter
λ ∈ [0, 1] (Correa et al., *J. Chem. Theory Comput.* 18, 5876, 2022):

| term | scales | global parameter |
|------|--------|------------------|
| C | capped Lennard-Jones | `lambdaLJ` |
| D | decapping terms, Lennard-Jones and Coulomb | `lambdaD` |
| E | electrostatics (Coulomb, capped below σ) | `lambdaQ` |

The energy is linear in the three parameters, U = U_rest + h_C U_C + h_D U_D + h_E U_E, and the plugin
computes U_C, U_D and U_E along with the energy. So the reduced potential of any state is known from the
three slice energies of each sample. Replica exchange gets the energies of every replica in every state
from one evaluation, and the search can reweight the samples to any protocol.

Each term follows S_{m0,m1}(x) = x^(m0+1) P(x), the polynomial step whose first m0 derivatives vanish at
x = 0 and first m1 derivatives at x = 1. Written as text, a protocol is, e.g.,

    C(1,0) D(11,1)[0.25] E(2,0)

that is, h_C = λ², h_E = λ³, and h_D zero up to a = 0.25 and S_{11,1} over [0.25, 1]. `D(inf)[0.25]` is a
C∞ step, and `paper` is the windowed S_{2,2} protocol of Correa et al.

By default, the search keeps only switching functions whose first derivative vanishes at both ends of
their interval (m0 ≥ 1 and m1 ≥ 1). Then dh/dλ, and with it the λ-force, is zero at λ = 0 and 1, which
λ-dynamics needs. In the pilot search (a = 0.2), this cost nothing in thermodynamic length: the best
protocol by L, C(1,1) D(6,2) E(2,1), already satisfied it.

The window keeps h_D, and the λ-force of decapping, exactly zero while the capped terms are weak. It
also keeps the hazard of a configuration where solvent overlaps the solute's capped core, weighted by a
tiny but nonzero h_D, away from small λ. The window start is a fixed choice, not a searched variable: the
scores measure efficiency, and overlaps are too rare for reweighted variances to see.

## Files

| file | what it does |
|------|--------------|
| `lbf.py` | switching functions, `Protocol`, state placement, the replica-exchange move, the system builder, the run loader |
| `simulate.py` | replica exchange of λ among the states of a protocol → `samples.npz` |
| `search.py` | pools runs, solves MBAR, scores protocol families → ranked table, `results.npz` and the next round's states |
| `validate.py` | ΔG of independent runs of one or more protocols, and its spread over runs |
| `test_lbf.py` | unit tests (`python -m pytest test_lbf.py`) |

Requirements: OpenMM with this plugin (with Coulomb decapping), `openff-toolkit`, `openmmforcefields`,
`pymbar` 4, `scipy`.

## A round

```bash
# 1. sample 21 uniformly spaced states of a starting protocol (one Context per state, on one GPU)
python simulate.py run0 --protocol "C(1,1) D(6,2)[0.25] E(2,1)"

# 2. score the protocol families against everything sampled so far, and place the next round's states
python search.py run0 --a 0.25 --next next.json

# 3. sample the chosen protocol at those states, and score again with both runs
python simulate.py run1 --plan next.json
python search.py run0 run1 --a 0.25 --next next.json

# 4. compare independent replicas of the final protocol with a reference
python simulate.py final_1 --plan next.json --seed 1   # and so on for more replicas
python simulate.py paper_1 --protocol paper --seed 1
python validate.py final=final_1,final_2,final_3 paper=paper_1,paper_2,paper_3
```

The solute and force fields are options of `simulate.py` (`--smiles`, `--forcefield` with a SMIRNOFF or GAFF
name, `--water`), as are the sampling settings; `python simulate.py --help` lists them.

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

- L = ∫ sd(du/dλ) dλ, the thermodynamic length. It is the cost of a protocol whose states are spaced at
  equal increments of length, which is how `--next` places them. This is the default ranking.
- I_var = ∫ var(du/dλ) dλ, the cost with uniformly spaced states, and the relevant measure when λ is
  sampled uniformly, as in λ-dynamics with a flat target (`--rank I_var`).
- I_var/L² ≥ 1, which is 1 when sd(du/dλ) is flat: how much a protocol depends on the spacing of its states.

Every protocol must integrate to the same ΔG, which checks the quadrature. A protocol counts as explored
when the Kish effective sample size is at least `--ess` at every node; protocols far from the sampled
states are not trusted. `--min_h_at_a C=0.1` keeps only protocols whose capped Lennard-Jones is already
that strong where decapping starts.
