# Plan: choosing LBF switching functions for Lambda-UFED

The procedure to run from scratch, built on what the phenol pilot (October 2026) settled. The pilot's
findings are inputs here, not things to search again; what the pilot showed to be unnecessary is left out.

## Fixed by the pilot

**Basis.** Four slice energies, C (capped Lennard-Jones), D (Lennard-Jones decapping), Q (Coulomb
decapping) and E (capped Coulomb), recorded for every sample. The protocol family is bare Coulomb,
h_Q = h_E, with the charges following the decapping. The Coulomb cap stays in the basis only so that capped
protocols can be scored as references: under the ordering rule it buys nothing (same length either way).

**Safety constraints, imposed a priori.**

- Decapping onset power k = m0 + 1 ≥ 8 at the window start, so that var(du/dλ) stays finite there.
- Charging hazard max h_E/h_C ≤ 0.25 wherever h_D < 10⁻³. Above about 0.5 a water fuses onto the phenol
  hydroxyl (O–Ow 1.1 Å). Reweighting cannot see the collapse, since the samples contain no collapsed
  configurations, so this constraint is the only thing that prevents it, and it must be checked on
  configurations, not on scores.

**Switching families.** C from a family with free end slopes, the exponential
h_C(λ) = (e^(bλ) − 1)/(e^b − 1). D and E as windowed polynomial steps S_{m0,m1}[a], with the E window at
or after the D window. The rule that dh/dλ vanish at both ends is dropped for C: it excluded every
switching function that makes sd(du/dλ) flat, and the pilot reached the exponential only by a detour
through the thermodynamic-length coordinate. For phenol, S_{2,1} → exponential with b = 2.5 takes I_var
from 240 to 204 with the hazard at its limit.

**The cost measure.** I_var = ∫ var(du/dλ) dλ, the cost of uniform sampling in λ, which is what
Lambda-UFED does with a flat target. L = ∫ sd(du/dλ) dλ and I_var/L² ≥ 1 are reported; a family is good
when I_var/L² is near 1 without a change of coordinate. (I_var/L² was 1.29 for the polynomial steps and
1.10 for the exponential C.)

**State placement for replica exchange.** Equal increments of thermodynamic length, with the replica flow
(feedback method of Katzgraber, Trebst, Huse and Troyer) as the check; the two agreed for phenol.

## The procedure

1. **Build and verify the system.** Solute, Sage 2.2.1 with AM1-BCC, TIP3P, the same box and PME
   settings as the pilot (`simulate.py` defaults). Check that the four-term system at full coupling
   reproduces the plain force field to numerical precision. One energy evaluation.

2. **Add a collapse diagnostic to the sampler.** Per frame, record the shortest distance between each
   solute hydrogen-bond donor or acceptor and a water oxygen, and store it with the slice energies. A
   distance below about 2.4 Å at a state with weak h_D is a collapse. This turns the hazard from an
   argument into a measurement.

3. **Reference run.** One replica-exchange run of a seed protocol that already satisfies the constraints
   and is close to the expected optimum: C exponential with b = 2, D(7,2)[0.4], E(2,1)[0.4], Q = E;
   21 states uniform in λ, 2 ns per state, K³ swap attempts per picosecond. Check the diagnostic before
   using the samples.

4. **Search by reweighting.** Score the family over b ∈ [1, 4]; D with m0 from 7 to 11 and m1 from 1
   to 3; window starts a ∈ {0.3, 0.4, 0.5}; E with m0, m1 from 1 to 3 and a_E ≥ a. Keep only protocols
   with Kish ESS ≥ 100 at every quadrature node and hazard ≤ 0.25. Rank by I_var. Verify that every
   protocol integrates to the same ΔG, which checks the quadrature. Score the paper protocol and the
   pilot's polynomial-step protocols as references.

5. **Second round.** Resample the best protocol with states at equal thermodynamic length from the
   reweighted sd profile. Pool with the reference run and rescore. If the ranking holds and the diagnostic
   shows no collapse, the protocol is chosen. If I_var/L² exceeds about 1.15, the family is missing a
   shape: extend it (for instance with a shared parameter that slows D and E together, which keeps their
   order and so the hazard) before going on.

6. **Validate in the target method.** The step the pilot never reached. Run Lambda-UFED with the chosen
   protocol and with the paper protocol, several independent replicas each, same wall time. Compare the
   error of ΔG against simulation time and the number of round trips in λ. Two things are tested here
   that reweighting cannot see: the correlation time along λ, and whether the finite force at λ = 0 and 1
   from the exponential causes trouble at the ends. If it does, replace the exponential by a form with the
   same interior and soft ends, and accept the small loss. Replica-exchange replicas (`validate.py`) are
   the cheap preliminary check.

7. **Generality.** Repeat steps 3 to 5 for two or three solutes that differ in what matters: a nonpolar
   one (methane), a larger polar one, and one with acceptors but no donor. Compare the optimal b, windows
   and orders. If they agree, publish them as defaults and the per-solute search becomes optional. If
   they do not, the deliverable is the one-round procedure, at the cost of one reference run per solute.

## Results for phenol (31 October 2026)

Steps 1 to 5 are done. Runs and tables are in `~/github_repos/lbf_runs/`.

1. **System.** At full coupling the four-term system matches the plain force field to 5e-9 relative in
   energy and 2e-10 kJ/mol/nm in forces (CUDA, double precision).
2. **Diagnostic.** Its first rule (any contact below 0.24 nm) flagged single-frame overlaps through the
   capped core, which are normal. The rule now counts the fraction of frames within 0.2 nm and flags it
   above 5% where the charges are on and h_D < 10⁻³. The mechanism of the collapse was found on the way. A
   water hydrogen has no Lennard-Jones sphere, so with bare Coulomb a protocol that charges while h_D ≈ 0
   crashes with NaN coordinates, and with capped Coulomb it samples a water fused onto the hydroxyl.
3. **Reference run** (`run_exp`) of `C(exp 2.5) D(7,3)[0.4] E(2,1)[0.4] Q=E`, the best protocol predicted
   from the pilot run (the seed of b = 2 above was superseded by that search). No state is flagged.
4. **Search** on `run_exp` (`search_s4_a0{3,4,5}.txt`). The seed ranks first at a = 0.4, with I_var 202.4
   against 202.2 predicted; a = 0.3 gives 200.6 at the hazard limit, and a = 0.5 gives 208. The hazard limit
   was checked with 2 ns single-state runs of family members: 0.25 and 0.35 are stable, 0.5 crashes once,
   and 0.7 and 0.9 collapse. The limit of 0.25 keeps a margin of about 2.
5. **Second round** (`run_exp2`, 21 states at equal thermodynamic length). Pooled with `run_exp`, the
   ranking holds (I_var 202.3, I_var/L² 1.08) and nothing is flagged. Neighbor acceptance is 0.62 to
   0.67, against 0.49 to 0.84 for uniform states. Round trips in 2000 iterations: 141, against 128 for
   uniform states and 99 for the pilot's polynomial-step protocol. ΔG over the three runs of the bare
   basis: −10.68, −10.56 and −10.65 kT.

**Chosen protocol for phenol: `C(exp 2.5) D(7,3)[0.4] E(2,1)[0.4] Q=E`.** Step 6 (Lambda-UFED) is postponed.

7. **Generality** (`run_methane`, `run_anisole`, `run_paracetamol`: reference runs of the phenol protocol,
   uniform states; searches `search_<solute>_a0{3,4,5}.txt`, with b from 0.5; `common.py` compares). No run
   flags a collapse. I_var in kT²:

   | solute | ΔG (kT) | paper | pilot C(2,1) | phenol protocol | own best |
   |---|---|---|---|---|---|
   | phenol | −10.6 | 409 | 240 | 202 | 201, C(exp 2.5) D(11,3)[0.3] E(3,1)[0.3] |
   | methane | +4.1 | 101 | 63 | 56 | 48, C(exp 2.5) D(9,1)[0.5] E(3,1)[0.5] |
   | anisole | −4.3 | 409 | 241 | 197 | 188, C(exp 3) D(8,1)[0.4] E(3,1)[0.4] |
   | paracetamol | −26.7 | 681 | 401 | 344 | 335, C(exp 2.5) D(7,3)[0.3] E(2,1)[0.3] |

   The optima agree on b = 2.5 to 3 and differ in D and E orders that barely matter. Over the protocols
   explored for all four, the one closest to every optimum is **`C(exp 2.5) D(8,2)[0.4] E(3,1)[0.4] Q=E`**:
   within 8% of each solute's best, with hazard 0.13. The phenol protocol is within 3% for the three polar
   solutes and 18% off for methane. The per-solute search is therefore optional: the default loses at most
   8%, which is less than the spread of ΔG between independent runs suggests is resolvable. A second-round
   run of the default on paracetamol, the hardest case (`run_paracetamol2`), checks it directly.

## Code changes

Done on this branch, all in `applications/lbf_optimization`:

- `lbf.py`: the exponential family in the protocol text (`C(exp 2.5)`), with its h and dh, and end orders
  (0, 0); `polar_atoms` and `closest_contact` for the diagnostic.
- `search.py`: the family of step 4 as the defaults (`--bC` 1 to 4, D with m0 7 to 11 and m1 1 to 3, E
  with m0 and m1 1 to 3 at `--aE` = a, `--Q E`); `--max_hazard 0.25` and `--rank I_var` by default;
  polynomial steps for C only on request (`--m0C`, `--m1C`).
- `simulate.py`: the collapse diagnostic, `contact` in `samples.npz` (per iteration and replica, nm), and
  the per-state report at the end of a run.

Still to write:

- The Lambda-UFED validation of step 6, once the Lambda-UFED driver is available.

## What to leave out

The λ(c) reparametrization (`reparametrize.py`), the Bernstein, tanh and Gompertz fits, the λ¹² forms,
the search over Coulomb caps, and the feedback optimization as anything but a check. They were how the
pilot found the family and the constraints. With those known, the procedure is a direct search in λ.

## Decisions pending

- Which solutes to use in step 7.
- The GPU budget for steps 6 and 7; the validation replicas are the largest item.
- Whether finite λ-forces at the ends are acceptable in the Lambda-UFED driver (settled by step 6).

## Pilot data

Runs and results in `~/github_repos/lbf_runs/` (`run_bare` is the reference run of
`C(2,1) D(7,2)[0.4] E(2,1)[0.4] Q=E`, 21 states, 2.2 ns; `search_bare.txt` and `results_bare.npz` the
search on it). The older runs (pilot, run0, run_k8, the cap runs, run_d12) are in the legacy three-term
basis and only weight protocols with h_Q = h_D.
