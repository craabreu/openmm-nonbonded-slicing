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

## Code changes needed

Small, all in `applications/lbf_optimization`:

- `lbf.py`: the exponential family for C in the protocol text (e.g. `C(exp 2.5)`), with its h and dh.
- `search.py`: the exponential in the candidate family; `--max_hazard 0.25` on by default; I_var as the
  default ranking; the end-slope rule not applied to C.
- `simulate.py`: the collapse diagnostic, stored in `samples.npz`; a check that prints the smallest
  distance per state at the end of a run.
- `validate.py` or a new script: the Lambda-UFED validation of step 6, once the Lambda-UFED driver is
  available.

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
