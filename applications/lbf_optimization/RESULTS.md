# How the default protocol was chosen

The default protocol of `simulate.py`, `C(exp 2.5) D(8,2)[0.4] E(2,1)[0.5] Q=E`, comes from a pilot on
phenol and a search on three more solutes. This file records what the pilot settled, the
procedure, and the results, for anyone who wants to repeat it for other solutes, force fields or families.
All runs used OpenFF Sage 2.2.1 with AM1-BCC charges, TIP3P, a 3 nm box, PME with a 1.0 nm cutoff switched
from 0.9 nm, hydrogen mass repartitioning to 3 amu, 4 fs steps, 298.15 K and 1 bar, 21 states and 2 ns per state: the
defaults of `simulate.py` (the cutoff and switching distance are fixed in `lbf.build_system`).

## Summary

| solute | ΔG (kT) | ΔG (kcal/mol) | I_var (kT²): paper | default | own best | default over own best |
|---|---|---|---|---|---|---|
| phenol | −10.64 | −6.30 | 408.3 | 211.8 | 203.3 | 4.2% |
| methane | +4.10 | +2.43 | 101.3 | 51.4 | 47.6 | 8.0% |
| anisole | −4.26 | −2.52 | 408.7 | 197.2 | 190.5 | 3.5% |
| paracetamol | −26.74 | −15.84 | 680.9 | 368.0 | 342.0 | 7.6% |

ΔG is the hydration free energy. I_var is the cost of uniform sampling in λ: the simulation time needed for
a given error scales with it. All values are reweighted from each solute's reference run (results 3 and 5
below; for phenol its three runs pooled), with 512 quadrature nodes split at the window starts. "Own best"
is the best protocol found for that solute alone under the same hazard limit of 0.15. The default costs
about half as much as the protocol of Correa et al. (2022) and is within 8% of each own best
(`compare_solutes.py`). Its charging hazard is 0.07, half the 0.13 of the protocol that charges with the
decapping and is otherwise the same, `C(exp 2.5) D(8,2)[0.4] E(3,1)[0.4] Q=E`, which costs up to 2% less
(result 6).

## What the pilot settled

**Basis.** Four slice energies, C (capped Lennard-Jones), D (Lennard-Jones decapping), Q (Coulomb
decapping) and E (capped Coulomb), recorded for every sample. The family is bare Coulomb, h_Q = h_E, with
the charges following the decapping. The Coulomb cap stays in the basis only so that capped protocols can
be scored as references: once the charges follow the decapping, it buys nothing (same length either way).

**Constraints, imposed a priori.**

- Decapping onset power k = m0 + 1 ≥ 8 at the window start, so that var(du/dλ) stays finite there.
- Charging hazard max h_E/h_C ≤ 0.15 wherever h_D < 10⁻³. A water hydrogen has no Lennard-Jones sphere,
  so while the core is capped only the capped O–O repulsion holds it off a solute acceptor. With bare
  Coulomb, a protocol that charges too early crashes or holds a water fused onto the acceptor (O–Ow ≈ 1 Å).
  Reweighting sees this only once the pooled samples contain a fused configuration, which is rare, so the
  limit is imposed rather than searched, and checked on configurations (results, steps 2 and 4).

**Switching families.** C from a family with free end slopes, the exponential
h_C(λ) = (e^(bλ) − 1)/(e^b − 1). D and E as windowed polynomial steps S_{m0,m1}[a], with the E window at
or after the D window. The rule that dh/dλ vanish at both ends is dropped for C: it excluded every
switching function that makes sd(du/dλ) flat, and the pilot reached the exponential only by a detour
through the thermodynamic-length coordinate. For phenol, S_{2,1} → exponential with b = 2.5 takes I_var
from 240 to 204.

**The cost measure.** I_var = ∫ var(du/dλ) dλ, the cost of uniform sampling in λ, as in λ-dynamics
with a flat target. L = ∫ sd(du/dλ) dλ and I_var/L² ≥ 1 are reported; a family is good
when I_var/L² is near 1 without a change of coordinate (1.29 for the polynomial steps, 1.08 to 1.16 for
the exponential C).

**State placement for replica exchange.** Equal increments of thermodynamic length, with the replica flow
(feedback method of Katzgraber, Trebst, Huse and Troyer) as the check; the two agreed for phenol.

**Tools, not steps.** The λ(c) reparametrization (`reparametrize.py`) and the feedback optimization
(`flow.py`) helped the pilot find the family and check the placement; with those known, the procedure is a
direct search in λ.

## The procedure

1. **Build and verify the system.** Check that the four-term system at full coupling reproduces the plain
   force field to numerical precision. One energy evaluation.
2. **Reference run.** One replica-exchange run of a seed protocol that satisfies the constraints, the
   default or a solute's earlier optimum, on 21 states uniform in λ. Check the collapse diagnostic before
   using the samples.
3. **Search by reweighting.** Score the family over b ∈ [1, 4]; D with m0 from 7 to 11 and m1 from 1 to 3;
   window starts a ∈ {0.3, 0.4, 0.5}, one search per a; E with m0, m1 from 1 to 3 and a_E = a, and for
   a = 0.3 and 0.4 also a_E up to a + 0.2 in steps of 0.05 (the `search.py` defaults go to a + 0.1).
   Keep protocols with Kish ESS ≥ 100 at every quadrature node and hazard ≤ 0.15, and rank by I_var.
4. **Second round.** Resample the best protocol at equal thermodynamic length (`search.py --next`), pool
   with the reference run and rescore. If the ranking holds and nothing is flagged, the protocol is
   chosen. If I_var/L² exceeds about 1.15, the family is missing a shape and should be extended.
5. **Several solutes.** Repeat steps 2 and 3 per solute and compare the searches (`compare_solutes.py`).

## Results

Steps 1 to 4 were done for phenol, then step 5 for methane, anisole and paracetamol.

1. **System.** At full coupling the four-term system matches the plain force field to 5e-9 relative in
   energy and 2e-10 kJ/mol/nm in forces (CUDA, double precision).
2. **Collapse diagnostic.** A single short contact is normal through the capped core, so the diagnostic
   counts the fraction of frames within 0.2 nm and flags it above 5% where the charges are on and
   h_D < 10⁻³. Single-state runs (`--no_exchange`) of
   family members at the λ where h_D reaches 10⁻⁶ and at the peak of h_E/h_C showed the mechanism. With
   bare Coulomb, h_E = 0.02 at h_D = 0 crashed phenol within 40 ps; with capped Coulomb, the same protocol
   held a water fused onto the hydroxyl. Hazards of 0.25 and 0.35 were stable for 2 ns. At 0.5 one state
   crashed after 900 ps and another had a water within 0.2 nm in 7% of frames; at 0.7 and 0.9 the water
   fused within 100 ps and the runs crashed. These runs measure how fast a collapse happens, not whether
   the fused state is favorable.
3. **Reference run and search for phenol.** The pilot's search predicted
   `C(exp 2.5) D(7,3)[0.4] E(2,1)[0.4] Q=E` (hazard 0.18; the limit was then 0.25, see 4). Its run
   ranked it first again, with I_var 202.4 against 202.2 predicted; a = 0.3 gave 200.6 and a = 0.5 gave 208.
   Its equal-length round confirmed it (pooled I_var 202.3, I_var/L² 1.08, nothing flagged). Neighbor
   acceptance was 0.62 to 0.67, against 0.49 to 0.84 for uniform states, and round trips in 2000
   iterations were 141, against 128 for uniform states and 99 for the pilot's polynomial-step protocol.
4. **The hazard limit.** A third phenol run, with `C(exp 2.5) D(8,2)[0.4] E(3,1)[0.4] Q=E`, added a fused
   configuration to the pool, sampled at λ = 0.17 where water passes through the uncharged capped core
   freely. Pooled, protocols above hazard 0.15 mostly lost their effective sample size at some λ (22 of 76
   kept it, against 50 of 51 below). Among them was the phenol protocol of step 3: at λ = 0.48 its fused
   water has a reduced energy of −3.5 kT, against +5.8 and +21 kT at its sampled states 0.45 and 0.5.
   Replica exchange never visits it, but a continuous walk in λ can. The limit was lowered to 0.15, which
   rules that protocol out.
5. **Other solutes.** Reference runs of the phenol protocol for methane (nonpolar), anisole (an acceptor
   but no donor) and paracetamol (larger, with donors and acceptors), on uniform states, and a search for
   each. The seed was the phenol protocol of 3, run before the limit was lowered; a search does not depend
   on which protocol sampled, only on how well the samples overlap the protocols it scores. No run flagged
   a collapse. The optima agree on b = 2.5 to 3 and differ in D and E orders that
   barely matter, which is why one default serves all four (summary table).

   A paracetamol run of `C(exp 2.5) D(8,2)[0.4] E(3,1)[0.4] Q=E` on equal-length states checked the
   placement: nothing was flagged, the weakest neighbor acceptance was 0.52 against 0.36 for the uniform
   run of the phenol protocol, and there were 76 round trips in 2000 iterations against 60.
6. **Charging after the decapping starts.** The searches of 3 and 5 kept a_E = a. Searched again with a_E
   from a to a + 0.2, every solute's own best charges later than it decaps, and no protocol is
   meaningfully cheaper for all four than the protocol of 5, but late charging lowers the hazard at little
   cost. Starting E(2,1) at 0.5 instead of E(3,1) at 0.4 keeps the largest ratio to the own bests (1.08,
   methane) and halves the hazard, from 0.13 to 0.07; that protocol became the default.

   What the halving buys shows in the phenol pool of 4. The weight its fused configuration (O–Ow 0.099 nm)
   gets at the worst charged state is e⁻¹¹ under the earlier protocol, where the charges lower it by 13 kT
   at h_E = 0.003, and e⁻⁶² under the default; it is e⁻¹² at hazard 0.145 and 0.5 to 0.8 at 0.16 to 0.18.
   A fused state is thus either negligible or dominant in a narrow band of λ, and the default is 60 kT
   from that edge where the earlier protocol was 11 kT. These weights rest on one sampled configuration.

   The default's paracetamol run, on equal-length states, is the direct check. Nothing was flagged, the
   weakest neighbor acceptance was 0.52 and there were 75 round trips, as for the earlier protocol.
   Pooled with the uniform run, it scores I_var 368.6 against 368.0 predicted (the earlier protocol 361.6
   in the same pool) and I_var/L² 1.17.
7. **Growing the core from λ = 0.** h_D = λ¹² (`D(11,0)[0]`) grows the decapped core's radius, σh_D^(1/12),
   linearly in λ, and charges safely from 0.5 (hazard 0.03), but costs 13 to 34% more than the default: it
   decaps where the capped core is transparent to solvent. Windowed, ((λ − a)/(1 − a))¹² is cheap for
   methane but reaches h_D = 10⁻³ only 56% into its window, so the hazard is 0.4 to 0.6.

## Open questions

- **ΔG of paracetamol.** Its three runs give −26.72 ± 0.10, −26.99 ± 0.09 and −26.81 ± 0.09 kT (mean
  −26.84, standard deviation 0.14), more spread than their standard errors suggest; paracetamol may have
  slow degrees of freedom that 2 ns per state does not average. Phenol's three runs agree (−10.68, −10.56,
  −10.65).
- **The hazard for other solutes.** No paracetamol run sampled a fused configuration deep enough to
  probe the hazard as phenol's did, so for paracetamol the limit rests on the phenol evidence.
- **Validation in λ-dynamics.** Run λ-dynamics with the default and with the paper protocol, several
  independent replicas each, same wall time, and compare the error of ΔG against simulation time
  and the round trips in λ. This tests what reweighting cannot see: the correlation time along λ, whether
  the finite λ-force of the exponential at λ = 0 and 1 causes trouble, and whether the walk finds fused
  states between the hazard limit and the replica-exchange states.
