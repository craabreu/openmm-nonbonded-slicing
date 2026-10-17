# Inner force group for multiple-time-step integration

Design spec, 2026-10-17. Status: draft for review.

## 1. Goal

Let a `SlicedNonbondedForce` evaluate a short-range, force-switched copy of its direct-space pair
interactions in a separate force group, so that multiple-time-step (MTS) integrators can advance
the short-range part with a small step and the remainder with a large one. The split is the
force-directed RESPA2 scheme used in Abreu & Tuckerman, JCTC 2020 (doi 10.1021/acs.jctc.0c00698,
Secs. 3.1 and SI.2), implemented in the plugin's kernels instead of through `CustomNonbondedForce`.

Why inside the plugin: slice scaling parameters (λ) must multiply the inner force exactly as they
multiply the full force, otherwise a slice at small λ receives a full-strength inner force at the
fast level that is cancelled only at the slow level. A Python construction on top of a
`SlicedNonbondedForce` cannot do this. Parameter offsets and 1-4 exceptions come along for free.

## 2. Scope

In scope:

- One additional force group ("inner") per `SlicedNonbondedForce`, holding the switched
  short-range part of all direct-space pair interactions, including the 1-4 exceptions.
- The direct-space group ("outer") holds everything it holds today minus the inner part.
  Reciprocal space, exclusion corrections, Ewald self/background energies, and the dispersion
  correction do not move.
- Every group reports an energy whose gradient is its force. The inner energy is the exact
  modified potential V̂ (SI.2), shifted to vanish at the inner cutoff.
- All nonbonded methods, parameter offsets, slice scaling parameters, and energy parameter
  derivatives.
- Reference, CUDA, and OpenCL platforms (common kernels), serialization, Python wrapper, docs.

Out of scope (decided, not to be revisited in this spec):

- Morrone's RESPA1 (screened near force) and Leimkuhler's reciprocal-space (k-split) RESPA1.
- More than two direct-space levels.
- An "energy ≡ 0" inner group.
- Changing ufedmm. (`ufedmm.add_inner_nonbonded_force` can later delegate to this feature; that is
  a separate change.)

## 3. Definitions

Notation: r is the pair distance, r_s the inner switching distance, r_c the inner cutoff,
z = (r − r_s)/(r_c − r_s). k_e = ONE_4PI_EPS0. Each pair (i, j) belongs to slice IJ with scaling
factors λ_IJ^vdW and λ_IJ^elec (both 1 unless a scaling parameter is defined).

Bare pair potential (no screening, no LJ switch, no LJPME terms, no reaction field):

    V_ij(r) = λ^vdW · 4ε (σ¹²/r¹² − σ⁶/r⁶) + λ^elec · k_e q_i q_j / r

with σ = (σ_i + σ_j)/2 and ε = √(ε_i ε_j) for particle pairs, or the exception's own σ, ε, and
q_i q_j for 1-4 pairs. In the series form V(r) = Σ_n c_n r^{−n}, n ∈ {12, 6, 1}:
c_12 = λ^vdW·4εσ¹², c_6 = −λ^vdW·4εσ⁶, c_1 = λ^elec·k_e q_i q_j.

Switching function (quintic, applied to the force):

    S(z) = 1                      z ≤ 0
    S(z) = 1 − 10z³ + 15z⁴ − 6z⁵  0 < z < 1
    S(z) = 0                      z ≥ 1

Its derivative with respect to r is S'(r) = −(30/(r_c − r_s)⁵) P(r) on r_s < r < r_c, where
P(r) = (r − r_s)²(r_c − r)² = r⁴ − 2a r³ + (a² + 2b) r² − 2ab r + b², a = r_s + r_c, b = r_s r_c.

Inner force on particle i from pair (i, j):

    F_ij^inner(r) = −S(r) V_ij'(r) r̂_ij

Inner energy (the potential whose gradient is the inner force, SI.2 Eq. SI.1, shifted):

    V̂_ij(r) = V_ij(r) + V_pert(r) − V̂_ij(r_c)     r < r_c
    V̂_ij(r) = 0                                   r ≥ r_c

    V_pert(r) = 0                                                  r ≤ r_s
    V_pert(r) = (30/(r_c − r_s)⁵) Σ_n c_n [ G_n(r) − G_n(r_s) − r^{−n} (G_0(r) − G_0(r_s)) ]   r_s < r < r_c

    G_n(r) = ∫ P(s) s^{−n} ds
           = φ_{n−5}(r) − 2a φ_{n−4}(r) + (a² + 2b) φ_{n−3}(r) − 2ab φ_{n−2}(r) + b² φ_{n−1}(r)
    φ_m(r) = −r^{−m}/m  (m ≠ 0),   φ_0(r) = ln r

(This is SI.3–SI.4 with r_a = a/2 and r_g² = b.) The shift constant is linear in the c_n:

    V̂_ij(r_c) = Σ_n c_n K_n,
    K_n = r_c^{−n} + (30/(r_c − r_s)⁵) [ G_n(r_c) − G_n(r_s) − r_c^{−n} (G_0(r_c) − G_0(r_s)) ]

so K_12, K_6, K_1 are three host-side doubles computed once from (r_s, r_c), and the kernel
evaluates V̂ as V(r) + V_pert(r) − (c_12 K_12 + c_6 K_6 + c_1 K_1). Because every term is
linear in λ^vdW or λ^elec, the energy parameter derivatives of the inner group are the unscaled
vdW and Coulomb parts of V̂, which is exactly what the existing `COMPUTE_DERIVATIVES` machinery
accumulates from `ljEnergy` and `clEnergy`.

Outer (direct-space) group: force = (today's direct-space force) − F^inner; energy = (today's
direct-space energy) − V̂. Reciprocal group: unchanged. Hence for every pair and every group,
force = −∇(energy), and the sum over groups equals today's total.

Since F^inner is a central pair force, it is conservative with or without V̂; V̂ only makes the
reported per-group energy consistent with it. That consistency is required so that per-group
finite-difference tests and energy-based users (minimizers on subsets, derivative reporting) work.

## 4. Public API

C++ (`SlicedNonbondedForce.h`), mirroring `setReciprocalSpaceForceGroup`:

```cpp
int  getInnerForceGroup() const;            // default -1: feature disabled
void setInnerForceGroup(int group);         // -1 or 0..31
double getInnerCutoffDistance() const;      // default 0
void   setInnerCutoffDistance(double rc);
double getInnerSwitchingDistance() const;   // default 0
void   setInnerSwitchingDistance(double rs);
```

Validation (in `SlicedNonbondedForceImpl::initialize`, so errors surface at Context creation):

- `innerForceGroup` is −1 (disabled; the other two settings are ignored) or in [0, 31].
- When enabled: 0 < r_s < r_c; r_c ≤ `getCutoffDistance()` when the method uses a cutoff. (r_s must
  be strictly positive: with r_s = 0 the Lennard-Jones part of V_pert, ∫₀ P(s) s⁻¹² ds, diverges, so
  V̂ does not exist.)
- When enabled: `innerForceGroup != getForceGroup()` and `innerForceGroup != resolved reciprocal
  group` (both would put the fast part together with slow terms, which is never intended).
- When enabled: `getIncludeDirectSpace()` must be true (if direct space is supplied by another
  force, an inner copy of it here makes no sense).

The `SlicedNonbondedForce(const NonbondedForce&, int)` constructor leaves the feature disabled.
`updateParametersInContext` does not propagate changes to the three settings; like the cutoff and
force groups, they take effect on a new Context (or `Context.reinitialize`). Document this.

Python: the SWIG interface includes the header, so the methods appear automatically; add docstring
blocks in `python/nonbondedslicing.i` next to the reciprocal-space-group documentation.

Serialization (`SlicedNonbondedForceProxy.cpp`): bump `version` to 2; write `innerForceGroup`,
`innerCutoff`, `innerSwitchingDistance`; when reading version 1, use the defaults. Add the three
settings to the serialization test.

## 5. Implementation

### 5.1 API layer

- `SlicedNonbondedForce`: three private fields with the defaults above; getters/setters as in
  OpenMM style (`ASSERT_VALID` for the group range).
- `SlicedNonbondedForceImpl::calcForcesAndEnergy`: compute
  `includeInner = inner >= 0 && (groups & (1 << inner)) != 0` next to `includeDirect` and
  `includeReciprocal`, and pass it on.
- `CalcSlicedNonbondedForceKernel::execute` (`NonbondedSlicingKernels.h`): add
  `bool includeInner` after `includeDirect`. Thread the argument through
  `CommonCalcSlicedNonbondedForceKernel`, `ReferenceCalcSlicedNonbondedForceKernel`, and the CUDA
  and OpenCL parallel kernels (pass-through only).

### 5.2 Common platform (CUDA and OpenCL)

Mechanism: OpenMM's `NonbondedUtilities` compiles one kernel per requested force-group set from the
sources of the interactions registered in those groups (`createKernelsForGroups`), over a single
neighbor list built with the largest registered cutoff. Registering the pair kernel twice, in two
groups, therefore costs no second neighbor list and no second parameter array.

In `commonInitialize`, when the feature is enabled:

1. Compute host constants: `INNER_CUTOFF`, `INNER_CUTOFF_SQUARED`, `INNER_SWITCH`,
   `INNER_SWITCH_SCALE = 1/(r_c − r_s)`, the quintic coefficients, the five polynomial
   coefficients of P scaled by 30/(r_c − r_s)⁵, G_n(r_s) and G_0(r_s) for n ∈ {12, 6, 1}, and
   K_12, K_6, K_1.
2. Register the direct-space interaction (`coulombLennardJones.cc`) in `getForceGroup()` with
   `defines["SUBTRACT_INNER"] = "1"` plus the constants, and a second copy in the inner group with
   `defines["INNER_GROUP"] = "1"` plus the constants, `addInteraction(..., cutoffDistance = r_c,
   ..., forceGroup = inner, ...)`. Both copies use the same parameter arrays (`charges`,
   `sigmaEpsilon`, `subsets`, `sliceLambdas`), so parameter offsets apply to both automatically.
   The inner copy must test `r2 < INNER_CUTOFF_SQUARED` in its own source and not rely on the
   kernel-set cutoff define.
3. Do the same for the exceptions kernel (`nonbondedExceptions.cc`) through `BondedUtilities`:
   one registration per group, with the same defines.

Kernel source changes (`coulombLennardJones.cc`): the existing code computes, per pair, the force
factor `tempForce` (later multiplied by `invR*invR` into `dEdR`), `clEnergy`, and `ljEnergy`, then
applies `ljLambda`/`clLambda`. Add one block, compiled under `INNER_GROUP || SUBTRACT_INNER`,
that computes for `r2 < INNER_CUTOFF_SQUARED` (name the switch value `innerSwitch`; `switchValue`
is already taken by the LJ switching function):

- `innerForceLJ = innerSwitch * epssig6*(12 sig6 − 6)`, `innerForceCl = innerSwitch * prefactor`
  (the bare forms already present, before any erfc/RF/LJ-switch/LJPME terms);
- `innerEnergyLJ`, `innerEnergyCl` = the LJ and Coulomb parts of V̂ (computed only under
  `INCLUDE_ENERGY`): for `r <= INNER_SWITCH`, bare energy minus the K shift; otherwise bare energy
  plus V_pert minus the K shift, using Horner forms of the φ_m terms.

Under `INNER_GROUP`, these replace the force and energy outputs (`tempForce = ljLambda*innerForceLJ
+ clLambda*innerForceCl`, `ljEnergy = innerEnergyLJ`, `clEnergy = innerEnergyCl`), and the
`includeInteraction` mask is `!isExcluded && r2 < INNER_CUTOFF_SQUARED`. Under `SUBTRACT_INNER`,
they are subtracted from the usual outputs when `r2 < INNER_CUTOFF_SQUARED`. The LJ-switch,
LJPME, reaction-field, and Ewald branches are untouched, so the outer kernel remains "today's
kernel minus the inner part". `COMPUTE_DERIVATIVES` needs no change: it reads `clEnergy` and
`ljEnergy`.

`nonbondedExceptions.cc`: same two blocks, written against `exceptionParams`.

`execute()`: wherever `includeDirect` gates host-side preparation that the inner kernels also
need (the parameter kernel that applies offsets to `charges`/`sigmaEpsilon`/`exceptionParams`,
and any "force was requested" bookkeeping), extend the condition with `includeInner`. The
interaction kernels themselves are selected by group mask inside `NonbondedUtilities` and
`BondedUtilities`, so no explicit dispatch is needed. Energy accumulation into the context's
energy buffer and the NaN marking of derivatives follow the existing pattern (`includeDirect ||
includeInner || includeReciprocal`).

### 5.3 Reference platform

- `ReferenceSlicedLJCoulombIxn`: add `setInnerSplit(rs, rc)` and a private
  `calculateInnerPair(ii, jj, ..., double& innerFactor, double innerEnergy[2])` helper that
  returns the inner force factor (to multiply `deltaR`) and the vdW/Coulomb parts of V̂ for a
  pair. In the direct loop (`calculateEwaldIxn` and `calculatePairIxn`), when the split is
  enabled, subtract the helper's outputs. Add `calculateInnerIxn(...)` that loops over the
  neighbor list (or all pairs for `NoCutoff`) and adds them; it is called from the kernel when
  `includeInner` is true. Pair geometry (periodic delta, switching, slice index) is shared.
- `ReferenceSlicedLJCoulomb14`: add a mode argument (full / inner / remainder) and apply the same
  split to 1-4 pairs. The kernel calls it with `inner` when `includeInner`, with `remainder`
  when `includeDirect` and the split is enabled, and with `full` otherwise.
- `ReferenceCalcSlicedNonbondedForceKernel::execute`: build the neighbor list when
  `includeDirect || includeInner`; everything else unchanged.

### 5.4 Things that do not move

Dispersion correction (direct group), Ewald self energy and neutralizing-background terms
(reciprocal group, as fixed in PR #46), exclusion corrections (reciprocal group), reciprocal sums.
`setIncludeDirectSpace(false)` plus an enabled inner group is rejected at initialization.

## 6. Validation

All in `tests/TestSlicedNonbondedForce.h` (shared by the three platform test executables), plus
Python tests where noted. "Reference force" below means a system built from OpenMM forces that
reproduces one group exactly.

1. **Inner group vs. exact reference.** A `CustomNonbondedForce` with the ufedmm expression
   (`ufedmm.integrators.add_inner_nonbonded_force` builds V̂ for the same quintic switch, except
   for the shift) plus a `CustomBondForce` for the 1-4 pairs, with per-particle subset parameters
   and global λ's reproducing the slices. Compare energy, forces, and energy parameter derivatives
   of the inner group, for PME and CutoffPeriodic, with and without exceptions, offsets, and
   non-unit λ. Account for the per-pair shift in the reference by adding the constant term.
2. **Per-group consistency.** For inner and outer groups separately, compare forces to central
   finite differences of the group's energy along random displacements, in double precision, for
   PME, LJPME, Ewald, CutoffPeriodic (with reaction field), CutoffNonPeriodic, NoCutoff; with LJ
   switching on the outer force; with exceptions and offsets. This is the test that the exact
   V̂ exists for.
3. **Sum rule.** Groups {inner, outer, reciprocal} together equal the unrestricted state, which
   equals a plain `NonbondedForce` with the same parameters (λ = 1), for every method.
4. **Continuity at the inner cutoff.** For a two-particle system, evaluate the inner and outer
   groups at r = r_c − δ, r_c, r_c + δ: the inner force and energy go to zero continuously, the
   outer force and energy are continuous, and the sums equal the unsplit values.
5. **MTS smoke test.** Flexible-water box (or the plugin's existing periodic test system) with
   `MTSIntegrator` using groups (bonded, inner, outer+reciprocal); run a few hundred steps and
   check that the energy drift is bounded and that `getState(groups=...)` partial energies sum
   to the total. Keep it short; it is a wiring test, not a physics study.
6. **Validation errors.** Each rule in Sec. 4 raises `OpenMMException` with a clear message.
7. **Serialization.** Round-trip the three settings; a version-1 XML loads with the feature
   disabled.
8. **Python.** Mirror test 1 and test 3 in `python/tests/TestSlicedNonbondedForce.py`, and add a
   doctest-style usage example to the docstring.

Run on Reference (all precisions), CUDA mixed/double, OpenCL mixed/double before the PR; CI covers
Reference and OpenCL.

## 7. Documentation

- `docs/overview.rst`: new section "Force splitting for multiple time-step integration" with the
  definitions of Sec. 3 (switch, inner force, V̂, group contents), the MTS usage example, and the
  statement that all slice scaling applies to the inner group too.
- `python/nonbondedslicing.i`: docstrings for the six methods, including the "new Context
  required" note and the cutoff/group validation rules.
- `docs/changelog.rst`: entry under 0.5.0.

Usage example (illustrative):

```python
force = nbs.SlicedNonbondedForce(nonbonded, 2)
force.setForceGroup(2)               # outer: direct remainder + reciprocal + exclusions
force.setInnerForceGroup(1)
force.setInnerSwitchingDistance(0.5*unit.nanometer)
force.setInnerCutoffDistance(0.8*unit.nanometer)
# bonded forces in group 0
integrator = mm.MTSIntegrator(90*unit.femtoseconds, [(2, 1), (1, 30), (0, 6)])  # 90 / 3 / 0.5 fs
```

(Verify the tuple order and substep semantics against the OpenMM `MTSIntegrator` documentation
when writing the docs; the example is illustrative.)

## 8. Performance

Expected: the inner kernel at the intermediate level is a hand-written pair kernel with the switch
on the force only (no energy series during dynamics, since `NonbondedUtilities` compiles separate
force-only and force+energy variants), over the shared neighbor list. The outer step costs one
kernel launch for the remainder instead of a full `NonbondedForce` plus a `CustomNonbondedForce`
evaluation. Both are improvements over the Python construction; neither is quantified yet.

Measure before merging: (a) ufedmm's inner `CustomNonbondedForce` group alone, (b) the plugin's
inner group alone, (c) the plugin's direct group alone, on a 512–4000 SPC-Fw water box on the GPU,
mixed precision. Report ns/day-equivalent per level in the PR.

Neighbor list: the inner kernel iterates tiles built for the full cutoff and rejects pairs beyond
r_c. A dedicated short-cutoff list could be added later if (b) shows the tile overhead dominates;
it is out of scope here.

## 9. Risks and open points

- **Single precision of V̂.** The V_pert polynomial in r and 1/r on [r_s, r_c] has no large
  cancellations for the distances of interest (0.3–1.0 nm), but test 2 should be run in single
  precision too and the tolerance recorded. Forces do not depend on V̂.
- **NoCutoff with the inner split.** Allowed and tested; on Reference it loops over all pairs.
- **Group changes at runtime.** As for other force groups, they require Context reinitialization.
  Documented, not worked around.
- **Interaction with the LJ switching function of the outer force.** Independent: the inner
  part is bare LJ; the outer remainder includes the LJ switch minus the bare inner part. Test 2
  covers it.

## 10. Rollout

Branch `feat/inner-force-group` (worktree). Lands after PR #46. Implementation plan to follow
this spec, split into: API + serialization; Reference implementation + tests 1–3, 6, 7; common
kernels + GPU runs; docs + Python tests + MTS smoke test + benchmark.
