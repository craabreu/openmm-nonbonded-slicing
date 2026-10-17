# Inner Force Group Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `SlicedNonbondedForce` evaluate a force-switched, short-range copy of its direct-space pair interactions (particle pairs and 1-4 exceptions) in a separate "inner" force group, with the direct-space group holding the remainder, so MTS integrators can step the inner part with a small time step.

**Architecture:** Three new settings on `SlicedNonbondedForce` (inner group, inner cutoff r_c, inner switching distance r_s) flow through the Impl as a fourth `includeInner` flag into the kernels. Host-side constants for the switch and for the exact inner potential V̂ live in one shared header used by the Reference platform, by the GPU kernel defines, and by nothing else. On the GPU platforms the existing pair and exception kernel sources are registered twice through `NonbondedUtilities`/`BondedUtilities`, once per group, with `INNER_GROUP`/`SUBTRACT_INNER` defines; the Reference platform gets one pair helper used by both loops.

**Tech Stack:** C++17 OpenMM plugin (OpenMM ≥ 8.4), CUDA/OpenCL kernels via OpenMM's common platform layer, SWIG Python wrapper, CMake, pytest, Sphinx.

**Spec:** `docs/superpowers/specs/2026-10-17-inner-force-group-design.md` (same branch). Read it first; Sec. 3 defines every formula used below.

## Global Constraints

- Branch: `feat/inner-force-group` in worktree `/tmp/claude-1020/-data3-craabreu-github-repos/fd8ccff1-699a-4ca8-9c40-7736f4e90790/scratchpad/nbs-inner-group-spec`. Never touch `/data3/craabreu/github_repos/openmm-nonbonded-slicing`.
- Build: out-of-source in `build/` inside the worktree. The conda env `/data3/craabreu/envs/openxps-test` has OpenMM 8.6.1 + CUDA 13.2 but no cmake/compiler/swig; `/data3/craabreu/envs/openmm-colvars` has cmake 4.4, gcc 15.3, swig 4.5.1. A working CMake configuration for exactly this setup (including CUDA and the scratch `nvrtc.h`) is in `/tmp/claude-1020/-data3-craabreu-github-repos/fd8ccff1-699a-4ca8-9c40-7736f4e90790/scratchpad/nbs-exclusions-fix/build/CMakeCache.txt`; copy its `OPENMM_DIR`, compiler, CUDA, SWIG and Python entries. Do not install anything into shared envs. `build/` is git-ignored.
- Tests: C++ tests live in `tests/TestSlicedNonbondedForce.h` (shared by Reference/CUDA/OpenCL drivers) and are registered by calling them from `main` in that file. Run `./platforms/reference/tests/TestReferenceSlicedNonbondedForce` (no argument) and `./platforms/cuda/tests/TestCudaSlicedNonbondedForce mixed` (also `double`); `platform` is a global in the drivers, precision via `platform.getPropertyDefaultValue("Precision")`. Python tests: `pytest python/tests` with `OPENMM_PLUGIN_DIR` pointing at the built plugin directory, as set up in the exclusions worktree.
- Style: match surrounding code exactly (4-space indent, OpenMM brace style, `/*---` comment banners in Reference files, sparse comments). No unrelated refactoring.
- Every commit message ends with exactly:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012Px2sYfcPG4WRLy6RJd3HP
  ```
- Validation rules (spec Sec. 4, exact): inner group ∈ {−1} ∪ [0, 31]; when enabled, 0 < r_s < r_c, r_c ≤ cutoff when the method uses a cutoff, inner group ≠ direct group, inner group ≠ resolved reciprocal group, `getIncludeDirectSpace()` must be true.
- Every force group's energy must be the potential of its force (spec Sec. 1). The inner energy is V̂ shifted to zero at r_c; never an "energy ≡ 0" shortcut.
- Nothing moves between the direct and reciprocal groups: reciprocal sums, exclusion corrections, Ewald self/background energies and the dispersion correction stay where they are today.

## Review Focus

1. **r_s = 0** — the user sets a switching distance of zero expecting "switch over the whole range"; V̂'s LJ part diverges, so Context creation must raise a clear `OpenMMException`, not produce NaN. Test: Task 1 (`testInnerGroupErrors`).
2. **Pure exclusions (1-2/1-3 exceptions with chargeProd = 0, epsilon = 0)** — must contribute nothing to the inner group and must not break the −erf cancellation in the outer group. Test: Task 3 (`testInnerGroupForces` with water-like molecules).
3. **LJ-only or Coulomb-only systems** (`hasLJ` or `hasCoulomb` false in the kernels) — the inner kernel must compile and give the switched LJ or Coulomb force alone. Test: Task 3 (`testInnerGroupForces` with `lj=false` / all charges zero).
4. **Parameter offsets changed through `setParameter` and `updateParametersInContext`** — the inner group must follow the updated charges/sigma/epsilon, since it reads the same arrays. Test: Task 3 (`testInnerGroupOffsets`).
5. **r_c equal to the outer cutoff** — allowed; the outer direct force then vanishes beyond r_c and the sum rule must still hold. Test: Task 3 (`testInnerGroupForces` parametrized with `rc == cutoff`).

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `openmmapi/include/SlicedNonbondedForce.h` | modify | 3 settings, 6 accessors |
| `openmmapi/src/SlicedNonbondedForce.cpp` | modify | defaults, setter range check |
| `openmmapi/src/SlicedNonbondedForceImpl.cpp` | modify | validation at init; `includeInner` flag |
| `openmmapi/include/NonbondedSlicingKernels.h` | modify | kernel `execute` gets `includeInner` |
| `openmmapi/include/internal/InnerSplit.h` | create | host-side math: switch, V̂ constants, V_pert (double precision, header-only) |
| `serialization/src/SlicedNonbondedForceProxy.cpp`, `serialization/tests/TestSerializeSlicedNonbondedForce.cpp` | modify | version 2 with the 3 settings |
| `platforms/reference/include/internal/ReferenceSlicedLJCoulombIxn.h`, `platforms/reference/src/ReferenceSlicedLJCoulombIxn.cpp` | modify | inner pair helper; subtract in direct loops; `calculateInnerIxn` |
| `platforms/reference/include/internal/ReferenceSlicedLJCoulomb14.h`, `platforms/reference/src/ReferenceSlicedLJCoulomb14.cpp` | modify | mode argument: full / inner / remainder |
| `platforms/reference/include/ReferenceNonbondedSlicingKernels.h`, `platforms/reference/src/ReferenceNonbondedSlicingKernels.cpp` | modify | store settings; dispatch with `includeInner` |
| `platforms/common/include/CommonNonbondedSlicingKernels.h`, `platforms/common/src/CommonNonbondedSlicingKernels.cpp` | modify | register kernels twice; `includeInner` in `execute` |
| `platforms/common/src/kernels/coulombLennardJones.cc`, `platforms/common/src/kernels/nonbondedExceptions.cc` | modify | `INNER_GROUP` / `SUBTRACT_INNER` code paths |
| `platforms/cuda/{include,src}/CudaParallelNonbondedSlicingKernels.*`, `platforms/opencl/{include,src}/OpenCLParallelNonbondedSlicingKernels.*` | modify | pass `includeInner` through |
| `tests/TestSlicedNonbondedForce.h` | modify | new tests (errors, forces, energy quadrature, FD consistency, derivatives, offsets, continuity) |
| `python/nonbondedslicing.i` | modify | declare + document the 6 methods |
| `python/tests/TestSlicedNonbondedForce.py` | modify | inner-group tests + MTS wiring test |
| `docs/overview.rst`, `docs/changelog.rst` | modify | user documentation |
| `devtools/benchmarks/benchmark.py` | modify | `--inner-group` timing |

---

### Task 1: API, validation, kernel-flag plumbing, serialization

**Files:**
- Modify: `openmmapi/include/SlicedNonbondedForce.h:52-68`
- Modify: `openmmapi/src/SlicedNonbondedForce.cpp:31-82`
- Modify: `openmmapi/src/SlicedNonbondedForceImpl.cpp:127-142`
- Modify: `openmmapi/include/NonbondedSlicingKernels.h:49-59`
- Modify: `platforms/common/include/CommonNonbondedSlicingKernels.h:60`, `platforms/common/src/CommonNonbondedSlicingKernels.cpp:842`
- Modify: `platforms/reference/include/ReferenceNonbondedSlicingKernels.h:40`, `platforms/reference/src/ReferenceNonbondedSlicingKernels.cpp:188`
- Modify: `platforms/cuda/include/CudaParallelNonbondedSlicingKernels.h:47`, `platforms/cuda/src/CudaParallelNonbondedSlicingKernels.cpp:20-50`
- Modify: `platforms/opencl/include/OpenCLParallelNonbondedSlicingKernels.h:47`, `platforms/opencl/src/OpenCLParallelNonbondedSlicingKernels.cpp:18-48`
- Modify: `serialization/src/SlicedNonbondedForceProxy.cpp:25,52,96-124`
- Test: `serialization/tests/TestSerializeSlicedNonbondedForce.cpp`, `tests/TestSlicedNonbondedForce.h`

**Interfaces:**
- Produces (used by every later task):
  ```cpp
  int SlicedNonbondedForce::getInnerForceGroup() const;           // -1 = disabled (default)
  void SlicedNonbondedForce::setInnerForceGroup(int group);       // -1 or 0..31, else OpenMMException
  double SlicedNonbondedForce::getInnerCutoffDistance() const;    // default 0.0
  void SlicedNonbondedForce::setInnerCutoffDistance(double distance);
  double SlicedNonbondedForce::getInnerSwitchingDistance() const; // default 0.0
  void SlicedNonbondedForce::setInnerSwitchingDistance(double distance);
  virtual double CalcSlicedNonbondedForceKernel::execute(ContextImpl& context, bool includeForces,
          bool includeEnergy, bool includeDirect, bool includeReciprocal, bool includeInner) = 0;
  ```

- [ ] **Step 1: Write the failing serialization test**

In `serialization/tests/TestSerializeSlicedNonbondedForce.cpp`, after the line `force.setReciprocalSpaceForceGroup(2);` add:

```cpp
    force.setInnerForceGroup(3);
    force.setInnerCutoffDistance(0.8);
    force.setInnerSwitchingDistance(0.5);
```

and after `ASSERT_EQUAL(force.getExceptionsUsePeriodicBoundaryConditions(), force2.getExceptionsUsePeriodicBoundaryConditions());` add:

```cpp
    ASSERT_EQUAL(force.getInnerForceGroup(), force2.getInnerForceGroup());
    ASSERT_EQUAL(force.getInnerCutoffDistance(), force2.getInnerCutoffDistance());
    ASSERT_EQUAL(force.getInnerSwitchingDistance(), force2.getInnerSwitchingDistance());
```

- [ ] **Step 2: Write the failing validation test**

In `tests/TestSlicedNonbondedForce.h`, insert before `void testErrors()`:

```cpp
void testInnerGroupErrors() {
    // Each configure() call should make Context creation fail; the unmodified setup must succeed.

    auto build = [](function<void(SlicedNonbondedForce&)> configure, bool expectFailure) {
        System system;
        system.addParticle(1.0);
        system.addParticle(1.0);
        system.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
        SlicedNonbondedForce* force = new SlicedNonbondedForce(1);
        force->setNonbondedMethod(SlicedNonbondedForce::PME);
        force->setCutoffDistance(1.0);
        force->addParticle(1.0, 0.3, 0.5);
        force->addParticle(-1.0, 0.3, 0.5);
        force->setInnerForceGroup(1);
        force->setInnerSwitchingDistance(0.5);
        force->setInnerCutoffDistance(0.8);
        configure(*force);
        system.addForce(force);
        VerletIntegrator integrator(0.001);
        bool threw = false;
        try {
            Context context(system, integrator, platform);
        }
        catch (const OpenMMException&) {
            threw = true;
        }
        ASSERT_EQUAL(expectFailure, threw);
    };
    build([](SlicedNonbondedForce& f) {}, false);
    build([](SlicedNonbondedForce& f) { f.setInnerSwitchingDistance(0.0); }, true);   // r_s must be > 0
    build([](SlicedNonbondedForce& f) { f.setInnerSwitchingDistance(0.8); }, true);   // r_s must be < r_c
    build([](SlicedNonbondedForce& f) { f.setInnerCutoffDistance(1.2); }, true);      // r_c must be <= cutoff
    build([](SlicedNonbondedForce& f) { f.setInnerForceGroup(0); }, true);            // same as direct group
    build([](SlicedNonbondedForce& f) { f.setReciprocalSpaceForceGroup(1); }, true);  // same as reciprocal group
    build([](SlicedNonbondedForce& f) { f.setIncludeDirectSpace(false); }, true);
    build([](SlicedNonbondedForce& f) { f.setInnerForceGroup(-1); f.setInnerSwitchingDistance(0.0); }, false);  // disabled: settings ignored
    bool threw = false;
    try {
        SlicedNonbondedForce force(1);
        force.setInnerForceGroup(32);
    }
    catch (const OpenMMException&) {
        threw = true;
    }
    ASSERT(threw);
}
```

Register it in `main` right after `testErrors();`: add the line `testInnerGroupErrors();`.

- [ ] **Step 3: Build and run to verify the tests fail to compile**

Run: `cd build && make -j8 2>&1 | tail -5`
Expected: compilation errors mentioning `setInnerForceGroup` (no such member).

- [ ] **Step 4: Add the settings to the header**

In `openmmapi/include/SlicedNonbondedForce.h`, after the `setUseCuFFT` method (line 57) add:

```cpp
    /**
     * Get the force group of the inner part of the direct-space interactions, or -1 if the inner
     * force group is disabled (the default).  When enabled, every direct-space pair interaction
     * (including 1-4 exceptions) is split into a short-range part, switched off smoothly between
     * the inner switching distance and the inner cutoff distance, which is computed in the inner
     * force group, and a remainder, which stays in the force group of this force.
     */
    int getInnerForceGroup() const {
        return innerForceGroup;
    }
    void setInnerForceGroup(int group);
    double getInnerCutoffDistance() const {
        return innerCutoff;
    }
    void setInnerCutoffDistance(double distance);
    double getInnerSwitchingDistance() const {
        return innerSwitchingDistance;
    }
    void setInnerSwitchingDistance(double distance);
```

and in the private section after `bool useCuFFT;` add:

```cpp
    int innerForceGroup;
    double innerCutoff, innerSwitchingDistance;
```

- [ ] **Step 5: Implement defaults and setters**

In `openmmapi/src/SlicedNonbondedForce.cpp`, change the first constructor to

```cpp
SlicedNonbondedForce::SlicedNonbondedForce(int numSubsets) : NonbondedForce(), numSubsets(numSubsets), useCuFFT(true),
        innerForceGroup(-1), innerCutoff(0.0), innerSwitchingDistance(0.0) {
}
```

in the second constructor, after `useCuFFT = true;` add

```cpp
    innerForceGroup = -1;
    innerCutoff = 0.0;
    innerSwitchingDistance = 0.0;
```

and after `getParticleSubset` add

```cpp
void SlicedNonbondedForce::setInnerForceGroup(int group) {
    if (group < -1 || group > 31)
        throw OpenMMException("SlicedNonbondedForce: the inner force group must be -1 (disabled) or between 0 and 31");
    innerForceGroup = group;
}

void SlicedNonbondedForce::setInnerCutoffDistance(double distance) {
    innerCutoff = distance;
}

void SlicedNonbondedForce::setInnerSwitchingDistance(double distance) {
    innerSwitchingDistance = distance;
}
```

- [ ] **Step 6: Validate at initialization and compute the flag**

In `openmmapi/src/SlicedNonbondedForceImpl.cpp`, immediately before `kernel.getAs<CalcSlicedNonbondedForceKernel>().initialize(context.getSystem(), owner);` (line 132) add:

```cpp
    if (owner.getInnerForceGroup() >= 0) {
        double innerSwitch = owner.getInnerSwitchingDistance();
        double innerCutoff = owner.getInnerCutoffDistance();
        if (!(innerSwitch > 0.0 && innerSwitch < innerCutoff))
            throw OpenMMException("SlicedNonbondedForce: the inner switching distance must satisfy 0 < r_switch < r_cutoff");
        if (owner.getNonbondedMethod() != SlicedNonbondedForce::NoCutoff && innerCutoff > owner.getCutoffDistance())
            throw OpenMMException("SlicedNonbondedForce: the inner cutoff distance cannot be greater than the cutoff distance");
        int reciprocalGroup = owner.getReciprocalSpaceForceGroup();
        if (reciprocalGroup < 0)
            reciprocalGroup = owner.getForceGroup();
        if (owner.getInnerForceGroup() == owner.getForceGroup() || owner.getInnerForceGroup() == reciprocalGroup)
            throw OpenMMException("SlicedNonbondedForce: the inner force group must differ from the direct-space and reciprocal-space force groups");
        if (!owner.getIncludeDirectSpace())
            throw OpenMMException("SlicedNonbondedForce: the inner force group requires the direct space to be included");
    }
```

Replace `calcForcesAndEnergy` with:

```cpp
double SlicedNonbondedForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    bool includeDirect = (owner.getIncludeDirectSpace() && (groups&(1<<owner.getForceGroup())) != 0);
    int reciprocalGroup = owner.getReciprocalSpaceForceGroup();
    if (reciprocalGroup < 0)
        reciprocalGroup = owner.getForceGroup();
    bool includeReciprocal = ((groups&(1<<reciprocalGroup)) != 0);
    int innerGroup = owner.getInnerForceGroup();
    bool includeInner = (innerGroup >= 0 && (groups&(1<<innerGroup)) != 0);
    return kernel.getAs<CalcSlicedNonbondedForceKernel>().execute(context, includeForces, includeEnergy, includeDirect, includeReciprocal, includeInner);
}
```

- [ ] **Step 7: Thread `includeInner` through every kernel (ignored for now)**

`openmmapi/include/NonbondedSlicingKernels.h`: add ` * @param includeInner  true if the inner force group should be included` to the doc comment and change the signature to `virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal, bool includeInner) = 0;`.

`platforms/common/include/CommonNonbondedSlicingKernels.h:60` and `platforms/reference/include/ReferenceNonbondedSlicingKernels.h:40`: same signature change (add `, bool includeInner`). Update the matching definitions at `CommonNonbondedSlicingKernels.cpp:842` and `ReferenceNonbondedSlicingKernels.cpp:188`; the bodies do not use the flag yet.

`platforms/cuda/src/CudaParallelNonbondedSlicingKernels.cpp` (and the OpenCL twin with the `OpenCL` prefix): in class `Task`, add `bool includeInner` as the constructor parameter after `includeReciprocal`, initialize it in the initializer list, add it to the member declaration `bool includeForce, includeEnergy, includeDirect, includeReciprocal, includeInner;`, and call `kernel.execute(context, includeForce, includeEnergy, includeDirect, includeReciprocal, includeInner)`. Change the outer `execute` to take `bool includeInner` after `includeReciprocal` (header too) and pass it when constructing each `Task`.

- [ ] **Step 8: Serialize version 2**

In `serialization/src/SlicedNonbondedForceProxy.cpp`: change `node.setIntProperty("version", 1);` to `2`; after the `recipForceGroup` property add

```cpp
    node.setIntProperty("innerForceGroup", force.getInnerForceGroup());
    node.setDoubleProperty("innerCutoff", force.getInnerCutoffDistance());
    node.setDoubleProperty("innerSwitchingDistance", force.getInnerSwitchingDistance());
```

In `deserialize`, change the version check to `if (version < 1 || version > 2)` and after `force->setReciprocalSpaceForceGroup(...)` add

```cpp
        force->setInnerForceGroup(node.getIntProperty("innerForceGroup", -1));
        force->setInnerCutoffDistance(node.getDoubleProperty("innerCutoff", 0.0));
        force->setInnerSwitchingDistance(node.getDoubleProperty("innerSwitchingDistance", 0.0));
```

- [ ] **Step 9: Build and run the tests**

Run: `cd build && make -j8 && ./serialization/tests/TestSerializeSlicedNonbondedForce && ./platforms/reference/tests/TestReferenceSlicedNonbondedForce && ./platforms/cuda/tests/TestCudaSlicedNonbondedForce mixed`
Expected: all three print no failure; `testInnerGroupErrors` passes (the valid configuration constructs a Context, every invalid one throws).

- [ ] **Step 10: Commit**

```bash
git add openmmapi serialization platforms/common/include platforms/common/src/CommonNonbondedSlicingKernels.cpp platforms/reference platforms/cuda platforms/opencl tests
git commit -m "Add inner force group settings, validation and kernel plumbing"
```

---

### Task 2: Shared inner-split math and the Reference platform

**Files:**
- Create: `openmmapi/include/internal/InnerSplit.h`
- Modify: `platforms/reference/include/internal/ReferenceSlicedLJCoulombIxn.h:26-65,182-207`
- Modify: `platforms/reference/src/ReferenceSlicedLJCoulombIxn.cpp:362-460,526-629`
- Modify: `platforms/reference/include/internal/ReferenceSlicedLJCoulomb14.h`, `platforms/reference/src/ReferenceSlicedLJCoulomb14.cpp:61-95`
- Modify: `platforms/reference/include/ReferenceNonbondedSlicingKernels.h:80-97`, `platforms/reference/src/ReferenceNonbondedSlicingKernels.cpp:150-260`
- Test: `tests/TestSlicedNonbondedForce.h`

**Interfaces:**
- Consumes: the six accessors and `includeInner` from Task 1.
- Produces (used by Tasks 3 and 4):
  ```cpp
  // openmmapi/include/internal/InnerSplit.h — header-only, double precision
  class NonbondedSlicing::InnerSplit {
      InnerSplit();                      // enabled == false
      InnerSplit(double rs, double rc);  // precomputes everything below
      bool enabled; double rs, rc;
      double p[5];          // P(s) = (s-rs)^2 (rc-s)^2 = sum_k p[k] s^k
      double prefactor;     // 30/(rc-rs)^5
      int n[3];             // {12, 6, 1}
      double gs[3], g0s;    // G_n(rs), G_0(rs)
      double K[3];          // shift constants K_n (spec Sec. 3)
      double switchValue(double r) const;          // S(r)
      double G0(double r) const;                   // int P(s) ds
      double G(int n, double r) const;             // int P(s) s^-n ds
      double perturbation(int i, double r) const;  // coefficient of c_n in Vpert(r); 0 outside (rs, rc)
      double energyTerm(int i, double r) const;    // r^-n + perturbation - K_n for r < rc, else 0
  };
  ```
  The inner energy of a pair is `c12*energyTerm(0,r) + c6*energyTerm(1,r) + c1*energyTerm(2,r)` with
  c12 = eps*sigma^12, c6 = -eps*sigma^6 (eps = 4ε in the kernels' convention), c1 = ONE_4PI_EPS0*q1*q2.

- [ ] **Step 1: Write the failing pair test (energy by quadrature, force = S·F, continuity at r_c)**

In `tests/TestSlicedNonbondedForce.h`, insert after `testInnerGroupErrors`:

```cpp
void testInnerGroupPair(bool asException) {
    // Two particles. The inner energy must equal the integral of the switched bare force from r to the
    // inner cutoff (V-hat vanishes there), the inner force must be S(r) times the bare force, and the three
    // groups must add up to the unsplit result, including across the inner cutoff.

    const double rs = 0.5, rc = 0.8, cutoff = 1.0, q1 = 0.6, q2 = -0.4, sigma = 0.3, epsilon = 0.7;
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
    SlicedNonbondedForce* force = new SlicedNonbondedForce(1);
    force->setNonbondedMethod(SlicedNonbondedForce::PME);
    force->setCutoffDistance(cutoff);
    force->setReciprocalSpaceForceGroup(2);
    force->setInnerForceGroup(1);
    force->setInnerSwitchingDistance(rs);
    force->setInnerCutoffDistance(rc);
    if (asException) {
        force->addParticle(0.0, sigma, 0.0);
        force->addParticle(0.0, sigma, 0.0);
        force->addException(0, 1, q1*q2, sigma, epsilon);
    }
    else {
        force->addParticle(q1, sigma, epsilon);
        force->addParticle(q2, sigma, epsilon);
    }
    system.addForce(force);
    VerletIntegrator integrator(0.001);
    Context context(system, integrator, platform);
    auto bareForce = [&](double r) {
        double s6 = pow(sigma/r, 6);
        return 4*epsilon*(12*s6*s6-6*s6)/r + ONE_4PI_EPS0*q1*q2/(r*r);  // -dV/dr
    };
    auto switchValue = [&](double r) {
        if (r <= rs)
            return 1.0;
        if (r >= rc)
            return 0.0;
        double z = (r-rs)/(rc-rs);
        return 1+z*z*z*(-10+z*(15-z*6));
    };
    bool precise = platform.getName() == "Reference" || platform.getPropertyDefaultValue("Precision") == "double";
    double tol = precise ? 1e-6 : 1e-4;
    for (double r : {0.3, 0.45, 0.5, 0.55, 0.65, 0.75, 0.7999, 0.8, 0.8001, 0.9}) {
        context.setPositions({Vec3(0, 0, 0), Vec3(r, 0, 0)});
        State inner = context.getState(State::Energy | State::Forces, false, 1<<1);
        assertEqualTo(switchValue(r)*bareForce(r), inner.getForces()[1][0], tol);
        double expected = 0.0;
        if (r < rc) {
            const int n = 2000;
            double h = (rc-r)/n;
            for (int i = 0; i <= n; i++) {
                double s = r+i*h;
                double w = (i == 0 || i == n) ? 1 : (i%2 ? 4 : 2);
                expected += w*switchValue(s)*bareForce(s);
            }
            expected *= h/3;
        }
        assertEqualTo(expected, inner.getPotentialEnergy(), tol);
        State all = context.getState(State::Energy | State::Forces);
        State direct = context.getState(State::Energy | State::Forces, false, 1<<0);
        State recip = context.getState(State::Energy | State::Forces, false, 1<<2);
        assertEqualTo(all.getPotentialEnergy(), direct.getPotentialEnergy()+inner.getPotentialEnergy()+recip.getPotentialEnergy(), tol);
        for (int i = 0; i < 2; i++)
            assertEqualVec(all.getForces()[i], direct.getForces()[i]+inner.getForces()[i]+recip.getForces()[i], tol);
    }
}
```

Register in `main` after `testInnerGroupErrors();`:

```cpp
        testInnerGroupPair(false);
        testInnerGroupPair(true);
```

- [ ] **Step 2: Run the Reference test to verify it fails**

Run: `cd build && make -j8 && ./platforms/reference/tests/TestReferenceSlicedNonbondedForce`
Expected: failure inside `testInnerGroupPair` (inner group currently yields zero force/energy).

- [ ] **Step 3: Create `openmmapi/include/internal/InnerSplit.h`**

```cpp
#ifndef NONBONDED_SLICING_INNER_SPLIT_H_
#define NONBONDED_SLICING_INNER_SPLIT_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2026 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include <cmath>

namespace NonbondedSlicing {

/**
 * Host-side constants of the inner force group split.  The inner force of a pair is S(r) times the
 * bare force, where S is a quintic switch that goes from 1 at rs to 0 at rc.  The inner energy is the
 * potential whose gradient is that force, shifted to vanish at rc.  Writing the bare pair potential as
 * V(r) = sum_n c_n r^-n for n in {12, 6, 1}, the inner energy is sum_n c_n energyTerm(n, r).
 */
class InnerSplit {
public:
    InnerSplit() : enabled(false), rs(0), rc(0) {
    }
    InnerSplit(double rs, double rc) : enabled(true), rs(rs), rc(rc) {
        double a = rs+rc, b = rs*rc;
        p[0] = b*b;
        p[1] = -2*a*b;
        p[2] = a*a+2*b;
        p[3] = -2*a;
        p[4] = 1.0;
        n[0] = 12;
        n[1] = 6;
        n[2] = 1;
        prefactor = 30.0/std::pow(rc-rs, 5);
        g0s = G0(rs);
        for (int i = 0; i < 3; i++) {
            gs[i] = G(n[i], rs);
            K[i] = std::pow(rc, -n[i]) + prefactor*(G(n[i], rc)-gs[i]-std::pow(rc, -n[i])*(G0(rc)-g0s));
        }
    }
    bool enabled;
    double rs, rc, p[5], prefactor, g0s, gs[3], K[3];
    int n[3];
    double switchValue(double r) const {
        if (r <= rs)
            return 1.0;
        if (r >= rc)
            return 0.0;
        double z = (r-rs)/(rc-rs);
        return 1+z*z*z*(-10+z*(15-z*6));
    }
    double G0(double r) const {
        double sum = 0, rk = r;
        for (int k = 0; k < 5; k++) {
            sum += p[k]*rk/(k+1);
            rk *= r;
        }
        return sum;
    }
    double G(int n, double r) const {
        double sum = 0;
        for (int k = 0; k < 5; k++) {
            int m = n-k-1;
            sum += p[k]*(m == 0 ? std::log(r) : -std::pow(r, -m)/m);
        }
        return sum;
    }
    double perturbation(int i, double r) const {
        if (r <= rs || r >= rc)
            return 0.0;
        return prefactor*(G(n[i], r)-gs[i]-std::pow(r, -n[i])*(G0(r)-g0s));
    }
    double energyTerm(int i, double r) const {
        if (r >= rc)
            return 0.0;
        return std::pow(r, -n[i]) + perturbation(i, r) - K[i];
    }
};

} // namespace NonbondedSlicing

#endif /*NONBONDED_SLICING_INNER_SPLIT_H_*/
```

- [ ] **Step 4: Reference pair interactions**

`ReferenceSlicedLJCoulombIxn.h`: add `#include "internal/InnerSplit.h"`; add the member `InnerSplit innerSplit;` next to `cutoffDistance`; add the public method `void setInnerSplit(const InnerSplit& split);` and the public method

```cpp
      void calculateInnerIxn(int numberOfAtoms, vector<OpenMM::Vec3>& atomCoordinates, const vector<int>& atomSubsets,
                           const vector<vector<double>>& atomParameters, const vector<vector<double>>& sliceLambdas,
                           const vector<set<int> >& exclusions, vector<OpenMM::Vec3>& forces,
                           vector<vector<double>>& sliceEnergies) const;
```

and the private helper

```cpp
      void calculateInnerPair(int ii, int jj, const vector<vector<double>>& atomParameters, double r,
                              double& forceFactorVdw, double& forceFactorCoul, double& vdwEnergy, double& coulEnergy) const;
```

`ReferenceSlicedLJCoulombIxn.cpp`: add

```cpp
void ReferenceSlicedLJCoulombIxn::setInnerSplit(const InnerSplit& split) {
    innerSplit = split;
}

/**---------------------------------------------------------------------------------------

   Compute the inner part of a pair interaction: the switched bare force factors (to be multiplied by
   the slice scaling factors and by deltaR) and the shifted inner energies, all zero beyond the inner cutoff

   --------------------------------------------------------------------------------------- */

void ReferenceSlicedLJCoulombIxn::calculateInnerPair(int ii, int jj, const vector<vector<double>>& atomParameters, double r,
                                                     double& forceFactorVdw, double& forceFactorCoul, double& vdwEnergy, double& coulEnergy) const {
    forceFactorVdw = forceFactorCoul = vdwEnergy = coulEnergy = 0.0;
    if (r >= innerSplit.rc)
        return;
    double inverseR = 1.0/r;
    double sig = atomParameters[ii][SigIndex] + atomParameters[jj][SigIndex];
    double eps = atomParameters[ii][EpsIndex]*atomParameters[jj][EpsIndex];
    double qq = ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex];
    double sig2 = inverseR*sig;
    sig2 *= sig2;
    double sig6 = sig2*sig2*sig2;
    double switchValue = innerSplit.switchValue(r);
    forceFactorVdw = switchValue*eps*(12.0*sig6 - 6.0)*sig6*inverseR*inverseR;
    forceFactorCoul = switchValue*qq*inverseR*inverseR*inverseR;
    double sigma6 = sig*sig;
    sigma6 = sigma6*sigma6*sigma6;
    vdwEnergy = eps*(sigma6*sigma6*innerSplit.energyTerm(0, r) - sigma6*innerSplit.energyTerm(1, r));
    coulEnergy = qq*innerSplit.energyTerm(2, r);
}

void ReferenceSlicedLJCoulombIxn::calculateInnerIxn(
    int numberOfAtoms, vector<Vec3>& atomCoordinates, const vector<int>& atomSubsets,
    const vector<vector<double>>& atomParameters, const vector<vector<double>>& sliceLambdas, const vector<set<int>>& exclusions,
    vector<Vec3>& forces, vector<vector<double>>& sliceEnergies
) const {
    auto addPair = [&](int ii, int jj) {
        int si = atomSubsets[ii];
        int sj = atomSubsets[jj];
        int slice = si > sj ? si*(si+1)/2+sj : sj*(sj+1)/2+si;
        double deltaR[2][ReferenceForce::LastDeltaRIndex];
        if (periodic)
            ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
        else
            ReferenceForce::getDeltaR(atomCoordinates[jj], atomCoordinates[ii], deltaR[0]);
        double forceFactorVdw, forceFactorCoul, vdwEnergy, coulEnergy;
        calculateInnerPair(ii, jj, atomParameters, deltaR[0][ReferenceForce::RIndex], forceFactorVdw, forceFactorCoul, vdwEnergy, coulEnergy);
        double factor = sliceLambdas[slice][vdW]*forceFactorVdw + sliceLambdas[slice][Coul]*forceFactorCoul;
        for (int kk = 0; kk < 3; kk++) {
            double force = factor*deltaR[0][kk];
            forces[ii][kk] += force;
            forces[jj][kk] -= force;
        }
        sliceEnergies[slice][vdW] += vdwEnergy;
        sliceEnergies[slice][Coul] += coulEnergy;
    };
    if (cutoff) {
        for (auto& pair : *neighborList)
            addPair(pair.first, pair.second);
    }
    else {
        for (int ii = 0; ii < numberOfAtoms; ii++)
            for (int jj = ii+1; jj < numberOfAtoms; jj++)
                if (exclusions[jj].find(ii) == exclusions[jj].end())
                    addPair(ii, jj);
    }
}
```

In the direct loop of `calculateEwaldIxn` (the `for (auto& pair : *neighborList)` loop starting at line 365), right before `// accumulate forces`, add:

```cpp
        if (innerSplit.enabled) {
            double forceFactorVdw, forceFactorCoul, innerVdwEnergy, innerCoulEnergy;
            calculateInnerPair(ii, jj, atomParameters, r, forceFactorVdw, forceFactorCoul, innerVdwEnergy, innerCoulEnergy);
            dEdRvdW -= forceFactorVdw;
            dEdRCoul -= forceFactorCoul;
            vdwEnergy -= innerVdwEnergy;
            coulEnergyCorrection = innerCoulEnergy;
        }
```

declare `double coulEnergyCorrection = 0.0;` at the top of that loop body, and change the Coulomb energy accumulation to `sliceEnergies[slice][Coul] += ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR*erfc(alphaR) - coulEnergyCorrection;`.

In `calculateOneIxn`, right before `sliceEnergies[slice][vdW] += energy;`, add the same block but with `energy -= innerVdwEnergy;` instead of `vdwEnergy -= ...` and `double r = deltaR[0][ReferenceForce::RIndex];` computed first; subtract `innerCoulEnergy` from the Coulomb energy accumulated in both the cutoff and the no-cutoff branches.

- [ ] **Step 5: Reference 1-4 exceptions**

`ReferenceSlicedLJCoulomb14.h`: add `#include "internal/InnerSplit.h"`; add `enum Mode {Full = 0, Inner = 1, Remainder = 2};` in the public section; add `void setInnerSplit(const InnerSplit& split);`; add `Mode mode` as the last parameter of `calculateBondIxn` (default `Full`); add private member `InnerSplit innerSplit;`.

`ReferenceSlicedLJCoulomb14.cpp`: add `setInnerSplit` (assigns the member). In `calculateBondIxn`, after `double sig6 = ...` insert

```cpp
    double r = deltaR[0][ReferenceForce::RIndex];
    double innerVdwForce = 0.0, innerCoulForce = 0.0, innerVdwEnergy = 0.0, innerCoulEnergy = 0.0;
    if (mode != Full && r < innerSplit.rc) {
        double switchValue = innerSplit.switchValue(r);
        innerVdwForce = switchValue*parameters[1]*(12.0*sig6 - 6.0)*sig6*inverseR*inverseR;
        innerCoulForce = switchValue*ONE_4PI_EPS0*parameters[2]*inverseR*inverseR*inverseR;
        double sigma6 = pow(parameters[0], 6);
        innerVdwEnergy = parameters[1]*(sigma6*sigma6*innerSplit.energyTerm(0, r) - sigma6*innerSplit.energyTerm(1, r));
        innerCoulEnergy = ONE_4PI_EPS0*parameters[2]*innerSplit.energyTerm(2, r);
    }
    double vdwForce = parameters[1]*(12.0*sig6 - 6.0)*sig6*inverseR*inverseR;
    double coulForce = ONE_4PI_EPS0*parameters[2]*inverseR*inverseR*inverseR;
    double vdwEnergy = parameters[1]*(sig6 - 1.0)*sig6;
    double coulEnergy = ONE_4PI_EPS0*parameters[2]*inverseR;
    if (mode == Inner) {
        vdwForce = innerVdwForce;
        coulForce = innerCoulForce;
        vdwEnergy = innerVdwEnergy;
        coulEnergy = innerCoulEnergy;
    }
    else if (mode == Remainder) {
        vdwForce -= innerVdwForce;
        coulForce -= innerCoulForce;
        vdwEnergy -= innerVdwEnergy;
        coulEnergy -= innerCoulEnergy;
    }
    double dEdR = sliceLambdas[vdW]*vdwForce + sliceLambdas[Coul]*coulForce;
```

replacing the existing three `dEdR` lines, and replace the two energy accumulations with `sliceEnergies[vdW] += vdwEnergy; sliceEnergies[Coul] += coulEnergy;`.

- [ ] **Step 6: Reference kernel dispatch**

`ReferenceNonbondedSlicingKernels.h`: add `#include "internal/InnerSplit.h"` and the member `InnerSplit innerSplit;`.

`ReferenceNonbondedSlicingKernels.cpp`, in `initialize` after `rfDielectric = ...`: add

```cpp
    if (force.getInnerForceGroup() >= 0)
        innerSplit = InnerSplit(force.getInnerSwitchingDistance(), force.getInnerCutoffDistance());
```

In `execute`: after `clj.setPeriodicExceptions(...)`/method setup and before `calculatePairIxn`, add `clj.setInnerSplit(innerSplit);`. After the `calculatePairIxn` call add

```cpp
    if (includeInner)
        clj.calculateInnerIxn(numParticles, posData, subsets, particleParamArray, sliceLambdas, exclusions, forceData, sliceEnergies);
```

Replace the `if (includeDirect) { ... }` block that handles 1-4 terms and the dispersion correction with

```cpp
    if (includeDirect || includeInner) {
        ReferenceSlicedLJCoulomb14 nonbonded14;
        nonbonded14.setInnerSplit(innerSplit);
        if (exceptionsArePeriodic) {
            Vec3* boxVectors = extractBoxVectors(context);
            nonbonded14.setPeriodic(boxVectors);
        }
        vector<ReferenceSlicedLJCoulomb14::Mode> modes;
        if (includeDirect)
            modes.push_back(innerSplit.enabled ? ReferenceSlicedLJCoulomb14::Remainder : ReferenceSlicedLJCoulomb14::Full);
        if (includeInner)
            modes.push_back(ReferenceSlicedLJCoulomb14::Inner);
        for (auto mode : modes)
            for (int k = 0; k < num14; k++) {
                int slice = bonded14SliceArray[k];
                nonbonded14.calculateBondIxn(bonded14IndexArray[k], posData, bonded14ParamArray[k],
                                             forceData, sliceLambdas[slice], sliceEnergies[slice], mode);
            }
        if (includeDirect && (periodic || ewald || pme)) {
            Vec3* boxVectors = extractBoxVectors(context);
            double volume = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
            for (int slice = 0; slice < numSlices; slice++)
                sliceEnergies[slice][vdW] += dispersionCoefficients[slice]/volume;
        }
    }
```

- [ ] **Step 7: Run the Reference tests**

Run: `cd build && make -j8 && ./platforms/reference/tests/TestReferenceSlicedNonbondedForce`
Expected: passes, including `testInnerGroupPair(false)` and `testInnerGroupPair(true)` and every pre-existing test (the split is disabled by default, so nothing else changes).

- [ ] **Step 8: Commit**

```bash
git add openmmapi/include/internal/InnerSplit.h platforms/reference tests/TestSlicedNonbondedForce.h
git commit -m "Implement the inner force group on the Reference platform"
```

---

### Task 3: Many-particle tests on the Reference platform (forces, consistency, derivatives, offsets)

**Files:**
- Test: `tests/TestSlicedNonbondedForce.h` (all new code goes before `void testErrors()`; registrations in `main`)

**Interfaces:**
- Consumes: Task 1 accessors, Task 2 Reference implementation.
- Produces: the test fixture `InnerTestSystem` reused unchanged by the GPU runs of Task 4.

- [ ] **Step 1: Write the fixture and the failing tests**

```cpp
// A box of three-site molecules (1-2/1-3 exclusions) with extra scaled 1-4 exceptions, subsets, slice
// scaling parameters and parameter offsets, built either with the inner force group (split=true) or without.
// Group 0 = direct space, 1 = inner, 2 = reciprocal space (for Ewald/PME/LJPME).
class InnerTestSystem {
public:
    enum Kind {Both, LJOnly, CoulombOnly};
    InnerTestSystem(OpenMM_SFMT::SFMT& sfmt, NonbondedForce::NonbondedMethod method, int numSubsets, Kind kind, bool offsets, bool innerAtCutoff) :
            method(method), numSubsets(numSubsets), kind(kind), offsets(offsets), rc(innerAtCutoff ? cutoff : 0.8) {
        periodic = (method != NonbondedForce::NoCutoff && method != NonbondedForce::CutoffNonPeriodic);
        for (int m = 0; m < numMolecules; m++) {
            Vec3 center(boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt), boxSize*genrand_real2(sfmt));
            positions.push_back(center);
            positions.push_back(center+Vec3(0.1, 0, 0));
            positions.push_back(center+Vec3(0, 0.1, 0));
        }
        for (int i = 0; i < numParticles; i++) {
            bool oxygen = (i%3 == 0);
            charge.push_back(kind == LJOnly ? 0.0 : (oxygen ? -0.8 : 0.4));
            sigma.push_back(oxygen ? 0.31 : 0.2);
            epsilon.push_back(kind == CoulombOnly ? 0.0 : (oxygen ? 0.6 : 0.1));
            subset.push_back(i%numSubsets);
        }
        for (int m = 0; m < numMolecules; m++) {
            exclusions.push_back({3*m, 3*m+1});
            exclusions.push_back({3*m, 3*m+2});
            exclusions.push_back({3*m+1, 3*m+2});
        }
        for (int m = 0; m+1 < numMolecules; m += 4)
            exceptions14.push_back({3*m+2, 3*m+4, kind == LJOnly ? 0.0 : 0.3, 0.25, kind == CoulombOnly ? 0.0 : 0.2});
    }
    System* build(bool split, SlicedNonbondedForce** forceOut) {
        System* system = new System();
        for (int i = 0; i < numParticles; i++)
            system->addParticle(1.0);
        system->setDefaultPeriodicBoxVectors(Vec3(boxSize, 0, 0), Vec3(0, boxSize, 0), Vec3(0, 0, boxSize));
        SlicedNonbondedForce* force = new SlicedNonbondedForce(numSubsets);
        force->setNonbondedMethod((SlicedNonbondedForce::NonbondedMethod) method);
        force->setCutoffDistance(cutoff);
        if (method == NonbondedForce::CutoffPeriodic || method == NonbondedForce::PME || method == NonbondedForce::Ewald) {
            force->setUseSwitchingFunction(true);
            force->setSwitchingDistance(0.9);
        }
        if (periodic && method != NonbondedForce::CutoffPeriodic)
            force->setReciprocalSpaceForceGroup(2);
        for (int i = 0; i < numParticles; i++) {
            force->addParticle(charge[i], sigma[i], epsilon[i]);
            force->setParticleSubset(i, subset[i]);
        }
        for (auto& e : exclusions)
            force->addException(e[0], e[1], 0.0, 1.0, 0.0);
        for (auto& e : exceptions14)
            force->addException((int) e[0], (int) e[1], e[2], e[3], e[4]);
        force->addGlobalParameter("lambdaA", lambdaA);
        force->addScalingParameter("lambdaA", 0, numSubsets-1, true, true);
        force->addEnergyParameterDerivative("lambdaA");
        if (numSubsets > 1) {
            force->addGlobalParameter("lambdaB", lambdaB);
            force->addScalingParameter("lambdaB", 1, 1, true, false);
            force->addEnergyParameterDerivative("lambdaB");
        }
        if (offsets) {
            force->addGlobalParameter("offset", offsetValue);
            force->addParticleParameterOffset("offset", 0, 0.2, 0.0, 0.1);
            force->addParticleParameterOffset("offset", 4, -0.1, 0.02, 0.0);
            force->addExceptionParameterOffset("offset", (int) exclusions.size(), 0.1, 0.0, 0.05);
        }
        if (split) {
            force->setInnerForceGroup(1);
            force->setInnerSwitchingDistance(rs);
            force->setInnerCutoffDistance(rc);
        }
        system->addForce(force);
        *forceOut = force;
        return system;
    }
    // Inner-group forces computed independently of the plugin: S(r) times the bare force, with the
    // current values of the scaling parameters and offsets applied.
    vector<Vec3> expectedInnerForces(double offsetNow) const {
        vector<double> q = charge, sig = sigma, eps = epsilon;
        vector<array<double, 5>> ex14 = exceptions14;
        if (offsets) {
            q[0] += offsetNow*0.2;
            eps[0] += offsetNow*0.1;
            q[4] += offsetNow*(-0.1);
            sig[4] += offsetNow*0.02;
            ex14[0][2] += offsetNow*0.1;
            ex14[0][4] += offsetNow*0.05;
        }
        set<pair<int, int>> excluded;
        for (auto& e : exclusions)
            excluded.insert({min(e[0], e[1]), max(e[0], e[1])});
        for (auto& e : ex14)
            excluded.insert({min((int) e[0], (int) e[1]), max((int) e[0], (int) e[1])});
        vector<Vec3> forces(numParticles, Vec3());
        auto addPair = [&](int i, int j, Vec3 delta, double qq, double sigmaIJ, double epsIJ) {
            double r = sqrt(delta.dot(delta));
            if (r >= rc)
                return;
            double z = (r-rs)/(rc-rs);
            double s = r <= rs ? 1.0 : 1+z*z*z*(-10+z*(15-z*6));
            double s6 = pow(sigmaIJ/r, 6);
            double f = lambdaVdw(i, j)*4*epsIJ*(12*s6*s6-6*s6)/r + lambdaCoul(i, j)*ONE_4PI_EPS0*qq/(r*r);
            Vec3 force = delta*(s*f/r);
            forces[i] += force;
            forces[j] -= force;
        };
        for (int i = 0; i < numParticles; i++)
            for (int j = i+1; j < numParticles; j++) {
                if (excluded.count({i, j}))
                    continue;
                Vec3 delta = positions[i]-positions[j];
                if (periodic)
                    for (int k = 0; k < 3; k++)
                        delta[k] -= boxSize*round(delta[k]/boxSize);
                addPair(i, j, delta, q[i]*q[j], 0.5*(sig[i]+sig[j]), sqrt(eps[i]*eps[j]));
            }
        for (auto& e : ex14)
            addPair((int) e[0], (int) e[1], positions[(int) e[0]]-positions[(int) e[1]], e[2], e[3], e[4]);
        return forces;
    }
    double lambdaVdw(int i, int j) const {
        return sliceIndex(subset[i], subset[j]) == sliceIndex(0, numSubsets-1) ? lambdaA : 1.0;
    }
    double lambdaCoul(int i, int j) const {
        int slice = sliceIndex(subset[i], subset[j]);
        if (slice == sliceIndex(0, numSubsets-1))
            return lambdaA;
        if (numSubsets > 1 && slice == sliceIndex(1, 1))
            return lambdaB;
        return 1.0;
    }
    static constexpr int numMolecules = 20, numParticles = 60;
    static constexpr double boxSize = 3.0, cutoff = 1.0, rs = 0.5, lambdaA = 0.7, lambdaB = 0.4, offsetValue = 0.5;
    NonbondedForce::NonbondedMethod method;
    int numSubsets;
    Kind kind;
    bool offsets, periodic;
    double rc;
    vector<Vec3> positions;
    vector<double> charge, sigma, epsilon;
    vector<int> subset;
    vector<array<int, 2>> exclusions;
    vector<array<double, 5>> exceptions14;
};

static double groupTolerance() {
    return (platform.getName() == "Reference" || platform.getPropertyDefaultValue("Precision") == "double") ? 1e-6 : 1e-4;
}

void testInnerGroupForces(OpenMM_SFMT::SFMT& sfmt, NonbondedForce::NonbondedMethod method, int numSubsets, InnerTestSystem::Kind kind, bool offsets, bool innerAtCutoff) {
    InnerTestSystem data(sfmt, method, numSubsets, kind, offsets, innerAtCutoff);
    SlicedNonbondedForce *splitForce, *plainForce;
    System* splitSystem = data.build(true, &splitForce);
    System* plainSystem = data.build(false, &plainForce);
    VerletIntegrator integrator1(0.001), integrator2(0.001);
    Context split(*splitSystem, integrator1, platform);
    Context plain(*plainSystem, integrator2, platform);
    split.setPositions(data.positions);
    plain.setPositions(data.positions);
    double tol = groupTolerance();

    // The inner group alone against the independent reference.

    State inner = split.getState(State::Forces | State::Energy, false, 1<<1);
    vector<Vec3> expected = data.expectedInnerForces(InnerTestSystem::offsetValue);
    for (int i = 0; i < data.numParticles; i++)
        assertEqualVec(expected[i], inner.getForces()[i], tol);
    ASSERT(inner.getPotentialEnergy() != 0.0);

    // All groups together against the unsplit force: energy, forces and derivatives.

    State all = split.getState(State::Forces | State::Energy | State::ParameterDerivatives);
    State reference = plain.getState(State::Forces | State::Energy | State::ParameterDerivatives);
    assertEnergy(reference, all, tol);
    assertForces(reference, all, tol);
    for (auto& entry : reference.getEnergyParameterDerivatives())
        assertEqualTo(entry.second, all.getEnergyParameterDerivatives().at(entry.first), tol);

    // The direct and inner groups are complementary: their sum is the unsplit direct group.

    State direct = split.getState(State::Forces | State::Energy, false, 1<<0);
    State plainDirect = plain.getState(State::Forces | State::Energy, false, 1<<0);
    assertEqualTo(plainDirect.getPotentialEnergy(), direct.getPotentialEnergy()+inner.getPotentialEnergy(), tol);
    for (int i = 0; i < data.numParticles; i++)
        assertEqualVec(plainDirect.getForces()[i], direct.getForces()[i]+inner.getForces()[i], tol);
    delete splitSystem;
    delete plainSystem;
}

void testInnerGroupConsistency(OpenMM_SFMT::SFMT& sfmt, NonbondedForce::NonbondedMethod method) {
    // Each group's force must be minus the gradient of that group's energy (central differences).

    if (platform.getName() != "Reference" && platform.getPropertyDefaultValue("Precision") != "double")
        return;
    InnerTestSystem data(sfmt, method, 2, InnerTestSystem::Both, true, false);
    SlicedNonbondedForce* force;
    System* system = data.build(true, &force);
    VerletIntegrator integrator(0.001);
    Context context(*system, integrator, platform);
    const double h = 1e-5;
    for (int group : {0, 1}) {
        context.setPositions(data.positions);
        vector<Vec3> forces = context.getState(State::Forces, false, 1<<group).getForces();
        for (int i : {0, 1, 2, 7, 23, 44})
            for (int k = 0; k < 3; k++) {
                vector<Vec3> displaced = data.positions;
                displaced[i][k] += h;
                context.setPositions(displaced);
                double plus = context.getState(State::Energy, false, 1<<group).getPotentialEnergy();
                displaced[i][k] -= 2*h;
                context.setPositions(displaced);
                double minus = context.getState(State::Energy, false, 1<<group).getPotentialEnergy();
                assertEqualTo(-(plus-minus)/(2*h), forces[i][k], 1e-4);
            }
    }
    delete system;
}

void testInnerGroupDerivatives(OpenMM_SFMT::SFMT& sfmt) {
    // The energy of every group is linear in each scaling parameter, so dE/dlambda = E(1) - E(0) per group.

    InnerTestSystem data(sfmt, NonbondedForce::PME, 2, InnerTestSystem::Both, false, false);
    SlicedNonbondedForce* force;
    System* system = data.build(true, &force);
    VerletIntegrator integrator(0.001);
    Context context(*system, integrator, platform);
    context.setPositions(data.positions);
    double tol = groupTolerance();
    for (int group : {0, 1, 2})
        for (string name : {"lambdaA", "lambdaB"}) {
            context.setParameter(name, InnerTestSystem::lambdaA);
            double derivative = context.getState(State::Energy | State::ParameterDerivatives, false, 1<<group).getEnergyParameterDerivatives().at(name);
            context.setParameter(name, 1.0);
            double e1 = context.getState(State::Energy, false, 1<<group).getPotentialEnergy();
            context.setParameter(name, 0.0);
            double e0 = context.getState(State::Energy, false, 1<<group).getPotentialEnergy();
            context.setParameter(name, name == "lambdaA" ? InnerTestSystem::lambdaA : InnerTestSystem::lambdaB);
            assertEqualTo(e1-e0, derivative, 10*tol);
            if (group == 1)
                ASSERT(fabs(derivative) > 1e-3);
        }
    delete system;
}

void testInnerGroupOffsets(OpenMM_SFMT::SFMT& sfmt) {
    // Offsets changed through setParameter, and parameters changed through updateParametersInContext,
    // must both reach the inner group.

    InnerTestSystem data(sfmt, NonbondedForce::PME, 2, InnerTestSystem::Both, true, false);
    SlicedNonbondedForce* force;
    System* system = data.build(true, &force);
    VerletIntegrator integrator(0.001);
    Context context(*system, integrator, platform);
    context.setPositions(data.positions);
    double tol = groupTolerance();
    context.setParameter("offset", 1.2);
    vector<Vec3> expected = data.expectedInnerForces(1.2);
    vector<Vec3> found = context.getState(State::Forces, false, 1<<1).getForces();
    for (int i = 0; i < data.numParticles; i++)
        assertEqualVec(expected[i], found[i], tol);
    data.charge[3] = -0.5;
    data.sigma[3] = 0.28;
    data.epsilon[3] = 0.9;
    force->setParticleParameters(3, data.charge[3], data.sigma[3], data.epsilon[3]);
    data.exceptions14[1][2] = -0.2;
    force->setExceptionParameters((int) data.exclusions.size()+1, (int) data.exceptions14[1][0], (int) data.exceptions14[1][1], -0.2, 0.25, 0.2);
    force->updateParametersInContext(context);
    expected = data.expectedInnerForces(1.2);
    found = context.getState(State::Forces, false, 1<<1).getForces();
    for (int i = 0; i < data.numParticles; i++)
        assertEqualVec(expected[i], found[i], tol);
    delete system;
}
```

`sliceIndex` is the macro from `SlicedNonbondedForce.h`; `array`, `set`, `min`/`max` need `#include <array>` and `#include <set>` at the top of the test header. Register in `main` after `testInnerGroupPair(true);`:

```cpp
        for (auto method : nonbondedMethods)
            for (int numSubsets : {1, 3})
                for (auto kind : {InnerTestSystem::Both, InnerTestSystem::LJOnly, InnerTestSystem::CoulombOnly})
                    testInnerGroupForces(sfmt, method, numSubsets, kind, numSubsets == 3, false);
        testInnerGroupForces(sfmt, NonbondedForce::PME, 2, InnerTestSystem::Both, true, true);
        for (auto method : nonbondedMethods)
            testInnerGroupConsistency(sfmt, method);
        testInnerGroupDerivatives(sfmt);
        testInnerGroupOffsets(sfmt);
```

(`nonbondedMethods` and `sfmt` already exist in `main`; check their exact names there and reuse them.)

- [ ] **Step 2: Run on Reference and make it pass**

Run: `cd build && make -j8 && ./platforms/reference/tests/TestReferenceSlicedNonbondedForce`
Expected: all pass. If `testInnerGroupForces` fails only for `LJOnly`/`CoulombOnly`, the Reference `calculateInnerPair` is fine and the failure is in parameter handling (charges of zero make `hasCoulomb` false in the kernels; check `computeParameters`). If `testInnerGroupConsistency` fails for the direct group only, the subtraction of the inner energy in `calculateEwaldIxn`/`calculateOneIxn` is missing a term (compare with the force subtraction, which is tested by `testInnerGroupForces`).

- [ ] **Step 3: Commit**

```bash
git add tests/TestSlicedNonbondedForce.h platforms/reference
git commit -m "Test the inner force group against independent references on the Reference platform"
```

---

### Task 4: CUDA and OpenCL (common kernels)

**Files:**
- Modify: `platforms/common/src/kernels/coulombLennardJones.cc` (full replacement below)
- Modify: `platforms/common/src/kernels/nonbondedExceptions.cc` (full replacement below)
- Modify: `platforms/common/src/CommonNonbondedSlicingKernels.cpp:368-372` (defines), `:681-719` (pair kernel registration), `:722-757` (exceptions registration), `:846` (NaN marking)
- Test: the tests of Tasks 1–3 run through the CUDA and OpenCL drivers

**Interfaces:**
- Consumes: `InnerSplit` (Task 2) for the host constants; `includeInner` (Task 1).
- Produces: nothing new; the GPU kernels must match the Reference platform within `groupTolerance()`.

Mechanism (spec Sec. 5.2): the same pair source is registered twice with `NonbondedUtilities::addInteraction`, once in the direct group compiled with `SUBTRACT_INNER=1`, once in the inner group compiled with `INNER_GROUP=1`; likewise the exceptions source through `BondedUtilities`. Both share the parameter arrays and the neighbor list. The slice index is guarded with `isExcluded` (padding atoms carry stale subsets on CPU OpenCL devices; PR #46 adds the same guard on `main`, keep exactly one copy after rebasing).

- [ ] **Step 1: Run the CUDA tests to see the current failure**

Run: `cd build && make -j8 && ./platforms/cuda/tests/TestCudaSlicedNonbondedForce mixed`
Expected: `testInnerGroupPair` fails (inner group returns zero on the GPU).

- [ ] **Step 2: Replace `platforms/common/src/kernels/coulombLennardJones.cc`**

```c
{
    // Excluded pairs include those with a padding atom, whose subset may be garbage (e.g., stale local
    // memory in OpenCL CPU kernels), so they must not index LAMBDA. Their terms are masked out below.
    int slice = isExcluded ? 0 : (SUBSET1>SUBSET2 ? SUBSET1*(SUBSET1+1)/2+SUBSET2 : SUBSET2*(SUBSET2+1)/2+SUBSET1);
    real clLambda = LAMBDA[slice].x;
    real ljLambda = LAMBDA[slice].y;
#if HAS_COULOMB
    const real qq = ONE_4PI_EPS0*CHARGE1*CHARGE2;
#else
    const real qq = 0.0f;
#endif
#if HAS_LENNARD_JONES
    real sig = SIGMA_EPSILON1.x + SIGMA_EPSILON2.x;
    real sig2 = invR*sig;
    sig2 *= sig2;
    real sig6 = sig2*sig2*sig2;
    real eps = SIGMA_EPSILON1.y*SIGMA_EPSILON2.y;
    real epssig6 = sig6*eps;
#endif
#if INNER_GROUP || SUBTRACT_INNER
    // Inner part of the interaction: S(r) times the bare force, and the shifted potential whose
    // gradient it is (the energy terms are only needed when the energy is requested).
    real innerForce = 0.0f, innerEnergyLJ = 0.0f, innerEnergyCl = 0.0f;
    const bool insideInner = (!isExcluded && r2 < INNER_CUTOFF_SQUARED);
    if (insideInner) {
        real innerSwitch = 1.0f;
        if (r > INNER_SWITCH) {
            const real z = (r-INNER_SWITCH)*INNER_SWITCH_SCALE;
            innerSwitch = 1.0f+z*z*z*(-10.0f+z*(15.0f-z*6.0f));
        }
        real innerForceLJ = 0.0f;
#if HAS_LENNARD_JONES
        innerForceLJ = epssig6*(12.0f*sig6 - 6.0f);
#endif
        innerForce = innerSwitch*(ljLambda*innerForceLJ + clLambda*qq*invR);
#ifdef INCLUDE_ENERGY
        real pert12 = 0.0f, pert6 = 0.0f, pert1 = 0.0f;
        if (r > INNER_SWITCH) {
            const real invR2 = invR*invR, invR3 = invR2*invR, invR4 = invR2*invR2, invR5 = invR4*invR;
            const real invR7 = invR5*invR2, invR8 = invR4*invR4, invR9 = invR8*invR, invR10 = invR5*invR5, invR11 = invR10*invR;
            const real r3 = r2*r, r4 = r2*r2, r5 = r4*r;
            const real g0 = INNER_P0*r + INNER_P1*r2*0.5f + INNER_P2*r3*(1.0f/3.0f) + INNER_P3*r4*0.25f + INNER_P4*r5*0.2f - INNER_G0S;
            const real g12 = -(INNER_P0*invR11*(1.0f/11.0f) + INNER_P1*invR10*0.1f + INNER_P2*invR9*(1.0f/9.0f) + INNER_P3*invR8*0.125f + INNER_P4*invR7*(1.0f/7.0f)) - INNER_G12S;
            const real g6 = -(INNER_P0*invR5*0.2f + INNER_P1*invR4*0.25f + INNER_P2*invR3*(1.0f/3.0f) + INNER_P3*invR2*0.5f + INNER_P4*invR) - INNER_G6S;
            const real g1 = INNER_P0*LOG(r) + INNER_P1*r + INNER_P2*r2*0.5f + INNER_P3*r3*(1.0f/3.0f) + INNER_P4*r4*0.25f - INNER_G1S;
            pert12 = INNER_PREFACTOR*(g12 - invR11*invR*g0);
            pert6 = INNER_PREFACTOR*(g6 - invR5*invR*g0);
            pert1 = INNER_PREFACTOR*(g1 - invR*g0);
        }
        innerEnergyCl = qq*(invR + pert1 - INNER_K1);
#if HAS_LENNARD_JONES
        real sigma6 = sig*sig;
        sigma6 = sigma6*sigma6*sigma6;
        innerEnergyLJ = epssig6*(sig6 - 1.0f) + eps*(sigma6*sigma6*(pert12 - INNER_K12) - sigma6*(pert6 - INNER_K6));
#endif
#endif
    }
#endif
#if INNER_GROUP
    real tempForce = innerForce;
    real ljEnergy = innerEnergyLJ;
    real clEnergy = innerEnergyCl;
    tempEnergy += ljLambda*ljEnergy + clLambda*clEnergy;
    dEdR += insideInner ? tempForce*invR*invR : 0;
#else
#if USE_EWALD
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
    const real alphaR = EWALD_ALPHA*r;
    const real expAlphaRSqr = EXP(-alphaR*alphaR);
    const real prefactor = qq*invR;

#ifdef USE_DOUBLE_PRECISION
    const real erfcAlphaR = erfc(alphaR);
#else
    // This approximation for erfc is from Abramowitz and Stegun (1964) p. 299.  They cite the following as
    // the original source: C. Hastings, Jr., Approximations for Digital Computers (1955).  It has a maximum
    // error of 1.5e-7.

    const real t = RECIP(1.0f+0.3275911f*alphaR);
    const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
    real clEnergy = includeInteraction ? prefactor*erfcAlphaR : 0;
    real tempForce = 0.0f;
#if HAS_LENNARD_JONES
    tempForce = epssig6*(12.0f*sig6 - 6.0f);
    real ljEnergy = includeInteraction ? epssig6*(sig6 - 1.0f) : 0;
    #if USE_LJ_SWITCH
    if (r > LJ_SWITCH_CUTOFF) {
        real x = r-LJ_SWITCH_CUTOFF;
        real switchValue = 1+x*x*x*(LJ_SWITCH_C3+x*(LJ_SWITCH_C4+x*LJ_SWITCH_C5));
        real switchDeriv = x*x*(3*LJ_SWITCH_C3+x*(4*LJ_SWITCH_C4+x*5*LJ_SWITCH_C5));
        tempForce = tempForce*switchValue - ljEnergy*switchDeriv*r;
        ljEnergy *= switchValue;
    }
    #endif
#if DO_LJPME
    // The multiplicative term to correct for the multiplicative terms that are always
    // present in reciprocal space.
    const real dispersionAlphaR = EWALD_DISPERSION_ALPHA*r;
    const real dar2 = dispersionAlphaR*dispersionAlphaR;
    const real dar4 = dar2*dar2;
    const real dar6 = dar4*dar2;
    const real invR2 = invR*invR;
    const real expDar2 = EXP(-dar2);
    const float2 sigExpProd = SIGMA_EPSILON1*SIGMA_EPSILON2;
    const real c6 = 64*sigExpProd.x*sigExpProd.x*sigExpProd.x*sigExpProd.y;
    const real coef = invR2*invR2*invR2*c6;
    const real eprefac = 1.0f + dar2 + 0.5f*dar4;
    const real dprefac = eprefac + dar6/6.0f;
    // The multiplicative grid term
    real ljPmeEnergy = coef*(1.0f - expDar2*eprefac);
    tempForce += 6.0f*coef*(1.0f - expDar2*dprefac);
    // The potential shift accounts for the step at the cutoff introduced by the
    // transition from additive to multiplicative combintion rules and is only
    // needed for the real (not excluded) terms.  By addin these terms to ljEnergy
    // instead of tempEnergy here, the includeInteraction mask is correctly applied.
    sig2 = sig*sig;
    sig6 = sig2*sig2*sig2*INVCUT6;
    epssig6 = eps*sig6;
    // The additive part of the potential shift
    ljPmeEnergy += epssig6*(1.0f - sig6);
    // The multiplicative part of the potential shift
    ljPmeEnergy += MULTSHIFT6*c6;
    ljEnergy += includeInteraction ? ljPmeEnergy : 0;
#endif
    tempForce *= ljLambda;
    tempForce += clLambda*prefactor*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
#if SUBTRACT_INNER
    tempForce -= innerForce;
    ljEnergy -= innerEnergyLJ;
    clEnergy -= innerEnergyCl;
#endif
    tempEnergy += ljLambda*ljEnergy + clLambda*clEnergy;
#else
    real tempForce = clLambda*prefactor*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
#if SUBTRACT_INNER
    tempForce -= innerForce;
    clEnergy -= innerEnergyCl;
#endif
    tempEnergy += clLambda*clEnergy;
#endif
#else
#ifdef USE_CUTOFF
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
#else
    unsigned int includeInteraction = (!isExcluded);
#endif
    real tempForce = 0.0f;
#if HAS_LENNARD_JONES
    tempForce = epssig6*(12.0f*sig6 - 6.0f);
    real ljEnergy = includeInteraction ? epssig6*(sig6 - 1) : 0;
    #if USE_LJ_SWITCH
    if (r > LJ_SWITCH_CUTOFF) {
        real x = r-LJ_SWITCH_CUTOFF;
        real switchValue = 1+x*x*x*(LJ_SWITCH_C3+x*(LJ_SWITCH_C4+x*LJ_SWITCH_C5));
        real switchDeriv = x*x*(3*LJ_SWITCH_C3+x*(4*LJ_SWITCH_C4+x*5*LJ_SWITCH_C5));
        tempForce = tempForce*switchValue - ljEnergy*switchDeriv*r;
        ljEnergy *= switchValue;
    }
    #endif
    tempForce *= ljLambda;
#if SUBTRACT_INNER
    ljEnergy -= innerEnergyLJ;
#endif
    tempEnergy += ljLambda*ljEnergy;
#endif
#if HAS_COULOMB
  #ifdef USE_CUTOFF
    const real prefactor = qq;
    real clEnergy = includeInteraction ? prefactor*(invR + REACTION_FIELD_K*r2 - REACTION_FIELD_C) : 0;
    tempForce += clLambda*prefactor*(invR - 2.0f*REACTION_FIELD_K*r2);
  #else
    const real prefactor = qq*invR;
    real clEnergy = includeInteraction ? prefactor : 0;
    tempForce += clLambda*prefactor;
  #endif
#if SUBTRACT_INNER
    clEnergy -= innerEnergyCl;
#endif
    tempEnergy += clLambda*clEnergy;
#endif
#if SUBTRACT_INNER
    tempForce -= innerForce;
#endif
#endif
dEdR += includeInteraction ? tempForce*invR*invR : 0;
#endif
COMPUTE_DERIVATIVES
}
```

Check against the original file line by line: every pre-existing branch keeps its arithmetic; only `prefactor` is now written in terms of `qq`, the LJ quantities are hoisted to the top, and the `SUBTRACT_INNER` lines are new. When `INNER_GROUP` and `SUBTRACT_INNER` are both `0` (the default) the generated code is equivalent to today's.

- [ ] **Step 3: Replace `platforms/common/src/kernels/nonbondedExceptions.cc`**

```c
float4 exceptionParams = PARAMS[index];
union {int i; float f;} u;
u.f = exceptionParams.w;
int slice = u.i;
real clLambda = LAMBDAS[slice].x;
real ljLambda = LAMBDAS[slice].y;
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if APPLY_PERIODIC
APPLY_PERIODIC_TO_DELTA(delta)
#endif
real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
real invR = RSQRT(r2);
real sig2 = invR*exceptionParams.y;
sig2 *= sig2;
real sig6 = sig2*sig2*sig2;
real ljForce = exceptionParams.z*(12.0f*sig6-6.0f)*sig6;
real clForce = exceptionParams.x*invR;
real ljEnergy = exceptionParams.z*(sig6-1.0f)*sig6;
real clEnergy = exceptionParams.x*invR;
#if INNER_GROUP || SUBTRACT_INNER
// Inner part: S(r) times the bare force and the shifted potential whose gradient it is
real innerForceLJ = 0.0f, innerForceCl = 0.0f, innerEnergyLJ = 0.0f, innerEnergyCl = 0.0f;
if (r2 < INNER_CUTOFF_SQUARED) {
    const real r = SQRT(r2);
    real innerSwitch = 1.0f;
    real pert12 = 0.0f, pert6 = 0.0f, pert1 = 0.0f;
    if (r > INNER_SWITCH) {
        const real z = (r-INNER_SWITCH)*INNER_SWITCH_SCALE;
        innerSwitch = 1.0f+z*z*z*(-10.0f+z*(15.0f-z*6.0f));
        const real invR2 = invR*invR, invR3 = invR2*invR, invR4 = invR2*invR2, invR5 = invR4*invR;
        const real invR7 = invR5*invR2, invR8 = invR4*invR4, invR9 = invR8*invR, invR10 = invR5*invR5, invR11 = invR10*invR;
        const real r3 = r2*r, r4 = r2*r2, r5 = r4*r;
        const real g0 = INNER_P0*r + INNER_P1*r2*0.5f + INNER_P2*r3*(1.0f/3.0f) + INNER_P3*r4*0.25f + INNER_P4*r5*0.2f - INNER_G0S;
        const real g12 = -(INNER_P0*invR11*(1.0f/11.0f) + INNER_P1*invR10*0.1f + INNER_P2*invR9*(1.0f/9.0f) + INNER_P3*invR8*0.125f + INNER_P4*invR7*(1.0f/7.0f)) - INNER_G12S;
        const real g6 = -(INNER_P0*invR5*0.2f + INNER_P1*invR4*0.25f + INNER_P2*invR3*(1.0f/3.0f) + INNER_P3*invR2*0.5f + INNER_P4*invR) - INNER_G6S;
        const real g1 = INNER_P0*LOG(r) + INNER_P1*r + INNER_P2*r2*0.5f + INNER_P3*r3*(1.0f/3.0f) + INNER_P4*r4*0.25f - INNER_G1S;
        pert12 = INNER_PREFACTOR*(g12 - invR11*invR*g0);
        pert6 = INNER_PREFACTOR*(g6 - invR5*invR*g0);
        pert1 = INNER_PREFACTOR*(g1 - invR*g0);
    }
    innerForceLJ = innerSwitch*ljForce;
    innerForceCl = innerSwitch*clForce;
    real sigma6 = exceptionParams.y*exceptionParams.y;
    sigma6 = sigma6*sigma6*sigma6;
    innerEnergyLJ = ljEnergy + exceptionParams.z*(sigma6*sigma6*(pert12 - INNER_K12) - sigma6*(pert6 - INNER_K6));
    innerEnergyCl = exceptionParams.x*(invR + pert1 - INNER_K1);
}
#endif
#if INNER_GROUP
ljForce = innerForceLJ;
clForce = innerForceCl;
ljEnergy = innerEnergyLJ;
clEnergy = innerEnergyCl;
#elif SUBTRACT_INNER
ljForce -= innerForceLJ;
clForce -= innerForceCl;
ljEnergy -= innerEnergyLJ;
clEnergy -= innerEnergyCl;
#endif
real dEdR = (ljLambda*ljForce + clLambda*clForce)*invR*invR;
energy += clLambda*clEnergy + ljLambda*ljEnergy;
delta *= dEdR;
real3 force1 = -delta;
real3 force2 = delta;
COMPUTE_DERIVATIVES
```

- [ ] **Step 4: Host-side registration in `CommonNonbondedSlicingKernels.cpp`**

Add `#include "internal/InnerSplit.h"` with the other includes. In `commonInitialize`, right after `defines["USE_LJ_SWITCH"] = ...;` (line 371) add:

```cpp
    bool hasInner = (force.getInnerForceGroup() >= 0);
    map<string, string> innerDefines;
    innerDefines["INNER_GROUP"] = "0";
    innerDefines["SUBTRACT_INNER"] = hasInner ? "1" : "0";
    if (hasInner) {
        InnerSplit split(force.getInnerSwitchingDistance(), force.getInnerCutoffDistance());
        innerDefines["INNER_CUTOFF_SQUARED"] = cc.doubleToString(split.rc*split.rc);
        innerDefines["INNER_SWITCH"] = cc.doubleToString(split.rs);
        innerDefines["INNER_SWITCH_SCALE"] = cc.doubleToString(1.0/(split.rc-split.rs));
        innerDefines["INNER_PREFACTOR"] = cc.doubleToString(split.prefactor);
        for (int k = 0; k < 5; k++)
            innerDefines["INNER_P"+cc.intToString(k)] = cc.doubleToString(split.p[k]);
        innerDefines["INNER_G12S"] = cc.doubleToString(split.gs[0]);
        innerDefines["INNER_G6S"] = cc.doubleToString(split.gs[1]);
        innerDefines["INNER_G1S"] = cc.doubleToString(split.gs[2]);
        innerDefines["INNER_G0S"] = cc.doubleToString(split.g0s);
        innerDefines["INNER_K12"] = cc.doubleToString(split.K[0]);
        innerDefines["INNER_K6"] = cc.doubleToString(split.K[1]);
        innerDefines["INNER_K1"] = cc.doubleToString(split.K[2]);
    }
    defines.insert(innerDefines.begin(), innerDefines.end());
```

Replace the pair-kernel registration (the `if (force.getIncludeDirectSpace()) cc.getNonbondedUtilities().addInteraction(...)` statement at line 718-719) with:

```cpp
    if (force.getIncludeDirectSpace())
        cc.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup(), numParticles > 3000, true);
    if (hasInner) {
        map<string, string> innerKernelDefines = defines;
        innerKernelDefines["INNER_GROUP"] = "1";
        innerKernelDefines["SUBTRACT_INNER"] = "0";
        string innerSource = cc.replaceStrings(CommonNonbondedSlicingKernelSources::coulombLennardJones, innerKernelDefines);
        innerSource = cc.replaceStrings(innerSource, replacements);
        cc.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getInnerCutoffDistance(), exclusionList, innerSource, force.getInnerForceGroup(), numParticles > 3000, true);
    }
```

In the exceptions block, after `replacements["COMPUTE_DERIVATIVES"] = code.str();` (line 755) add `replacements.insert(innerDefines.begin(), innerDefines.end());` and replace the registration statement with:

```cpp
        if (force.getIncludeDirectSpace())
            cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonNonbondedSlicingKernelSources::nonbondedExceptions, replacements), force.getForceGroup());
        if (hasInner) {
            map<string, string> innerReplacements = replacements;
            innerReplacements["INNER_GROUP"] = "1";
            innerReplacements["SUBTRACT_INNER"] = "0";
            cc.getBondedUtilities().addInteraction(atoms, cc.replaceStrings(CommonNonbondedSlicingKernelSources::nonbondedExceptions, innerReplacements), force.getInnerForceGroup());
        }
```

In `execute`, change the NaN-marking condition at line 846 to `(includeDirect || includeReciprocal || includeInner)`. Nothing else in `execute` needs the flag: the pair and exception kernels are selected by group mask inside `NonbondedUtilities`/`BondedUtilities`, and the parameter kernel already runs whenever parameters changed.

- [ ] **Step 5: Run the GPU tests**

Run: `cd build && make -j8 && ./platforms/cuda/tests/TestCudaSlicedNonbondedForce mixed && ./platforms/cuda/tests/TestCudaSlicedNonbondedForce double && ./platforms/opencl/tests/TestOpenCLSlicedNonbondedForce mixed && ./platforms/reference/tests/TestReferenceSlicedNonbondedForce`
Expected: all pass. If `testInnerGroupPair` passes but `testInnerGroupForces` fails only for `LJOnly` or `CoulombOnly`, check the `HAS_LENNARD_JONES`/`HAS_COULOMB` guards in the new kernel blocks. If forces pass but energies fail only in single/mixed precision near r_c, the `LOG` and high inverse powers lost precision: compute `g12`, `g6`, `g1`, `g0` in `double` when `USE_DOUBLE_PRECISION` is not set is NOT an option on all devices; instead subtract `INNER_G*S` before multiplying by `INNER_PREFACTOR` (already done) and verify the host constants are emitted with full precision by `cc.doubleToString`.

- [ ] **Step 6: Full matrix and commit**

Run: `cd build && ctest --output-on-failure -j4`
Expected: every entry passes.

```bash
git add platforms/common
git commit -m "Implement the inner force group on the CUDA and OpenCL platforms"
```

---

### Task 5: Python wrapper and Python tests (including the MTS wiring test)

**Files:**
- Modify: `python/nonbondedslicing.i:394-404` (declare the six methods inside the `class SlicedNonbondedForce` block, before `getUseCuFFT`)
- Test: `python/tests/TestSlicedNonbondedForce.py`

**Interfaces:**
- Consumes: Tasks 1–4. SWIG wraps only what the `.i` declares, so without this task the methods are invisible from Python.

- [ ] **Step 1: Write the failing Python tests**

Append to `python/tests/TestSlicedNonbondedForce.py`:

```python
def _innerGroupSystem(split):
    # Lattice of charged LJ particles with 1-2 exclusions and scaled 1-4 exceptions; group 0 is direct
    # space, 1 the inner group (when split), 2 reciprocal space.
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(1.0)
    positions = []
    for i in range(4):
        for j in range(4):
            for k in range(2):
                positions.append(mm.Vec3(0.8*i + 0.1*k, 0.8*j + 0.05*k, 0.4*k))
                nonbonded.addParticle(0.5 if (i + j + k) % 2 else -0.5, 0.3, 0.5)
    for i in range(0, len(positions), 2):
        nonbonded.addException(i, i + 1, 0.0, 1.0, 0.0)
    for i in range(1, len(positions) - 1, 4):
        nonbonded.addException(i, i + 1, 0.1, 0.25, 0.2)
    force = plugin.SlicedNonbondedForce(nonbonded, 2)
    for i in range(len(positions)):
        force.setParticleSubset(i, i % 2)
    force.addGlobalParameter("lambda", 0.5)
    force.addScalingParameter("lambda", 0, 1, True, True)
    force.addEnergyParameterDerivative("lambda")
    force.setReciprocalSpaceForceGroup(2)
    if split:
        force.setInnerForceGroup(1)
        force.setInnerSwitchingDistance(0.5)
        force.setInnerCutoffDistance(0.8)
    system = mm.System()
    for _ in positions:
        system.addParticle(10.0)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3.2, 0, 0), mm.Vec3(0, 3.2, 0), mm.Vec3(0, 0, 3.2))
    system.addForce(force)
    return system, force, positions


def testInnerGroupAccessors():
    force = plugin.SlicedNonbondedForce(1)
    assert force.getInnerForceGroup() == -1
    force.setInnerForceGroup(3)
    force.setInnerCutoffDistance(0.8*unit.nanometer)
    force.setInnerSwitchingDistance(0.5*unit.nanometer)
    assert force.getInnerForceGroup() == 3
    ASSERT_EQUAL_TOL(0.8, force.getInnerCutoffDistance(), 1e-12)
    ASSERT_EQUAL_TOL(0.5, force.getInnerSwitchingDistance(), 1e-12)
    with pytest.raises(Exception):
        force.setInnerForceGroup(32)
    copy_ = plugin.SlicedNonbondedForce._deserialize(mm.XmlSerializer.serialize(force))
    assert copy_.getInnerForceGroup() == 3
    ASSERT_EQUAL_TOL(0.5, copy_.getInnerSwitchingDistance(), 1e-12)


@pytest.mark.parametrize("platformName, precision", cases)
def testInnerGroupSplit(platformName, precision):
    platform = mm.Platform.getPlatformByName(platformName)
    properties = {} if platformName == "Reference" else {"Precision": precision}
    tol = 1e-5 if platformName == "Reference" or precision == "double" else 1e-4
    contexts = {}
    for split in (True, False):
        system, force, positions = _innerGroupSystem(split)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform, properties)
        context.setPositions(positions)
        contexts[split] = context
    whole = contexts[False].getState(getEnergy=True, getForces=True, getParameterDerivatives=True)
    parts = [contexts[True].getState(getEnergy=True, getForces=True, getParameterDerivatives=True, groups={g}) for g in (0, 1, 2)]
    ASSERT_EQUAL_TOL(whole.getPotentialEnergy(), sum(value(p.getPotentialEnergy()) for p in parts), tol)
    ASSERT_EQUAL_TOL(whole.getEnergyParameterDerivatives()["lambda"], sum(p.getEnergyParameterDerivatives()["lambda"] for p in parts), tol)
    for i, force in enumerate(whole.getForces()):
        total = parts[0].getForces()[i] + parts[1].getForces()[i] + parts[2].getForces()[i]
        ASSERT_EQUAL_VEC(force, total, tol)
    inner = parts[1]
    assert abs(value(inner.getPotentialEnergy())) > 1.0
    # The inner group is linear in lambda: derivative equals E(1) - E(0).
    contexts[True].setParameter("lambda", 1.0)
    e1 = value(contexts[True].getState(getEnergy=True, groups={1}).getPotentialEnergy())
    contexts[True].setParameter("lambda", 0.0)
    e0 = value(contexts[True].getState(getEnergy=True, groups={1}).getPotentialEnergy())
    ASSERT_EQUAL_TOL(e1 - e0, inner.getEnergyParameterDerivatives()["lambda"], 10*tol)


@pytest.mark.parametrize("platformName, precision", cases)
def testInnerGroupMTSIntegration(platformName, precision):
    # Wiring test: an MTSIntegrator stepping the inner group four times per outer step runs, and the
    # group energies keep adding up to the total afterwards.
    platform = mm.Platform.getPlatformByName(platformName)
    properties = {} if platformName == "Reference" else {"Precision": precision}
    system, force, positions = _innerGroupSystem(True)
    integrator = mm.MTSIntegrator(2*unit.femtoseconds, [(2, 1), (0, 1), (1, 4)])
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(300*unit.kelvin, 1)
    integrator.step(50)
    state = context.getState(getEnergy=True)
    total = value(state.getPotentialEnergy())
    assert math.isfinite(total) and math.isfinite(value(state.getKineticEnergy()))
    parts = sum(value(context.getState(getEnergy=True, groups={g}).getPotentialEnergy()) for g in (0, 1, 2))
    tol = 1e-5 if platformName == "Reference" or precision == "double" else 1e-4
    ASSERT_EQUAL_TOL(total, parts, tol)
```

Check the `MTSIntegrator` group-tuple convention in the installed OpenMM docstring (`help(mm.MTSIntegrator)`) before relying on the order above; adjust the tuple list if the convention is "slowest first" versus "fastest first", keeping group 1 as the one with 4 substeps.

- [ ] **Step 2: Run the Python tests to verify they fail**

Run: `pytest python/tests/TestSlicedNonbondedForce.py -k InnerGroup -q`
Expected: `AttributeError: ... has no attribute 'setInnerForceGroup'`.

- [ ] **Step 3: Declare and document the methods in `python/nonbondedslicing.i`**

Insert before the `getUseCuFFT` docstring block (around line 390):

```cpp
    /**
     * Get the force group of the inner part of the direct-space interactions, or -1 if the inner
     * force group is disabled.
     */
    int getInnerForceGroup() const;
    /**
     * Set the force group of the inner part of the direct-space interactions.  When enabled, every
     * direct-space pair interaction, including 1-4 exceptions, is split into a short-range part and
     * a remainder.  The short-range part is the bare Lennard-Jones plus Coulomb force multiplied by
     * a quintic switching function that goes from 1 at the inner switching distance to 0 at the
     * inner cutoff distance; its energy is the potential whose gradient is that force, shifted to
     * vanish at the inner cutoff.  The remainder stays in the force group of this force.  Slice
     * scaling parameters and parameter offsets apply to both parts.  This is the force splitting
     * used for multiple time-step integration in :cite:`Abreu_2020`.
     *
     * The inner force group must differ from the force group of this force and from the reciprocal
     * space force group, and requires 0 < inner switching distance < inner cutoff distance <=
     * cutoff distance.  Like other force group settings, changes take effect when a new
     * :OpenMM:`Context` is created or an existing one is reinitialized.
     *
     * Parameters
     * ----------
     *     group : int
     *         the force group, between 0 and 31, or -1 to disable the inner force group
     */
    void setInnerForceGroup(int group);
    /**
     * Get the inner cutoff distance, beyond which the inner part of an interaction is zero.
     */
    double getInnerCutoffDistance() const;
    /**
     * Set the inner cutoff distance.
     *
     * Parameters
     * ----------
     *     distance : double
     *         the inner cutoff distance, measured in nm
     */
    void setInnerCutoffDistance(double distance);
    /**
     * Get the inner switching distance, where the inner part of an interaction begins to switch off.
     */
    double getInnerSwitchingDistance() const;
    /**
     * Set the inner switching distance.
     *
     * Parameters
     * ----------
     *     distance : double
     *         the inner switching distance, measured in nm
     */
    void setInnerSwitchingDistance(double distance);
```

If the file uses `%feature("docstring")` or unit typemaps for distance setters elsewhere (check how `setCutoffDistance` arrives from OpenMM's typemaps: `swig/typemaps.i` is included at the top), follow the same mechanism so `unit.Quantity` arguments are accepted; the accessor test passes nanometer quantities.

- [ ] **Step 4: Rebuild the wrapper and run**

Run: `cd build && make -j8 PythonInstall` (or the target the CMake configuration defines for the Python module; check `make help | grep -i python`), then `pytest python/tests/TestSlicedNonbondedForce.py -q`
Expected: all pass, including the three new tests on every available platform/precision.

- [ ] **Step 5: Commit**

```bash
git add python
git commit -m "Expose the inner force group in Python and test it"
```

---

### Task 6: Documentation and changelog

**Files:**
- Modify: `docs/overview.rst` (new section before "Installation", line 37)
- Modify: `docs/changelog.rst` (entry under the "0.5.0 (unreleased)" section; create it above "0.4.0" if PR #46 has not been merged yet)
- Modify: `docs/refs.bib` (add `Abreu_2020` if absent)

- [ ] **Step 1: Write the overview section**

Insert before the `Installation` heading:

```rst
Force splitting for multiple time-step integration
==================================================

A :class:`SlicedNonbondedForce` can place a short-range, force-switched copy of its direct-space
pair interactions in a separate *inner* force group, so that a multiple time-step integrator can
advance it with a small step while the remainder (the rest of direct space, the reciprocal-space
sum and its exclusion corrections) is advanced with a large one. For a pair at distance :math:`r`
with bare potential :math:`V(r)` (Lennard-Jones plus Coulomb, scaled by the slice's parameters),
the inner force is

.. math::
   \mathbf{F}^\mathrm{inner}(r) = -S(r)\,V'(r)\,\hat{\mathbf{r}}, \qquad
   S(z) = 1 - 10 z^3 + 15 z^4 - 6 z^5, \quad z = \frac{r - r_s}{r_c - r_s},

with :math:`S = 1` for :math:`r \le r_s` and :math:`S = 0` for :math:`r \ge r_c`, where
:math:`r_s` and :math:`r_c` are the inner switching and cutoff distances. The inner energy is the
potential :math:`\hat V` whose gradient is this force, shifted so that :math:`\hat V(r_c) = 0`
:cite:`Abreu_2020`; the direct-space group reports :math:`V - \hat V` and the corresponding force.
Every group therefore has an energy consistent with its force, and the groups add up to the
unsplit result. The same applies to 1-4 exceptions. Scaling parameters, their derivatives and
parameter offsets apply to both parts.

.. code-block:: python

   force = nonbondedslicing.SlicedNonbondedForce(nonbonded, 2)
   force.setForceGroup(2)               # direct-space remainder; reciprocal space goes there too
   force.setInnerForceGroup(1)
   force.setInnerSwitchingDistance(0.5*unit.nanometer)
   force.setInnerCutoffDistance(0.8*unit.nanometer)
   # bonded forces in group 0
   integrator = openmm.MTSIntegrator(90*unit.femtoseconds, [(2, 1), (1, 30), (0, 6)])

The inner force group must differ from the force group of the force and from its reciprocal-space
force group, and requires :math:`0 < r_s < r_c \le` cutoff distance. Changing these settings
requires a new :OpenMM:`Context`, as for any force group.
```

Use the tuple order verified in Task 5 for the `MTSIntegrator` example.

- [ ] **Step 2: Changelog and bibliography**

Under `0.5.0 (unreleased)` add a `**New features**` item: "Inner force group: a force-switched short-range copy of the direct-space interactions (particle pairs and 1-4 exceptions) can be computed in its own force group for multiple time-step integration, with an energy consistent with its force (``setInnerForceGroup``, ``setInnerCutoffDistance``, ``setInnerSwitchingDistance``)." Add to `docs/refs.bib`, if missing:

```bibtex
@article{Abreu_2020,
  author  = {Abreu, Charlles R. A. and Tuckerman, Mark E.},
  title   = {Molecular Dynamics with Very Large Time Steps for the Calculation of Solvation Free Energies},
  journal = {Journal of Chemical Theory and Computation},
  volume  = {16},
  number  = {12},
  pages   = {7314--7327},
  year    = {2020},
  doi     = {10.1021/acs.jctc.0c00698}
}
```

(Verify volume/pages against the DOI before committing; the DOI is certain.)

- [ ] **Step 3: Build the docs if Sphinx is available, otherwise rely on the Doc CI check**

Run: `conda run -p /data3/craabreu/envs/openmm-colvars python -c "import sphinx"` — if it imports, build with the commands in `.github/workflows/Doc.yml`; otherwise note in the PR that the Doc CI job is the verification.

- [ ] **Step 4: Commit**

```bash
git add docs
git commit -m "Document the inner force group"
```

---

### Task 7: Benchmark

**Files:**
- Modify: `devtools/benchmarks/benchmark.py`

- [ ] **Step 1: Add an `--inner-group` option**

Following the script's structure (`sliced()` builds the force, `measure()` times steps), add `parser.add_argument("--inner-group", action="store_true", help="time the inner force group (0.5-0.8 nm switch) and the direct-space remainder separately")`. When set, after building the sliced system, configure `force.setInnerForceGroup(1)`, `force.setInnerSwitchingDistance(0.5)`, `force.setInnerCutoffDistance(0.8)`, direct group 0, reciprocal group 2, and time three things with the existing `measure` pattern using a `CustomIntegrator`-free approach: `context.getState(getForces=True, groups={g})` repeated `steps` times for g in (1,), (0,), (0, 2), reporting ms per evaluation. Also add a fourth column: a `CustomNonbondedForce` built by `ufedmm.integrators.add_inner_nonbonded_force` (import guarded; skip the column if ufedmm is absent) on a plain `NonbondedForce` copy of the water system, timed the same way, so the comparison the spec asks for (Sec. 8) is in one table.

- [ ] **Step 2: Run on the water system**

Run: `conda run -p /data3/craabreu/envs/openxps-test python devtools/benchmarks/benchmark.py water --inner-group --steps 200`
Expected: a table with ms/evaluation for inner, direct remainder, direct+reciprocal, and (if available) ufedmm's CustomNonbondedForce inner group. Record the numbers in the PR description.

- [ ] **Step 3: Commit**

```bash
git add devtools/benchmarks/benchmark.py
git commit -m "Benchmark the inner force group"
```

---

## Self-review notes

- Spec coverage: API/validation (Task 1), serialization (Task 1), math + Reference (Task 2), GPU (Task 4), tests 1–8 of spec Sec. 6 (Tasks 2, 3, 5; test 4 "continuity" is in `testInnerGroupPair`; test 5 "MTS smoke" in Task 5; test 6 "errors" in Task 1; test 7 "serialization" in Tasks 1 and 5), docs (Task 6), benchmark (Task 7).
- Rebase onto `main` after PR #46 merges before opening the PR; resolve the duplicated `isExcluded` slice guard in `coulombLennardJones.cc` by keeping one copy, and merge the changelog sections.
- Names used consistently across tasks: `getInnerForceGroup/setInnerForceGroup`, `getInnerCutoffDistance/setInnerCutoffDistance`, `getInnerSwitchingDistance/setInnerSwitchingDistance`, `includeInner`, `InnerSplit` (`rs, rc, p[5], prefactor, gs[3], g0s, K[3], n[3], switchValue, G0, G, perturbation, energyTerm`), kernel defines `INNER_GROUP`, `SUBTRACT_INNER`, `INNER_CUTOFF_SQUARED`, `INNER_SWITCH`, `INNER_SWITCH_SCALE`, `INNER_PREFACTOR`, `INNER_P0..4`, `INNER_G12S/G6S/G1S/G0S`, `INNER_K12/K6/K1`, Reference `calculateInnerPair`, `calculateInnerIxn`, `setInnerSplit`, `ReferenceSlicedLJCoulomb14::Mode {Full, Inner, Remainder}`.
