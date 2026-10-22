# Reciprocal-space force group and exclusion corrections

## What it does

With Ewald, PME or LJPME, the reciprocal-space sum can be computed in a separate force group by calling
`setReciprocalSpaceForceGroup`, as in OpenMM's {OpenMM}`NonbondedForce`. A multiple time-step integrator
can then evaluate the reciprocal-space sum, which varies slowly, less often than the direct-space
interactions.

The reciprocal-space sum implicitly includes the interactions of excluded pairs and exceptions, and the
terms that subtract them are, by default, computed in the direct-space force group, as NonbondedForce does.
Since these terms only cancel part of the reciprocal-space sum, they should be integrated along with it
whenever the two groups are evaluated at different times; otherwise, the cancellation is incomplete and the
slow force acquires a fast, intramolecular component {cite}`Procacci_1998,Morrone_2010`. Calling
`setReciprocalExclusions(True)` moves these terms to the reciprocal-space force group.

## Setting it up

```python
force.setNonbondedMethod(nbs.SlicedNonbondedForce.PME)
force.setForceGroup(1)                   # direct space
force.setReciprocalSpaceForceGroup(2)    # reciprocal space
force.setReciprocalExclusions(True)      # and the exclusion corrections with it
```

## Restrictions

* `setReciprocalExclusions` only matters with Ewald, PME or LJPME, when the reciprocal-space force group
  differs from the direct-space one. The default, `False`, keeps the behavior of NonbondedForce.
* With `setReciprocalExclusions(True)`, the exclusion corrections are also kept when direct space is turned
  off with `setIncludeDirectSpace(False)`. A replacement for the direct-space calculation, such as a
  {OpenMM}`CustomNonbondedForce`, must then not include them, or they are counted twice.
