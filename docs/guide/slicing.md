# Subsets, slices and scaling parameters

## What it does

A `SlicedNonbondedForce` computes the same Coulomb and Lennard-Jones interactions as a
{OpenMM}`NonbondedForce`, with the same nonbonded methods, cutoffs, exceptions and parameter offsets. In
addition, each particle belongs to one of $n$ subsets, and the energy is sliced into $n(n+1)/2$ parts,
one for each pair of subsets $I \le J$: slice $I, J$ comprises all pairs formed by a particle in subset
$I$ and a particle in subset $J$. The Coulomb and Lennard-Jones contributions of every slice can be
multiplied by a *scaling parameter*, the value of a {OpenMM}`Context` global parameter:

$$
E = \sum_{I \le J} \left( \lambda^{vdW}_{I,J} E^{vdW}_{I,J} + \lambda^{elec}_{I,J} E^{elec}_{I,J} \right).
$$

Scaling parameters are constant and equal to 1 unless you say otherwise, so a force with no scaling
parameters yields the energy of the NonbondedForce it replaces. Lennard-Jones interactions use the
Lorentz-Berthelot combining rule: the arithmetic mean of the sigmas and the geometric mean of the epsilons
of the two particles.

## Setting it up

A `SlicedNonbondedForce` is created either from scratch, by specifying the number of subsets and calling
`addParticle` for every particle, or from an existing NonbondedForce, whose particles, exceptions, global
parameters, offsets and settings it inherits. Every particle starts in subset 0; `setParticleSubset` moves
it. A scaling parameter is a global parameter, declared with `addGlobalParameter`, whose name is passed to
`addScalingParameter` together with the slice it applies to and the contributions it multiplies:

```python
import openmm as mm
import nonbondedslicing as nbs

force = nbs.SlicedNonbondedForce(nonbonded, 2)   # built from a NonbondedForce, 2 subsets
for atom in ligand_atoms:
    force.setParticleSubset(atom, 1)             # the rest stays in subset 0
force.addGlobalParameter("lambda", 1.0)
force.addScalingParameter("lambda", 0, 1, True, True)   # scale Coulomb and LJ of slice (0, 1)
system.removeForce(index)                        # index of the NonbondedForce being replaced
system.addForce(force)

context = mm.Context(system, integrator)
context.setParameter("lambda", 0.5)              # halve the ligand-environment interactions
```

The order of the two subsets in `addScalingParameter` is irrelevant. The same parameter can be applied to
any number of slices, and a slice can take one parameter for its Coulomb part and another for its
Lennard-Jones part. A sixth argument, `softcoreAlpha`, makes a parameter also soften the Lennard-Jones
interactions of its slice; see {doc}`softcore`.

## Restrictions

* A scaling parameter must have been added with `addGlobalParameter`, and it cannot double as the
  parameter of a particle or exception parameter offset.
* Every call to `addScalingParameter` must give the parameter at least one role in the slice: Coulomb
  (`includeCoulomb`), Lennard-Jones (`includeLJ`) or softening (`softcoreAlpha > 0`), and each role of a
  slice belongs to at most one parameter. A second parameter for a role that is already taken is an
  error.
* The set of scaling parameters, their roles and their `softcoreAlpha` values are read when a Context is
  created. Changes made afterwards with `addScalingParameter` or `setScalingParameter` take effect only
  after the Context's `reinitialize()` method is called. The *values* of the parameters, in contrast, are
  read at every evaluation and can be changed at any time with the Context's `setParameter` method.
* Particle parameters, exception parameters and particle subsets can be changed in an existing Context with
  `updateParametersInContext`, with the same limitations as in NonbondedForce: other settings, such as the
  nonbonded method and the cutoff, are not updated (reinitialize the Context instead), the pair of
  particles of an exception cannot change, and particles or exceptions cannot be added.
