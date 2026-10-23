# Overview

This OpenMM plugin implements `SlicedNonbondedForce`, a variant of OpenMM's {OpenMM}`NonbondedForce`.
By partitioning all particles among $n$ disjoint subsets, the total nonbonded potential energy becomes a
linear combination of contributions from pairs of subsets,

$$
E = \sum_{I=0}^{n-1} \sum_{J=I}^{n-1}
\left( \lambda^{vdW}_{I,J} E^{vdW}_{I,J} + \lambda^{elec}_{I,J} E^{elec}_{I,J} \right),
$$

where each *slice* is defined by subsets $I$ and $J$, superscripts *vdW* and *elec* denote van der Waals
and electrostatic contributions, $E_{I,J}$ is the potential energy of all particle pairs formed by one
particle in subset $I$ and one in subset $J$, and $\lambda_{I,J}$ is a scaling parameter. With this
decomposition in place, the plugin offers four capabilities.

**Scaling slices.** By default, every scaling parameter is 1 and the force yields the same energy as the
NonbondedForce it replaces. Selected scaling parameters can instead be tied to {OpenMM}`Context` global
parameters, so that the van der Waals and electrostatic contributions of chosen slices are scaled up or
down while a simulation runs. See {doc}`guide/slicing`.

**Soft-core Lennard-Jones potential.** A scaling parameter can also soften the Lennard-Jones interactions
of its slice with the Beutler soft-core potential, which remains finite as particles overlap, so that a
slice can be decoupled alchemically without the singularity of the usual potential. See
{doc}`guide/softcore`.

**Energy parameter derivatives.** The derivative of the energy with respect to a scaling parameter can be
requested along with the energy, which gives the energy of a slice or of a group of slices. See
{doc}`guide/derivatives`.

**Force-group splitting.** With Ewald, PME and LJPME, the reciprocal-space sum can be computed in its own
force group, together with the terms that correct it for excluded pairs and exceptions
({doc}`guide/reciprocal`). With any nonbonded method, a short-range, force-switched copy of the
direct-space interactions can be computed in an *inner* force group, so that a multiple time-step
integrator advances it with a smaller step than the rest ({doc}`guide/mts`).

The formulation behind the sliced lattice sums and the inner force group is in {doc}`theory`, measured
costs are in {doc}`performance`, and the class is documented in {doc}`pythonapi/index`.
