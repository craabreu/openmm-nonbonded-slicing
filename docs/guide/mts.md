# Inner force group for multiple time stepping

## What it does

A `SlicedNonbondedForce` can place a short-range, force-switched copy of its direct-space pair
interactions in a separate *inner* force group, so that a multiple time-step integrator can advance it
with a small step while the remainder (the rest of direct space, the reciprocal-space sum and its exclusion
corrections) is advanced with a large one. For a pair at distance $r$ with bare potential $V(r)$
(Lennard-Jones plus Coulomb, scaled by the slice's parameters), the inner force is

$$
\mathbf{F}^\mathrm{inner}(r) = -S(r)\,V'(r)\,\hat{\mathbf{r}}, \qquad
S = 1 - 10 z^3 + 15 z^4 - 6 z^5, \quad z = \frac{r - r_s}{r_c - r_s},
$$

with $S = 1$ for $r \le r_s$ and $S = 0$ for $r \ge r_c$, where $r_s$ and $r_c$ are the inner switching
and cutoff distances. The inner energy is the potential $\hat V$ whose gradient is this force, shifted so
that $\hat V(r_c) = 0$ {cite}`Abreu_2020`; the direct-space group reports $V - \hat V$ and the
corresponding force. Every group therefore has an energy consistent with its force, and the groups add up
to the unsplit result. The same applies to 1-4 exceptions, and scaling parameters, their derivatives and
parameter offsets apply to both parts. The closed form of $\hat V$ used by all platforms is derived in
{doc}`../theory`.

## Setting it up

```python
force = nbs.SlicedNonbondedForce(nonbonded, 2)
force.setForceGroup(2)               # direct-space remainder; reciprocal space goes there too
force.setInnerForceGroup(1)
force.setInnerSwitchingDistance(0.5*unit.nanometer)
force.setInnerCutoffDistance(0.8*unit.nanometer)
# bonded forces in group 0; steps of 90, 3 and 0.5 fs
integrator = mm.MTSIntegrator(90*unit.femtoseconds, [(2, 1), (1, 30), (0, 180)])
```

With Ewald, PME or LJPME, consider computing the exclusion corrections along with the reciprocal-space sum
({doc}`reciprocal`), so that the slow group has no fast intramolecular component.

## Restrictions

* The inner force group must differ from the force group of the force and, with Ewald, PME or LJPME, from
  its reciprocal-space force group.
* The distances must satisfy $0 < r_s < r_c \le$ cutoff distance and, when the Lennard-Jones switching
  function is used, $r_c \le$ switching distance: the inner force is the bare Lennard-Jones force times
  $S(r)$, which beyond the switching distance would exceed the switched force and leave the remainder with
  a correction of opposite sign.
* Since the split assumes the usual Lennard-Jones potential, the inner force group cannot be used with
  soft-core scaling parameters ({doc}`softcore`).
* The inner force group, switching distance and cutoff distance are fixed when a {OpenMM}`Context` is
  created; changing them afterwards takes effect only after the Context's `reinitialize()` method is
  called.
* On the CUDA and OpenCL platforms, all nonbonded interactions in one force group must use the same
  cutoff, so a Context fails to be created if the inner force group also contains another cutoff-based
  nonbonded force, such as a {OpenMM}`CustomNonbondedForce`, with a different cutoff. Put the inner force
  group in a group of its own, or in one whose other nonbonded forces use the inner cutoff.

## Performance

On the CUDA and OpenCL platforms, the cost of an inner-group evaluation depends little on the inner cutoff,
because the inner kernels traverse the neighbor list built for the full cutoff and only skip the pairs
beyond $r_c$. In a box of 50,442 water atoms with a 1.2 nm cutoff, an inner-group evaluation on an NVIDIA
RTX 4090 took 0.167 ms with $r_c = 0.5$ nm and 0.188 ms with $r_c = 1.2$ nm, against 0.21–0.23 ms for
the direct-space group alone. The inner group is therefore not much cheaper than the direct space it is
carved from, which limits the speedup of multiple time stepping; with the inner force group disabled, a
step costs the same as without the feature.
