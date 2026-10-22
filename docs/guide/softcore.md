# Soft-core Lennard-Jones potential

## What it does

Driving the Lennard-Jones scaling parameter of a slice to zero with the usual potential leaves a
singularity at $r = 0$ that makes free-energy calculations hard to converge. A scaling parameter added with
a positive `softcoreAlpha` switches its slice to the soft-core form of {cite}`Beutler_1994`, in which each
pair interacts through

$$
U(r) = 4 \lambda_E \epsilon \left( s^2 - s \right), \qquad
s = \frac{1}{\alpha (1 - \lambda_s) + (r/\sigma)^6},
$$

where $\lambda_E$ is the scaling parameter that multiplies the Lennard-Jones contributions of the slice
(1 if there is none), $\lambda_s$ is the softening parameter and $\alpha$ is `softcoreAlpha`, typically
0.5. The potential remains finite as $r \to 0$ whenever $\lambda_s < 1$ and $\alpha > 0$, and it reduces
to the usual one when $\lambda_s = 1$ or $\alpha = 0$.

## Setting it up

The energy and softening parameters can be the same,

```python
force.addScalingParameter("lambda", 0, 1, False, True, 0.5)
```

or different,

```python
force.addScalingParameter("lambdaE", 0, 1, False, True)
force.addScalingParameter("lambdaS", 0, 1, False, False, 0.5)
```

and a slice can be softened without scaling its energy at all (only the second line above). Since only the
Lennard-Jones term is softened, scale the Coulomb contribution of the slice with its own parameter
(`includeCoulomb=True, includeLJ=False`) and turn it off before softening the Lennard-Jones term. Like the
other properties of a scaling parameter, $\alpha$ is fixed when a {OpenMM}`Context` is created, while
$\lambda_E$ and $\lambda_s$ are its global parameters.

The soft-core form applies to the direct-space Lennard-Jones term and to 1-4 exceptions, including the
additive potential shift at the cutoff with LJPME. The reciprocal-space part of LJPME and the long-range
dispersion correction are not softened, an approximation accurate to
$O\left(\alpha (1-\lambda_s) (\sigma/r_c)^6\right)$.

## Restrictions

* Keep $\lambda_s$ between 0 and 1. For $\lambda_s > 1$, the term $\alpha (1 - \lambda_s)$ is negative and
  the potential becomes singular at $r = \left[ \alpha (\lambda_s - 1) \right]^{1/6} \sigma$.
* Only one scaling parameter per slice can soften it.
* As with the usual potential, do not combine LJPME with a switching function: the Reference platform
  ignores the switching function, while the CUDA and OpenCL platforms apply it to the direct-space
  Lennard-Jones term, as OpenMM's NonbondedForce does on each platform.
* Soft-core slices cannot be combined with an inner force group ({doc}`mts`), whose split assumes the
  usual Lennard-Jones potential.

## What the derivatives mean

The derivatives with respect to the two parameters are

$$
\frac{\partial U}{\partial \lambda_E} = 4 \epsilon (s^2 - s), \qquad
\frac{\partial U}{\partial \lambda_s} = 4 \lambda_E \epsilon \alpha s^2 (2 s - 1),
$$

and a parameter that plays both roles gets their sum. The first is the soft-core van der Waals energy of
the slice, which differs from the usual one whenever $\lambda_s < 1$; the second is not an energy, and
neither is their sum, even at $\lambda = 1$. The table in {doc}`derivatives` states what is reported in
each case.
