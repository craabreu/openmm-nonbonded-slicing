# Capped Lennard-Jones potential

## What it does

Driving the Lennard-Jones scaling parameter of a slice to zero with the usual potential leaves a singularity
at $r = 0$ that makes free-energy calculations hard to converge. A scaling parameter added with
`includeDecapping=True` multiplies the *decapping* term of its slice, which splits each Lennard-Jones pair
of the slice into a capped term and a decapping term {cite}`Correa_2022`, in the spirit of the capping of
{cite}`Buelens_2012` and the linear basis functions of {cite}`Naden_2014`:

$$
U(r) = \lambda_{LJ}\, u_{cap}(r) + \lambda_D \left[ u_{LJ}(r) - u_{cap}(r) \right],
$$

where $u_{LJ}$ is the usual potential, $\lambda_{LJ}$ is the scaling parameter that multiplies the
Lennard-Jones contributions of the slice (1 if there is none), $\lambda_D$ is the decapping parameter, and

$$
u_{cap}(r) = \begin{cases}
4 \epsilon \left( 29.8 - 360 x^4 + 547.2 x^5 - 217 x^6 \right), & x = r/\sigma < 1 \\
u_{LJ}(r), & r \ge \sigma
\end{cases}
$$

is a polynomial core that joins the usual potential smoothly (up to the second derivative) at $r = \sigma$
and takes the finite value $119.2\,\epsilon$ at $r = 0$. The decapping term vanishes for $r \ge \sigma$, and
$\lambda_D = \lambda_{LJ}$ gives the usual potential.

The energy is linear in both parameters. The derivative with respect to $\lambda_{LJ}$ is the capped van der
Waals energy of the slice, the one with respect to $\lambda_D$ is its decapping energy, and their sum is the
usual van der Waals energy ({doc}`derivatives`). One evaluation therefore gives the energy at any pair of
values, which is what reweighting estimators such as MBAR need.

## Setting it up

```python
force.addGlobalParameter("lambdaQ", 1.0)
force.addGlobalParameter("lambdaLJ", 1.0)
force.addGlobalParameter("lambdaD", 1.0)
force.addScalingParameter("lambdaQ", 0, 1, True, False)
force.addScalingParameter("lambdaLJ", 0, 1, False, True)
force.addScalingParameter("lambdaD", 0, 1, False, False, True)
```

A coupling path switches on the capped term first ($\lambda_{LJ}$ from 0 to 1 with $\lambda_D = 0$), then
the decapping term, and finally the Coulomb term. The stages can overlap: {cite}`Correa_2022` couples all
three in one sweep of a master parameter $\lambda$ with smoothstep functions $h_k(\lambda)$ that start and
finish at $(0, 0.7)$, $(0.4, 0.8)$ and $(0.6, 1.0)$ for the capped, decapping and Coulomb terms. OpenMM
cannot make one global parameter a function of another, so set `lambdaLJ`, `lambdaD` and `lambdaQ` from
$\lambda$ at each state; for $\lambda$-dynamics, apply the chain rule to the three derivatives.

The capped term is the usual potential beyond $\sigma$, so the cutoff, the switching function, LJPME and the
dispersion correction apply to it exactly and are multiplied by $\lambda_{LJ}$. The decapping term is not
switched, so with a switching function the $\sigma$ of every Lennard-Jones pair of a decapped slice (pairs
with $\epsilon = 0$ excluded) cannot exceed the switching distance; this is checked when a Context is
created and by `updateParametersInContext`, but not for parameter offsets. Decapping is available on all
platforms and with every nonbonded method, for direct-space pairs and 1-4 exceptions. Like the other
properties of a scaling parameter, the decapping role is fixed when a {OpenMM}`Context` is created, while
$\lambda_{LJ}$ and $\lambda_D$ are its global parameters.

## Restrictions

* Only one scaling parameter per slice can multiply its decapping term.
* The capped core is finite, so charges of opposite sign can collapse onto it. Turn the Coulomb contribution
  of a slice on only while $\lambda_D > 0$.
* The height of the core, $119.2\,\epsilon$, is only a few $k_B T$ for atoms with small $\epsilon$, which
  then overlap often at $\lambda_D = 0$. The decapping energy of such overlaps is large, so place enough
  states near $\lambda_D = 0$ to keep the estimates converged.
* With an inner force group ({doc}`mts`), the inner group takes all of the capped and decapping terms below
  $\sigma$. The $\sigma$ of every Lennard-Jones pair of a decapped slice (pairs with $\epsilon = 0$
  excluded) cannot exceed the inner switching distance. This is checked when a Context is created and by
  `updateParametersInContext`, but not for parameter offsets.
