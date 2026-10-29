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
4 \epsilon\, c(r/\sigma), & r < \sigma \\
u_{LJ}(r), & r \ge \sigma
\end{cases}
$$

is a polynomial core. It is defined by two orders, $m_0$ and $m_1$: $c(x)$ is the polynomial of degree
$m_0 + m_1$ whose first $m_0$ derivatives vanish at $x = 0$ and whose value and first $m_1$ derivatives
match those of $x^{-12} - x^{-6}$ at $x = 1$. The first condition leaves the terms $a_0$ and $a_k x^k$ for
$k = m_0 + 1, \dots, m_0 + m_1$, and the second fixes their $m_1 + 1$ coefficients. The core thus joins the
usual potential smoothly, up to the $m_1$-th derivative, at $r = \sigma$ and takes the finite value
$4 \epsilon\, a_0$ at $r = 0$. The default orders, $m_0 = m_1 = 3$, give

$$
c(x) = 29.8 - 360 x^4 + 547.2 x^5 - 217 x^6,
$$

with a core of $119.2\,\epsilon$. The decapping term vanishes for $r \ge \sigma$, and
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

The orders are set for the whole force with `setCappingOrders`, from 1 to 6 each. Lowering $m_0$ or raising
$m_1$ raises the core, $4 \epsilon\, a_0$, given here in units of $\epsilon$ (rows: $m_0$; columns: $m_1$):

| | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **1** | 12.0 | 92.0 | 440.0 | 1640.0 | 5184.0 | 14523.4 |
| **2** | 8.0 | 50.0 | 206.0 | 684.0 | 1969.7 | 5108.1 |
| **3** | 6.0 | 32.4 | 119.2 | 361.3 | 964.4 | 2345.7 |
| **4** | 4.8 | 23.2 | 78.1 | 219.7 | 550.7 | 1268.7 |
| **5** | 4.0 | 17.7 | 55.4 | 146.7 | 348.7 | 766.8 |
| **6** | 3.4 | 14.1 | 41.7 | 104.7 | 237.8 | 502.3 |

```python
force.setCappingOrders(2, 4)    # a core of 684 epsilon
```

## Coulomb decapping

The capped core is finite while the Coulomb attraction of opposite charges is not, so with the decapping term off
they can collapse onto it wherever the Coulomb term is on. One answer is to keep the Coulomb term off until the
decapping term is on; the other is to cap the Coulomb interaction too:

```python
force.setCoulombDecapping(True)
```

With this option, the Coulomb term of every pair in a decapped slice, whatever its $\epsilon$, is capped below
$r = \sigma$ in the same way as the Lennard-Jones one. The $1/r$ part of its direct-space potential is replaced by
$c_Q(r/\sigma)/\sigma$, where $c_Q(x)$ is the polynomial of the same orders $m_0$ and $m_1$ whose first $m_0$
derivatives vanish at $x = 0$ and whose value and first $m_1$ derivatives match those of $1/x$ at $x = 1$. The
default orders give

$$
c_Q(x) = 1.75 - 5.25 x^4 + 7 x^5 - 2.5 x^6,
$$

a core 1.75 times the bare potential at $r = \sigma$. The cap is a statement about $1/r$ alone, as the Lennard-Jones
one is about $x^{-12} - x^{-6}$: the rest of the direct-space potential, which is $-\mathrm{erf}(\alpha r)/r$ with
Ewald, PME and LJPME or the reaction-field terms with a cutoff, and the reciprocal space are unchanged, and the
decapping term of the slice gains

$$
\frac{q_i q_j}{4 \pi \epsilon_0} \left[ \frac{1}{r} - \frac{c_Q(r/\sigma)}{\sigma} \right]
$$

for $r < \sigma$, which is the same for every nonbonded method and for exceptions. The potential is then bounded
below everywhere in the $(\lambda_{LJ}, \lambda_D, \lambda_Q)$ cube, so the three terms can be coupled in any
order. The energy stays linear in every parameter: the derivative with respect to $\lambda_Q$ is the capped Coulomb
energy of the slice (its capped direct-space part plus the reciprocal one), and the one with respect to $\lambda_D$
is its decapping energy, Coulomb part included.

Since every pair of a decapped slice is now capped, $\sigma$ matters for particles with $\epsilon = 0$ too, and the
sigma restriction below covers them. A TIP3P hydrogen, for instance, carries a placeholder $\sigma = 1$ nm that
exceeds any switching distance; set it to the Coulomb cap radius you want (its $\epsilon = 0$ keeps it out of every
Lennard-Jones term), such as zero, which makes $\sigma_{ij}$ half the oxygen's for its pairs. For opposite charges
the capped Coulomb potential is a finite well, deepest at $r = 0$, so decapping should still not lag too far behind
the Coulomb term; raising $m_0$ lowers the core (1.33 at $m_0 = 5$, $m_1 = 2$).

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
properties of a scaling parameter, the decapping role and the capping orders are fixed when a
{OpenMM}`Context` is created, while $\lambda_{LJ}$ and $\lambda_D$ are its global parameters.

## Restrictions

* Only one scaling parameter per slice can multiply its decapping term.
* The capped core is finite, so charges of opposite sign can collapse onto it. Turn the Coulomb contribution
  of a slice on only while $\lambda_D > 0$, or cap the Coulomb interaction too with `setCoulombDecapping`.
* With Coulomb decapping, the $\sigma$ of every particle of the subsets of a decapped slice and of every exception
  with a nonzero charge product, those with $\epsilon = 0$ included, cannot exceed the switching distance.
  Coulomb decapping cannot be combined with an inner force group.
* With the default orders, the height of the core, $119.2\,\epsilon$, is only a few $k_B T$ for atoms with
  small $\epsilon$, which then overlap often at $\lambda_D = 0$. The decapping energy of such overlaps is
  large, so raise the core with `setCappingOrders` or place enough states near $\lambda_D = 0$ to keep the
  estimates converged. Higher orders give larger coefficients that partly cancel near $r = \sigma$, which
  costs some single-precision accuracy there.
* With an inner force group ({doc}`mts`), the inner group takes all of the capped and decapping terms below
  $\sigma$. The $\sigma$ of every Lennard-Jones pair of a decapped slice (pairs with $\epsilon = 0$
  excluded) cannot exceed the inner switching distance. This is checked when a Context is created and by
  `updateParametersInContext`, but not for parameter offsets.
