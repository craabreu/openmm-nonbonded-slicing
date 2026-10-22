# Energy parameter derivatives

## What it does

`addEnergyParameterDerivative` requests the derivative of the force's energy with respect to a scaling
parameter. The derivatives are returned by the `getEnergyParameterDerivatives` method of a
{OpenMM}`State` obtained with both `getEnergy=True` and `getParameterDerivatives=True`:

```python
force.addGlobalParameter("lambda", 1.0)
force.addScalingParameter("lambda", 0, 1, True, True)
force.addEnergyParameterDerivative("lambda")
...
state = context.getState(getEnergy=True, getParameterDerivatives=True)
energy_01 = state.getEnergyParameterDerivatives()["lambda"]
```

Only scaling parameters qualify: requesting the derivative with respect to another global parameter raises
an exception.

## What the derivatives mean

For a parameter that multiplies the contributions of slices with the usual Lennard-Jones potential, the
energy is linear in the parameter, and its derivative is the sum of the unscaled contributions it
multiplies,

$$
\frac{\partial E}{\partial \lambda} = \sum_{(I,J) \in S^{elec}_\lambda} E^{elec}_{I,J}
+ \sum_{(I,J) \in S^{vdW}_\lambda} E^{vdW}_{I,J},
$$

where $S^{elec}_\lambda$ and $S^{vdW}_\lambda$ are the sets of slices whose electrostatic and van der
Waals contributions are scaled by $\lambda$. This derivative does not depend on the values of the scaling
parameters. A parameter that scales both contributions of a single slice $I, J$ yields its full energy
$E^{vdW}_{I,J} + E^{elec}_{I,J}$, while one that scales the electrostatic contributions of several slices
yields their sum. This is how the energy of a slice, or of a group of slices, can be reported.

A parameter that softens a slice ({doc}`softcore`) makes the energy non-linear in it. For a pair of the
slice, the derivatives with respect to the energy and softening parameters are

$$
\frac{\partial U}{\partial \lambda_E} = 4 \epsilon (s^2 - s), \qquad
\frac{\partial U}{\partial \lambda_s} = 4 \lambda_E \epsilon \alpha s^2 (2 s - 1),
$$

and the reported derivative depends on the roles the parameter plays:

| Parameter | Reported derivative |
|---|---|
| Multiplies the Coulomb contributions of any slices, or the Lennard-Jones contributions of unsoftened slices | The sum of the unscaled contributions it multiplies, independent of the parameter values |
| Multiplies the Lennard-Jones contribution of a softened slice, with a separate softening parameter | $\partial U / \partial \lambda_E$ summed over its pairs: the soft-core van der Waals energy of the slice, which differs from the usual energy whenever $\lambda_s < 1$ |
| Softens a slice | $\partial U / \partial \lambda_s$ summed over its pairs, which is not an energy |
| Multiplies the Lennard-Jones contribution of a slice and softens it | The sum of the two rows above, which is not an energy, even when the parameter is 1 and the soft-core and usual potentials coincide |

The pair terms in the last three rows are multiplied by the switching function when it is used and, with
LJPME, have the same term evaluated at the cutoff subtracted. To obtain the usual van der Waals energy of a
softened slice, evaluate it in a separate force without the soft-core potential.

## Restrictions

```{important}
Derivatives are only computed along with the energy. A State requested with
`getParameterDerivatives=True` but without `getEnergy=True` holds NaN for every requested derivative, on
all platforms. If other forces in the System also contribute derivatives with respect to the same
parameter, their sum is NaN too. In exchange, steps that compute only forces, such as regular molecular
dynamics steps, do not pay for the derivatives. Expressions like `deriv(energy, lambda)` in a
{OpenMM}`CustomIntegrator` always compute the energy, so they work as usual.
```

The cost of an evaluation with derivatives grows with the number of requested derivatives, not with the
number of slices, since a parameter that scales several slices adds a single derivative; see
{doc}`../performance`. Request derivatives only for the parameters whose energies you need.
