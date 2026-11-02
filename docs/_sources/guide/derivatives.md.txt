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

For a parameter that multiplies the contributions of slices, the energy is linear in the parameter, and its
derivative is the sum of the unscaled contributions it multiplies,

$$
\frac{\partial E}{\partial \lambda} = \sum_{(I,J) \in S^{elec}_\lambda} E^{elec}_{I,J}
+ \sum_{(I,J) \in S^{vdW}_\lambda} E^{vdW}_{I,J},
$$

where $S^{elec}_\lambda$ and $S^{vdW}_\lambda$ are the sets of slices whose electrostatic and van der
Waals contributions are scaled by $\lambda$. This derivative does not depend on the values of the scaling
parameters. A parameter that scales both contributions of a single slice $I, J$ yields its full energy
$E^{vdW}_{I,J} + E^{elec}_{I,J}$, while one that scales the electrostatic contributions of several slices
yields their sum. This is how the energy of a slice, or of a group of slices, can be reported.

The energy remains linear with decapped slices ({doc}`decapping`). The derivative with respect to the
Lennard-Jones scaling parameter of a decapped slice is its capped van der Waals energy, and the derivative
with respect to its decapping parameter is its decapping energy; their sum is the usual van der Waals energy
of the slice. A parameter that plays both roles in a slice yields the usual van der Waals energy, as it
would without decapping. With Coulomb decapping, the derivative with respect to the Coulomb scaling parameter
is the capped Coulomb energy of the slice and the one with respect to its Coulomb decapping parameter is its
Coulomb decapping energy, so that the derivatives still sum to the usual energy of the slice.

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
