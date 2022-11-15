const float charge = CHARGE_PRODS[index];
int slice = SLICES[index];
real lambda = LAMBDAS[slice];
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if APPLY_PERIODIC
    APPLY_PERIODIC_TO_DELTA(delta)
#endif
const real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
const real r = SQRT(r2);
const real invR = RECIP(r);
const real alphaR = EWALD_ALPHA*r;
const real expAlphaRSqr = EXP(-alphaR*alphaR);
real tempForce, tempEnergy;
if (alphaR > 1e-6f) {
    const real erfAlphaR = ERF(alphaR);
    const real prefactor = -charge*invR;
    tempForce = prefactor*(erfAlphaR-alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
    tempEnergy = prefactor*erfAlphaR;
}
else {
    tempForce = 0.0f;
    tempEnergy = -TWO_OVER_SQRT_PI*EWALD_ALPHA*charge;
}
energy += lambda*tempEnergy;
if (r > 0)
    delta *= lambda*tempForce*invR*invR;
real3 force1 = -delta;
real3 force2 = delta;
#if HAS_DERIVATIVES
    int derivIndex = DERIV_INDICES[slice];
    if (derivIndex != -1)
        energyParamDerivs[GLOBAL_ID*NUM_ALL_DERIVS+derivIndex] += tempEnergy;
#endif
