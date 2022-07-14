const float exclusionChargeProds = PARAMS[index];
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if USE_PERIODIC
    APPLY_PERIODIC_TO_DELTA(delta)
#endif
const real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
const real r = SQRT(r2);
const real invR = RECIP(r);
const real alphaR = EWALD_ALPHA*r;
const real expAlphaRSqr = EXP(-alphaR*alphaR);
real tempForce = 0.0f;
if (alphaR > 1e-6f) {
    const real erfAlphaR = ERF(alphaR);
    const real prefactor = exclusionChargeProds*invR;
    tempForce = -prefactor*(erfAlphaR-alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
    energy -= prefactor*erfAlphaR;
}
else {
    energy -= TWO_OVER_SQRT_PI*EWALD_ALPHA*exclusionChargeProds;
}
if (r > 0)
    delta *= tempForce*invR*invR;
real3 force1 = -delta;
real3 force2 = delta;
