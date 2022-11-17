float chargeProd = CHARGE_PRODS[index];
int slice = SLICES[index];
real lambda = LAMBDAS[slice];
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if APPLY_PERIODIC
    APPLY_PERIODIC_TO_DELTA(delta)
#endif
real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
real invR = RSQRT(r2);
real dEdR = chargeProd*invR;
dEdR *= lambda*invR*invR;
real tempEnergy = chargeProd*invR;
energy += lambda*tempEnergy;
delta *= dEdR;
real3 force1 = -delta;
real3 force2 = delta;

COMPUTE_DERIVATIVES