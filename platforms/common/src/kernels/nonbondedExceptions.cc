float4 exceptionParams = PARAMS[index];
union {int i; float f;} u;
u.f = exceptionParams.w;
int slice = u.i;
real clLambda = LAMBDAS[slice].x;
real ljLambda = LAMBDAS[slice].y;
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if APPLY_PERIODIC
APPLY_PERIODIC_TO_DELTA(delta)
#endif
real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
real invR = RSQRT(r2);
real sig2 = invR*exceptionParams.y;
sig2 *= sig2;
real sig6 = sig2*sig2*sig2;
#if HAS_SOFTCORE
real softcoreAlpha = SOFTCORE_ALPHAS[slice];
real softLambda = SOFTCORE_LAMBDAS[slice];
real q = RECIP(1.0f + softcoreAlpha*(1.0f-softLambda)*sig6);
real s = sig6*q;
real epss = exceptionParams.z*s;
real ljForce = 6.0f*epss*(2.0f*s-1.0f)*q;
real ljEnergy = epss*(s-1.0f);
real ljSoftDeriv = ljLambda*softcoreAlpha*epss*s*(2.0f*s-1.0f);
#else
real ljForce = exceptionParams.z*(12.0f*sig6-6.0f)*sig6;
real ljEnergy = exceptionParams.z*(sig6-1.0f)*sig6;
#endif
real clForce = exceptionParams.x*invR;
real clEnergy = exceptionParams.x*invR;
#if INNER_GROUP || SUBTRACT_INNER
// Inner part: S(r) times the bare force and the shifted potential whose gradient it is. The inner
// group cannot be used with soft-core slices, so HAS_SOFTCORE is 0 here and ljForce is bare.
real innerForceLJ = 0.0f, innerForceCl = 0.0f, innerEnergyLJ = 0.0f, innerEnergyCl = 0.0f;
if (r2 < INNER_CUTOFF_SQUARED) {
    const real r = SQRT(r2);
    real innerSwitch = 1.0f;
    if (r > INNER_SWITCH) {
        const real z = (r-INNER_SWITCH)*INNER_SWITCH_SCALE;
        innerSwitch = 1.0f+z*z*z*(-10.0f+z*(15.0f-z*6.0f));
    }
    // Shifted inner energy terms e_12, e_6 and e_1 (see InnerSplit.h): with x = (rc-r)/r, the Lennard-Jones terms
    // are x^4 times a polynomial in x; the Coulomb term uses Y = 1-r/rc and the positive series L_m(Y) = sum(Y^j/j, j>=m).
    real e12, e6, e1;
    if (r > INNER_SWITCH) {
        const real x = (INNER_CUTOFF-r)*invR;
        real x4 = x*x;
        x4 *= x4;
        e12 = x4*(INNER_A12_0+x*(INNER_A12_1+x*(INNER_A12_2+x*(INNER_A12_3+x*(INNER_A12_4+x*(INNER_A12_5+x*(INNER_A12_6+x*(INNER_A12_7+x*INNER_A12_8))))))));
        e6 = x4*(INNER_A6_0+x*(INNER_A6_1+x*INNER_A6_2));
        const real Y = (INNER_CUTOFF-r)*INNER_INV_CUTOFF;
    const real Y2 = Y*Y, Y4 = Y2*Y2, Y5 = Y4*Y, Y6 = Y4*Y2;
    const real invOneMinusY = INNER_CUTOFF*invR;
    real L6;
    if (Y < 0.8f) {
        // L6/Y^6 = sum(Y^m/(m+6), m>=0) from its Chebyshev expansion on [0, 0.8] (Clenshaw recurrence)
        const real ell[21] = {INNER_ELL_0, INNER_ELL_1, INNER_ELL_2, INNER_ELL_3, INNER_ELL_4, INNER_ELL_5, INNER_ELL_6,
                              INNER_ELL_7, INNER_ELL_8, INNER_ELL_9, INNER_ELL_10, INNER_ELL_11, INNER_ELL_12, INNER_ELL_13,
                              INNER_ELL_14, INNER_ELL_15, INNER_ELL_16, INNER_ELL_17, INNER_ELL_18, INNER_ELL_19, INNER_ELL_20};
        const real u = 2.5f*Y-1.0f, u2 = 2.0f*u;
        real b1 = 0.0f, b2 = 0.0f;
        for (int j = 20; j > 0; j--) {
            const real b0 = u2*b1-b2+ell[j];
            b2 = b1;
            b1 = b0;
        }
        L6 = Y6*(u*b1-b2+ell[0]);
    }
    else
        L6 = LOG(INNER_CUTOFF*invR)-(Y+Y2*0.5f+Y2*Y*(1.0f/3.0f)+Y4*0.25f+Y5*0.2f);
    e1 = INNER_C1*(Y4*(10.0f-15.0f*INNER_GAMMA*Y+6.0f*INNER_GAMMA*INNER_GAMMA*Y2)*invOneMinusY-30.0f*(INNER_GAMMA1SQ*L6+(1.0f-2.0f*INNER_GAMMA)*Y5*0.2f+Y4*0.25f));
    }
    else {
        const real invR2 = invR*invR;
        const real invR6 = invR2*invR2*invR2;
        e12 = invR6*invR6-INNER_K12;
        e6 = invR6-INNER_K6;
        e1 = invR-INNER_K1;
    }
    innerForceLJ = innerSwitch*ljForce;
    innerForceCl = innerSwitch*clForce;
    real sigma6 = exceptionParams.y*exceptionParams.y;
    sigma6 = sigma6*sigma6*sigma6;
    innerEnergyLJ = exceptionParams.z*(sigma6*sigma6*e12 - sigma6*e6);
    innerEnergyCl = exceptionParams.x*e1;
}
#endif
#if INNER_GROUP
ljForce = innerForceLJ;
clForce = innerForceCl;
ljEnergy = innerEnergyLJ;
clEnergy = innerEnergyCl;
#elif SUBTRACT_INNER
ljForce -= innerForceLJ;
clForce -= innerForceCl;
ljEnergy -= innerEnergyLJ;
clEnergy -= innerEnergyCl;
#endif
real dEdR = (ljLambda*ljForce + clLambda*clForce)*invR*invR;
energy += clLambda*clEnergy + ljLambda*ljEnergy;
delta *= dEdR;
real3 force1 = -delta;
real3 force2 = delta;
COMPUTE_DERIVATIVES
