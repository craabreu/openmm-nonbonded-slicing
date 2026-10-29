float4 exceptionParams = PARAMS[index];
union {int i; float f;} u;
u.f = exceptionParams.w;
int slice = u.i;
real clLambda = LAMBDAS[slice].x;
real ljLambda = LAMBDAS[slice].y;
#if HAS_DECAPPING
const real decapLambda = DECAP_LAMBDAS[slice];
real decapEnergy = 0.0f, decapForce = 0.0f;
#endif
real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
#if APPLY_PERIODIC
APPLY_PERIODIC_TO_DELTA(delta)
#endif
real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
real invR = RSQRT(r2);
real sig2 = invR*exceptionParams.y;
sig2 *= sig2;
real sig6 = sig2*sig2*sig2;
real ljForce = exceptionParams.z*(12.0f*sig6-6.0f)*sig6;
real ljEnergy = exceptionParams.z*(sig6-1.0f)*sig6;
real clForce = exceptionParams.x*invR;
real clEnergy = exceptionParams.x*invR;
#if INNER_GROUP || SUBTRACT_INNER
// Inner part: S(r) times the bare force and the shifted potential whose gradient it is.
// ljForce is the bare force here.
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
#if SUBTRACT_INNER
        e12 = e6 = e1 = 0.0f;
#else
        const real invR2 = invR*invR;
        const real invR6 = invR2*invR2*invR2;
        e12 = invR6*invR6-INNER_K12;
        e6 = invR6-INNER_K6;
        e1 = invR-INNER_K1;
#endif
    }
    innerForceLJ = innerSwitch*ljForce;
    innerForceCl = innerSwitch*clForce;
    real sigma6 = exceptionParams.y*exceptionParams.y;
    sigma6 = sigma6*sigma6*sigma6;
    innerEnergyLJ = exceptionParams.z*(sigma6*sigma6*e12 - sigma6*e6);
    innerEnergyCl = exceptionParams.x*e1;
#if SUBTRACT_INNER
    // For r <= r_s the inner force is the whole bare force, so the remainder's share of the bare potential
    // is the constant sum_n c_n K_n. It replaces the bare energies rather than being computed as the
    // difference of two large, nearly equal terms (the inner energies are zero above).
    if (r <= INNER_SWITCH) {
        ljEnergy = exceptionParams.z*sigma6*(sigma6*INNER_K12 - INNER_K6);
        clEnergy = exceptionParams.x*INNER_K1;
    }
#endif
}
#endif
#if SUBTRACT_INNER && HAS_DECAPPING
// Below sigma, the remainder of a decapped slice has no Lennard-Jones force; dropping the bare force from both sides of
// the difference keeps it from becoming inf-inf in single precision as r -> 0
if (IS_DECAPPED && sig6 > 1.0f)
    ljForce = innerForceLJ = 0.0f;
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
#if HAS_DECAPPING && !SUBTRACT_INNER
// Below sigma, a decapped slice takes the capped potential and the decapping term (see CappedLJ.h). Here ljForce and
// ljEnergy are bare, and with an inner group sigma cannot exceed INNER_SWITCH, where the inner part is the whole force
// and its energy is the bare one minus the constant subtracted below.
if (IS_DECAPPED && sig6 > 1.0f) {
    const real x = RECIP(invR*exceptionParams.y);
    const real xp = CAP_XPOW;
    const real capEnergy = exceptionParams.z*(CAP_C0 + xp*(CAP_POLY));
    const real capForce = -exceptionParams.z*xp*(CAP_RPOLY);
    const real sig6d = min(sig6, (real) CAP_MAX_SIG6);   // bounded to stay finite (see CappedLJ.h)
    decapForce += exceptionParams.z*(12.0f*sig6d-6.0f)*sig6d - capForce;
    decapEnergy += exceptionParams.z*(sig6d-1.0f)*sig6d - capEnergy;
    ljForce = capForce;
    ljEnergy = capEnergy;
#if INNER_GROUP
    real sigma6 = exceptionParams.y*exceptionParams.y;
    sigma6 = sigma6*sigma6*sigma6;
    ljEnergy -= exceptionParams.z*sigma6*(sigma6*INNER_K12 - INNER_K6);
#endif
}
#endif
#if COULOMB_DECAPPING && !SUBTRACT_INNER
// With Coulomb decapping (never combined with an inner force group), the 1/r term is capped below sigma too
if (IS_DECAPPED && sig6 > 1.0f) {
    const real x = RECIP(invR*exceptionParams.y);
    const real xp = CAP_XPOW;
    const real qqOverSig = exceptionParams.x*RECIP(exceptionParams.y);
    const real capEnergy = qqOverSig*(CAPQ_C0 + xp*(CAPQ_POLY));
    const real capForce = -qqOverSig*xp*(CAPQ_RPOLY);
    const real bare = exceptionParams.x*min(invR, (real) CAP_MAX_INVX*RECIP(exceptionParams.y));   // bounded to stay finite
    decapForce += bare - capForce;
    decapEnergy += bare - capEnergy;
    clForce = capForce;
    clEnergy = capEnergy;
}
#endif
#if HAS_DECAPPING
real dEdR = (ljLambda*ljForce + clLambda*clForce + decapLambda*decapForce)*invR*invR;
energy += clLambda*clEnergy + ljLambda*ljEnergy + decapLambda*decapEnergy;
#else
real dEdR = (ljLambda*ljForce + clLambda*clForce)*invR*invR;
energy += clLambda*clEnergy + ljLambda*ljEnergy;
#endif
delta *= dEdR;
real3 force1 = -delta;
real3 force2 = delta;
COMPUTE_DERIVATIVES
