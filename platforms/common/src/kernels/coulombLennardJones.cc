{
    // Excluded pairs include those with a padding atom, whose subset may be garbage (e.g., stale
    // local memory in OpenCL CPU kernels), so they must not index LAMBDA. Their terms are masked out.
    int slice = isExcluded ? 0 : SUBSET1>SUBSET2 ? SUBSET1*(SUBSET1+1)/2+SUBSET2 : SUBSET2*(SUBSET2+1)/2+SUBSET1;
    real clLambda = LAMBDA[slice].x;
    real ljLambda = LAMBDA[slice].y;
#if HAS_DECAPPING
    // The Lennard-Jones (x) and Coulomb (y) decapping terms of a decapped slice (see CappedLJ.h), zero unless r < sigma
    const real2 decapLambda = DECAP_LAMBDA[slice];
    real decapEnergy = 0.0f, decapForce = 0.0f, decapQEnergy = 0.0f, decapQForce = 0.0f;
#endif
#if HAS_COULOMB
    const real qq = ONE_4PI_EPS0*CHARGE1*CHARGE2;
#else
    const real qq = 0.0f;
#endif
#if HAS_LENNARD_JONES || COULOMB_DECAPPING
    real sig = SIGMA_EPSILON1.x + SIGMA_EPSILON2.x;
    real sig2 = invR*sig;
    sig2 *= sig2;
    real sig6 = sig2*sig2*sig2;
#endif
#if HAS_LENNARD_JONES
    real eps = SIGMA_EPSILON1.y*SIGMA_EPSILON2.y;
    real epssig6 = sig6*eps;
#endif
#if INNER_GROUP || SUBTRACT_INNER
    // Inner part of the interaction: S(r) times the bare force, and the shifted potential whose
    // gradient it is (the energy terms are only needed when the energy is requested). It is always
    // the bare Lennard-Jones potential.
    real innerForce = 0.0f, innerEnergyLJ = 0.0f, innerEnergyCl = 0.0f;
    const bool insideInner = (!isExcluded && r2 < INNER_CUTOFF_SQUARED);
#if SUBTRACT_INNER
    // For r <= r_s the inner force is the whole bare force, so the remainder has no bare force and its share of
    // the bare potential is the constant sum_n c_n K_n (plus -erf(alpha*r)/r with Ewald). They are set directly
    // below rather than computed as differences of large, nearly equal terms, and the inner terms are then zero.
    const bool innerIsBare = (insideInner && r <= INNER_SWITCH);
#endif
#if INNER_GROUP && LJ_DECAPPING
    bool decapped = false;
    real capEnergy = 0.0f;
#endif
#if INNER_GROUP && COULOMB_DECAPPING
    bool decappedQ = false;
    real capQEnergy = 0.0f;
#endif
    if (insideInner) {
        real innerSwitch = 1.0f;
        if (r > INNER_SWITCH) {
            const real z = (r-INNER_SWITCH)*INNER_SWITCH_SCALE;
            innerSwitch = 1.0f+z*z*z*(-10.0f+z*(15.0f-z*6.0f));
        }
        real innerForceLJ = 0.0f, innerForceCl = qq*invR;
#if HAS_LENNARD_JONES
        innerForceLJ = epssig6*(12.0f*sig6 - 6.0f);
#if INNER_GROUP && LJ_DECAPPING
        // Below sigma, which cannot exceed INNER_SWITCH, the inner force is the whole force, so the inner group takes
        // the capped force of a decapped slice and all of its decapping term; the remainder gets neither.
        decapped = (IS_LJ_DECAPPED && sig6 > 1.0f);
        if (decapped) {
            const real x = RECIP(invR*sig);
            const real xp = CAP_XPOW;
            capEnergy = eps*(CAP_C0 + xp*(CAP_POLY));
            const real capForce = -eps*xp*(CAP_RPOLY);
            const real sig6d = min(sig6, (real) CAP_MAX_SIG6);   // bounded to stay finite (see CappedLJ.h)
            decapForce = eps*sig6d*(12.0f*sig6d - 6.0f) - capForce;
            decapEnergy = eps*sig6d*(sig6d - 1.0f) - capEnergy;
            innerForceLJ = capForce;
        }
#endif
#endif
#if INNER_GROUP && COULOMB_DECAPPING
        // Likewise, the inner group takes the capped Coulomb force of a Coulomb-decapped slice and all of its decapping term
        decappedQ = (IS_COULOMB_DECAPPED && sig6 > 1.0f);
        if (decappedQ) {
            const real x = RECIP(invR*sig);
            const real xp = CAP_XPOW;
            const real qqOverSig = qq*RECIP(sig);
            capQEnergy = qqOverSig*(CAPQ_C0 + xp*(CAPQ_POLY));
            const real capForce = -qqOverSig*xp*(CAPQ_RPOLY);
            const real bare = qq*min(invR, (real) CAP_MAX_INVX*RECIP(sig));   // bounded to stay finite
            decapQForce = bare - capForce;
            decapQEnergy = bare - capQEnergy;
            innerForceCl = capForce;
        }
#endif
#if SUBTRACT_INNER
        // The remainder's forces below r_s are set directly, so the bare forces are dropped from both sides of the
        // difference, which keeps it from becoming inf-inf in single precision as r -> 0
        if (innerIsBare)
            innerForceLJ = innerForceCl = 0.0f;
#endif
        innerForce = innerSwitch*(ljLambda*innerForceLJ + clLambda*innerForceCl);
#ifdef INCLUDE_ENERGY
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
        innerEnergyCl = qq*e1;
#if INNER_GROUP && COULOMB_DECAPPING
        if (decappedQ)
            innerEnergyCl = capQEnergy - qq*INNER_K1;
#endif
#if HAS_LENNARD_JONES
        real sigma6 = sig*sig;
        sigma6 = sigma6*sigma6*sigma6;
        innerEnergyLJ = eps*(sigma6*sigma6*e12 - sigma6*e6);
#if INNER_GROUP && LJ_DECAPPING
        if (decapped)
            innerEnergyLJ = capEnergy - eps*sigma6*(sigma6*INNER_K12 - INNER_K6);
#endif
#endif
#endif
    }
#endif
#if INNER_GROUP
    real tempForce = innerForce;
    real ljEnergy = innerEnergyLJ;
    real clEnergy = innerEnergyCl;
    tempEnergy += ljLambda*ljEnergy + clLambda*clEnergy;
#if HAS_DECAPPING
    tempForce += decapLambda.x*decapForce + decapLambda.y*decapQForce;
    tempEnergy += decapLambda.x*decapEnergy + decapLambda.y*decapQEnergy;
#endif
    dEdR += insideInner ? tempForce*invR*invR : 0;
#else
#if USE_EWALD
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
    const real alphaR = EWALD_ALPHA*r;
    const real expAlphaRSqr = EXP(-alphaR*alphaR);
    const real prefactor = qq*invR;

#ifdef USE_DOUBLE_PRECISION
    const real erfcAlphaR = erfc(alphaR);
#else
    // This approximation for erfc is from Abramowitz and Stegun (1964) p. 299.  They cite the following as
    // the original source: C. Hastings, Jr., Approximations for Digital Computers (1955).  It has a maximum
    // error of 1.5e-7.

    const real t = RECIP(1.0f+0.3275911f*alphaR);
    const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
    real clEnergy = includeInteraction ? prefactor*erfcAlphaR : 0;
    real clForce = prefactor*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
#if SUBTRACT_INNER || COULOMB_DECAPPING
    // The 1/r part of erfc(alpha*r)/r goes to the inner force group below the inner switching distance, or to the capped
    // potential of a Coulomb-decapped slice below sigma. What is left, -erf(alpha*r)/r, is finite at r = 0 and is computed
    // from its series there (see CappedLJ.h) rather than as the difference of two diverging terms.
#if SUBTRACT_INNER
    if (innerIsBare) {
#else
    if (includeInteraction && IS_COULOMB_DECAPPED && sig6 > 1.0f) {
#endif
        real erfOverR;
        if (alphaR < 0.1f) {
            const real a2 = alphaR*alphaR;
            erfOverR = EWALD_ALPHA*TWO_OVER_SQRT_PI*(1.0f - a2*(1.0f/3.0f - a2*(0.1f - a2*(1.0f/42.0f))));
        }
        else
            erfOverR = (1.0f-erfcAlphaR)*invR;
        const real erfForce = qq*(erfOverR - EWALD_ALPHA*TWO_OVER_SQRT_PI*expAlphaRSqr);
#if SUBTRACT_INNER
        // The remainder also has the constant K_1 of the inner split
        clEnergy = qq*(INNER_K1 - erfOverR);
        clForce = -erfForce;
#else
        const real x = RECIP(invR*sig);
        const real xp = CAP_XPOW;
        const real qqOverSig = qq*RECIP(sig);
        const real capEnergy = qqOverSig*(CAPQ_C0 + xp*(CAPQ_POLY));
        const real capForce = -qqOverSig*xp*(CAPQ_RPOLY);
        const real bare = qq*min(invR, (real) CAP_MAX_INVX*RECIP(sig));   // bounded to stay finite
        decapQForce = bare - capForce;
        decapQEnergy = bare - capEnergy;
        clEnergy = capEnergy - qq*erfOverR;
        clForce = capForce - erfForce;
#endif
    }
#endif
    real tempForce = 0.0f;
#if HAS_LENNARD_JONES
    tempForce = epssig6*(12.0f*sig6 - 6.0f);
    real ljEnergy = includeInteraction ? epssig6*(sig6 - 1.0f) : 0;
#if LJ_DECAPPING && !SUBTRACT_INNER
    // Below sigma, a decapped slice takes the capped potential, computed directly rather than as the difference of
    // two terms that diverge as r -> 0, and the decapping term. With an inner force group, both belong to it.
    if (includeInteraction && IS_LJ_DECAPPED && sig6 > 1.0f) {
        const real x = RECIP(invR*sig);
        const real xp = CAP_XPOW;
        const real capEnergy = eps*(CAP_C0 + xp*(CAP_POLY));
        const real capForce = -eps*xp*(CAP_RPOLY);
        const real sig6d = min(sig6, (real) CAP_MAX_SIG6);   // bounded to stay finite (see CappedLJ.h)
        decapForce = eps*sig6d*(12.0f*sig6d - 6.0f) - capForce;
        decapEnergy = eps*sig6d*(sig6d - 1.0f) - capEnergy;
        tempForce = capForce;
        ljEnergy = capEnergy;
    }
#endif
#if SUBTRACT_INNER
    // Below the inner switching distance, the remainder has no Lennard-Jones force, only the constant of the inner split
    if (innerIsBare) {
        real sigma6 = sig*sig;
        sigma6 = sigma6*sigma6*sigma6;
        tempForce = 0.0f;
        ljEnergy = eps*sigma6*(sigma6*INNER_K12 - INNER_K6);
    }
#endif
    #if USE_LJ_SWITCH
    if (r > LJ_SWITCH_CUTOFF) {
        real x = r-LJ_SWITCH_CUTOFF;
        real switchValue = 1+x*x*x*(LJ_SWITCH_C3+x*(LJ_SWITCH_C4+x*LJ_SWITCH_C5));
        real switchDeriv = x*x*(3*LJ_SWITCH_C3+x*(4*LJ_SWITCH_C4+x*5*LJ_SWITCH_C5));
        tempForce = tempForce*switchValue - ljEnergy*switchDeriv*r;
        ljEnergy *= switchValue;
    }
    #endif
#if DO_LJPME
    // The multiplicative term to correct for the multiplicative terms that are always
    // present in reciprocal space.
    const real dispersionAlphaR = EWALD_DISPERSION_ALPHA*r;
    const real dar2 = dispersionAlphaR*dispersionAlphaR;
    const real dar4 = dar2*dar2;
    const real dar6 = dar4*dar2;
    const real invR2 = invR*invR;
    const real expDar2 = EXP(-dar2);
    const float2 sigExpProd = SIGMA_EPSILON1*SIGMA_EPSILON2;
    const real c6 = 64*sigExpProd.x*sigExpProd.x*sigExpProd.x*sigExpProd.y;
    const real coef = invR2*invR2*invR2*c6;
    const real eprefac = 1.0f + dar2 + 0.5f*dar4;
    const real dprefac = eprefac + dar6/6.0f;
    // The multiplicative grid term
    real ljPmeEnergy = coef*(1.0f - expDar2*eprefac);
    tempForce += 6.0f*coef*(1.0f - expDar2*dprefac);
    // The potential shift accounts for the step at the cutoff introduced by the
    // transition from additive to multiplicative combintion rules and is only
    // needed for the real (not excluded) terms.  By addin these terms to ljEnergy
    // instead of tempEnergy here, the includeInteraction mask is correctly applied.
    sig2 = sig*sig;
    sig6 = sig2*sig2*sig2*INVCUT6;
    epssig6 = eps*sig6;
    // The additive part of the potential shift
    ljPmeEnergy += epssig6*(1.0f - sig6);
    // The multiplicative part of the potential shift
    ljPmeEnergy += MULTSHIFT6*c6;
    ljEnergy += includeInteraction ? ljPmeEnergy : 0;
#endif
    tempForce *= ljLambda;
#if HAS_DECAPPING && !SUBTRACT_INNER
    tempForce += decapLambda.x*decapForce + decapLambda.y*decapQForce;
    tempEnergy += decapLambda.x*decapEnergy + decapLambda.y*decapQEnergy;
#endif
    tempForce += clLambda*clForce;
#if SUBTRACT_INNER
    tempForce -= innerForce;
    ljEnergy -= innerEnergyLJ;
    clEnergy -= innerEnergyCl;
#endif
    tempEnergy += ljLambda*ljEnergy + clLambda*clEnergy;
#else
    tempForce = clLambda*clForce;
#if HAS_DECAPPING && !SUBTRACT_INNER
    tempForce += decapLambda.x*decapForce + decapLambda.y*decapQForce;
    tempEnergy += decapLambda.x*decapEnergy + decapLambda.y*decapQEnergy;
#endif
#if SUBTRACT_INNER
    tempForce -= innerForce;
    clEnergy -= innerEnergyCl;
#endif
    tempEnergy += clLambda*clEnergy;
#endif
#else
#ifdef USE_CUTOFF
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
#else
    unsigned int includeInteraction = (!isExcluded);
#endif
    real tempForce = 0.0f;
#if HAS_LENNARD_JONES
    tempForce = epssig6*(12.0f*sig6 - 6.0f);
    real ljEnergy = includeInteraction ? epssig6*(sig6 - 1) : 0;
#if LJ_DECAPPING && !SUBTRACT_INNER
    // Below sigma, a decapped slice takes the capped potential, computed directly rather than as the difference of
    // two terms that diverge as r -> 0, and the decapping term. With an inner force group, both belong to it.
    if (includeInteraction && IS_LJ_DECAPPED && sig6 > 1.0f) {
        const real x = RECIP(invR*sig);
        const real xp = CAP_XPOW;
        const real capEnergy = eps*(CAP_C0 + xp*(CAP_POLY));
        const real capForce = -eps*xp*(CAP_RPOLY);
        const real sig6d = min(sig6, (real) CAP_MAX_SIG6);   // bounded to stay finite (see CappedLJ.h)
        decapForce = eps*sig6d*(12.0f*sig6d - 6.0f) - capForce;
        decapEnergy = eps*sig6d*(sig6d - 1.0f) - capEnergy;
        tempForce = capForce;
        ljEnergy = capEnergy;
    }
#endif
#if SUBTRACT_INNER
    // Below the inner switching distance, the remainder has no Lennard-Jones force, only the constant of the inner split
    if (innerIsBare) {
        real sigma6 = sig*sig;
        sigma6 = sigma6*sigma6*sigma6;
        tempForce = 0.0f;
        ljEnergy = eps*sigma6*(sigma6*INNER_K12 - INNER_K6);
    }
#endif
    #if USE_LJ_SWITCH
    if (r > LJ_SWITCH_CUTOFF) {
        real x = r-LJ_SWITCH_CUTOFF;
        real switchValue = 1+x*x*x*(LJ_SWITCH_C3+x*(LJ_SWITCH_C4+x*LJ_SWITCH_C5));
        real switchDeriv = x*x*(3*LJ_SWITCH_C3+x*(4*LJ_SWITCH_C4+x*5*LJ_SWITCH_C5));
        tempForce = tempForce*switchValue - ljEnergy*switchDeriv*r;
        ljEnergy *= switchValue;
    }
    #endif
    tempForce *= ljLambda;
#if SUBTRACT_INNER
    ljEnergy -= innerEnergyLJ;
#endif
    tempEnergy += ljLambda*ljEnergy;
#endif
#if HAS_COULOMB
  #ifdef USE_CUTOFF
    const real prefactor = qq;
    real clEnergy = includeInteraction ? prefactor*(invR + REACTION_FIELD_K*r2 - REACTION_FIELD_C) : 0;
    real clForce = prefactor*(invR - 2.0f*REACTION_FIELD_K*r2);
#if SUBTRACT_INNER
    // Below the inner switching distance, the remainder is the constant of the inner split plus the reaction-field terms
    if (innerIsBare) {
        clEnergy = prefactor*(INNER_K1_MINUS_RF_C + REACTION_FIELD_K*r2);
        clForce = -2.0f*prefactor*REACTION_FIELD_K*r2;
    }
#endif
  #else
    const real prefactor = qq*invR;
    real clEnergy = includeInteraction ? prefactor : 0;
    real clForce = prefactor;
#if SUBTRACT_INNER
    // Below the inner switching distance, the remainder is the constant of the inner split
    if (innerIsBare) {
        clEnergy = qq*INNER_K1;
        clForce = 0.0f;
    }
#endif
  #endif
#if COULOMB_DECAPPING && !SUBTRACT_INNER
    // Below sigma, a Coulomb-decapped slice takes the capped Coulomb potential in place of the 1/r term (see CappedLJ.h).
    // With an inner force group, both belong to it (see the inner part above).
    if (includeInteraction && IS_COULOMB_DECAPPED && sig6 > 1.0f) {
        const real x = RECIP(invR*sig);
        const real xp = CAP_XPOW;
        const real qqOverSig = qq*RECIP(sig);
        const real capEnergy = qqOverSig*(CAPQ_C0 + xp*(CAPQ_POLY));
        const real capForce = -qqOverSig*xp*(CAPQ_RPOLY);
        const real bare = qq*min(invR, (real) CAP_MAX_INVX*RECIP(sig));   // bounded to stay finite
        decapQForce = bare - capForce;
        decapQEnergy = bare - capEnergy;
  #ifdef USE_CUTOFF
        clEnergy = capEnergy + qq*(REACTION_FIELD_K*r2 - REACTION_FIELD_C);
        clForce = capForce - 2.0f*qq*REACTION_FIELD_K*r2;
  #else
        clEnergy = capEnergy;
        clForce = capForce;
  #endif
    }
#endif
    tempForce += clLambda*clForce;
#if SUBTRACT_INNER
    clEnergy -= innerEnergyCl;
#endif
    tempEnergy += clLambda*clEnergy;
#endif
#if HAS_DECAPPING && !SUBTRACT_INNER
    // The Lennard-Jones and Coulomb decapping terms
    tempForce += decapLambda.x*decapForce + decapLambda.y*decapQForce;
    tempEnergy += decapLambda.x*decapEnergy + decapLambda.y*decapQEnergy;
#endif
#if SUBTRACT_INNER
    tempForce -= innerForce;
#endif
#endif
dEdR += includeInteraction ? tempForce*invR*invR : 0;
#endif
COMPUTE_DERIVATIVES
}
