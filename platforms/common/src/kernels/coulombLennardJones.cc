{
    // Excluded pairs include those with a padding atom, whose subset may be garbage (e.g., stale local
    // memory in OpenCL CPU kernels), so they must not index LAMBDA. Their terms are masked out below.
    int slice = isExcluded ? 0 : (SUBSET1>SUBSET2 ? SUBSET1*(SUBSET1+1)/2+SUBSET2 : SUBSET2*(SUBSET2+1)/2+SUBSET1);
    real clLambda = LAMBDA[slice].x;
    real ljLambda = LAMBDA[slice].y;
#if HAS_COULOMB
    const real qq = ONE_4PI_EPS0*CHARGE1*CHARGE2;
#else
    const real qq = 0.0f;
#endif
#if HAS_LENNARD_JONES
    real sig = SIGMA_EPSILON1.x + SIGMA_EPSILON2.x;
    real sig2 = invR*sig;
    sig2 *= sig2;
    real sig6 = sig2*sig2*sig2;
    real eps = SIGMA_EPSILON1.y*SIGMA_EPSILON2.y;
    real epssig6 = sig6*eps;
#endif
#if INNER_GROUP || SUBTRACT_INNER
    // Inner part of the interaction: S(r) times the bare force, and the shifted potential whose
    // gradient it is (the energy terms are only needed when the energy is requested).
    real innerForce = 0.0f, innerEnergyLJ = 0.0f, innerEnergyCl = 0.0f;
    const bool insideInner = (!isExcluded && r2 < INNER_CUTOFF_SQUARED);
    if (insideInner) {
        real innerSwitch = 1.0f;
        if (r > INNER_SWITCH) {
            const real z = (r-INNER_SWITCH)*INNER_SWITCH_SCALE;
            innerSwitch = 1.0f+z*z*z*(-10.0f+z*(15.0f-z*6.0f));
        }
        real innerForceLJ = 0.0f;
#if HAS_LENNARD_JONES
        innerForceLJ = epssig6*(12.0f*sig6 - 6.0f);
#endif
        innerForce = innerSwitch*(ljLambda*innerForceLJ + clLambda*qq*invR);
#ifdef INCLUDE_ENERGY
        // Shifted inner energy terms for the powers 12, 6 and 1: e_n = n*integral(S(s)*s^-(n+1), s=r..rc) beyond the
        // switching distance, computed by Gauss-Legendre quadrature in log(s) (no cancellation in any precision),
        // and r^-n - K_n below it.
        real e12, e6, e1;
        if (r > INNER_SWITCH) {
            e12 = e6 = e1 = 0.0f;
            const real logR = LOG(r);
            const real panel = (INNER_LOG_CUTOFF-logR)/INNER_PANELS;
            for (int p = 0; p < INNER_PANELS; p++) {
                const real center = logR+(p+0.5f)*panel;
                for (int k = 0; k < 3; k++) {
                    const real node = 0.5f*panel*(k == 0 ? INNER_GL_X0 : (k == 1 ? INNER_GL_X1 : INNER_GL_X2));
                    const real weight = (k == 0 ? INNER_GL_W0 : (k == 1 ? INNER_GL_W1 : INNER_GL_W2));
                    for (int side = -1; side <= 1; side += 2) {
                        const real s = EXP(center+side*node);
                        const real z = (s-INNER_SWITCH)*INNER_SWITCH_SCALE;
                        const real ws = weight*(1.0f+z*z*z*(-10.0f+z*(15.0f-z*6.0f)));
                        const real invS = RECIP(s);
                        const real invS2 = invS*invS;
                        const real invS6 = invS2*invS2*invS2;
                        e12 += ws*invS6*invS6;
                        e6 += ws*invS6;
                        e1 += ws*invS;
                    }
                }
            }
            e12 *= 6.0f*panel;
            e6 *= 3.0f*panel;
            e1 *= 0.5f*panel;
        }
        else {
            const real invR2 = invR*invR;
            const real invR6 = invR2*invR2*invR2;
            e12 = invR6*invR6-INNER_K12;
            e6 = invR6-INNER_K6;
            e1 = invR-INNER_K1;
        }
        innerEnergyCl = qq*e1;
#if HAS_LENNARD_JONES
        real sigma6 = sig*sig;
        sigma6 = sigma6*sigma6*sigma6;
        innerEnergyLJ = eps*(sigma6*sigma6*e12 - sigma6*e6);
#endif
#endif
    }
#endif
#if INNER_GROUP
    real tempForce = innerForce;
    real ljEnergy = innerEnergyLJ;
    real clEnergy = innerEnergyCl;
    tempEnergy += ljLambda*ljEnergy + clLambda*clEnergy;
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
    real tempForce = 0.0f;
#if HAS_LENNARD_JONES
    tempForce = epssig6*(12.0f*sig6 - 6.0f);
    real ljEnergy = includeInteraction ? epssig6*(sig6 - 1.0f) : 0;
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
    tempForce += clLambda*prefactor*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
#if SUBTRACT_INNER
    tempForce -= innerForce;
    ljEnergy -= innerEnergyLJ;
    clEnergy -= innerEnergyCl;
#endif
    tempEnergy += ljLambda*ljEnergy + clLambda*clEnergy;
#else
    tempForce = clLambda*prefactor*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
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
    tempForce += clLambda*prefactor*(invR - 2.0f*REACTION_FIELD_K*r2);
  #else
    const real prefactor = qq*invR;
    real clEnergy = includeInteraction ? prefactor : 0;
    tempForce += clLambda*prefactor;
  #endif
#if SUBTRACT_INNER
    clEnergy -= innerEnergyCl;
#endif
    tempEnergy += clLambda*clEnergy;
#endif
#if SUBTRACT_INNER
    tempForce -= innerForce;
#endif
#endif
dEdR += includeInteraction ? tempForce*invR*invR : 0;
#endif
COMPUTE_DERIVATIVES
}
