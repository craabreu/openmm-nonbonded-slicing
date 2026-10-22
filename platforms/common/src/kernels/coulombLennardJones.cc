{
    // Excluded pairs include those with a padding atom, whose subset may be garbage (e.g., stale
    // local memory in OpenCL CPU kernels), so they must not index LAMBDA. Their terms are masked out.
    int slice = isExcluded ? 0 : SUBSET1>SUBSET2 ? SUBSET1*(SUBSET1+1)/2+SUBSET2 : SUBSET2*(SUBSET2+1)/2+SUBSET1;
    real clLambda = LAMBDA[slice].x;
    real ljLambda = LAMBDA[slice].y;
#if USE_EWALD
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
    const real alphaR = EWALD_ALPHA*r;
    const real expAlphaRSqr = EXP(-alphaR*alphaR);
#if HAS_COULOMB
    const real prefactor = ONE_4PI_EPS0*CHARGE1*CHARGE2*invR;
#else
    const real prefactor = 0.0f;
#endif

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
    real sig = SIGMA_EPSILON1.x + SIGMA_EPSILON2.x;
    real sig2 = invR*sig;
    sig2 *= sig2;
    real sig6 = sig2*sig2*sig2;
    real eps = SIGMA_EPSILON1.y*SIGMA_EPSILON2.y;
#if HAS_SOFTCORE
    // Beutler soft-core potential: s replaces sig6 and reduces to it when alpha = 0 or lambda = 1.
    // ljEnergyDeriv is the derivative of lambda*ljEnergy with respect to lambda.
    const real softcoreAlpha = SOFTCORE_ALPHA[slice];
    const real q = RECIP(1.0f + softcoreAlpha*(1.0f-ljLambda)*sig6);
    const real s = sig6*q;
    real epss = eps*s;
    tempForce = 6.0f*epss*(2.0f*s - 1.0f)*q;
    real ljEnergy = includeInteraction ? epss*(s - 1.0f) : 0;
    real ljEnergyDeriv = includeInteraction ? ljEnergy + ljLambda*softcoreAlpha*epss*s*(2.0f*s - 1.0f) : 0;
#else
    real epssig6 = sig6*eps;
    tempForce = epssig6*(12.0f*sig6 - 6.0f);
    real ljEnergy = includeInteraction ? epssig6*(sig6 - 1.0f) : 0;
    real ljEnergyDeriv = ljEnergy;
#endif
    #if USE_LJ_SWITCH
    if (r > LJ_SWITCH_CUTOFF) {
        real x = r-LJ_SWITCH_CUTOFF;
        real switchValue = 1+x*x*x*(LJ_SWITCH_C3+x*(LJ_SWITCH_C4+x*LJ_SWITCH_C5));
        real switchDeriv = x*x*(3*LJ_SWITCH_C3+x*(4*LJ_SWITCH_C4+x*5*LJ_SWITCH_C5));
        tempForce = tempForce*switchValue - ljEnergy*switchDeriv*r;
        ljEnergy *= switchValue;
        ljEnergyDeriv *= switchValue;
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
#if HAS_SOFTCORE
    // The additive part of the potential shift, in the soft-core form so that the energy
    // stays continuous at the cutoff
    const real qc = RECIP(1.0f + softcoreAlpha*(1.0f-ljLambda)*sig6);
    const real sc = sig6*qc;
    epss = eps*sc;
    ljPmeEnergy += epss*(1.0f - sc);
    const real shiftDeriv = ljLambda*softcoreAlpha*epss*sc*(2.0f*sc - 1.0f);
#else
    epssig6 = eps*sig6;
    // The additive part of the potential shift
    ljPmeEnergy += epssig6*(1.0f - sig6);
#endif
    // The multiplicative part of the potential shift
    ljPmeEnergy += MULTSHIFT6*c6;
    ljEnergy += includeInteraction ? ljPmeEnergy : 0;
#if HAS_SOFTCORE
    ljEnergyDeriv += includeInteraction ? ljPmeEnergy - shiftDeriv : 0;
#else
    ljEnergyDeriv = ljEnergy;
#endif
#endif
    tempForce *= ljLambda;
    tempForce += clLambda*prefactor*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
    tempEnergy += ljLambda*ljEnergy + clLambda*clEnergy;
#else
    tempForce = clLambda*prefactor*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
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
    real sig = SIGMA_EPSILON1.x + SIGMA_EPSILON2.x;
    real sig2 = invR*sig;
    sig2 *= sig2;
    real sig6 = sig2*sig2*sig2;
    real eps = SIGMA_EPSILON1.y*SIGMA_EPSILON2.y;
#if HAS_SOFTCORE
    const real softcoreAlpha = SOFTCORE_ALPHA[slice];
    const real q = RECIP(1.0f + softcoreAlpha*(1.0f-ljLambda)*sig6);
    const real s = sig6*q;
    const real epss = eps*s;
    tempForce = 6.0f*epss*(2.0f*s - 1.0f)*q;
    real ljEnergy = includeInteraction ? epss*(s - 1.0f) : 0;
    real ljEnergyDeriv = includeInteraction ? ljEnergy + ljLambda*softcoreAlpha*epss*s*(2.0f*s - 1.0f) : 0;
#else
    real epssig6 = sig6*eps;
    tempForce = epssig6*(12.0f*sig6 - 6.0f);
    real ljEnergy = includeInteraction ? epssig6*(sig6 - 1) : 0;
    real ljEnergyDeriv = ljEnergy;
#endif
    #if USE_LJ_SWITCH
    if (r > LJ_SWITCH_CUTOFF) {
        real x = r-LJ_SWITCH_CUTOFF;
        real switchValue = 1+x*x*x*(LJ_SWITCH_C3+x*(LJ_SWITCH_C4+x*LJ_SWITCH_C5));
        real switchDeriv = x*x*(3*LJ_SWITCH_C3+x*(4*LJ_SWITCH_C4+x*5*LJ_SWITCH_C5));
        tempForce = tempForce*switchValue - ljEnergy*switchDeriv*r;
        ljEnergy *= switchValue;
        ljEnergyDeriv *= switchValue;
    }
    #endif
    tempForce *= ljLambda;
    tempEnergy += ljLambda*ljEnergy;
#endif
#if HAS_COULOMB
  #ifdef USE_CUTOFF
    const real prefactor = ONE_4PI_EPS0*CHARGE1*CHARGE2;
    real clEnergy = includeInteraction ? prefactor*(invR + REACTION_FIELD_K*r2 - REACTION_FIELD_C) : 0;
    tempForce += clLambda*prefactor*(invR - 2.0f*REACTION_FIELD_K*r2);
  #else
    const real prefactor = ONE_4PI_EPS0*CHARGE1*CHARGE2*invR;
    real clEnergy = includeInteraction ? prefactor : 0;
    tempForce += clLambda*prefactor;
  #endif
    tempEnergy += clLambda*clEnergy;
#endif
#endif
dEdR += includeInteraction ? tempForce*invR*invR : 0;
COMPUTE_DERIVATIVES
}
