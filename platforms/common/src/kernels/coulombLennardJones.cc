{
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
    real tempForce = 0.0f;
    tempForce = prefactor*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
    tempEnergy += includeInteraction ? prefactor*erfcAlphaR : 0;
    dEdR += includeInteraction ? tempForce*invR*invR : 0;
#else
#ifdef USE_CUTOFF
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
#else
    unsigned int includeInteraction = (!isExcluded);
#endif
    real tempForce = 0.0f;
#if HAS_COULOMB
    const real prefactor = ONE_4PI_EPS0*CHARGE1*CHARGE2*invR;
    tempForce += prefactor;
    tempEnergy += includeInteraction ? prefactor : 0;
#endif
    dEdR += includeInteraction ? tempForce*invR*invR : 0;
#endif
}
