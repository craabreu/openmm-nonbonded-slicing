{
unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
const real alphaR = EWALD_ALPHA*r;
const real expAlphaRSqr = EXP(-alphaR*alphaR);
const real prefactor = ONE_4PI_EPS0*CHARGE1*CHARGE2*invR;
#ifdef USE_DOUBLE_PRECISION
    const real erfcAlphaR = erfc(alphaR);
#else
    // This approximation for erfc is from Abramowitz and Stegun (1964) p. 299.  They cite the following as
    // the original source: C. Hastings, Jr., Approximations for Digital Computers (1955).  It has a maximum
    // error of 1.5e-7.

    const real t = RECIP(1.0f+0.3275911f*alphaR);
    const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
real tempForce = prefactor*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
#define SLICE(i, j) i > j ? i*(i+1)/2+j : j*(j+1)/2+i
int slice = SLICE(SUBSET1, SUBSET2);
tempEnergy = includeInteraction ? prefactor*erfcAlphaR : 0;
COMPUTE_DERIVATIVES
tempEnergy *= LAMBDA[slice];
dEdR += includeInteraction ? LAMBDA[slice]*tempForce*invR*invR : 0;
}
