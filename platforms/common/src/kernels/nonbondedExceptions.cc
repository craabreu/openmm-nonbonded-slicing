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
real ljForce = exceptionParams.z*(12.0f*sig6-6.0f)*sig6;
real clForce = exceptionParams.x*invR;
real ljEnergy = exceptionParams.z*(sig6-1.0f)*sig6;
real clEnergy = exceptionParams.x*invR;
#if INNER_GROUP || SUBTRACT_INNER
// Inner part: S(r) times the bare force and the shifted potential whose gradient it is
real innerForceLJ = 0.0f, innerForceCl = 0.0f, innerEnergyLJ = 0.0f, innerEnergyCl = 0.0f;
if (r2 < INNER_CUTOFF_SQUARED) {
    const real r = SQRT(r2);
    real innerSwitch = 1.0f;
    if (r > INNER_SWITCH) {
        const real z = (r-INNER_SWITCH)*INNER_SWITCH_SCALE;
        innerSwitch = 1.0f+z*z*z*(-10.0f+z*(15.0f-z*6.0f));
    }
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
