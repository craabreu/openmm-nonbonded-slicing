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
real dEdR = ljLambda*6.0f*epss*(2.0f*s-1.0f)*q;
real ljEnergy = epss*(s-1.0f);
real ljSoftDeriv = ljLambda*softcoreAlpha*epss*s*(2.0f*s-1.0f);
#else
real dEdR = ljLambda*exceptionParams.z*(12.0f*sig6-6.0f)*sig6;
real ljEnergy = exceptionParams.z*(sig6-1.0f)*sig6;
#endif
dEdR += clLambda*exceptionParams.x*invR;
dEdR *= invR*invR;
real clEnergy = exceptionParams.x*invR;
energy += clLambda*clEnergy + ljLambda*ljEnergy;
delta *= dEdR;
real3 force1 = -delta;
real3 force2 = delta;
COMPUTE_DERIVATIVES
