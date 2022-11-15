KERNEL void computeBonds(GLOBAL const real4* RESTRICT posq, GLOBAL mixed* RESTRICT energyBuffer, GLOBAL mm_ulong* RESTRICT forceBuffer,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        GLOBAL const uint2* RESTRICT exclusionAtoms, GLOBAL const int* RESTRICT exclusionSlices, GLOBAL const float* RESTRICT exclusionChargeProds,
        GLOBAL const uint2* RESTRICT exceptionAtoms, GLOBAL const int* RESTRICT exceptionSlices, GLOBAL const float* RESTRICT exceptionChargeProds,
        GLOBAL const real* RESTRICT sliceLambda) {

    mixed energy[NUM_SLICES] = {0};

#if NUM_EXCLUSIONS > 0
    for (int index = GLOBAL_ID; index < NUM_EXCLUSIONS; index += GLOBAL_SIZE) {
        uint2 atoms = exclusionAtoms[index];
        int slice = exclusionSlices[index];
        const float chargeProd = exclusionChargeProds[index];
        real4 pos1 = posq[atoms.x];
        real4 pos2 = posq[atoms.y];
        real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
        #if USE_PERIODIC
            APPLY_PERIODIC_TO_DELTA(delta)
        #endif
        const real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        const real r = SQRT(r2);
        const real invR = RECIP(r);
        const real alphaR = EWALD_ALPHA*r;
        const real expAlphaRSqr = EXP(-alphaR*alphaR);
        real tempForce = 0.0f;
        if (alphaR > 1e-6f) {
            const real erfAlphaR = ERF(alphaR);
            const real prefactor = chargeProd*invR;
            tempForce = -prefactor*(erfAlphaR-alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
            energy[slice] -= prefactor*erfAlphaR;
        }
        else {
            energy[slice] -= TWO_OVER_SQRT_PI*EWALD_ALPHA*chargeProd;
        }
        if (r > 0)
            delta *= sliceLambda[slice]*tempForce*invR*invR;
        real3 force1 = -delta;
        real3 force2 = delta;

        ATOMIC_ADD(&forceBuffer[atoms.x], (mm_ulong) realToFixedPoint(force1.x));
        ATOMIC_ADD(&forceBuffer[atoms.x+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force1.y));
        ATOMIC_ADD(&forceBuffer[atoms.x+PADDED_NUM_ATOMS*2], (mm_ulong) realToFixedPoint(force1.z));
        MEM_FENCE;

        ATOMIC_ADD(&forceBuffer[atoms.y], (mm_ulong) realToFixedPoint(force2.x));
        ATOMIC_ADD(&forceBuffer[atoms.y+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force2.y));
        ATOMIC_ADD(&forceBuffer[atoms.y+PADDED_NUM_ATOMS*2], (mm_ulong) realToFixedPoint(force2.z));
        MEM_FENCE;
    }
#endif

#if NUM_EXCEPTIONS > 0
    for (int index = GLOBAL_ID; index < NUM_EXCEPTIONS; index += GLOBAL_SIZE) {
        uint2 atoms = exceptionAtoms[index];
        int slice = exceptionSlices[index];
        const float chargeProd = exceptionChargeProds[index];
        real4 pos1 = posq[atoms.x];
        real4 pos2 = posq[atoms.y];
        real3 delta = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
        #if USE_PERIODIC
        APPLY_PERIODIC_TO_DELTA(delta)
        #endif
        real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
        real invR = RSQRT(r2);
        real tempEnergy = chargeProd*invR;
        energy[slice] += tempEnergy;
        delta *= sliceLambda[slice]*tempEnergy*invR*invR;
        real3 force1 = -delta;
        real3 force2 = delta;

        ATOMIC_ADD(&forceBuffer[atoms.x], (mm_ulong) realToFixedPoint(force1.x));
        ATOMIC_ADD(&forceBuffer[atoms.x+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force1.y));
        ATOMIC_ADD(&forceBuffer[atoms.x+PADDED_NUM_ATOMS*2], (mm_ulong) realToFixedPoint(force1.z));
        MEM_FENCE;

        ATOMIC_ADD(&forceBuffer[atoms.y], (mm_ulong) realToFixedPoint(force2.x));
        ATOMIC_ADD(&forceBuffer[atoms.y+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(force2.y));
        ATOMIC_ADD(&forceBuffer[atoms.y+PADDED_NUM_ATOMS*2], (mm_ulong) realToFixedPoint(force2.z));
        MEM_FENCE;
    }
#endif

    for (int slice = 0; slice < NUM_SLICES; slice++)
        energyBuffer[GLOBAL_ID] += sliceLambda[slice]*energy[slice];
}
