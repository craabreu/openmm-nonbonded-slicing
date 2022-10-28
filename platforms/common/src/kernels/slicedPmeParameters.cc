# define SLICE(i, j) (i > j ? i*(i+1)/2+j : j*(j+1)/2+i)

/**
 * Compute the nonbonded parameters for particles and exceptions.
 */
KERNEL void computeParameters(GLOBAL mixed* RESTRICT energyBuffer, int includeSelfEnergy, GLOBAL real* RESTRICT globalParams,
        int numAtoms, GLOBAL const float* RESTRICT baseParticleCharges, GLOBAL real4* RESTRICT posq, GLOBAL real* RESTRICT charge,
        GLOBAL float2* RESTRICT particleParamOffsets, GLOBAL int* RESTRICT particleOffsetIndices,
        GLOBAL const int* RESTRICT subsets
#ifdef HAS_EXCEPTIONS
        , int numExceptions, GLOBAL const float* RESTRICT baseExceptionChargeProds, GLOBAL float* RESTRICT exceptionChargeProds,
        GLOBAL float2* RESTRICT exceptionParamOffsets, GLOBAL int* RESTRICT exceptionOffsetIndices,
        GLOBAL const int2* RESTRICT exceptionAtoms, GLOBAL int* RESTRICT exceptionSlices
#endif
        ) {
    mixed energy = 0;

    // Compute particle parameters.
    
    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE) {
        float q = baseParticleCharges[i];
#ifdef HAS_PARTICLE_OFFSETS
        int start = particleOffsetIndices[i], end = particleOffsetIndices[i+1];
        for (int j = start; j < end; j++) {
            float2 offset = particleParamOffsets[j];
            q += globalParams[(int) offset.y]*offset.x;
        }
#endif
#ifdef USE_POSQ_CHARGES
        posq[i].w = q;
#else
        charge[i] = q;
#endif
#ifdef HAS_OFFSETS
    #ifdef INCLUDE_EWALD
        energy -= EWALD_SELF_ENERGY_SCALE*q*q;
    #endif
#endif
    }

    // Compute exception parameters.
    
#ifdef HAS_EXCEPTIONS
    for (int i = GLOBAL_ID; i < numExceptions; i += GLOBAL_SIZE) {
        float chargeProd = baseExceptionChargeProds[i];
#ifdef HAS_EXCEPTION_OFFSETS
        int start = exceptionOffsetIndices[i], end = exceptionOffsetIndices[i+1];
        for (int j = start; j < end; j++) {
            float2 offset = exceptionParamOffsets[j];
            chargeProd += globalParams[(int) offset.y]*offset.x;
        }
#endif
        exceptionChargeProds[i] = (float) (ONE_4PI_EPS0*chargeProd);

        int2 atoms = exceptionAtoms[i];
        int subset1 = subsets[atoms.x];
        int subset2 = subsets[atoms.y];
        exceptionSlices[i] = SLICE(subset1, subset2);
    }
#endif
    if (includeSelfEnergy)
        energyBuffer[GLOBAL_ID] += energy;
}

/**
 * Compute parameters for subtracting the reciprocal part of excluded interactions.
 */
KERNEL void computeExclusionParameters(GLOBAL real4* RESTRICT posq, GLOBAL real* RESTRICT charge,
        int numExclusions, GLOBAL const int2* RESTRICT exclusionAtoms, GLOBAL const int* RESTRICT subsets,
        GLOBAL int* RESTRICT exclusionSlices, GLOBAL float* RESTRICT exclusionChargeProds) {
    for (int i = GLOBAL_ID; i < numExclusions; i += GLOBAL_SIZE) {
        int2 atoms = exclusionAtoms[i];
#ifdef USE_POSQ_CHARGES
        real chargeProd = posq[atoms.x].w*posq[atoms.y].w;
#else
        real chargeProd = charge[atoms.x]*charge[atoms.y];
#endif
        exclusionChargeProds[i] = (float) (ONE_4PI_EPS0*chargeProd);
        int subset1 = subsets[atoms.x];
        int subset2 = subsets[atoms.y];
        exclusionSlices[i] = SLICE(subset1, subset2);
    }
}