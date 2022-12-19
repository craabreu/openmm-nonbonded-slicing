/**
 * Compute the nonbonded parameters for particles and exceptions.
 */
KERNEL void computeParameters(GLOBAL mixed* RESTRICT energyBuffer, int includeSelfEnergy, GLOBAL real* RESTRICT globalParams,
        int numAtoms, GLOBAL const float4* RESTRICT baseParticleParams, GLOBAL real4* RESTRICT posq, GLOBAL real* RESTRICT charge,
        GLOBAL float2* RESTRICT sigmaEpsilon, GLOBAL float4* RESTRICT particleParamOffsets, GLOBAL int* RESTRICT particleOffsetIndices,
        GLOBAL const int* RESTRICT subsets, GLOBAL const real2* RESTRICT sliceLambdas
#ifdef HAS_EXCEPTIONS
        , int numExceptions, GLOBAL const int2* RESTRICT exceptionPairs, GLOBAL const float4* RESTRICT baseExceptionParams,
        GLOBAL int* RESTRICT exceptionSlices, GLOBAL float4* RESTRICT exceptionParams,
        GLOBAL float4* RESTRICT exceptionParamOffsets, GLOBAL int* RESTRICT exceptionOffsetIndices
#endif
        ) {
    mixed clEnergy[NUM_SUBSETS] = {0};
    mixed ljEnergy[NUM_SUBSETS] = {0};

    // Compute particle parameters.

    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE) {
        float4 params = baseParticleParams[i];
#ifdef HAS_PARTICLE_OFFSETS
        int start = particleOffsetIndices[i], end = particleOffsetIndices[i+1];
        for (int j = start; j < end; j++) {
            float4 offset = particleParamOffsets[j];
            real value = globalParams[(int) offset.w];
            params.x += value*offset.x;
            params.y += value*offset.y;
            params.z += value*offset.z;
        }
#endif
#ifdef USE_POSQ_CHARGES
        posq[i].w = params.x;
#else
        charge[i] = params.x;
#endif
        sigmaEpsilon[i] = make_float2(0.5f*params.y, 2*SQRT(params.z));
#ifdef HAS_OFFSETS
    #ifdef INCLUDE_EWALD
        clEnergy[subsets[i]] -= EWALD_SELF_ENERGY_SCALE*params.x*params.x;
    #endif
    #ifdef INCLUDE_LJPME
        real sig3 = params.y*params.y*params.y;
        ljEnergy[subsets[i]] += LJPME_SELF_ENERGY_SCALE*sig3*sig3*params.z;
    #endif
#endif
    }

    // Compute exception parameters.

#ifdef HAS_EXCEPTIONS
    for (int i = GLOBAL_ID; i < numExceptions; i += GLOBAL_SIZE) {
        float4 params = baseExceptionParams[i];
#ifdef HAS_EXCEPTION_OFFSETS
        int start = exceptionOffsetIndices[i], end = exceptionOffsetIndices[i+1];
        for (int j = start; j < end; j++) {
            float4 offset = exceptionParamOffsets[j];
            real value = globalParams[(int) offset.w];
            params.x += value*offset.x;
            params.y += value*offset.y;
            params.z += value*offset.z;
        }
#endif
        int j = subsets[exceptionPairs[i].x];
        int k = subsets[exceptionPairs[i].y];
        int slice = j>k ? j*(j+1)/2+k : k*(k+1)/2+j;
        float sliceAsFloat = *((float*) &slice);
        exceptionParams[i] = make_float4((float) (ONE_4PI_EPS0*params.x), (float) params.y, (float) (4*params.z), sliceAsFloat);
    }
#endif
    if (includeSelfEnergy) {
        mixed energy = 0;
        for (int j = 0; j < NUM_SUBSETS; j++) {
            int slice = j*(j+3)/2;
            energy += sliceLambdas[slice].x*clEnergy[j] + sliceLambdas[slice].y*ljEnergy[j];
        }
        energyBuffer[GLOBAL_ID] += energy;
    }
}

/**
 * Compute parameters for subtracting the reciprocal part of excluded interactions.
 */
KERNEL void computeExclusionParameters(GLOBAL real4* RESTRICT posq, GLOBAL real* RESTRICT charge,
        GLOBAL float2* RESTRICT sigmaEpsilon, GLOBAL const int* RESTRICT subsets,
        int numExclusions, GLOBAL const int2* RESTRICT exclusionAtoms, GLOBAL float4* RESTRICT exclusionParams) {
    for (int i = GLOBAL_ID; i < numExclusions; i += GLOBAL_SIZE) {
        int2 atoms = exclusionAtoms[i];
#ifdef USE_POSQ_CHARGES
        real chargeProd = posq[atoms.x].w*posq[atoms.y].w;
#else
        real chargeProd = charge[atoms.x]*charge[atoms.y];
#endif
#ifdef INCLUDE_LJPME_EXCEPTIONS
        float2 sigEps1 = sigmaEpsilon[atoms.x];
        float2 sigEps2 = sigmaEpsilon[atoms.y];
        float sigma = sigEps1.x*sigEps2.x;
        float epsilon = sigEps1.y*sigEps2.y;
#else
        float sigma = 0;
        float epsilon = 0;
#endif
        int j = subsets[atoms.x];
        int k = subsets[atoms.y];
        int slice = j>k ? j*(j+1)/2+k : k*(k+1)/2+j;
        float sliceAsFloat = *((float*) &slice);
        exclusionParams[i] = make_float4((float) (ONE_4PI_EPS0*chargeProd), sigma, epsilon, sliceAsFloat);
    }
}