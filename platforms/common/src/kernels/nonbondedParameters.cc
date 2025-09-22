/**
 * Compute the nonbonded parameters for particles and exceptions.
 */
KERNEL void computeParameters(GLOBAL mixed* RESTRICT energyBuffer, int includeSelfEnergy
#if HANDLE_RECIPROCAL
        , GLOBAL real* RESTRICT subsetSumsBuffer
#endif
        , GLOBAL real* RESTRICT globalParams, int numAtoms, GLOBAL const float4* RESTRICT baseParticleParams, GLOBAL real4* RESTRICT posq, GLOBAL real* RESTRICT charge,
        GLOBAL float2* RESTRICT sigmaEpsilon, GLOBAL float4* RESTRICT particleParamOffsets, GLOBAL int* RESTRICT particleOffsetIndices,
        GLOBAL real* RESTRICT chargeBuffer, GLOBAL const int* RESTRICT subsets, GLOBAL const real2* RESTRICT sliceLambdas
#ifdef HAS_EXCEPTIONS
        , int numExceptions, GLOBAL const float4* RESTRICT baseExceptionParams, GLOBAL float4* RESTRICT exceptionParams,
        GLOBAL float4* RESTRICT exceptionParamOffsets, GLOBAL int* RESTRICT exceptionOffsetIndices
#endif
) {
    mixed clEnergy[NUM_SUBSETS] = {0};
    mixed ljEnergy[NUM_SUBSETS] = {0};
    real subsetCharge[NUM_SUBSETS] = {0};

#if HANDLE_RECIPROCAL
    const int NUM_SUBSET_SUMS = NUM_SUBSETS*SUMS_PER_SUBSET;
    real subsetSums[NUM_SUBSET_SUMS] = {0};
#endif

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
#if HANDLE_RECIPROCAL
    int subset = subsets[i];
    int offset = subset*SUMS_PER_SUBSET;
    #ifdef INCLUDE_EWALD
        clEnergy[subset] -= EWALD_SELF_ENERGY_SCALE*params.x*params.x;
        subsetCharge[subset] += params.x;
        subsetSums[offset] += params.x;
        subsetSums[offset+1] += params.x*params.x;
    #endif
    #ifdef INCLUDE_LJPME
        real sig3 = params.y*params.y*params.y;
        ljEnergy[subset] += LJPME_SELF_ENERGY_SCALE*sig3*sig3*params.z;
        subsetSums[offset+2] += sig3*sig3*params.z;
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
        exceptionParams[i] = make_float4((float) (ONE_4PI_EPS0*params.x), (float) params.y, (float) (4*params.z), (float) params.w);
    }
#endif
    if (includeSelfEnergy) {
        mixed energy = 0;
        for (int subset = 0; subset < NUM_SUBSETS; subset++) {
            int slice = subset*(subset+3)/2;
            energy += sliceLambdas[slice].x*clEnergy[subset] + sliceLambdas[slice].y*ljEnergy[subset];
        }
        energyBuffer[GLOBAL_ID] += energy;
    }

    // Record the total charge from particles processed by this block.

#if HANDLE_RECIPROCAL
    LOCAL real temp[WORK_GROUP_SIZE][NUM_SUBSETS];
    for (int subset = 0; subset < NUM_SUBSETS; subset++)
        temp[LOCAL_ID][subset] = subsetCharge[subset];
    for (int i = 1; i < WORK_GROUP_SIZE; i *= 2) {
        SYNC_THREADS;
        if (LOCAL_ID%(i*2) == 0 && LOCAL_ID+i < WORK_GROUP_SIZE)
            for (int subset = 0; subset < NUM_SUBSETS; subset++)
                temp[LOCAL_ID][subset] += temp[LOCAL_ID+i][subset];
    }
    if (LOCAL_ID == 0)
        for (int subset = 0; subset < NUM_SUBSETS; subset++)
            chargeBuffer[GROUP_ID*NUM_SUBSETS + subset] = temp[0][subset];
#endif
#if HANDLE_RECIPROCAL
    LOCAL real tempSubsetSums[WORK_GROUP_SIZE][NUM_SUBSET_SUMS];
    for (int j = 0; j < NUM_SUBSET_SUMS; j++)
        tempSubsetSums[LOCAL_ID][j] = subsetSums[j];
    for (int i = 1; i < WORK_GROUP_SIZE; i *= 2) {
        SYNC_THREADS;
        if (LOCAL_ID%(i*2) == 0 && LOCAL_ID+i < WORK_GROUP_SIZE)
            for (int j = 0; j < NUM_SUBSET_SUMS; j++)
                tempSubsetSums[LOCAL_ID][j] += tempSubsetSums[LOCAL_ID+i][j];
    }
    if (LOCAL_ID == 0)
        for (int j = 0; j < NUM_SUBSET_SUMS; j++)
            subsetSumsBuffer[GROUP_ID*NUM_SUBSET_SUMS+j] = tempSubsetSums[0][j];
#endif
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
        union {int i; float f;} slice;
        slice.i = j>k ? j*(j+1)/2+k : k*(k+1)/2+j;
        exclusionParams[i] = make_float4((float) (ONE_4PI_EPS0*chargeProd), sigma, epsilon, slice.f);
    }
}

/**
 * When using Ewald or PME with parameter offsets, the total charge can change each step.
 * We therefore need to compute the correction for the neutralizing plasma on the GPU.
 * This kernel is executed by a single thread block.
 */
KERNEL void computePlasmaCorrection(GLOBAL real* RESTRICT chargeBuffer, GLOBAL mixed* RESTRICT energyBuffer,
    real alpha, real volume, GLOBAL const real2* RESTRICT sliceLambdas) {  // TODO: Compute background energy times volume for each slice
    LOCAL real subsetCharge[WORK_GROUP_SIZE][NUM_SUBSETS];
    real sum[NUM_SUBSETS] = {0};
    for (unsigned int index = LOCAL_ID; index < NUM_GROUPS; index += LOCAL_SIZE)
        for (int subset = 0; subset < NUM_SUBSETS; subset++)
            sum[subset] += chargeBuffer[index*NUM_SUBSETS + subset];
    for (int subset = 0; subset < NUM_SUBSETS; subset++)
        subsetCharge[LOCAL_ID][subset] = sum[subset];
    for (int i = 1; i < WORK_GROUP_SIZE; i *= 2) {
        SYNC_THREADS;
        if (LOCAL_ID%(i*2) == 0 && LOCAL_ID+i < WORK_GROUP_SIZE)
            for (int subset = 0; subset < NUM_SUBSETS; subset++)
                subsetCharge[LOCAL_ID][subset] += subsetCharge[LOCAL_ID+i][subset];
    }
    if (LOCAL_ID == 0) {
        mixed energy = 0;
        for (int i = 0; i < NUM_SUBSETS; i++) {
            real factor = -subsetCharge[0][i]/(8*EPSILON0*volume*alpha*alpha);
            int offset = i*(i+1)/2;
            for (int j = 0; j < i; j++)
                energy += 2*sliceLambdas[offset+j].x*subsetCharge[0][j]*factor;
            energy += sliceLambdas[offset+i].x*subsetCharge[0][i]*factor;
        }
        energyBuffer[0] += energy;
    }
}

#if HANDLE_RECIPROCAL
KERNEL void computeSubsetSums(GLOBAL real* RESTRICT subsetSums, GLOBAL real* RESTRICT subsetSumsBuffer) {
    const int NUM_SUBSET_SUMS = NUM_SUBSETS*SUMS_PER_SUBSET;
    LOCAL real tempSubsetSums[WORK_GROUP_SIZE][NUM_SUBSET_SUMS];
    real sum[NUM_SUBSET_SUMS] = {0};
    for (unsigned int index = LOCAL_ID; index < NUM_GROUPS*NUM_SUBSET_SUMS; index += LOCAL_SIZE)
        for (int i = 0; i < NUM_SUBSET_SUMS; i++)
            sum[i] += subsetSumsBuffer[index*NUM_SUBSET_SUMS+i];
    for (int i = 0; i < NUM_SUBSET_SUMS; i++)
        tempSubsetSums[LOCAL_ID][i] = sum[i];
    for (int j = 1; j < WORK_GROUP_SIZE; j *= 2) {
        SYNC_THREADS;
        if (LOCAL_ID%(j*2) == 0 && LOCAL_ID+j < WORK_GROUP_SIZE)
            for (int i = 0; i < NUM_SUBSET_SUMS; i++)
                tempSubsetSums[LOCAL_ID][i] += tempSubsetSums[LOCAL_ID+j][i];
    }
    if (LOCAL_ID == 0) {
        for (int i = 0; i < NUM_SUBSET_SUMS; i++)
            subsetSums[i] = tempSubsetSums[0][i];
    }
}
#endif