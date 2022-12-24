DEVICE real2 multofReal2(real2 a, real2 b) {
    return make_real2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

/**
 * Precompute the cosine and sine sums which appear in each force term.
 */

KERNEL void calculateEwaldCosSinSums(GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real4* RESTRICT posq,
                GLOBAL const int* RESTRICT subsets, GLOBAL real2* RESTRICT cosSinSum, real4 periodicBoxSize) {
    const unsigned int ksizex = 2*KMAX_X-1;
    const unsigned int ksizey = 2*KMAX_Y-1;
    const unsigned int ksizez = 2*KMAX_Z-1;
    const unsigned int totalK = ksizex*ksizey*ksizez;
    real3 reciprocalBoxSize = make_real3(2*M_PI/periodicBoxSize.x, 2*M_PI/periodicBoxSize.y, 2*M_PI/periodicBoxSize.z);
    real reciprocalCoefficient = ONE_4PI_EPS0*4*M_PI/(periodicBoxSize.x*periodicBoxSize.y*periodicBoxSize.z);
    unsigned int index = GLOBAL_ID;
    mixed energy[NUM_SLICES] = {0};
    while (index < (KMAX_Y-1)*ksizez+KMAX_Z)
        index += GLOBAL_SIZE;
    while (index < totalK) {
        // Find the wave vector (kx, ky, kz) this index corresponds to.

        int rx = index/(ksizey*ksizez);
        int remainder = index - rx*ksizey*ksizez;
        int ry = remainder/ksizez;
        int rz = remainder - ry*ksizez - KMAX_Z + 1;
        ry += -KMAX_Y + 1;
        real kx = rx*reciprocalBoxSize.x;
        real ky = ry*reciprocalBoxSize.y;
        real kz = rz*reciprocalBoxSize.z;

        // Compute the sum for this wave vector.

        real2 sum[NUM_SUBSETS] = {make_real2(0)};
        for (int atom = 0; atom < NUM_ATOMS; atom++) {
            real4 apos = posq[atom];
            real phase = apos.x*kx;
            real2 structureFactor = make_real2(COS(phase), SIN(phase));
            phase = apos.y*ky;
            structureFactor = multofReal2(structureFactor, make_real2(COS(phase), SIN(phase)));
            phase = apos.z*kz;
            structureFactor = multofReal2(structureFactor, make_real2(COS(phase), SIN(phase)));
            sum[subsets[atom]] += apos.w*structureFactor;
        }

        real k2 = kx*kx + ky*ky + kz*kz;
        real ak = EXP(k2*EXP_COEFFICIENT) / k2;

        for (int j = 0; j < NUM_SUBSETS; j++) {
            real2 sum_j = sum[j];

            cosSinSum[NUM_SUBSETS*index+j] = sum_j;

            // Compute the contribution to the energy.

            for (int i = 0; i < j; i++)
                energy[j*(j+1)/2+i] += 2*ak*(sum[i].x*sum_j.x + sum[i].y*sum_j.y);
            energy[j*(j+3)/2] += ak*(sum_j.x*sum_j.x + sum_j.y*sum_j.y);
        }
        index += GLOBAL_SIZE;
    }
    for (int slice = 0; slice < NUM_SLICES; slice++)
        energyBuffer[GLOBAL_ID*NUM_SLICES+slice] = reciprocalCoefficient*energy[slice];
}

/**
 * Compute the reciprocal space part of the Ewald force, using the precomputed sums from the
 * previous routine.
 */

KERNEL void calculateEwaldForces(GLOBAL mm_long* RESTRICT forceBuffers, GLOBAL const real4* RESTRICT posq, GLOBAL const real2* RESTRICT cosSinSum,
            GLOBAL const int* RESTRICT subsets, GLOBAL const real2* RESTRICT sliceLambdas, real4 periodicBoxSize) {
    unsigned int atom = GLOBAL_ID;
    real3 reciprocalBoxSize = make_real3(2*M_PI/periodicBoxSize.x, 2*M_PI/periodicBoxSize.y, 2*M_PI/periodicBoxSize.z);
    real reciprocalCoefficient = ONE_4PI_EPS0*4*M_PI/(periodicBoxSize.x*periodicBoxSize.y*periodicBoxSize.z);
    while (atom < NUM_ATOMS) {
        real3 force = make_real3(0);
        real4 apos = posq[atom];

        // Loop over all wave vectors.

        int lowry = 0;
        int lowrz = 1;
        for (int rx = 0; rx < KMAX_X; rx++) {
            real kx = rx*reciprocalBoxSize.x;
            for (int ry = lowry; ry < KMAX_Y; ry++) {
                real ky = ry*reciprocalBoxSize.y;
                real phase = apos.x*kx;
                real2 tab_xy = make_real2(COS(phase), SIN(phase));
                phase = apos.y*ky;
                tab_xy = multofReal2(tab_xy, make_real2(COS(phase), SIN(phase)));
                for (int rz = lowrz; rz < KMAX_Z; rz++) {
                    real kz = rz*reciprocalBoxSize.z;

                    // Compute the force contribution of this wave vector.

                    int index = rx*(KMAX_Y*2-1)*(KMAX_Z*2-1) + (ry+KMAX_Y-1)*(KMAX_Z*2-1) + (rz+KMAX_Z-1);
                    real k2 = kx*kx + ky*ky + kz*kz;
                    real ak = EXP(k2*EXP_COEFFICIENT)/k2;
                    phase = apos.z*kz;
                    real2 structureFactor = multofReal2(tab_xy, make_real2(COS(phase), SIN(phase)));

                    real sum = 0;
                    int i = subsets[atom];
                    for (int j = 0; j < NUM_SUBSETS; j++) {
                        real2 sum_j = cosSinSum[NUM_SUBSETS*index+j];
                        int slice = j > i ? j*(j+1)/2+i : i*(i+1)/2+j;
                        sum += sliceLambdas[slice].x*(sum_j.x*structureFactor.y - sum_j.y*structureFactor.x);
                    }
                    real dEdR = 2*reciprocalCoefficient*ak*apos.w*sum;

                    force.x += dEdR*kx;
                    force.y += dEdR*ky;
                    force.z += dEdR*kz;
                    lowrz = 1 - KMAX_Z;
                }
                lowry = 1 - KMAX_Y;
            }
        }

        // Record the force on the atom.

        forceBuffers[atom] += realToFixedPoint(force.x);
        forceBuffers[atom+PADDED_NUM_ATOMS] += realToFixedPoint(force.y);
        forceBuffers[atom+2*PADDED_NUM_ATOMS] += realToFixedPoint(force.z);
        atom += GLOBAL_SIZE;
    }
}
