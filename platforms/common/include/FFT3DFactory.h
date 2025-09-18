#ifndef COMMON_FFT3D_FACTORY_H_
#define COMMON_FFT3D_FACTORY_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2025 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "openmm/common/FFT3D.h"
#include "openmm/common/ComputeContext.h"

using namespace OpenMM;

namespace NonbondedSlicing {

class FFT3DFactory {
public:
    virtual FFT3D createFFT3D(ComputeContext& context, int xsize, int ysize, int zsize, int numBatches, bool realToComplex=false) = 0;
    /**
     * Get the smallest legal size for a dimension of the grid (that is, a size with no prime
     * factors other than 2, 3, 5, ..., maxPrimeFactor).
     *
     * @param minimum   the minimum size the return value must be greater than or equal to
     * @param maxPrimeFactor  the maximum supported prime number factor (default=13)
     */
    int findLegalDimension(int minimum, int maxPrimeFactor) {
        if (minimum < 1)
            return 1;
        while (true) {
            int unfactored = minimum;
            for (int factor = 2; factor <= maxPrimeFactor; factor++) {
                while (unfactored > 1 && unfactored%factor == 0)
                    unfactored /= factor;
            }
            if (unfactored == 1)
                return minimum;
            minimum++;
        }
    }
    virtual int findLegalDimension(int minimum) {
        return findLegalDimension(minimum, 13);
    }
};

} // namespace NonbondedSlicing

#endif // COMMON_FFT3D_FACTORY_H_