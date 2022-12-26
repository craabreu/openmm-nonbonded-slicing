#ifndef __OPENMM_CUDAFFT3D_H__
#define __OPENMM_CUDAFFT3D_H__

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaContext.h"

using namespace OpenMM;

namespace NonbondedSlicing {

/**
 * This class performs three dimensional Fast Fourier Transforms using VkFFT by
 * Dmitrii Tolmachev (https://github.com/DTolm/VkFFT).
 *
 * Note that this class performs an unnormalized transform.  That means that if you perform
 * a forward transform followed immediately by an inverse transform, the effect is to
 * multiply every value of the original data set by the total number of data points.
 */

class CudaFFT3D {
public:
    /**
     * Create an CudaFFT3D object for performing transforms of a particular size.
     *
     * The transform cannot be done in-place: the input and output
     * arrays must be different.  Also, the input array is used as workspace, so its contents
     * are destroyed.  This also means that both arrays must be large enough to hold complex values,
     * even when performing a real-to-complex transform.
     *
     * When performing a real-to-complex transform, the output data is of size xsize*ysize*(zsize/2+1)
     * and contains only the non-redundant elements.
     *
     * @param context the context in which to perform calculations
     * @param stream  the CUDA stream doing the calculations
     * @param xsize   the first dimension of the data sets on which FFTs will be performed
     * @param ysize   the second dimension of the data sets on which FFTs will be performed
     * @param zsize   the third dimension of the data sets on which FFTs will be performed
     * @param batch   the number of FFTs
     * @param realToComplex  if true, a real-to-complex transform will be done.  Otherwise, it is complex-to-complex.
     * @param in      the data to transform, ordered such that in[x*ysize*zsize + y*zsize + z] contains element (x, y, z)
     * @param out     on exit, this contains the transformed data
     */
    CudaFFT3D(CudaContext& context, CUstream& stream, int xsize, int ysize, int zsize, int batch, bool realToComplex, CudaArray& in, CudaArray& out) :
        realToComplex(realToComplex), doublePrecision(context.getUseDoublePrecision()),
        inputBuffer(in.getDevicePointer()), outputBuffer(out.getDevicePointer()) { }
    virtual ~CudaFFT3D() {};
    /**
     * Perform a Fourier transform.
     *
     * @param forward  true to perform a forward transform, false to perform an inverse transform
     */
    virtual void execFFT(bool forward) {};
    /**
     * Get the smallest legal size for a dimension of the grid (that is, a size with no prime
     * factors other than 2, 3, 5, ..., maxPrimeFactor).
     *
     * @param minimum   the minimum size the return value must be greater than or equal to
     * @param maxPrimeFactor  the maximum supported prime number factor (default=7)
     */
    static int findLegalDimension(int minimum, int maxPrimeFactor=7) {  // VkFFT allows maxPrimeFactor up to 13
        if (minimum < 1)
            return 1;
        while (true) {
            // Attempt to factor the current value.

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
protected:
    CUdeviceptr inputBuffer;
    CUdeviceptr outputBuffer;
    bool realToComplex;
    bool doublePrecision;
};

} // namespace NonbondedSlicing

#endif // __OPENMM_CUDAFFT3D_H__
