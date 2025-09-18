/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2025 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "CudaVkFFT3D.h"
#include "openmm/cuda/CudaContext.h"

using namespace NonbondedSlicing;
using namespace OpenMM;

CudaVkFFT::CudaVkFFT(
    CudaContext& context, int xsize, int ysize, int zsize, int numBatches, bool realToComplex
) : context(context), realToComplex(realToComplex), hasInitialized(false) {
    cufftType type1, type2;
    if (realToComplex) {
        if (context.getUseDoublePrecision()) {
            type1 = CUFFT_D2Z;
            type2 = CUFFT_Z2D;
        }
        else {
            type1 = CUFFT_R2C;
            type2 = CUFFT_C2R;
        }
    }
    else {
        if (context.getUseDoublePrecision())
            type1 = type2 = CUFFT_Z2Z;
        else
            type1 = type2 = CUFFT_C2C;
    }
    int n[3] = {xsize, ysize, zsize};
    int outputZSize = realToComplex ? (zsize/2+1) : zsize;
    int inembed[] = {xsize, ysize, zsize};
    int onembed[] = {xsize, ysize, outputZSize};
    int idist = xsize*ysize*zsize;
    int odist = xsize*ysize*outputZSize;

    cufftResult result = cufftPlanMany(&fftForward, 3, n, inembed, 1, idist, onembed, 1, odist, type1, numBatches);
    if (result != CUFFT_SUCCESS)
        throw OpenMMException("Error initializing FFT: "+context.intToString(result));
    result = cufftPlanMany(&fftBackward, 3, n, onembed, 1, odist, inembed, 1, idist, type2, numBatches);
    if (result != CUFFT_SUCCESS)
        throw OpenMMException("Error initializing FFT: "+context.intToString(result));
        hasInitialized = true;
}

CudaVkFFT::~CudaVkFFT() {
    if (hasInitialized) {
        cufftDestroy(fftForward);
        cufftDestroy(fftBackward);
    }
}

void CudaVkFFT::execFFT(ArrayInterface& in, ArrayInterface& out, bool forward) {
    CUdeviceptr in2 = context.unwrap(in).getDevicePointer();
    CUdeviceptr out2 = context.unwrap(out).getDevicePointer();
    cufftResult result;
    if (forward) {
        cufftSetStream(fftForward, context.getCurrentStream());
        if (realToComplex) {
            if (context.getUseDoublePrecision())
                result = cufftExecD2Z(fftForward, (double*) in2, (double2*) out2);
            else
                result = cufftExecR2C(fftForward, (float*) in2, (float2*) out2);
        }
        else {
            if (context.getUseDoublePrecision())
                result = cufftExecZ2Z(fftForward, (double2*) in2, (double2*) out2, CUFFT_FORWARD);
            else
                result = cufftExecC2C(fftForward, (float2*) in2, (float2*) out2, CUFFT_FORWARD);
        }
    }
    else {
        cufftSetStream(fftBackward, context.getCurrentStream());
        if (realToComplex) {
            if (context.getUseDoublePrecision())
                result = cufftExecZ2D(fftBackward, (double2*) in2, (double*) out2);
            else
                result = cufftExecC2R(fftBackward, (float2*) in2, (float*) out2);
        }
        else {
            if (context.getUseDoublePrecision())
                result = cufftExecZ2Z(fftBackward, (double2*) in2, (double2*) out2, CUFFT_INVERSE);
            else
                result = cufftExecC2C(fftBackward, (float2*) in2, (float2*) out2, CUFFT_INVERSE);
        }
    }
    if (result != CUFFT_SUCCESS)
        throw OpenMMException("Error executing FFT: "+context.intToString(result));
}
