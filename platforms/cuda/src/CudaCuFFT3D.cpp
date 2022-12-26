/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "internal/CudaCuFFT3D.h"
#include "openmm/cuda/CudaContext.h"
#include <string>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

CudaCuFFT3D::CudaCuFFT3D(CudaContext& context, CUstream& stream, int xsize, int ysize, int zsize, int batch, bool realToComplex, CudaArray& in, CudaArray& out) :
        CudaFFT3D(context, stream, xsize, ysize, zsize, batch, realToComplex, in, out) {
    int outputZSize = realToComplex ? (zsize/2+1) : zsize;
    int n[3] = {xsize, ysize, zsize};
    int inembed[] = {xsize, ysize, zsize};
    int onembed[] = {xsize, ysize, outputZSize};
    int idist = xsize*ysize*zsize;
    int odist = xsize*ysize*outputZSize;

    cufftType_t forwardType, backwardType;
    if (realToComplex) {
        forwardType = doublePrecision ? CUFFT_D2Z : CUFFT_R2C;
        backwardType = doublePrecision ? CUFFT_Z2D : CUFFT_C2R;
    }
    else
        forwardType = backwardType = doublePrecision ? CUFFT_Z2Z : CUFFT_C2C;

    cufftResult result = cufftPlanMany(&fftForward, 3, n, inembed, 1, idist, onembed, 1, odist, forwardType, batch);
    if (result != CUFFT_SUCCESS)
        throw OpenMMException("Error initializing CuFFT: "+to_string(result));

    result = cufftPlanMany(&fftBackward, 3, n, onembed, 1, odist, inembed, 1, idist, backwardType, batch);
    if (result != CUFFT_SUCCESS)
        throw OpenMMException("Error initializing FFT: "+to_string(result));

    cufftSetStream(fftForward, stream);
    cufftSetStream(fftBackward, stream);
}

CudaCuFFT3D::~CudaCuFFT3D() {
    cufftDestroy(fftForward);
    cufftDestroy(fftBackward);
}

void CudaCuFFT3D::execFFT(bool forward) {
    cufftResult result;
    if (forward) {
        if (realToComplex) {
            if (doublePrecision)
                result = cufftExecD2Z(fftForward, (double*) inputBuffer, (double2*) outputBuffer);
            else
                result = cufftExecR2C(fftForward, (float*) inputBuffer, (float2*) outputBuffer);
        }
        else {
            if (doublePrecision)
                result = cufftExecZ2Z(fftForward, (double2*) inputBuffer, (double2*) outputBuffer, CUFFT_FORWARD);
            else
                result = cufftExecC2C(fftForward, (float2*) inputBuffer, (float2*) outputBuffer, CUFFT_FORWARD);
        }
    }
    else {
        if (realToComplex) {
            if (doublePrecision)
                result = cufftExecZ2D(fftBackward, (double2*) outputBuffer, (double*) inputBuffer);
            else
                result = cufftExecC2R(fftBackward, (float2*) outputBuffer, (float*) inputBuffer);
        }
        else {
            if (doublePrecision)
                result = cufftExecZ2Z(fftBackward, (double2*) outputBuffer, (double2*) inputBuffer, CUFFT_INVERSE);
            else
                result = cufftExecC2C(fftBackward, (float2*) outputBuffer, (float2*) inputBuffer, CUFFT_INVERSE);
        }
    }
    if (result != CUFFT_SUCCESS)
        throw OpenMMException("Error executing FFT: "+to_string(result));
}
