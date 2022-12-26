/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "internal/CudaVkFFT3D.h"
#include "openmm/cuda/CudaContext.h"
#include <string>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

CudaVkFFT3D::CudaVkFFT3D(CudaContext& context, CUstream& stream, int xsize, int ysize, int zsize, int batch, bool realToComplex, CudaArray& in, CudaArray& out) :
        CudaFFT3D(context, stream, xsize, ysize, zsize, batch, realToComplex, in, out) {
    int outputZSize = realToComplex ? (zsize/2+1) : zsize;
    size_t realTypeSize = doublePrecision ? sizeof(double) : sizeof(float);
    size_t inputElementSize = realToComplex ? realTypeSize : 2*realTypeSize;
    device = context.getDeviceIndex();
    inputBufferSize = inputElementSize*zsize*ysize*xsize*batch;
    outputBufferSize = 2*realTypeSize*outputZSize*ysize*xsize*batch;

    VkFFTConfiguration config = {};
    config.performR2C = realToComplex;
    config.device = &device;
    config.num_streams = 1;
    config.stream = &stream;
    config.doublePrecision = doublePrecision;

    config.FFTdim = 3;
    config.size[0] = zsize;
    config.size[1] = ysize;
    config.size[2] = xsize;
    config.numberBatches = batch;

    config.inverseReturnToInputBuffer = true;
    config.isInputFormatted = true;
    config.inputBufferSize = &inputBufferSize;
    config.inputBuffer = (void**) &inputBuffer;
    config.inputBufferStride[0] = zsize;
    config.inputBufferStride[1] = zsize*ysize;
    config.inputBufferStride[2] = zsize*ysize*xsize;

    config.bufferSize = &outputBufferSize;
    config.buffer = (void**) &outputBuffer;
    config.bufferStride[0] = outputZSize;
    config.bufferStride[1] = outputZSize*ysize;
    config.bufferStride[2] = outputZSize*ysize*xsize;

    app = new VkFFTApplication();
    VkFFTResult result = initializeVkFFT(app, config);
    if (result != VKFFT_SUCCESS)
        throw OpenMMException("Error initializing VkFFT: "+to_string(result));
}

CudaVkFFT3D::~CudaVkFFT3D() {
    deleteVkFFT(app);
    delete app;
}

void CudaVkFFT3D::execFFT(bool forward) {
    VkFFTResult result = VkFFTAppend(app, forward ? -1 : 1, NULL);
    if (result != VKFFT_SUCCESS)
        throw OpenMMException("Error executing VkFFT: "+to_string(result));
}
