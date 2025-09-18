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
) : context(context) {
    int outputZSize = realToComplex ? (zsize/2+1) : zsize;
    bool doublePrecision = context.getUseDoublePrecision();
    size_t realTypeSize = doublePrecision ? sizeof(double) : sizeof(float);
    size_t inputElementSize = realToComplex ? realTypeSize : 2*realTypeSize;
    device = context.getDeviceIndex();
    inputBufferSize = inputElementSize*zsize*ysize*xsize*numBatches;
    outputBufferSize = 2*realTypeSize*outputZSize*ysize*xsize*numBatches;

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
    config.numberBatches = numBatches;

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
        throw OpenMMException("Error initializing VkFFT: "+context.intToString(result));
}

CudaVkFFT::~CudaVkFFT() {
    deleteVkFFT(app);
    delete app;
}

void CudaVkFFT::execFFT(ArrayInterface& in, ArrayInterface& out, bool forward) {
    stream = context.getCurrentStream();
    if (forward) {
        inputBuffer = context.unwrap(in).getDevicePointer();
        outputBuffer = context.unwrap(out).getDevicePointer();
    }
    else {
        inputBuffer = context.unwrap(out).getDevicePointer();
        outputBuffer = context.unwrap(in).getDevicePointer();
    }
    VkFFTResult result = VkFFTAppend(app, forward ? -1 : 1, NULL);
    if (result != VKFFT_SUCCESS)
        throw OpenMMException("Error executing VkFFT: "+context.intToString(result));
}
