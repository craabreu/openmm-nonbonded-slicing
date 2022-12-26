/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "internal/OpenCLVkFFT3D.h"
#include "openmm/opencl/OpenCLContext.h"
#include <string>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

OpenCLVkFFT3D::OpenCLVkFFT3D(OpenCLContext& context, int xsize, int ysize, int zsize, int batch, bool realToComplex, OpenCLArray& in, OpenCLArray& out) {
    device = context.getDevice().get();
    cl = context.getContext().get();
    inputBuffer = in.getDeviceBuffer().get();
    outputBuffer = out.getDeviceBuffer().get();

    bool doublePrecision = context.getUseDoublePrecision();
    int outputZSize = realToComplex ? (zsize/2+1) : zsize;
    size_t realTypeSize = doublePrecision ? sizeof(double) : sizeof(float);
    size_t inputElementSize = realToComplex ? realTypeSize : 2*realTypeSize;
    inputBufferSize = inputElementSize*zsize*ysize*xsize*batch;
    outputBufferSize = 2*realTypeSize*outputZSize*ysize*xsize*batch;

    VkFFTConfiguration config = {};
    config.performR2C = realToComplex;
    config.device = &device;
    config.context = &cl;
    config.doublePrecision = doublePrecision;

    config.FFTdim = 3;
    config.size[0] = zsize;
    config.size[1] = ysize;
    config.size[2] = xsize;
    config.numberBatches = batch;

    config.inverseReturnToInputBuffer = true;
    config.isInputFormatted = true;
    config.inputBufferSize = &inputBufferSize;
    config.inputBuffer = &inputBuffer;
    config.inputBufferStride[0] = zsize;
    config.inputBufferStride[1] = zsize*ysize;
    config.inputBufferStride[2] = zsize*ysize*xsize;

    config.bufferSize = &outputBufferSize;
    config.buffer = &outputBuffer;
    config.bufferStride[0] = outputZSize;
    config.bufferStride[1] = outputZSize*ysize;
    config.bufferStride[2] = outputZSize*ysize*xsize;

    VkFFTResult result = initializeVkFFT(&app, config);
    if (result != VKFFT_SUCCESS)
        throw OpenMMException("Error initializing VkFFT: "+to_string(result));
}

OpenCLVkFFT3D::~OpenCLVkFFT3D() {
    deleteVkFFT(&app);
}

void OpenCLVkFFT3D::execFFT(bool forward, cl::CommandQueue queue) {
    cl_command_queue commandQueue = queue.get();
    VkFFTLaunchParams launchParams = {};
    launchParams.commandQueue = &commandQueue;
    VkFFTResult result = VkFFTAppend(&app, forward ? -1 : 1, &launchParams);
    if (result != VKFFT_SUCCESS)
        throw OpenMMException("Error executing VkFFT: "+to_string(result));
}
