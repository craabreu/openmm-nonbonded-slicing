/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2025 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "internal/OpenCLVkFFT3D.h"
#include "openmm/opencl/OpenCLContext.h"
#include <string>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

OpenCLVkFFT::OpenCLVkFFT(
    OpenCLContext& context, int xsize, int ysize, int zsize, int numBatches, bool realToComplex
) : context(context) {
    app = {};
    VkFFTConfiguration config = {};
    config.FFTdim = 3;
    config.size[0] = zsize;
    config.size[1] = ysize;
    config.size[2] = xsize;
    config.performR2C = realToComplex;
    config.doublePrecision = context.getUseDoublePrecision();
    config.device = &context.getDevice()();
    config.context = &context.getContext()();
    config.inverseReturnToInputBuffer = true;
    config.isInputFormatted = 1;
    config.inputBufferStride[0] = zsize;
    config.inputBufferStride[1] = ysize*zsize;
    config.inputBufferStride[2] = xsize*ysize*zsize;
    cl::Platform platform(context.getDevice().getInfo<CL_DEVICE_PLATFORM>());
    string platformVendor = platform.getInfo<CL_PLATFORM_VENDOR>();
    if (platformVendor.size() >= 5 && platformVendor.substr(0, 5) == "Intel") {
        // Intel's OpenCL uses low accuracy trig functions, so tell VkFFT to use lookup tables instead.
        config.useLUT = 1;
    }
    VkFFTResult result = initializeVkFFT(&app, config);
    if (result != VKFFT_SUCCESS)
        throw OpenMMException("Error initializing VkFFT: "+context.intToString(result));
}

OpenCLVkFFT::~OpenCLVkFFT() {
    deleteVkFFT(&app);
}

void OpenCLVkFFT::execFFT(ArrayInterface& in, ArrayInterface& out, bool forward) {
    VkFFTLaunchParams params = {};
    if (forward) {
        params.inputBuffer = &context.unwrap(in).getDeviceBuffer()();
        params.buffer = &context.unwrap(out).getDeviceBuffer()();
    }
    else {
        params.inputBuffer = &context.unwrap(out).getDeviceBuffer()();
        params.buffer = &context.unwrap(in).getDeviceBuffer()();
    }
    params.commandQueue = &context.getQueue()();
    VkFFTResult result = VkFFTAppend(&app, forward ? -1 : 1, &params);
    if (result != VKFFT_SUCCESS)
        throw OpenMMException("Error executing VkFFT: "+context.intToString(result));
}
