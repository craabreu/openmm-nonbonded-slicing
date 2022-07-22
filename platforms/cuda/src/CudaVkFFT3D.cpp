/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2009-2015 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "internal/CudaVkFFT3D.h"
#include "openmm/cuda/CudaContext.h"
#include <string>

using namespace PmeSlicing;
using namespace OpenMM;
using namespace std;

CudaVkFFT3D::CudaVkFFT3D(CudaContext& context, CUstream& stream, int xsize, int ysize, int zsize, int batch, bool realToComplex, CudaArray& in, CudaArray& out) :
        CudaFFT3D(context, stream, xsize, ysize, zsize, batch, realToComplex, in, out) {
    VkFFTConfiguration config = {};
    vkfftApp = new VkFFTApplication();

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
    config.inputBuffer = &inputBuffer;
    config.inputBufferStride[0] = zsize;
    config.inputBufferStride[1] = config.inputBufferStride[0]*ysize;
    config.inputBufferStride[2] = config.inputBufferStride[1]*xsize;

    config.bufferSize = &outputBufferSize;
    config.buffer = &outputBuffer;
    config.bufferStride[0] = realToComplex ? (zsize/2+1) : zsize;
    config.bufferStride[1] = config.bufferStride[0]*ysize;
    config.bufferStride[2] = config.bufferStride[1]*xsize;

    VkFFTResult result = initializeVkFFT(vkfftApp, config);
    if (result != VKFFT_SUCCESS)
        throw OpenMMException("Error initializing VkFFT: "+to_string(result));
}

CudaVkFFT3D::~CudaVkFFT3D() {
    deleteVkFFT(vkfftApp);
    delete vkfftApp;
}

void CudaVkFFT3D::execFFT(bool forward) {
    VkFFTResult result = VkFFTAppend(vkfftApp, forward ? -1 : 1, NULL);
    if (result != VKFFT_SUCCESS)
        throw OpenMMException("Error executing VkFFT: "+to_string(result));
}
