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

#include "internal/CudaCuFFT3D.h"
#include "openmm/cuda/CudaContext.h"
#include <string>

using namespace PmeSlicing;
using namespace OpenMM;
using namespace std;

CudaCuFFT3D::CudaCuFFT3D(CudaContext& context, CUstream& stream, int xsize, int ysize, int zsize, int batch, bool realToComplex, CudaArray& in, CudaArray& out) :
        CudaFFT3D(context, stream, xsize, ysize, zsize, batch, realToComplex, in, out) {

    int n[3] = {xsize, ysize, zsize};
    int inembed[] = {xsize, ysize, zsize};
    int onembed[] = {xsize, ysize, zsize/2+1};
    int idist = xsize*ysize*zsize;
    int odist = xsize*ysize*(zsize/2+1);

    if (realToComplex)
        forwardType = context.getUseDoublePrecision() ? CUFFT_D2Z : CUFFT_R2C;
    else
        forwardType = context.getUseDoublePrecision() ? CUFFT_Z2Z : CUFFT_C2C;

    cufftResult result = cufftPlanMany(&fftForward, 3, n, inembed, 1, idist, onembed, 1, odist, forwardType, batch);
    if (result != CUFFT_SUCCESS)
        throw OpenMMException("Error initializing CuFFT: "+to_string(result));

    if (realToComplex) {
        backwardType = context.getUseDoublePrecision() ? CUFFT_Z2D : CUFFT_C2R;
        result = cufftPlanMany(&fftBackward, 3, n, onembed, 1, odist, inembed, 1, idist, backwardType, batch);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+to_string(result));
    }

    cufftSetStream(fftForward, stream);
    cufftSetStream(fftBackward, stream);
}

CudaCuFFT3D::~CudaCuFFT3D() {
    cufftDestroy(fftForward);
    cufftDestroy(fftBackward);
}

void CudaCuFFT3D::execFFT(bool forward) {
    cufftResult result;
    if (forwardType == CUFFT_Z2Z)
        result = cufftExecZ2Z(fftForward, (double2*) inputBuffer, (double2*) outputBuffer, forward ? CUFFT_FORWARD : CUFFT_INVERSE);
    else if (forwardType == CUFFT_C2C)
        result = cufftExecC2C(fftForward, (float2*) inputBuffer, (float2*) outputBuffer, forward ? CUFFT_FORWARD : CUFFT_INVERSE);
    else if (forwardType == CUFFT_D2Z) {
        if (forward)
            result = cufftExecD2Z(fftForward, (double*) inputBuffer, (double2*) outputBuffer);
        else
            result = cufftExecZ2D(fftBackward, (double2*) outputBuffer, (double*) inputBuffer);
    }
    else {
        if (forward)
            result = cufftExecR2C(fftForward, (float*) inputBuffer, (float2*) outputBuffer);
        else
            result = cufftExecC2R(fftBackward, (float2*) outputBuffer, (float*) inputBuffer);
    }

    if (result != CUFFT_SUCCESS)
        throw OpenMMException("Error executing FFT: "+to_string(result));
}
