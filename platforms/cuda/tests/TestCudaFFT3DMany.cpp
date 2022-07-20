/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2011-2016 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

/**
 * This tests the CUDA implementation of multiple real-to-complex FFTs.
 */

#include "internal/CudaFFT3DMany.h"
#include "openmm/cuda/CudaFFT3D.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaSort.h"
#include "openmm/reference/fftpack.h"
#include "sfmt/SFMT.h"
#include "openmm/System.h"
#include <iostream>
#include <cmath>
#include <set>
#include <cufft.h>

using namespace PmeSlicing;
using namespace OpenMM;
using namespace std;

static CudaPlatform platform;

template <typename Real, class Real2>
void testTransform(int xsize, int ysize, int zsize, int batch) {
    printf("size=(%d, %d, %d)  batch=%d\n", xsize, ysize, zsize, batch);
    System system;
    system.addParticle(0.0);
    CudaPlatform::PlatformData platformData(NULL, system, "", "true", platform.getPropertyDefaultValue("CudaPrecision"), "false",
            platform.getPropertyDefaultValue(CudaPlatform::CudaCompiler()), platform.getPropertyDefaultValue(CudaPlatform::CudaTempDirectory()),
            platform.getPropertyDefaultValue(CudaPlatform::CudaHostCompiler()), platform.getPropertyDefaultValue(CudaPlatform::CudaDisablePmeStream()), "false", true, 1, NULL);
    CudaContext& context = *platformData.contexts[0];
    context.initialize();
    context.setAsCurrent();
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);

    // Allocate arrays
    int gridSize = xsize*ysize*zsize;
    vector<Real> original(gridSize*batch*2);
    vector<vector<t_complex>> reference(batch);
    for (int j = 0; j < batch; j++)
        reference[j].resize(gridSize);

    for (int j = 0; j < batch; j++)
        for (int i = 0; i < gridSize; i++) {
            float value = (float) genrand_real2(sfmt);
            reference[j][i] = t_complex(value, 0);
            original[j*gridSize+i] = value;
        }

    CudaArray grid1(context, 2*batch*gridSize, sizeof(Real), "grid1");
    // CudaArray grid2(context, batch*xsize*ysize*(zsize/2+1), sizeof(Real2), "grid2");
    CudaArray grid2(context, batch*gridSize, sizeof(Real2), "grid2");
    grid1.upload(original);

    // Initialize FFT:

    // CudaFFT3DMany fft(context, xsize, ysize, zsize, batch, true);
    CudaFFT3D fft(context, xsize, ysize, zsize, true);

    // int n[3] = {xsize, ysize, zsize};
    // int inembed[] = {xsize, ysize, zsize};
    // int onembed[] = {xsize, ysize, zsize/2+1};
    // int idist = xsize*ysize*zsize;
    // int odist = xsize*ysize*(zsize/2+1);
    // int istride = 1;
    // int ostride = 1;

    // cufftHandle fftForward;
    // cufftHandle fftBackward;
    // cufftResult resultForward, resultBackward;
    // bool doublePrecision = typeid(Real2) == typeid(double2);
    // if (doublePrecision) {
    //     resultForward = cufftPlanMany(&fftForward, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, batch);
    //     resultBackward = cufftPlanMany(&fftBackward, 3, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_Z2D, batch);
    // }
    // else {
    //     resultForward = cufftPlanMany(&fftForward, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
    //     resultBackward = cufftPlanMany(&fftBackward, 3, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2R, batch);
    // }
    // if (resultForward != CUFFT_SUCCESS || resultBackward != CUFFT_SUCCESS)
    //     throw OpenMMException("Error initializing FFT");

    // Perform a forward FFT, then verify the result is correct.

    fft.execFFT(grid1, grid2, true);

    // if (doublePrecision)
    //     resultForward = cufftExecD2Z(fftForward, (double*) grid1.getDevicePointer(), (double2*) grid2.getDevicePointer());
    // else
    //     resultForward = cufftExecR2C(fftForward, (float*) grid1.getDevicePointer(), (float2*) grid2.getDevicePointer());
    // if (resultForward != CUFFT_SUCCESS)
    //     throw OpenMMException("Error executing forward FFT");

    vector<Real2> result;
    grid2.download(result);

    fftpack_t plan;
    fftpack_init_3d(&plan, xsize, ysize, zsize);
    for (int j = 0; j < batch; j++) {
        fftpack_exec_3d(plan, FFTPACK_FORWARD, &reference[j][0], &reference[j][0]);

        for (int x = 0; x < xsize; x++)
            for (int y = 0; y < ysize; y++)
                for (int z = 0; z < zsize/2+1; z++) {
                    int index1 = (x*ysize + y)*zsize + z;
                    int index2 = ((j*xsize+x)*ysize + y)*(zsize/2+1) + z;
                    ASSERT_EQUAL_TOL(reference[j][index1].re, result[index2].x, 1e-3);
                    ASSERT_EQUAL_TOL(reference[j][index1].im, result[index2].y, 1e-3);
                }
    }
    fftpack_destroy(plan);

    // Perform a backward transform and see if we get the original values.

    fft.execFFT(grid2, grid1, false);

    // if (doublePrecision)
    //     resultBackward = cufftExecZ2D(fftBackward, (double2*) grid2.getDevicePointer(), (double*) grid1.getDevicePointer());
    // else
    //     resultBackward = cufftExecC2R(fftBackward, (float2*) grid2.getDevicePointer(), (float*) grid1.getDevicePointer());
    // if (resultBackward != CUFFT_SUCCESS)
    //     throw OpenMMException("Error executing backward FFT");

    vector<Real> bwResult;
    grid1.download(bwResult);
    double scale = 1.0/(xsize*ysize*zsize);
    for (int i = 0; i < gridSize; ++i)
        ASSERT_EQUAL_TOL(original[i], scale*bwResult[i], 1e-4);
}

template <typename Real, class Real2>
void run_tests(int batch) {
    testTransform<Real, Real2>(28, 25, 25, batch);
    testTransform<Real, Real2>(25, 28, 25, batch);
    testTransform<Real, Real2>(25, 25, 28, batch);
    testTransform<Real, Real2>(21, 25, 27, batch);
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 1)
            platform.setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        if (platform.getPropertyDefaultValue("CudaPrecision") == "double") {
            run_tests<double, double2>(1);
            run_tests<double, double2>(2);
            run_tests<double, double2>(3);
        }
        else {
            run_tests<float, float2>(1);
            run_tests<float, float2>(2);
            run_tests<float, float2>(3);
        }
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
