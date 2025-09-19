/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2025 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

/**
 * This tests the CUDA implementation of CudaCuFFT.
 */

#include "internal/CudaCuFFT3D.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaSort.h"
#include "sfmt/SFMT.h"
#include "openmm/System.h"
#include <cmath>
#include <complex>
#include <set>
#ifdef _MSC_VER
  #define POCKETFFT_NO_VECTORS
#endif
#include "internal/pocketfft_hdronly.h"

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

static CudaPlatform platform;

template <typename Real, class Real2>
void testTransform(bool realToComplex, int xsize, int ysize, int zsize, int numBatches) {
    System system;
    system.addParticle(0.0);

    CudaPlatform::PlatformData platformData(
        NULL,
        system,
        "",
        "true",
        platform.getPropertyDefaultValue("CudaPrecision"),
        "false",
        platform.getPropertyDefaultValue(CudaPlatform::CudaTempDirectory()),
        platform.getPropertyDefaultValue(CudaPlatform::CudaDisablePmeStream()),
        "false",
        1,
        NULL
    );
    CudaContext& context = *platformData.contexts[0];
    context.initialize();
    context.setAsCurrent();
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    int gridSize = xsize*ysize*zsize;
    int outputZSize = (realToComplex ? zsize/2+1 : zsize);

    vector<vector<complex<double>>> reference(numBatches);
    for (int j = 0; j < numBatches; j++) {
        reference[j].resize(gridSize);
        for (int i = 0; i < gridSize; i++) {
            Real x = (float) genrand_real2(sfmt);
            Real y = realToComplex ? 0 : (float) genrand_real2(sfmt);
            reference[j][i] = complex<double>(x, y);
        }
    }

    vector<Real2> complexOriginal(gridSize*numBatches);
    Real* realOriginal = (Real*) &complexOriginal[0];
    for (int j = 0; j < numBatches; j++)
        for (int i = 0; i < gridSize; i++) {
            int offset = j*gridSize;
            if (realToComplex)
                realOriginal[offset+i] = reference[j][i].real();
            else {
                complexOriginal[offset+i].x = reference[j][i].real();
                complexOriginal[offset+i].y = reference[j][i].imag();
            }
        }

    CudaArray grid1(context, complexOriginal.size(), sizeof(Real2), "grid1");
    CudaArray grid2(context, complexOriginal.size(), sizeof(Real2), "grid2");
    grid1.upload(complexOriginal);

    CudaCuFFT fft(context, xsize, ysize, zsize, numBatches, realToComplex);

    // Perform a forward FFT, then verify the result is correct.

    fft.execFFT(grid1, grid2, true);
    vector<Real2> result;
    grid2.download(result);

    vector<size_t> shape = {(size_t) xsize, (size_t) ysize, (size_t) zsize};
    vector<size_t> axes = {0, 1, 2};
    vector<ptrdiff_t> stride = {(ptrdiff_t) (ysize*zsize*sizeof(complex<double>)),
                                (ptrdiff_t) (zsize*sizeof(complex<double>)),
                                (ptrdiff_t) sizeof(complex<double>)};
    for (int j = 0; j < numBatches; j++) {
        pocketfft::c2c(shape, stride, stride, axes, true, reference[j].data(), reference[j].data(), 1.0);
        for (int x = 0; x < xsize; x++)
            for (int y = 0; y < ysize; y++)
                for (int z = 0; z < outputZSize; z++) {
                    int index1 = x*ysize*zsize + y*zsize + z;
                    int index2 = ((j*xsize + x)*ysize + y)*outputZSize + z;
                    ASSERT_EQUAL_TOL(reference[j][index1].real(), result[index2].x, 1e-3);
                    ASSERT_EQUAL_TOL(reference[j][index1].imag(), result[index2].y, 1e-3);
                }
    }

    // Perform a backward transform and see if we get the original values.

    fft.execFFT(grid2, grid1, false);
    grid1.download(result);
    double scale = 1.0/(xsize*ysize*zsize);
    Real* realResult = (Real*) &result[0];
    for (int j = 0; j < numBatches; j++)
        for (int i = 0; i < gridSize; i++) {
            int offset = j*gridSize;
            if (realToComplex) {
                ASSERT_EQUAL_TOL(realOriginal[offset+i], scale*realResult[offset+i], 1e-4);
            }
            else {
                ASSERT_EQUAL_TOL(complexOriginal[offset+i].x, scale*result[offset+i].x, 1e-4);
                ASSERT_EQUAL_TOL(complexOriginal[offset+i].y, scale*result[offset+i].y, 1e-4);
            }

        }
}

template <typename Real, class Real2>
void executeTests(int numBatches) {
    testTransform<Real, Real2>(false, 28, 25, 30, numBatches);
    testTransform<Real, Real2>(true, 28, 25, 25, numBatches);
    testTransform<Real, Real2>(true, 25, 28, 25, numBatches);
    testTransform<Real, Real2>(true, 25, 25, 28, numBatches);
    testTransform<Real, Real2>(true, 21, 25, 27, numBatches);
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 1)
            platform.setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        if (platform.getPropertyDefaultValue("CudaPrecision") == "double") {
            executeTests<double, double2>(1);
            executeTests<double, double2>(2);
            executeTests<double, double2>(3);
        }
        else {
            executeTests<float, float2>(1);
            executeTests<float, float2>(2);
            executeTests<float, float2>(3);
        }
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
