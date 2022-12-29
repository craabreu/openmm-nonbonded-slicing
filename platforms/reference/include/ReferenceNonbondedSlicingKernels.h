#ifndef REFERENCE_NONBONDED_SLICING_KERNELS_H_
#define REFERENCE_NONBONDED_SLICING_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "NonbondedSlicingKernels.h"
#include "openmm/Platform.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include <vector>
#include <array>
#include <map>

using namespace std;

namespace NonbondedSlicing {

/**
 * This kernel is invoked by SlicedNonbondedForce to calculate the forces acting on the system.
 */
class ReferenceCalcSlicedNonbondedForceKernel : public CalcSlicedNonbondedForceKernel {
public:
    ReferenceCalcSlicedNonbondedForceKernel(string name, const Platform& platform) : CalcSlicedNonbondedForceKernel(name, platform) {
    }
    ~ReferenceCalcSlicedNonbondedForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the SlicedNonbondedForce this kernel will be used for
     */
    void initialize(const System& system, const SlicedNonbondedForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @param includeReciprocal  true if reciprocal space interactions should be included
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the SlicedNonbondedForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const SlicedNonbondedForce& force);
    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Get the dispersion parameters being used for the dispersion term in LJPME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    static const int Coul = 0;
    static const int vdW = 1;
    class ScalingParameterInfo;
    void computeParameters(ContextImpl& context);
    int numParticles, num14;
    vector<vector<int>>bonded14IndexArray;
    vector<vector<double>> particleParamArray, bonded14ParamArray;
    vector<int> bonded14SliceArray;
    vector<array<double, 3>> baseParticleParams, baseExceptionParams;
    map<pair<string, int>, array<double, 3>> particleParamOffsets, exceptionParamOffsets;
    double nonbondedCutoff, switchingDistance, rfDielectric, ewaldAlpha, ewaldDispersionAlpha;
    vector<double> dispersionCoefficients;
    int kmax[3], gridSize[3], dispersionGridSize[3];
    bool useSwitchingFunction, exceptionsArePeriodic;
    vector<set<int>> exclusions;
    NonbondedMethod nonbondedMethod;
    NeighborList* neighborList;

    int numSubsets, numSlices;
    vector<int> subsets;
    vector<vector<double>> sliceLambdas;
    vector<vector<ScalingParameterInfo>> sliceScalingParams;
};

class ReferenceCalcSlicedNonbondedForceKernel::ScalingParameterInfo {
public:
    string name;
    bool hasDerivative;
    ScalingParameterInfo() : name(""), hasDerivative(false) {}
    ScalingParameterInfo(string name, bool hasDerivative) : name(name), hasDerivative(hasDerivative) {}
};

} // namespace NonbondedSlicing

#endif /*REFERENCE_NONBONDED_SLICING_KERNELS_H_*/
