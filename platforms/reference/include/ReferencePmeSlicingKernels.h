#ifndef REFERENCE_PMESLICING_KERNELS_H_
#define REFERENCE_PMESLICING_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential calculations on the basis *
 * of atom pairs and for applying scaling parameters to selected slices.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "PmeSlicingKernels.h"
#include "openmm/Platform.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include <vector>
#include <array>
#include <map>

using namespace std;

namespace NonbondedSlicing {

/**
 * This kernel is invoked by SlicedPmeForce to calculate the forces acting on the system.
 */
class ReferenceCalcSlicedPmeForceKernel : public CalcSlicedPmeForceKernel {
public:
    ReferenceCalcSlicedPmeForceKernel(string name, const OpenMM::Platform& platform) : CalcSlicedPmeForceKernel(name, platform), neighborList(NULL) {
    }
    ~ReferenceCalcSlicedPmeForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the SlicedPmeForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const SlicedPmeForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @param includeReciprocal  true if reciprocal space interactions should be included
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the SlicedPmeForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const SlicedPmeForce& force);
    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    void computeParameters(OpenMM::ContextImpl& context);
    int numParticles;
    vector<double> particleParamArray;
    vector<double> particleCharges;
    map<pair<string, int>, double> particleParamOffsets;
    int total14;
    vector<int> num14;
    vector<vector<int>> nb14s;
    vector<vector<vector<int>>> bonded14IndexArray;
    vector<vector<vector<double>>> bonded14ParamArray;
    vector<vector<double>> exceptionChargeProds;
    map<int, vector<pair<string, double>>> exceptionParamOffsets;
    double nonbondedCutoff, ewaldAlpha;
    int gridSize[3];
    bool exceptionsArePeriodic;
    vector<set<int>> exclusions;
    OpenMM::NeighborList* neighborList;

    int numSubsets, numSlices;
    vector<int> subsets;
    vector<int> sliceSwitchParamIndices;
    vector<string> switchParamName;
    vector<double> sliceLambda;

    vector<int> sliceSwitchParamDerivative;
};

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
    vector<string> scalingParams;
    vector<vector<double>> sliceLambdas;
    vector<vector<int>> sliceScalingParams;
    vector<vector<int>> sliceScalingParamDerivs;
};

} // namespace NonbondedSlicing

#endif /*REFERENCE_PMESLICING_KERNELS_H_*/
