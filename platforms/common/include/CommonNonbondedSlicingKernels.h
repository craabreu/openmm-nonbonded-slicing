#ifndef COMMON_NONBONDED_SLICING_KERNELS_H_
#define COMMON_NONBONDED_SLICING_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2025 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "NonbondedSlicingKernels.h"
#include "FFT3DFactory.h"
#include "openmm/kernels.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeEvent.h"
#include "openmm/common/ComputeQueue.h"
#include "openmm/common/ComputeSort.h"
#include "openmm/common/FFT3D.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace NonbondedSlicing {

/**
 * This kernel is invoked by SlicedNonbondedForce to calculate the forces acting on the system.
 */
class CommonCalcSlicedNonbondedForceKernel : public CalcSlicedNonbondedForceKernel {
    public:
        CommonCalcSlicedNonbondedForceKernel(std::string name, const Platform& platform, ComputeContext& cc, const System& system) : CalcSlicedNonbondedForceKernel(name, platform),
                hasInitializedKernel(false), cc(cc), pmeio(NULL) {
        }
        ~CommonCalcSlicedNonbondedForceKernel();
        /**
         * Initialize the kernel.  Subclasses should call this from their initialize() method.
         *
         * @param system       the System this kernel will be applied to
         * @param force        the SlicedNonbondedForce this kernel will be used for
         * @param fftFactory   the factory for creating FFT3D objects
         * @param usePmeQueue  whether to perform PME on a separate queue
         * @param deviceIsCpu  whether the device this calculation is running on is a CPU
         * @param useFixedPointChargeSpreading  whether PME charge spreading should be done in fixed point or floating point
         * @param useCpuPme    whether to perform the PME reciprocal space calculation on the CPU
         */
        void commonInitialize(const System& system, const SlicedNonbondedForce& force, FFT3DFactory& fftFactory, bool usePmeQueue, bool deviceIsCpu, bool useFixedPointChargeSpreading, bool useCpuPme);
        /**
         * Execute the kernel to calculate the forces and/or energy.
         *
         * @param context        the context in which to execute this kernel
         * @param includeForces  true if forces should be calculated
         * @param includeEnergy  true if the energy should be calculated
         * @param includeDirect  true if direct space interactions should be included
         * @param includeReciprocal  true if reciprocal space interactions should be included
         * @return the potential energy due to the force
         */
        double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
        /**
         * Copy changed parameters over to a context.
         *
         * @param context        the context to copy parameters to
         * @param force          the SlicedNonbondedForce to copy the parameters from
         * @param firstParticle  the index of the first particle whose parameters might have changed
         * @param lastParticle   the index of the last particle whose parameters might have changed
         * @param firstException the index of the first exception whose parameters might have changed
         * @param lastException  the index of the last exception whose parameters might have changed
         */
        void copyParametersToContext(
            ContextImpl& context, const SlicedNonbondedForce& force
            // , int firstParticle, int lastParticle, int firstException, int lastException
            // TODO: Implement the approach added to OpenMM in commit 78902be (PR #4610)
        );
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
         * Get the parameters being used for the dispersion term in LJPME.
         *
         * @param alpha   the separation parameter
         * @param nx      the number of grid points along the X axis
         * @param ny      the number of grid points along the Y axis
         * @param nz      the number of grid points along the Z axis
         */
        void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    private:
        class SortTrait : public ComputeSortImpl::SortTrait {
            int getDataSize() const {return 8;}
            int getKeySize() const {return 4;}
            const char* getDataType() const {return "int2";}
            const char* getKeyType() const {return "int";}
            const char* getMinKey() const {return "(-2147483647-1)";}
            const char* getMaxKey() const {return "2147483647";}
            const char* getMaxValue() const {return "make_int2(2147483647, 2147483647)";}
            const char* getSortKey() const {return "value.y";}
        };
        class ForceInfo;
        class ScalingParameterInfo;
        class PmeIO;
        class PmePreComputation;
        class PmePostComputation;
        class SyncQueuePreComputation;
        class SyncQueuePostComputation;
        class AddEnergyPostComputation;
        class DispersionCorrectionPostComputation;
        ComputeContext& cc;
        ForceInfo* info;
        bool hasInitializedKernel;
        ComputeArray charges;
        ComputeArray sigmaEpsilon;
        ComputeArray exceptionParams;
        ComputeArray exclusionAtoms;
        ComputeArray exclusionParams;
        ComputeArray baseParticleParams;
        ComputeArray baseExceptionParams;
        ComputeArray particleParamOffsets;
        ComputeArray exceptionParamOffsets;
        ComputeArray particleOffsetIndices;
        ComputeArray exceptionOffsetIndices;
        ComputeArray globalParams;
        ComputeArray cosSinSums;
        ComputeArray pmeGrid1;
        ComputeArray pmeGrid2;
        ComputeArray pmeBsplineModuliX;
        ComputeArray pmeBsplineModuliY;
        ComputeArray pmeBsplineModuliZ;
        ComputeArray pmeDispersionBsplineModuliX;
        ComputeArray pmeDispersionBsplineModuliY;
        ComputeArray pmeDispersionBsplineModuliZ;
        ComputeArray pmeAtomGridIndex;
        ComputeArray pmeEnergyBuffer;
        ComputeArray pmeEnergyParamDerivBuffer;
        ComputeArray ljpmeEnergyBuffer;
        ComputeArray chargeBuffer;
        ComputeArray subsets;
        ComputeArray sliceLambdas;
        ComputeSort sort;
        ComputeQueue pmeQueue;
        ComputeEvent pmeSyncEvent, paramsSyncEvent;
        FFT3D fft, dispersionFft;
        Kernel cpuPme;
        PmeIO* pmeio;
        SyncQueuePostComputation* syncQueue;
        ComputeKernel computeParamsKernel, computeExclusionParamsKernel, computePlasmaCorrectionKernel;
        ComputeKernel computeSubsetSumsKernel;
        ComputeKernel ewaldSumsKernel, ewaldForcesKernel;
        ComputeKernel pmeGridIndexKernel, pmeDispersionGridIndexKernel;
        ComputeKernel pmeSpreadChargeKernel, pmeDispersionSpreadChargeKernel;
        ComputeKernel pmeFinishSpreadChargeKernel, pmeDispersionFinishSpreadChargeKernel;
        ComputeKernel pmeConvolutionKernel, pmeDispersionConvolutionKernel;
        ComputeKernel pmeEvalEnergyKernel, pmeDispersionEvalEnergyKernel;
        ComputeKernel pmeInterpolateForceKernel, pmeDispersionInterpolateForceKernel;
        ComputeKernel addEnergyKernel;
        std::map<std::string, std::string> paramsDefines, ewaldDefines, pmeDefines;
        std::vector<std::pair<int, int> > exceptionAtoms;
        std::vector<std::string> paramNames;
        std::vector<double> paramValues;
        std::map<int, int> exceptionIndex;
        std::set<std::string> requestedDerivatives;
        double ewaldSelfEnergy, dispersionCoefficient, alpha, dispersionAlpha, backgroundEnergyVolume;
        int gridSizeX, gridSizeY, gridSizeZ;
        int dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ;
        bool usePmeQueue, deviceIsCpu, useFixedPointChargeSpreading, useCpuPme;
        bool hasCoulomb, hasLJ, doLJPME, usePosqCharges, recomputeParams, hasOffsets, hasReciprocal;
        NonbondedMethod nonbondedMethod;
        static const int PmeOrder = 5;

        int numSubsets, numSlices;
        bool hasDerivatives;
        vector<int> subsetsVec;
        vector<double> dispersionCoefficients, sliceBackgroundEnergyVolume;
        vector<mm_double2> sliceLambdasVec, subsetSelfEnergy;
        vector<ScalingParameterInfo> sliceScalingParams;
        AddEnergyPostComputation* addEnergy;

        std::string getDerivativeExpression(std::string param, bool conditionCoulomb, bool conditionLJ);
        std::string getCoulombDerivativeCode(ComputeContext& cc, vector<ScalingParameterInfo>& sliceScalingParams, bool assign);

        double totalCharge;
    };

class CommonCalcSlicedNonbondedForceKernel::ScalingParameterInfo {
    public:
        std::string nameCoulomb, nameLJ;
        bool includeCoulomb, includeLJ;
        bool hasDerivativeCoulomb, hasDerivativeLJ;
        ScalingParameterInfo() :
            nameCoulomb(""), nameLJ(""), includeCoulomb(false), includeLJ(false),
            hasDerivativeCoulomb(false), hasDerivativeLJ(false) {
        }
        void addInfo(std::string name, bool includeCoulomb, bool includeLJ, bool hasDerivative) {
            if (includeCoulomb) {
                this->includeCoulomb = true;
                nameCoulomb = name;
                hasDerivativeCoulomb = hasDerivative;
            }
            if (includeLJ) {
                this->includeLJ = true;
                nameLJ = name;
                hasDerivativeLJ = hasDerivative;
            }
        }
    };
    
} // namespace NonbondedSlicing

#endif /*COMMON_NONBONDED_SLICING_KERNELS_H_*/
