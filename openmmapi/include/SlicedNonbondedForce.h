#ifndef OPENMM_SLICEDNONBONDEDFORCE_H_
#define OPENMM_SLICEDNONBONDEDFORCE_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2025 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "internal/windowsExportNonbondedSlicing.h"
#include "openmm/NonbondedForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include <map>
#include <string>
#include <vector>

#define sliceIndex(i, j) (i>j ? i*(i+1)/2+j : j*(j+1)/2+i)

namespace NonbondedSlicing {

class OPENMM_EXPORT_NONBONDED_SLICING SlicedNonbondedForce : public OpenMM::NonbondedForce {
public:
    SlicedNonbondedForce(int numSubsets);
    SlicedNonbondedForce(const OpenMM::NonbondedForce& force, int numSubsets);
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    void getLJPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    void updateParametersInContext(OpenMM::Context& context);
    std::string getNonbondedMethodName() const;
    int getNumSubsets() const {
        return numSubsets;
    }
    int getNumSlices() const {
        return numSubsets*(numSubsets+1)/2;
    }
    int getNumScalingParameters() const {
        return scalingParameters.size();
    }
    int getNumEnergyParameterDerivatives() const {
        return energyParameterDerivatives.size();
    }
    void setParticleSubset(int index, int subset);
    int getParticleSubset(int index) const;
    int addScalingParameter(const std::string& parameter, int subset1, int subset2, bool includeCoulomb, bool includeLJ, bool includeDecapping = false);
    void getScalingParameter(int index, std::string& parameter, int& subset1, int& subset2, bool& includeCoulomb, bool& includeLJ) const;
    bool getScalingParameterIncludeDecapping(int index) const;
    void setScalingParameter(int index, const std::string& parameter, int subset1, int subset2, bool includeCoulomb, bool includeLJ, bool includeDecapping = false);
    int addEnergyParameterDerivative(const std::string& parameter);
    const std::string& getEnergyParameterDerivativeName(int index) const;
    bool getUseCuFFT() const {
        return useCuFFT;
    };
    void setUseCuFFT(bool use) {
        useCuFFT = use;
    };
    bool getReciprocalExclusions() const {
        return reciprocalExclusions;
    };
    void setReciprocalExclusions(bool reciprocal) {
        reciprocalExclusions = reciprocal;
    };
    /**
     * Get the force group of the inner part of the direct-space interactions, or -1 if the inner
     * force group is disabled (the default).  When enabled, every direct-space pair interaction
     * (including 1-4 exceptions) is split into a short-range part, switched off smoothly between
     * the inner switching distance and the inner cutoff distance, which is computed in the inner
     * force group, and a remainder, which stays in the force group of this force.  The inner force
     * group cannot be used with decapping scaling parameters, and the inner cutoff distance cannot
     * exceed the switching distance when a switching function is used.  The inner force group, cutoff
     * distance and switching distance are fixed when a Context is created (call reinitialize() to
     * change them).
     */
    int getInnerForceGroup() const {
        return innerForceGroup;
    }
    void setInnerForceGroup(int group);
    double getInnerCutoffDistance() const {
        return innerCutoff;
    }
    void setInnerCutoffDistance(double distance);
    double getInnerSwitchingDistance() const {
        return innerSwitchingDistance;
    }
    void setInnerSwitchingDistance(double distance);
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    int getGlobalParameterIndex(const std::string& parameter) const;
    int getScalingParameterIndex(const std::string& parameter) const;
    class ScalingParameterInfo;
    int numSubsets;
    std::map<int, int> subsets;
    std::vector<ScalingParameterInfo> scalingParameters;
    std::vector<int> energyParameterDerivatives;
    bool useCuFFT, reciprocalExclusions;
    int innerForceGroup;
    double innerCutoff, innerSwitchingDistance;
};

/**
 * This is an internal class used to record information about a scaling parameter.
 * @private
 */
class SlicedNonbondedForce::ScalingParameterInfo {
public:
    int globalParamIndex, subset1, subset2, slice;
    bool includeCoulomb, includeLJ, includeDecapping;
    ScalingParameterInfo() {
        globalParamIndex = subset1 = subset2 = -1;
        includeCoulomb = includeLJ = includeDecapping = false;
    }
    ScalingParameterInfo(int globalParamIndex, int subset1, int subset2, bool includeCoulomb, bool includeLJ, bool includeDecapping) :
            globalParamIndex(globalParamIndex), subset1(subset1), subset2(subset2),
            includeCoulomb(includeCoulomb), includeLJ(includeLJ), includeDecapping(includeDecapping) {
        if (!(includeCoulomb || includeLJ || includeDecapping))
            OpenMM::throwException(__FILE__, __LINE__, "A scaling parameter must apply to Coulomb interactions, to Lennard-Jones interactions, or to the decapping term");
    }
    int getSlice() const {
        return sliceIndex(subset1, subset2);
    }
    bool clashesWith(const ScalingParameterInfo& info) {
        return getSlice() == info.getSlice() && ((includeCoulomb && info.includeCoulomb) || (includeLJ && info.includeLJ)
                || (includeDecapping && info.includeDecapping));
    }
};

} // namespace NonbondedSlicing

#endif /*OPENMM_SLICEDNONBONDEDFORCE_H_*/
