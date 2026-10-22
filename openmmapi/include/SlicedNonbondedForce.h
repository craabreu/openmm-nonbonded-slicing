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
#include <cmath>
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
    int addScalingParameter(const std::string& parameter, int subset1, int subset2, bool includeCoulomb, bool includeLJ, double softcoreAlpha = 0.0);
    void getScalingParameter(int index, std::string& parameter, int& subset1, int& subset2, bool& includeCoulomb, bool& includeLJ) const;
    double getScalingParameterSoftcoreAlpha(int index) const;
    void setScalingParameter(int index, const std::string& parameter, int subset1, int subset2, bool includeCoulomb, bool includeLJ, double softcoreAlpha = 0.0);
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
     * force group, and a remainder, which stays in the force group of this force.
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
    bool includeCoulomb, includeLJ;
    double softcoreAlpha;
    ScalingParameterInfo() {
        globalParamIndex = subset1 = subset2 = -1;
        includeCoulomb = includeLJ = false;
        softcoreAlpha = 0.0;
    }
    ScalingParameterInfo(int globalParamIndex, int subset1, int subset2, bool includeCoulomb, bool includeLJ, double softcoreAlpha) :
            globalParamIndex(globalParamIndex), subset1(subset1), subset2(subset2),
            includeCoulomb(includeCoulomb), includeLJ(includeLJ), softcoreAlpha(softcoreAlpha) {
        if (!(softcoreAlpha >= 0) || !std::isfinite(softcoreAlpha))
            OpenMM::throwException(__FILE__, __LINE__, "The soft-core parameter alpha must be finite and non-negative");
        if (!(includeCoulomb || includeLJ || softcoreAlpha > 0))
            OpenMM::throwException(__FILE__, __LINE__, "A scaling parameter must apply to Coulomb interactions, to Lennard-Jones interactions, or to the soft-core potential");
    }
    int getSlice() const {
        return sliceIndex(subset1, subset2);
    }
    bool includesSoftcore() const {
        return softcoreAlpha > 0;
    }
    bool clashesWith(const ScalingParameterInfo& info) {
        return getSlice() == info.getSlice() && ((includeCoulomb && info.includeCoulomb) || (includeLJ && info.includeLJ)
                || (includesSoftcore() && info.includesSoftcore()));
    }
};

} // namespace NonbondedSlicing

#endif /*OPENMM_SLICEDNONBONDEDFORCE_H_*/
