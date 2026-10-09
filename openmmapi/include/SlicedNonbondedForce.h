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
    int addScalingParameter(const std::string& parameter, int subset1, int subset2, bool includeCoulomb, bool includeLJ);
    void getScalingParameter(int index, std::string& parameter, int& subset1, int& subset2, bool& includeCoulomb, bool& includeLJ) const;
    void setScalingParameter(int index, const std::string& parameter, int subset1, int subset2, bool includeCoulomb, bool includeLJ);
    int addEnergyParameterDerivative(const std::string& parameter);
    const std::string& getEnergyParameterDerivativeName(int index) const;
    bool getUseCuFFT() const {
        return useCuFFT;
    };
    void setUseCuFFT(bool use) {
        useCuFFT = use;
    };
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
    bool useCuFFT;
};

/**
 * This is an internal class used to record information about a scaling parameter.
 * @private
 */
class SlicedNonbondedForce::ScalingParameterInfo {
public:
    int globalParamIndex, subset1, subset2, slice;
    bool includeCoulomb, includeLJ;
    ScalingParameterInfo() {
        globalParamIndex = subset1 = subset2 = -1;
        includeCoulomb = includeLJ = false;
    }
    ScalingParameterInfo(int globalParamIndex, int subset1, int subset2, bool includeCoulomb, bool includeLJ) :
            globalParamIndex(globalParamIndex), subset1(subset1), subset2(subset2),
            includeCoulomb(includeCoulomb), includeLJ(includeLJ) {
        if (!(includeCoulomb || includeLJ))
            OpenMM::throwException(__FILE__, __LINE__, "Keywords 'includeCoulomb' and 'includeLJ' cannot be both false");
    }
    int getSlice() const {
        return sliceIndex(subset1, subset2);
    }
    bool clashesWith(const ScalingParameterInfo& info) {
        return getSlice() == info.getSlice() && ((includeCoulomb && info.includeCoulomb) || (includeLJ && info.includeLJ));
    }
};

} // namespace NonbondedSlicing

#endif /*OPENMM_SLICEDNONBONDEDFORCE_H_*/
