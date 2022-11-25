#ifndef OPENMM_SLICEDNONBONDEDFORCE_H_
#define OPENMM_SLICEDNONBONDEDFORCE_H_

/* -------------------------------------------------------------------------- *
 *                             OpenMM PME Slicing                             *
 *                             ==================                             *
 *                                                                            *
 * An OpenMM plugin for slicing Particle Mesh Ewald calculations on the basis *
 * of atom pairs and applying a different switching parameter to each slice.  *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-pme-slicing                             *
 * -------------------------------------------------------------------------- */

#include "internal/windowsExportPmeSlicing.h"
#include "openmm/NonbondedForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include <map>

using namespace OpenMM;
using namespace std;

namespace PmeSlicing {

class OPENMM_EXPORT_PMESLICING SlicedNonbondedForce : public NonbondedForce {
public:
    SlicedNonbondedForce(int numSubsets);
    SlicedNonbondedForce(const OpenMM::NonbondedForce& force, int numSubsets);
    int getNumSubsets() const {
        return numSubsets;
    }
    int getNumScalingParameters() const {
        return scalingParameters.size();
    }
    int getNumScalingParameterDerivatives() const {
        return scalingParameterDerivatives.size();
    }
    void setParticleSubset(int index, int subset);
    int getParticleSubset(int index) const;
    int addScalingParameter(const string& parameter, int subset1, int subset2, bool includeLJ, bool includeCoulomb);
    void getScalingParameter(int index, string& parameter, int& subset1, int& subset2, bool& includeLJ, bool& includeCoulomb) const;
    void setScalingParameter(int index, const string& parameter, int subset1, int subset2, bool includeLJ, bool includeCoulomb);
    int addScalingParameterDerivative(const string& parameter);
    const string& getScalingParameterDerivativeName(int index) const;
    void setScalingParameterDerivative(int index, const string& parameter);
    bool getUseCudaFFT() const {
        return useCudaFFT;
    };
    void setUseCuFFT(bool use) {
        useCudaFFT = use;
    };
protected:
    ForceImpl* createImpl() const;
private:
    int getGlobalParameterIndex(const string& parameter) const;
    int getScalingParameterIndex(const string& parameter) const;
    class ScalingParameterInfo;
    int numSubsets;
    map<int, int> subsets;
    vector<ScalingParameterInfo> scalingParameters;
    vector<int> scalingParameterDerivatives;
    bool useCudaFFT;
};

/**
 * This is an internal class used to record information about a scaling parameter.
 * @private
 */
class SlicedNonbondedForce::ScalingParameterInfo {
public:
    int globalParamIndex, subset1, subset2, slice;
    bool includeLJ, includeCoulomb;
    ScalingParameterInfo() {
        globalParamIndex = subset1 = subset2 = -1;
        includeLJ = includeCoulomb = false;
    }
    ScalingParameterInfo(int globalParamIndex, int subset1, int subset2, bool includeLJ, bool includeCoulomb) :
            globalParamIndex(globalParamIndex), subset1(subset1), subset2(subset2),
            includeLJ(includeLJ), includeCoulomb(includeCoulomb) {
        if (!(includeLJ || includeCoulomb))
            throwException(__FILE__, __LINE__, "Scaling at least one contribution, LJ or Coulomb, is mandatory");
    }
    int getSlice() const {
        int i = min(subset1, subset2);
        int j = max(subset1, subset2);
        return j*(j+1)/2+i;
    }
    bool clashesWith(const ScalingParameterInfo& info) {
        return getSlice() == info.getSlice() && (includeLJ && info.includeLJ || includeCoulomb && info.includeCoulomb);
    }
};

} // namespace PmeSlicing

#endif /*OPENMM_SLICEDNONBONDEDFORCE_H_*/
