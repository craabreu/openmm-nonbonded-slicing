#ifndef OPENMM_SLICEDNONBONDEDFORCE_H_
#define OPENMM_SLICEDNONBONDEDFORCE_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "internal/windowsExportNonbondedSlicing.h"
#include "openmm/NonbondedForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include <map>

using namespace OpenMM;
using namespace std;

#define sliceIndex(i, j) (i>j ? i*(i+1)/2+j : j*(j+1)/2+i)

namespace NonbondedSlicing {

class OPENMM_EXPORT_NONBONDED_SLICING SlicedNonbondedForce : public NonbondedForce {
public:
    SlicedNonbondedForce(int numSubsets);
    SlicedNonbondedForce(const OpenMM::NonbondedForce& force, int numSubsets);
    void getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    void getLJPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    void updateParametersInContext(Context& context);
    string getNonbondedMethodName() const;
    int getNumSubsets() const {
        return numSubsets;
    }
    int getNumSlices() const {
        return numSubsets*(numSubsets+1)/2;
    }
    int getNumScalingParameters() const {
        return scalingParameters.size();
    }
    int getNumScalingParameterDerivatives() const {
        return scalingParameterDerivatives.size();
    }
    void setParticleSubset(int index, int subset);
    int getParticleSubset(int index) const;
    int addScalingParameter(const string& parameter, int subset1, int subset2, bool includeCoulomb, bool includeLJ);
    void getScalingParameter(int index, string& parameter, int& subset1, int& subset2, bool& includeCoulomb, bool& includeLJ) const;
    void setScalingParameter(int index, const string& parameter, int subset1, int subset2, bool includeCoulomb, bool includeLJ);
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
    bool includeCoulomb, includeLJ;
    ScalingParameterInfo() {
        globalParamIndex = subset1 = subset2 = -1;
        includeCoulomb = includeLJ = false;
    }
    ScalingParameterInfo(int globalParamIndex, int subset1, int subset2, bool includeCoulomb, bool includeLJ) :
            globalParamIndex(globalParamIndex), subset1(subset1), subset2(subset2),
            includeCoulomb(includeCoulomb), includeLJ(includeLJ) {
        if (!(includeCoulomb || includeLJ))
            throwException(__FILE__, __LINE__, "Keywords 'includeCoulomb' and 'includeLJ' cannot be both false");
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
