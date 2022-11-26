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
    /**
     * Get the parameters being used for PME in a particular Context.  Because some platforms have restrictions
     * on the allowed grid sizes, the values that are actually used may be slightly different from those
     * specified with setPMEParameters(), or the standard values calculated based on the Ewald error tolerance.
     * See the manual for details.
     *
     * @param context      the Context for which to get the parameters
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Get the PME parameters being used for the dispersion term for LJPME in a particular Context.  Because some
     * platforms have restrictions on the allowed grid sizes, the values that are actually used may be slightly different
     * from those specified with setPMEParameters(), or the standard values calculated based on the Ewald error tolerance.
     * See the manual for details.
     *
     * @param context      the Context for which to get the parameters
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getLJPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Update the particle and exception parameters in a Context to match those stored in this Force object.  This method
     * provides an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setParticleParameters() and setExceptionParameters() to modify this object's parameters, then call
     * updateParametersInContext() to copy them over to the Context.
     *
     * This method has several limitations.  The only information it updates is the parameters of particles and exceptions.
     * All other aspects of the Force (the nonbonded method, the cutoff distance, etc.) are unaffected and can only be
     * changed by reinitializing the Context.  Furthermore, only the chargeProd, sigma, and epsilon values of an exception
     * can be changed; the pair of particles involved in the exception cannot change.  Finally, this method cannot be used
     * to add new particles or exceptions, only to change the parameters of existing ones.
     */
    void updateParametersInContext(Context& context);
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
