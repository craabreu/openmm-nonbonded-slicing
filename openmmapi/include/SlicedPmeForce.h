#ifndef OPENMM_SLICEDPMEFORCE_H_
#define OPENMM_SLICEDPMEFORCE_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "openmm/NonbondedForce.h"
#include <map>
#include <set>
#include <utility>
#include <vector>
#include "internal/windowsExportNonbondedSlicing.h"

namespace NonbondedSlicing {

/**
 * Documentation for this class is available in the file python/nonbondedslicing.i
 */

class OPENMM_EXPORT_NONBONDED_SLICING SlicedPmeForce : public OpenMM::Force {
public:
    SlicedPmeForce(int numSubsets);
    SlicedPmeForce(const OpenMM::NonbondedForce&, int numSubsets);
    int getNumSubsets() const {
        return numSubsets;
    }
    int getNumParticles() const {
        return particles.size();
    }
    int getNumExceptions() const {
        return exceptions.size();
    }
    int getNumGlobalParameters() const {
        return globalParameters.size();
    }
    int getNumParticleChargeOffsets() const {
        return particleOffsets.size();
    }
    int getNumExceptionChargeOffsets() const {
        return exceptionOffsets.size();
    }
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    double getEwaldErrorTolerance() const;
    void setEwaldErrorTolerance(double tol);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    int addParticle(double charge, int subset=0);
    int getParticleSubset(int index) const;
    void setParticleSubset(int index, int subset);
    double getParticleCharge(int index) const;
    void setParticleCharge(int index, double charge);
    int addException(int particle1, int particle2, double chargeProd, bool replace = false);
    void getExceptionParameters(int index, int& particle1, int& particle2, double& chargeProd) const;
    void setExceptionParameters(int index, int particle1, int particle2, double chargeProd);
    void createExceptionsFromBonds(const std::vector<std::pair<int, int> >& bonds, double coulomb14Scale, double lj14Scale);
    int addGlobalParameter(const std::string& name, double defaultValue);
    const std::string& getGlobalParameterName(int index) const;
    void setGlobalParameterName(int index, const std::string& name);
    double getGlobalParameterDefaultValue(int index) const;
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    int addSwitchingParameter(const std::string& parameter, int subset1, int subset2);
    int getNumSwitchingParameters() const;
    void getSwitchingParameter(int index, std::string& parameter, int& subset1, int& subset2) const;
    void setSwitchingParameter(int index, const std::string& parameter, int subset1, int subset2);
    int addSwitchingParameterDerivative(const std::string& parameter);
    int getNumSwitchingParameterDerivatives() const;
    const std::string& getSwitchingParameterDerivativeName(int index) const;
    void setSwitchingParameterDerivative(int index, const std::string& parameter);
    int addParticleChargeOffset(const std::string& parameter, int particleIndex, double chargeScale);
    void getParticleChargeOffset(int index, std::string& parameter, int& particleIndex, double& chargeScale) const;
    void setParticleChargeOffset(int index, const std::string& parameter, int particleIndex, double chargeScale);
    int addExceptionChargeOffset(const std::string& parameter, int exceptionIndex, double chargeProdScale);
    void getExceptionChargeOffset(int index, std::string& parameter, int& exceptionIndex, double& chargeProdScale) const;
    void setExceptionChargeOffset(int index, const std::string& parameter, int exceptionIndex, double chargeProdScale);
    int getReciprocalSpaceForceGroup() const;
    void setReciprocalSpaceForceGroup(int group);
    bool getIncludeDirectSpace() const;
    void setIncludeDirectSpace(bool include);
    void updateParametersInContext(OpenMM::Context& context);
    bool getExceptionsUsePeriodicBoundaryConditions() const;
    void setExceptionsUsePeriodicBoundaryConditions(bool periodic);
    bool getUseCudaFFT() const {
        return useCudaFFT;
    };
    void setUseCuFFT(bool use) {
        useCudaFFT = use;
    };
protected:
    OpenMM::ForceImpl* createImpl() const;
    bool usesPeriodicBoundaryConditions() const {return true;}
private:
    class ParticleInfo;
    class ExceptionInfo;
    class GlobalParameterInfo;
    class ParticleOffsetInfo;
    class ExceptionOffsetInfo;
    class SwitchingParameterInfo;
    int numSubsets;
    double cutoffDistance, ewaldErrorTol, alpha, dalpha;
    bool exceptionsUsePeriodic, includeDirectSpace;
    int recipForceGroup, nx, ny, nz, dnx, dny, dnz;
    bool useCudaFFT;
    void addExclusionsToSet(const std::vector<std::set<int> >& bonded12, std::set<int>& exclusions, int baseParticle, int fromParticle, int currentLevel) const;
    int getGlobalParameterIndex(const std::string& parameter) const;
    int getSwitchingParameterIndex(const std::string& parameter) const;
    std::vector<ParticleInfo> particles;
    std::vector<ExceptionInfo> exceptions;
    std::vector<GlobalParameterInfo> globalParameters;
    std::vector<ParticleOffsetInfo> particleOffsets;
    std::vector<ExceptionOffsetInfo> exceptionOffsets;
    std::map<std::pair<int, int>, int> exceptionMap;
    std::vector<SwitchingParameterInfo> switchingParameters;
    std::vector<double> switchingParameter;

    std::vector<int> switchParamDerivatives;
};

/**
 * This is an internal class used to record information about a particle.
 * @private
 */
class SlicedPmeForce::ParticleInfo {
public:
    ParticleInfo() {
        charge = 0.0;
        subset = 0;
    }
    ParticleInfo(double charge, int subset) :
        charge(charge), subset(subset) {
    }
    int subset;
    double charge;
};

/**
 * This is an internal class used to record information about an exception.
 * @private
 */
class SlicedPmeForce::ExceptionInfo {
public:
    ExceptionInfo() {
        particle1 = particle2 = -1;
        chargeProd = 0.0;
    }
    ExceptionInfo(int particle1, int particle2, double chargeProd) :
        particle1(particle1), particle2(particle2), chargeProd(chargeProd) {
    }
    int particle1, particle2;
    double chargeProd;
};

/**
 * This is an internal class used to record information about a global parameter.
 * @private
 */
class SlicedPmeForce::GlobalParameterInfo {
public:
    std::string name;
    double defaultValue;
    GlobalParameterInfo() {
    }
    GlobalParameterInfo(const std::string& name, double defaultValue) : name(name), defaultValue(defaultValue) {
    }
};

/**
 * This is an internal class used to record information about a particle charge offset.
 * @private
 */
class SlicedPmeForce::ParticleOffsetInfo {
public:
    int parameter, particle;
    double chargeScale;
    ParticleOffsetInfo() {
        particle = parameter = -1;
        chargeScale = 0.0;
    }
    ParticleOffsetInfo(int parameter, int particle, double chargeScale) :
        parameter(parameter), particle(particle), chargeScale(chargeScale) {
    }
};

/**
 * This is an internal class used to record information about an exception charge offset.
 * @private
 */
class SlicedPmeForce::ExceptionOffsetInfo {
public:
    int parameter, exception;
    double chargeProdScale;
    ExceptionOffsetInfo() {
        exception = parameter = -1;
        chargeProdScale = 0.0;
    }
    ExceptionOffsetInfo(int parameter, int exception, double chargeProdScale) :
        parameter(parameter), exception(exception), chargeProdScale(chargeProdScale) {
    }
};

/**
 * This is an internal class used to record information about a switching parameter.
 * @private
 */
class SlicedPmeForce::SwitchingParameterInfo {
public:
    int globalParamIndex, subset1, subset2, slice;
    SwitchingParameterInfo() {
        globalParamIndex = subset1 = subset2 = slice = -1;
    }
    SwitchingParameterInfo(int globalParamIndex, int subset1, int subset2) : globalParamIndex(globalParamIndex) {
        this->subset1 = subset1;
        this->subset2 = subset2;
        int i = std::min(subset1, subset2);
        int j = std::max(subset1, subset2);
        this->slice = j*(j+1)/2+i;
    }
};

} // namespace OpenMM

#endif /*OPENMM_SLICEDPMEFORCE_H_*/