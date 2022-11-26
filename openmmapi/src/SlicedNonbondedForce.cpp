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

#include "SlicedNonbondedForce.h"
#include "internal/SlicedNonbondedForceImpl.h"
#include "openmm/internal/AssertionUtilities.h"
#include <string.h>
#include <algorithm>

using namespace std;
using namespace OpenMM;
using namespace PmeSlicing;

#define ASSERT_VALID(name, value, number) {if (value < 0 || value >= number) throwException(__FILE__, __LINE__, name " out of range");};

SlicedNonbondedForce::SlicedNonbondedForce(int numSubsets) :
    NonbondedForce(), numSubsets(numSubsets), useCudaFFT(false) {
}

SlicedNonbondedForce::SlicedNonbondedForce(const NonbondedForce& force, int numSubsets) {
    setForceGroup(force.getForceGroup());
    setName(force.getName());
    setNonbondedMethod((SlicedNonbondedForce::NonbondedMethod) force.getNonbondedMethod());
    setCutoffDistance(force.getCutoffDistance());
    setUseSwitchingFunction(force.getUseSwitchingFunction());
    setSwitchingDistance(force.getSwitchingDistance());
    setEwaldErrorTolerance(force.getEwaldErrorTolerance());
    setReactionFieldDielectric(force.getReactionFieldDielectric());
    setUseDispersionCorrection(force.getUseDispersionCorrection());
    setIncludeDirectSpace(force.getIncludeDirectSpace());
    double alpha;
    int nx, ny, nz;
    force.getPMEParameters(alpha, nx, ny, nz);
    setPMEParameters(alpha, nx, ny, nz);
    force.getLJPMEParameters(alpha, nx, ny, nz);
    setReciprocalSpaceForceGroup(force.getReciprocalSpaceForceGroup());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        addParticle(charge, sigma, epsilon);
    }
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        addException(particle1, particle2, chargeProd, sigma, epsilon);
    }
    setExceptionsUsePeriodicBoundaryConditions(force.getExceptionsUsePeriodicBoundaryConditions());
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        addGlobalParameter(force.getGlobalParameterName(i), force.getGlobalParameterDefaultValue(i));
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string parameter;
        int index;
        double chargeScale, sigmaScale, epsilonScale;
        force.getParticleParameterOffset(i, parameter, index, chargeScale, sigmaScale, epsilonScale);
        addParticleParameterOffset(parameter, index, chargeScale, sigmaScale, epsilonScale);
    }
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        string parameter;
        int index;
        double chargeProdScale, sigmaScale, epsilonScale;
        force.getExceptionParameterOffset(i, parameter, index, chargeProdScale, sigmaScale, epsilonScale);
        addExceptionParameterOffset(parameter, index, chargeProdScale, sigmaScale, epsilonScale);
    }
}

void SlicedNonbondedForce::setParticleSubset(int index, int subset) {
    ASSERT_VALID("Index", index, getNumParticles());
    ASSERT_VALID("Subset", subset, numSubsets);
    subsets[index] = subset;
}

int SlicedNonbondedForce::getParticleSubset(int index) const {
    ASSERT_VALID("Index", index, getNumParticles());
    auto element = subsets.find(index);
    return element == subsets.end() ? 0 : element->second;
}

int SlicedNonbondedForce::getGlobalParameterIndex(const string& parameter) const {
    for (int i = 0; i < getNumGlobalParameters(); i++)
        if (getGlobalParameterName(i) == parameter)
            return i;
    throw OpenMMException("There is no global parameter called '"+parameter+"'");
}

int SlicedNonbondedForce::addScalingParameter(const string& parameter, int subset1, int subset2, bool includeLJ, bool includeCoulomb) {
    ASSERT_VALID("Subset", subset1, numSubsets);
    ASSERT_VALID("Subset", subset2, numSubsets);
    ScalingParameterInfo info = ScalingParameterInfo(getGlobalParameterIndex(parameter), subset1, subset2, includeLJ, includeCoulomb);
    for (auto param : scalingParameters)
        if (param.clashesWith(info))
            throwException(__FILE__, __LINE__, "A scaling parameter has already been defined for this slice & contribution(s)");
    scalingParameters.push_back(info);
    return scalingParameters.size()-1;
}

void SlicedNonbondedForce::getScalingParameter(int index, string& parameter, int& subset1, int& subset2, bool& includeLJ, bool& includeCoulomb) const {
    ASSERT_VALID("Index", index, scalingParameters.size());
    ScalingParameterInfo* info = (ScalingParameterInfo*) &scalingParameters[index];
    parameter = getGlobalParameterName(info->globalParamIndex);
    subset1 = info->subset1;
    subset2 = info->subset2;
    includeLJ = info->includeLJ;
    includeCoulomb = info->includeCoulomb;
}

void SlicedNonbondedForce::setScalingParameter(int index, const string& parameter, int subset1, int subset2, bool includeLJ, bool includeCoulomb) {
    ASSERT_VALID("Index", index, scalingParameters.size());
    ASSERT_VALID("Subset", subset1, numSubsets);
    ASSERT_VALID("Subset", subset2, numSubsets);
    ScalingParameterInfo info = ScalingParameterInfo(getGlobalParameterIndex(parameter), subset1, subset2, includeLJ, includeCoulomb);
    ScalingParameterInfo old = scalingParameters[index];
    if (!old.clashesWith(info))
        for (auto param : scalingParameters)
            if (param.clashesWith(info))
                throwException(__FILE__, __LINE__, "A scaling parameter has already been defined for this slice & contribution(s)");
    scalingParameters[index] = info;
}

int SlicedNonbondedForce::getScalingParameterIndex(const string& parameter) const {
    for (int i = 0; i < scalingParameters.size(); i++) {
        int index = scalingParameters[i].globalParamIndex;
        if (getGlobalParameterName(index) == parameter)
            return i;
    }
    throw OpenMMException("There is no scaling parameter called '"+parameter+"'");
}

int SlicedNonbondedForce::addScalingParameterDerivative(const string& parameter) {
    int scalingParameterIndex = getScalingParameterIndex(parameter);
    auto begin = scalingParameterDerivatives.begin();
    auto end = scalingParameterDerivatives.end();
    if (find(begin, end, scalingParameterIndex) != end)
        throwException(__FILE__, __LINE__, "This scaling parameter derivative has already been requested");
    scalingParameterDerivatives.push_back(scalingParameterIndex);
    return scalingParameterDerivatives.size()-1;
}

const string& SlicedNonbondedForce::getScalingParameterDerivativeName(int index) const {
    ASSERT_VALID("Index", index, scalingParameterDerivatives.size());
    int globalParamIndex = scalingParameters[scalingParameterDerivatives[index]].globalParamIndex;
    return getGlobalParameterName(globalParamIndex);
}

void SlicedNonbondedForce::setScalingParameterDerivative(int index, const string& parameter) {
    ASSERT_VALID("Index", index, scalingParameterDerivatives.size());
    int scalingParameterIndex = getScalingParameterIndex(parameter);
    if (scalingParameterDerivatives[index] != scalingParameterIndex) {
        auto begin = scalingParameterDerivatives.begin();
        auto end = scalingParameterDerivatives.end();
        if (find(begin, end, scalingParameterIndex) != end)
            throwException(__FILE__, __LINE__, "This scaling parameter derivative has already been requested");
        scalingParameterDerivatives[index] = scalingParameterIndex;
    }
}

ForceImpl* SlicedNonbondedForce::createImpl() const {
    return new SlicedNonbondedForceImpl(*this);
}

void SlicedNonbondedForce::getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const SlicedNonbondedForceImpl&>(getImplInContext(context)).getPMEParameters(alpha, nx, ny, nz);
}

void SlicedNonbondedForce::getLJPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const SlicedNonbondedForceImpl&>(getImplInContext(context)).getLJPMEParameters(alpha, nx, ny, nz);
}

void SlicedNonbondedForce::updateParametersInContext(Context& context) {
    dynamic_cast<SlicedNonbondedForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}