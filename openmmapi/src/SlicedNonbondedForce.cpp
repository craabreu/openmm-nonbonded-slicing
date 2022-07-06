/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2021 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "SlicedNonbondedForce.h"
#include "internal/SlicedNonbondedForceImpl.h"
#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <cmath>
#include <map>
#include <sstream>
#include <utility>

using namespace NonbondedSlicing;
using namespace OpenMM;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

#define ASSERT_VALID_SUBSET(subset) {if (subset < 0 || subset >= numSubsets) throwException(__FILE__, __LINE__, "Subset out of range");};

SlicedNonbondedForce::SlicedNonbondedForce(int numSubsets) : numSubsets(numSubsets), nonbondedMethod(NoCutoff), cutoffDistance(1.0), switchingDistance(-1.0), rfDielectric(78.3),
        ewaldErrorTol(5e-4), alpha(0.0), dalpha(0.0), useSwitchingFunction(false), useDispersionCorrection(true), exceptionsUsePeriodic(false), recipForceGroup(-1),
        includeDirectSpace(true), nx(0), ny(0), nz(0), dnx(0), dny(0), dnz(0) {
    vector<int> row(numSubsets, -1);
    for (int i = 0; i < numSubsets; i++)
        sliceForceGroup.push_back(row);
}

SlicedNonbondedForce::SlicedNonbondedForce(const NonbondedForce& force, int numSubsets) : numSubsets(numSubsets) {
    nonbondedMethod = static_cast<NonbondedMethod>(force.getNonbondedMethod());
    cutoffDistance = force.getCutoffDistance();
    switchingDistance = force.getSwitchingDistance();
    rfDielectric = force.getReactionFieldDielectric();
    ewaldErrorTol = force.getEwaldErrorTolerance();
    force.getPMEParameters(alpha, nx, ny, nz);
    force.getLJPMEParameters(dalpha, dnx, dny, dnz);
    useSwitchingFunction = force.getUseSwitchingFunction();
    useDispersionCorrection = force.getUseDispersionCorrection();
    exceptionsUsePeriodic = force.getExceptionsUsePeriodicBoundaryConditions();
    recipForceGroup = force.getReciprocalSpaceForceGroup();
    includeDirectSpace = force.getIncludeDirectSpace();

    for (int index = 0; index < force.getNumParticles(); index++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(index, charge, sigma, epsilon);
        addParticle(charge);
    }

    for (int index = 0; index < force.getNumExceptions(); index++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(index, particle1, particle2, chargeProd, sigma, epsilon);
        addException(particle1, particle2, chargeProd);
    }

    for (int index = 0; index < force.getNumGlobalParameters(); index++)
        addGlobalParameter(force.getGlobalParameterName(index),
                           force.getGlobalParameterDefaultValue(index));

    for (int index = 0; index < force.getNumParticleParameterOffsets(); index++) {
        std::string parameter;
        int particleIndex;
        double chargeScale, sigmaScale, epsilonScale;
        force.getParticleParameterOffset(index, parameter, particleIndex, chargeScale, sigmaScale, epsilonScale);
        addParticleParameterOffset(parameter, particleIndex, chargeScale);
    }

    for (int index = 0; index < force.getNumExceptionParameterOffsets(); index++) {
        std::string parameter;
        int exceptionIndex;
        double chargeProdScale, sigmaScale, epsilonScale;
        force.getExceptionParameterOffset(index, parameter, exceptionIndex, chargeProdScale, sigmaScale, epsilonScale);
        addExceptionParameterOffset(parameter, exceptionIndex, chargeProdScale);
    }
}

SlicedNonbondedForce::NonbondedMethod SlicedNonbondedForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void SlicedNonbondedForce::setNonbondedMethod(NonbondedMethod method) {
    if (method < 0 || method > 5)
        throw OpenMMException("SlicedNonbondedForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

double SlicedNonbondedForce::getCutoffDistance() const {
    return cutoffDistance;
}

void SlicedNonbondedForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

bool SlicedNonbondedForce::getUseSwitchingFunction() const {
    return useSwitchingFunction;
}

void SlicedNonbondedForce::setUseSwitchingFunction(bool use) {
    useSwitchingFunction = use;
}

double SlicedNonbondedForce::getSwitchingDistance() const {
    return switchingDistance;
}

void SlicedNonbondedForce::setSwitchingDistance(double distance) {
    switchingDistance = distance;
}

double SlicedNonbondedForce::getReactionFieldDielectric() const {
    return rfDielectric;
}

void SlicedNonbondedForce::setReactionFieldDielectric(double dielectric) {
    rfDielectric = dielectric;
}

double SlicedNonbondedForce::getEwaldErrorTolerance() const {
    return ewaldErrorTol;
}

void SlicedNonbondedForce::setEwaldErrorTolerance(double tol) {
    ewaldErrorTol = tol;
}

void SlicedNonbondedForce::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = this->nx;
    ny = this->ny;
    nz = this->nz;
}

void SlicedNonbondedForce::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->dalpha;
    nx = this->dnx;
    ny = this->dny;
    nz = this->dnz;
}

void SlicedNonbondedForce::setPMEParameters(double alpha, int nx, int ny, int nz) {
    this->alpha = alpha;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
}

void SlicedNonbondedForce::setLJPMEParameters(double alpha, int nx, int ny, int nz) {
    this->dalpha = alpha;
    this->dnx = nx;
    this->dny = ny;
    this->dnz = nz;
}

void SlicedNonbondedForce::getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const SlicedNonbondedForceImpl&>(getImplInContext(context)).getPMEParameters(alpha, nx, ny, nz);
}

void SlicedNonbondedForce::getLJPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const SlicedNonbondedForceImpl&>(getImplInContext(context)).getLJPMEParameters(alpha, nx, ny, nz);
}

int SlicedNonbondedForce::addParticle(double charge, int subset) {
    particles.push_back(ParticleInfo(charge, subset));
    return particles.size()-1;
}

int SlicedNonbondedForce::getParticleSubset(int index) const {
    ASSERT_VALID_INDEX(index, particles);
    return particles[index].subset;
}

void SlicedNonbondedForce::setParticleSubset(int index, int subset) {
    ASSERT_VALID_INDEX(index, particles);
    ASSERT_VALID_SUBSET(subset);
    particles[index].subset = subset;
}

double SlicedNonbondedForce::getParticleCharge(int index) const {
    ASSERT_VALID_INDEX(index, particles);
    return particles[index].charge;
}

void SlicedNonbondedForce::setParticleCharge(int index, double charge) {
    ASSERT_VALID_INDEX(index, particles);
    particles[index].charge = charge;
}

int SlicedNonbondedForce::addException(int particle1, int particle2, double chargeProd, bool replace) {
    map<pair<int, int>, int>::iterator iter = exceptionMap.find(pair<int, int>(particle1, particle2));
    int newIndex;
    if (iter == exceptionMap.end())
        iter = exceptionMap.find(pair<int, int>(particle2, particle1));
    if (iter != exceptionMap.end()) {
        if (!replace) {
            stringstream msg;
            msg << "SlicedNonbondedForce: There is already an exception for particles ";
            msg << particle1;
            msg << " and ";
            msg << particle2;
            throw OpenMMException(msg.str());
        }
        exceptions[iter->second] = ExceptionInfo(particle1, particle2, chargeProd);
        newIndex = iter->second;
        exceptionMap.erase(iter->first);
    }
    else {
        exceptions.push_back(ExceptionInfo(particle1, particle2, chargeProd));
        newIndex = exceptions.size()-1;
    }
    exceptionMap[pair<int, int>(particle1, particle2)] = newIndex;
    return newIndex;
}

void SlicedNonbondedForce::getExceptionParameters(int index, int& particle1, int& particle2, double& chargeProd) const {
    ASSERT_VALID_INDEX(index, exceptions);
    particle1 = exceptions[index].particle1;
    particle2 = exceptions[index].particle2;
    chargeProd = exceptions[index].chargeProd;
}

void SlicedNonbondedForce::setExceptionParameters(int index, int particle1, int particle2, double chargeProd) {
    ASSERT_VALID_INDEX(index, exceptions);
    exceptions[index].particle1 = particle1;
    exceptions[index].particle2 = particle2;
    exceptions[index].chargeProd = chargeProd;
}

ForceImpl* SlicedNonbondedForce::createImpl() const {
    return new SlicedNonbondedForceImpl(*this);
}

void SlicedNonbondedForce::createExceptionsFromBonds(const vector<pair<int, int> >& bonds, double coulomb14Scale, double lj14Scale) {
    for (auto& bond : bonds)
        if (bond.first < 0 || bond.second < 0 || bond.first >= particles.size() || bond.second >= particles.size())
            throw OpenMMException("createExceptionsFromBonds: Illegal particle index in list of bonds");

    // Find particles separated by 1, 2, or 3 bonds.

    vector<set<int> > exclusions(particles.size());
    vector<set<int> > bonded12(exclusions.size());
    for (auto& bond : bonds) {
        bonded12[bond.first].insert(bond.second);
        bonded12[bond.second].insert(bond.first);
    }
    for (int i = 0; i < (int) exclusions.size(); ++i)
        addExclusionsToSet(bonded12, exclusions[i], i, i, 2);

    // Find particles separated by 1 or 2 bonds and create the exceptions.

    for (int i = 0; i < (int) exclusions.size(); ++i) {
        set<int> bonded13;
        addExclusionsToSet(bonded12, bonded13, i, i, 1);
        for (int j : exclusions[i]) {
            if (j < i) {
                if (bonded13.find(j) == bonded13.end()) {
                    // This is a 1-4 interaction.

                    const ParticleInfo& particle1 = particles[j];
                    const ParticleInfo& particle2 = particles[i];
                    const double chargeProd = coulomb14Scale*particle1.charge*particle2.charge;
                    addException(j, i, chargeProd);
                }
                else {
                    // This interaction should be completely excluded.

                    addException(j, i, 0.0);
                }
            }
        }
    }
}

void SlicedNonbondedForce::addExclusionsToSet(const vector<set<int> >& bonded12, set<int>& exclusions, int baseParticle, int fromParticle, int currentLevel) const {
    for (int i : bonded12[fromParticle]) {
        if (i != baseParticle)
            exclusions.insert(i);
        if (currentLevel > 0)
            addExclusionsToSet(bonded12, exclusions, baseParticle, i, currentLevel-1);
    }
}

int SlicedNonbondedForce::addGlobalParameter(const string& name, double defaultValue) {
    globalParameters.push_back(GlobalParameterInfo(name, defaultValue));
    return globalParameters.size()-1;
}

const string& SlicedNonbondedForce::getGlobalParameterName(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].name;
}

void SlicedNonbondedForce::setGlobalParameterName(int index, const string& name) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].name = name;
}

double SlicedNonbondedForce::getGlobalParameterDefaultValue(int index) const {
    ASSERT_VALID_INDEX(index, globalParameters);
    return globalParameters[index].defaultValue;
}

void SlicedNonbondedForce::setGlobalParameterDefaultValue(int index, double defaultValue) {
    ASSERT_VALID_INDEX(index, globalParameters);
    globalParameters[index].defaultValue = defaultValue;
}

int SlicedNonbondedForce::getGlobalParameterIndex(const std::string& parameter) const {
    for (int i = 0; i < globalParameters.size(); i++)
        if (globalParameters[i].name == parameter)
            return i;
    throw OpenMMException("There is no global parameter called '"+parameter+"'");
}

int SlicedNonbondedForce::addParticleParameterOffset(const std::string& parameter, int particleIndex, double chargeScale) {
    particleOffsets.push_back(ParticleOffsetInfo(getGlobalParameterIndex(parameter), particleIndex, chargeScale));
    return particleOffsets.size()-1;
}

void SlicedNonbondedForce::getParticleParameterOffset(int index, std::string& parameter, int& particleIndex, double& chargeScale) const {
    ASSERT_VALID_INDEX(index, particleOffsets);
    parameter = globalParameters[particleOffsets[index].parameter].name;
    particleIndex = particleOffsets[index].particle;
    chargeScale = particleOffsets[index].chargeScale;
}

void SlicedNonbondedForce::setParticleParameterOffset(int index, const std::string& parameter, int particleIndex, double chargeScale) {
    ASSERT_VALID_INDEX(index, particleOffsets);
    particleOffsets[index].parameter = getGlobalParameterIndex(parameter);
    particleOffsets[index].particle = particleIndex;
    particleOffsets[index].chargeScale = chargeScale;
}

int SlicedNonbondedForce::addExceptionParameterOffset(const std::string& parameter, int exceptionIndex, double chargeProdScale) {
    exceptionOffsets.push_back(ExceptionOffsetInfo(getGlobalParameterIndex(parameter), exceptionIndex, chargeProdScale));
    return exceptionOffsets.size()-1;
}

void SlicedNonbondedForce::getExceptionParameterOffset(int index, std::string& parameter, int& exceptionIndex, double& chargeProdScale) const {
    ASSERT_VALID_INDEX(index, exceptionOffsets);
    parameter = globalParameters[exceptionOffsets[index].parameter].name;
    exceptionIndex = exceptionOffsets[index].exception;
    chargeProdScale = exceptionOffsets[index].chargeProdScale;
}

void SlicedNonbondedForce::setExceptionParameterOffset(int index, const std::string& parameter, int exceptionIndex, double chargeProdScale) {
    ASSERT_VALID_INDEX(index, exceptionOffsets);
    exceptionOffsets[index].parameter = getGlobalParameterIndex(parameter);
    exceptionOffsets[index].exception = exceptionIndex;
    exceptionOffsets[index].chargeProdScale = chargeProdScale;
}

int SlicedNonbondedForce::getReciprocalSpaceForceGroup() const {
    return recipForceGroup;
}

void SlicedNonbondedForce::setReciprocalSpaceForceGroup(int group) {
    if (group < -1 || group > 31)
        throw OpenMMException("Force group must be between -1 and 31");
    recipForceGroup = group;
}

bool SlicedNonbondedForce::getIncludeDirectSpace() const {
    return includeDirectSpace;
}

void SlicedNonbondedForce::setIncludeDirectSpace(bool include) {
    includeDirectSpace = include;
}

void SlicedNonbondedForce::updateParametersInContext(Context& context) {
    dynamic_cast<SlicedNonbondedForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

bool SlicedNonbondedForce::getExceptionsUsePeriodicBoundaryConditions() const {
    return exceptionsUsePeriodic;
}

void SlicedNonbondedForce::setExceptionsUsePeriodicBoundaryConditions(bool periodic) {
    exceptionsUsePeriodic = periodic;
}

int SlicedNonbondedForce::getSliceForceGroup(int subset1, int subset2) const {
    ASSERT_VALID_SUBSET(subset1);
    ASSERT_VALID_SUBSET(subset2);
    return sliceForceGroup[subset1][subset2];
}

void SlicedNonbondedForce::setSliceForceGroup(int subset1, int subset2, int group) {
    if (group < -1 || group > 31)
        throw OpenMMException("Argument group must be between -1 and 31");
    ASSERT_VALID_SUBSET(subset1);
    ASSERT_VALID_SUBSET(subset2);
    int i = std::min(subset1, subset2);
    int j = std::max(subset1, subset2);
    sliceForceGroup[i][j] = sliceForceGroup[j][i] = group;
}
