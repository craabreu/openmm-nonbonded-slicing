/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "internal/SlicedNonbondedForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "NonbondedSlicingKernels.h"
#include <cmath>
#include <map>
#include <sstream>
#include <algorithm>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

SlicedNonbondedForceImpl::SlicedNonbondedForceImpl(const SlicedNonbondedForce& owner) : NonbondedForceImpl(owner), owner(owner) {
}

SlicedNonbondedForceImpl::~SlicedNonbondedForceImpl() {
}

void SlicedNonbondedForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcSlicedNonbondedForceKernel::Name(), context);

    // Check for errors in the specification of exceptions.

    const System& system = context.getSystem();
    if (owner.getNumParticles() != system.getNumParticles())
        throw OpenMMException("SlicedNonbondedForce must have exactly as many particles as the System it belongs to.");
    if (owner.getUseSwitchingFunction()) {
        if (owner.getSwitchingDistance() < 0 || owner.getSwitchingDistance() >= owner.getCutoffDistance())
            throw OpenMMException("SlicedNonbondedForce: Switching distance must satisfy 0 <= r_switch < r_cutoff");
    }
    for (int i = 0; i < owner.getNumParticles(); i++) {
        double charge, sigma, epsilon;
        owner.getParticleParameters(i, charge, sigma, epsilon);
        if (sigma < 0)
            throw OpenMMException("SlicedNonbondedForce: sigma for a particle cannot be negative");
        if (epsilon < 0)
            throw OpenMMException("SlicedNonbondedForce: epsilon for a particle cannot be negative");
    }
    vector<set<int> > exceptions(owner.getNumParticles());
    for (int i = 0; i < owner.getNumExceptions(); i++) {
        int particle[2];
        double chargeProd, sigma, epsilon;
        owner.getExceptionParameters(i, particle[0], particle[1], chargeProd, sigma, epsilon);
        for (int j = 0; j < 2; j++) {
            if (particle[j] < 0 || particle[j] >= owner.getNumParticles()) {
                stringstream msg;
                msg << "SlicedNonbondedForce: Illegal particle index for an exception: ";
                msg << particle[j];
                throw OpenMMException(msg.str());
            }
        }
        if (exceptions[particle[0]].count(particle[1]) > 0 || exceptions[particle[1]].count(particle[0]) > 0) {
            stringstream msg;
            msg << "SlicedNonbondedForce: Multiple exceptions are specified for particles ";
            msg << particle[0];
            msg << " and ";
            msg << particle[1];
            throw OpenMMException(msg.str());
        }
        exceptions[particle[0]].insert(particle[1]);
        exceptions[particle[1]].insert(particle[0]);
        if (sigma < 0)
            throw OpenMMException("SlicedNonbondedForce: sigma for an exception cannot be negative");
        if (epsilon < 0)
            throw OpenMMException("SlicedNonbondedForce: epsilon for an exception cannot be negative");
    }
    for (int i = 0; i < owner.getNumParticleParameterOffsets(); i++) {
        string parameter;
        int particleIndex;
        double chargeScale, sigmaScale, epsilonScale;
        owner.getParticleParameterOffset(i, parameter, particleIndex, chargeScale, sigmaScale, epsilonScale);
        if (particleIndex < 0 || particleIndex >= owner.getNumParticles()) {
            stringstream msg;
            msg << "SlicedNonbondedForce: Illegal particle index for a particle parameter offset: ";
            msg << particleIndex;
            throw OpenMMException(msg.str());
        }
    }
    for (int i = 0; i < owner.getNumExceptionParameterOffsets(); i++) {
        string parameter;
        int exceptionIndex;
        double chargeScale, sigmaScale, epsilonScale;
        owner.getExceptionParameterOffset(i, parameter, exceptionIndex, chargeScale, sigmaScale, epsilonScale);
        if (exceptionIndex < 0 || exceptionIndex >= owner.getNumExceptions()) {
            stringstream msg;
            msg << "SlicedNonbondedForce: Illegal exception index for an exception parameter offset: ";
            msg << exceptionIndex;
            throw OpenMMException(msg.str());
        }
    }
    if (owner.getNonbondedMethod() != SlicedNonbondedForce::NoCutoff && owner.getNonbondedMethod() != SlicedNonbondedForce::CutoffNonPeriodic) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("SlicedNonbondedForce: The cutoff distance cannot be greater than half the periodic box size.");
        if (owner.getNonbondedMethod() == SlicedNonbondedForce::Ewald && (boxVectors[1][0] != 0.0 || boxVectors[2][0] != 0.0 || boxVectors[2][1] != 0))
            throw OpenMMException("SlicedNonbondedForce: Ewald is not supported with non-rectangular boxes.  Use PME instead.");
    }
    set<string> offsetParams;
    string parameter;
    int offset, subset1, subset2;
    bool includeCoulomb, includeLJ;
    double charge, sigma, epsilon;
    for (int index = 0; index < owner.getNumParticleParameterOffsets(); index++) {
        owner.getParticleParameterOffset(index, parameter, offset, charge, sigma, epsilon);
        offsetParams.insert(parameter);
    }
    for (int index = 0; index < owner.getNumExceptionParameterOffsets(); index++) {
        owner.getExceptionParameterOffset(index, parameter, offset, charge, sigma, epsilon);
        offsetParams.insert(parameter);
    }
    for (int index = 0; index < owner.getNumScalingParameters(); index++) {
        owner.getScalingParameter(index, parameter, subset1, subset2, includeCoulomb, includeLJ);
        if (offsetParams.find(parameter) != offsetParams.end())
            throw OpenMMException("SlicedNonbondedForce: Cannot use a global parameter for both slice energy scaling and parameter offset.");
    }
    kernel.getAs<CalcSlicedNonbondedForceKernel>().initialize(context.getSystem(), owner);
}

double SlicedNonbondedForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    bool includeDirect = (owner.getIncludeDirectSpace() && (groups&(1<<owner.getForceGroup())) != 0);
    int reciprocalGroup = owner.getReciprocalSpaceForceGroup();
    if (reciprocalGroup < 0)
        reciprocalGroup = owner.getForceGroup();
    bool includeReciprocal = ((groups&(1<<reciprocalGroup)) != 0);
    return kernel.getAs<CalcSlicedNonbondedForceKernel>().execute(context, includeForces, includeEnergy, includeDirect, includeReciprocal);
}

std::vector<std::string> SlicedNonbondedForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcSlicedNonbondedForceKernel::Name());
    return names;
}

double SlicedNonbondedForceImpl::evalIntegral(double r, double rs, double rc, double sigma) {
    // Compute the indefinite integral of the LJ interaction multiplied by the switching function.
    // This is a large and somewhat horrifying expression, though it does grow on you if you look
    // at it long enough.  Perhaps it could be simplified further, but I got tired of working on it.

    double A = 1/(rc-rs);
    double A2 = A*A;
    double A3 = A2*A;
    double sig2 = sigma*sigma;
    double sig6 = sig2*sig2*sig2;
    double rs2 = rs*rs;
    double rs3 = rs*rs2;
    double r2 = r*r;
    double r3 = r*r2;
    double r4 = r*r3;
    double r5 = r*r4;
    double r6 = r*r5;
    double r9 = r3*r6;
    return sig6*A3*((
        sig6*(
            + rs3*28*(6*rs2*A2 + 15*rs*A + 10)
            - r*rs2*945*(rs2*A2 + 2*rs*A + 1)
            + r2*rs*1080*(2*rs2*A2 + 3*rs*A + 1)
            - r3*420*(6*rs2*A2 + 6*rs*A + 1)
            + r4*756*(2*rs*A2 + A)
            - r5*378*A2)
        -r6*(
            + rs3*84*(6*rs2*A2 + 15*rs*A + 10)
            - r*rs2*3780*(rs2*A2 + 2*rs*A + 1)
            + r2*rs*7560*(2*rs2*A2 + 3*rs*A + 1))
        )/(252*r9)
     - log(r)*10*(6*rs2*A2 + 6*rs*A + 1)
     + r*15*(2*rs*A2 + A)
     - r2*3*A2
    );
}

double SlicedNonbondedForceImpl::calcDispersionCorrection(const System& system, const SlicedNonbondedForce& force) {
    if (force.getNonbondedMethod() == SlicedNonbondedForce::NoCutoff || force.getNonbondedMethod() == SlicedNonbondedForce::CutoffNonPeriodic)
        return 0.0;

    // Record sigma and epsilon for every particle, including the default value
    // for every offset parameter.

    vector<double> sigma(force.getNumParticles()), epsilon(force.getNumParticles());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge;
        force.getParticleParameters(i, charge, sigma[i], epsilon[i]);
    }
    map<string, double> param;
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        param[force.getGlobalParameterName(i)] = force.getGlobalParameterDefaultValue(i);
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string parameter;
        int index;
        double chargeScale, sigmaScale, epsilonScale;
        force.getParticleParameterOffset(i, parameter, index, chargeScale, sigmaScale, epsilonScale);
        sigma[index] += param[parameter]*sigmaScale;
        epsilon[index] += param[parameter]*epsilonScale;
    }

    // Identify all particle classes (defined by sigma and epsilon), and count the number of
    // particles in each class.

    map<pair<double, double>, int> classCounts;
    for (int i = 0; i < force.getNumParticles(); i++) {
        pair<double, double> key = make_pair(sigma[i], epsilon[i]);
        map<pair<double, double>, int>::iterator entry = classCounts.find(key);
        if (entry == classCounts.end())
            classCounts[key] = 1;
        else
            entry->second++;
    }

    // Loop over all pairs of classes to compute the coefficient.

    double sum1 = 0, sum2 = 0, sum3 = 0;
    bool useSwitch = force.getUseSwitchingFunction();
    double cutoff = force.getCutoffDistance();
    double switchDist = force.getSwitchingDistance();
    for (map<pair<double, double>, int>::const_iterator entry = classCounts.begin(); entry != classCounts.end(); ++entry) {
        double sigma = entry->first.first;
        double epsilon = entry->first.second;
        double count = (double) entry->second;
        count *= (count + 1) / 2;
        double sigma2 = sigma*sigma;
        double sigma6 = sigma2*sigma2*sigma2;
        sum1 += count*epsilon*sigma6*sigma6;
        sum2 += count*epsilon*sigma6;
        if (useSwitch)
            sum3 += count*epsilon*(evalIntegral(cutoff, switchDist, cutoff, sigma)-evalIntegral(switchDist, switchDist, cutoff, sigma));
    }
    for (map<pair<double, double>, int>::const_iterator class1 = classCounts.begin(); class1 != classCounts.end(); ++class1)
        for (map<pair<double, double>, int>::const_iterator class2 = classCounts.begin(); class2 != class1; ++class2) {
            double sigma = 0.5*(class1->first.first+class2->first.first);
            double epsilon = sqrt(class1->first.second*class2->first.second);
            double count = (double) class1->second;
            count *= (double) class2->second;
            double sigma2 = sigma*sigma;
            double sigma6 = sigma2*sigma2*sigma2;
            sum1 += count*epsilon*sigma6*sigma6;
            sum2 += count*epsilon*sigma6;
            if (useSwitch)
                sum3 += count*epsilon*(evalIntegral(cutoff, switchDist, cutoff, sigma)-evalIntegral(switchDist, switchDist, cutoff, sigma));
        }
    double numParticles = (double) system.getNumParticles();
    double numInteractions = (numParticles*(numParticles+1))/2;
    sum1 /= numInteractions;
    sum2 /= numInteractions;
    sum3 /= numInteractions;
    return 8*numParticles*numParticles*M_PI*(sum1/(9*pow(cutoff, 9))-sum2/(3*pow(cutoff, 3))+sum3);
}

vector<double> SlicedNonbondedForceImpl::calcDispersionCorrections(const System& system, const SlicedNonbondedForce& force) {
    int numSlices = force.getNumSlices();
    vector<double> dispersionCorrections(numSlices, 0.0);
    if (force.getNonbondedMethod() == SlicedNonbondedForce::NoCutoff ||
        force.getNonbondedMethod() == SlicedNonbondedForce::CutoffNonPeriodic)
        return dispersionCorrections;

    // Record sigma and epsilon for every particle, including the default value
    // for every offset parameter.

    int numParticles = system.getNumParticles();
    vector<double> sigma(numParticles), epsilon(numParticles);
    vector<int> subset(numParticles);
    for (int i = 0; i < numParticles; i++) {
        double charge;
        force.getParticleParameters(i, charge, sigma[i], epsilon[i]);
        subset[i] = force.getParticleSubset(i);
    }
    map<string, double> param;
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        param[force.getGlobalParameterName(i)] = force.getGlobalParameterDefaultValue(i);
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        string parameter;
        int index;
        double chargeScale, sigmaScale, epsilonScale;
        force.getParticleParameterOffset(i, parameter, index, chargeScale, sigmaScale, epsilonScale);
        sigma[index] += param[parameter]*sigmaScale;
        epsilon[index] += param[parameter]*epsilonScale;
    }

    // Identify all particle classes (defined by sigma, epsilon, and subset), and count the number of
    // particles in each class.

    typedef tuple<double, double, int> ParticleClass;
    map<ParticleClass, int> classCounts;
    for (int i = 0; i < force.getNumParticles(); i++) {
        ParticleClass key = make_tuple(sigma[i], epsilon[i], subset[i]);
        map<ParticleClass, int>::iterator entry = classCounts.find(key);
        if (entry == classCounts.end())
            classCounts[key] = 1;
        else
            entry->second++;
    }

    // Loop over all pairs of classes to compute the coefficients.

    vector<double> sum1(numSlices, 0), sum2(numSlices, 0), sum3(numSlices, 0);
    bool useSwitch = force.getUseSwitchingFunction();
    double cutoff = force.getCutoffDistance();
    double switchDist = force.getSwitchingDistance();
    for (map<ParticleClass, int>::const_iterator entry = classCounts.begin(); entry != classCounts.end(); ++entry) {
        double sigma = get<0>(entry->first);
        double epsilon = get<1>(entry->first);
        int subset = get<2>(entry->first);
        int count = entry->second*(entry->second+1)/2;
        double sigmaSq = sigma*sigma;
        double sigma6 = sigmaSq*sigmaSq*sigmaSq;
        int slice = subset*(subset+3)/2;
        sum1[slice] += count*epsilon*sigma6*sigma6;
        sum2[slice] += count*epsilon*sigma6;
        if (useSwitch)
            sum3[slice] += count*epsilon*(evalIntegral(cutoff, switchDist, cutoff, sigma)-evalIntegral(switchDist, switchDist, cutoff, sigma));
    }
    for (map<ParticleClass, int>::const_iterator class1 = classCounts.begin(); class1 != classCounts.end(); ++class1) {
        double sigma1 = get<0>(class1->first);
        double epsilon1 = get<1>(class1->first);
        int s1 = get<2>(class1->first);
        for (map<ParticleClass, int>::const_iterator class2 = classCounts.begin(); class2 != class1; ++class2) {
            double sigma2 = get<0>(class2->first);
            double epsilon2 = get<1>(class2->first);
            int s2 = get<2>(class2->first);
            double sigma = 0.5*(sigma1+sigma2);
            double epsilon = sqrt(epsilon1*epsilon2);
            int slice = sliceIndex(s1, s2);
            int count = class1->second*class2->second;
            double sigmaSq = sigma*sigma;
            double sigma6 = sigmaSq*sigmaSq*sigmaSq;
            sum1[slice] += count*epsilon*sigma6*sigma6;
            sum2[slice] += count*epsilon*sigma6;
            if (useSwitch)
                sum3[slice] += count*epsilon*(evalIntegral(cutoff, switchDist, cutoff, sigma)-evalIntegral(switchDist, switchDist, cutoff, sigma));
        }
    }
    double numInteractions = (numParticles*(numParticles+1))/2;
    for (int slice = 0; slice < numSlices; slice++) {
        sum1[slice] /= numInteractions;
        sum2[slice] /= numInteractions;
        sum3[slice] /= numInteractions;
        dispersionCorrections[slice] = 8*numParticles*numParticles*M_PI*(sum1[slice]/(9*pow(cutoff, 9))-sum2[slice]/(3*pow(cutoff, 3))+sum3[slice]);
    }
    return dispersionCorrections;
}

void SlicedNonbondedForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcSlicedNonbondedForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void SlicedNonbondedForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcSlicedNonbondedForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}

void SlicedNonbondedForceImpl::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcSlicedNonbondedForceKernel>().getLJPMEParameters(alpha, nx, ny, nz);
}
