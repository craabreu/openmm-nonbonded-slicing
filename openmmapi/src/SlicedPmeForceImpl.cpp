/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential calculations on the basis *
 * of atom pairs and for applying scaling parameters to selected slices.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "internal/SlicedPmeForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "PmeSlicingKernels.h"
#include <cmath>
#include <map>
#include <sstream>
#include <algorithm>
#include <iostream>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

SlicedPmeForceImpl::SlicedPmeForceImpl(const SlicedPmeForce& owner) : owner(owner) {
}

SlicedPmeForceImpl::~SlicedPmeForceImpl() {
}

void SlicedPmeForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcSlicedPmeForceKernel::Name(), context);

    // Check for errors in the specification of exceptions.

    const System& system = context.getSystem();
    if (owner.getNumParticles() != system.getNumParticles())
        throw OpenMMException("SlicedPmeForce must have exactly as many particles as the System it belongs to.");

    vector<set<int> > exceptions(owner.getNumParticles());
    for (int i = 0; i < owner.getNumExceptions(); i++) {
        int particle[2];
        double chargeProd;
        owner.getExceptionParameters(i, particle[0], particle[1], chargeProd);
        for (int j = 0; j < 2; j++) {
            if (particle[j] < 0 || particle[j] >= owner.getNumParticles()) {
                stringstream msg;
                msg << "SlicedPmeForce: Illegal particle index for an exception: ";
                msg << particle[j];
                throw OpenMMException(msg.str());
            }
        }
        if (exceptions[particle[0]].count(particle[1]) > 0 || exceptions[particle[1]].count(particle[0]) > 0) {
            stringstream msg;
            msg << "SlicedPmeForce: Multiple exceptions are specified for particles ";
            msg << particle[0];
            msg << " and ";
            msg << particle[1];
            throw OpenMMException(msg.str());
        }
        exceptions[particle[0]].insert(particle[1]);
        exceptions[particle[1]].insert(particle[0]);
    }

    for (int i = 0; i < owner.getNumParticleChargeOffsets(); i++) {
        string parameter;
        int particleIndex;
        double chargeScale;
        owner.getParticleChargeOffset(i, parameter, particleIndex, chargeScale);
        if (particleIndex < 0 || particleIndex >= owner.getNumParticles()) {
            stringstream msg;
            msg << "SlicedPmeForce: Illegal particle index for a particle charge offset: ";
            msg << particleIndex;
            throw OpenMMException(msg.str());
        }
    }

    for (int i = 0; i < owner.getNumExceptionChargeOffsets(); i++) {
        string parameter;
        int exceptionIndex;
        double chargeScale;
        owner.getExceptionChargeOffset(i, parameter, exceptionIndex, chargeScale);
        if (exceptionIndex < 0 || exceptionIndex >= owner.getNumExceptions()) {
            stringstream msg;
            msg << "SlicedPmeForce: Illegal exception index for an exception charge offset: ";
            msg << exceptionIndex;
            throw OpenMMException(msg.str());
        }
    }

    Vec3 boxVectors[3];
    system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    double cutoff = owner.getCutoffDistance();
    if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
        throw OpenMMException("SlicedPmeForce: The cutoff distance cannot be greater than half the periodic box size.");

    set<string> offsetParams;
    string parameter;
    int particle, subset1, subset2;
    double value;
    for (int index = 0; index < owner.getNumParticleChargeOffsets(); index++) {
        owner.getParticleChargeOffset(index, parameter, particle, value);
        offsetParams.insert(parameter);
    }
    for (int index = 0; index < owner.getNumExceptionChargeOffsets(); index++) {
        owner.getExceptionChargeOffset(index, parameter, particle, value);
        offsetParams.insert(parameter);
    }
    for (int index = 0; index < owner.getNumSwitchingParameters(); index++) {
        owner.getSwitchingParameter(index, parameter, subset1, subset2);
        if (offsetParams.find(parameter) != offsetParams.end())
            throw OpenMMException("SlicedPmeForce: A switching parameter cannot be used for charge offset.");
    }

    kernel.getAs<CalcSlicedPmeForceKernel>().initialize(context.getSystem(), owner);
}

double SlicedPmeForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    bool includeDirect = (owner.getIncludeDirectSpace() && (groups&(1<<owner.getForceGroup())) != 0);
    int reciprocalGroup = owner.getReciprocalSpaceForceGroup();
    if (reciprocalGroup < 0)
        reciprocalGroup = owner.getForceGroup();
    bool includeReciprocal = ((groups&(1<<reciprocalGroup)) != 0);
    return kernel.getAs<CalcSlicedPmeForceKernel>().execute(context, includeForces, includeEnergy, includeDirect, includeReciprocal);
}

map<string, double> SlicedPmeForceImpl::getDefaultParameters() {
    map<string, double> parameters;
    for (int i = 0; i < owner.getNumGlobalParameters(); i++)
        parameters[owner.getGlobalParameterName(i)] = owner.getGlobalParameterDefaultValue(i);
    return parameters;
}

std::vector<std::string> SlicedPmeForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcSlicedPmeForceKernel::Name());
    return names;
}

class SlicedPmeForceImpl::ErrorFunction {
public:
    virtual double getValue(int arg) const = 0;
};

class SlicedPmeForceImpl::EwaldErrorFunction : public ErrorFunction {
public:
    EwaldErrorFunction(double width, double alpha, double target) : width(width), alpha(alpha), target(target) {
    }
    double getValue(int arg) const {
        double temp = arg*M_PI/(width*alpha);
        return target-0.05*sqrt(width*alpha)*arg*exp(-temp*temp);
    }
private:
    double width, alpha, target;
};

void SlicedPmeForceImpl::calcEwaldParameters(const System& system, const SlicedPmeForce& force, double& alpha, int& kmaxx, int& kmaxy, int& kmaxz) {
    Vec3 boxVectors[3];
    system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    double tol = force.getEwaldErrorTolerance();
    alpha = (1.0/force.getCutoffDistance())*std::sqrt(-log(2.0*tol));
    kmaxx = findZero(EwaldErrorFunction(boxVectors[0][0], alpha, tol), 10);
    kmaxy = findZero(EwaldErrorFunction(boxVectors[1][1], alpha, tol), 10);
    kmaxz = findZero(EwaldErrorFunction(boxVectors[2][2], alpha, tol), 10);
    if (kmaxx%2 == 0)
        kmaxx++;
    if (kmaxy%2 == 0)
        kmaxy++;
    if (kmaxz%2 == 0)
        kmaxz++;
}

void SlicedPmeForceImpl::calcPMEParameters(const System& system, const SlicedPmeForce& force, double& alpha, int& xsize, int& ysize, int& zsize, bool lj) {
    force.getPMEParameters(alpha, xsize, ysize, zsize);
    if (alpha == 0.0) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double tol = force.getEwaldErrorTolerance();
        alpha = (1.0/force.getCutoffDistance())*std::sqrt(-log(2.0*tol));
        if (lj) {
            xsize = (int) ceil(alpha*boxVectors[0][0]/(3*pow(tol, 0.2)));
            ysize = (int) ceil(alpha*boxVectors[1][1]/(3*pow(tol, 0.2)));
            zsize = (int) ceil(alpha*boxVectors[2][2]/(3*pow(tol, 0.2)));
        }
        else {
            xsize = (int) ceil(2*alpha*boxVectors[0][0]/(3*pow(tol, 0.2)));
            ysize = (int) ceil(2*alpha*boxVectors[1][1]/(3*pow(tol, 0.2)));
            zsize = (int) ceil(2*alpha*boxVectors[2][2]/(3*pow(tol, 0.2)));
        }
        xsize = max(xsize, 6);
        ysize = max(ysize, 6);
        zsize = max(zsize, 6);
    }
}

int SlicedPmeForceImpl::findZero(const SlicedPmeForceImpl::ErrorFunction& f, int initialGuess) {
    int arg = initialGuess;
    double value = f.getValue(arg);
    if (value > 0.0) {
        while (value > 0.0 && arg > 0)
            value = f.getValue(--arg);
        return arg+1;
    }
    while (value < 0.0)
        value = f.getValue(++arg);
    return arg;
}

void SlicedPmeForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcSlicedPmeForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void SlicedPmeForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcSlicedPmeForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}
