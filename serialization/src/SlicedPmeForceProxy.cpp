/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "SlicedPmeForceProxy.h"
#include "SlicedPmeForce.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/Force.h"
#include <sstream>
#include <string>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

SlicedPmeForceProxy::SlicedPmeForceProxy() : SerializationProxy("SlicedPmeForce") {
}

void SlicedPmeForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const SlicedPmeForce& force = *reinterpret_cast<const SlicedPmeForce*>(object);
    int numSubsets = force.getNumSubsets();
    node.setIntProperty("numSubsets", numSubsets);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setDoubleProperty("cutoff", force.getCutoffDistance());
    node.setDoubleProperty("ewaldTolerance", force.getEwaldErrorTolerance());
    node.setIntProperty("exceptionsUsePeriodic", force.getExceptionsUsePeriodicBoundaryConditions());
    node.setBoolProperty("includeDirectSpace", force.getIncludeDirectSpace());
    double alpha;
    int nx, ny, nz;
    force.getPMEParameters(alpha, nx, ny, nz);
    node.setDoubleProperty("alpha", alpha);
    node.setIntProperty("nx", nx);
    node.setIntProperty("ny", ny);
    node.setIntProperty("nz", nz);
    node.setIntProperty("recipForceGroup", force.getReciprocalSpaceForceGroup());
    SerializationNode& globalParams = node.createChildNode("GlobalParameters");
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalParams.createChildNode("Parameter").setStringProperty("name", force.getGlobalParameterName(i)).setDoubleProperty("default", force.getGlobalParameterDefaultValue(i));
    SerializationNode& switchingParams = node.createChildNode("SwitchingParameters");
    for (int i = 0; i < force.getNumSwitchingParameters(); i++) {
        int subset1, subset2;
        string parameter;
        force.getSwitchingParameter(i, parameter, subset1, subset2);
        switchingParams.createChildNode("switchingParameter").setStringProperty("parameter", parameter).setIntProperty("subset1", subset1).setIntProperty("subset2", subset2);
    }
    SerializationNode& particleOffsets = node.createChildNode("ParticleOffsets");
    for (int i = 0; i < force.getNumParticleChargeOffsets(); i++) {
        int particle;
        double chargeScale;
        string parameter;
        force.getParticleChargeOffset(i, parameter, particle, chargeScale);
        particleOffsets.createChildNode("Offset").setStringProperty("parameter", parameter).setIntProperty("particle", particle).setDoubleProperty("q", chargeScale);
    }
    SerializationNode& exceptionOffsets = node.createChildNode("ExceptionOffsets");
    for (int i = 0; i < force.getNumExceptionChargeOffsets(); i++) {
        int exception;
        double chargeProdScale;
        string parameter;
        force.getExceptionChargeOffset(i, parameter, exception, chargeProdScale);
        exceptionOffsets.createChildNode("Offset").setStringProperty("parameter", parameter).setIntProperty("exception", exception).setDoubleProperty("q", chargeProdScale);
    }
    SerializationNode& particles = node.createChildNode("Particles");
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge = force.getParticleCharge(i);
        int subset = force.getParticleSubset(i);
        particles.createChildNode("Particle").setDoubleProperty("q", charge).setIntProperty("subset", subset);
    }
    SerializationNode& exceptions = node.createChildNode("Exceptions");
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd;
        force.getExceptionParameters(i, particle1, particle2, chargeProd);
        exceptions.createChildNode("Exception").setIntProperty("p1", particle1).setIntProperty("p2", particle2).setDoubleProperty("q", chargeProd);
    }
}

void* SlicedPmeForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version != 1)
        throw OpenMMException("Unsupported version number");
    int numSubsets = node.getIntProperty("numSubsets", 1);
    SlicedPmeForce* force = new SlicedPmeForce(numSubsets);
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setCutoffDistance(node.getDoubleProperty("cutoff"));
        force->setEwaldErrorTolerance(node.getDoubleProperty("ewaldTolerance"));
        force->setIncludeDirectSpace(node.getBoolProperty("includeDirectSpace"));
        double alpha = node.getDoubleProperty("alpha", 0.0);
        int nx = node.getIntProperty("nx", 0);
        int ny = node.getIntProperty("ny", 0);
        int nz = node.getIntProperty("nz", 0);
        force->setPMEParameters(alpha, nx, ny, nz);
        force->setReciprocalSpaceForceGroup(node.getIntProperty("recipForceGroup", -1));
        const SerializationNode& globalParams = node.getChildNode("GlobalParameters");
        for (auto& parameter : globalParams.getChildren())
            force->addGlobalParameter(parameter.getStringProperty("name"), parameter.getDoubleProperty("default"));
        const SerializationNode& switchingParameters = node.getChildNode("SwitchingParameters");
        for (auto& parameter : switchingParameters.getChildren())
            force->addSwitchingParameter(parameter.getStringProperty("parameter"), parameter.getIntProperty("subset1"), parameter.getIntProperty("subset2"));
        const SerializationNode& particleOffsets = node.getChildNode("ParticleOffsets");
        for (auto& offset : particleOffsets.getChildren())
            force->addParticleChargeOffset(offset.getStringProperty("parameter"), offset.getIntProperty("particle"), offset.getDoubleProperty("q"));
        const SerializationNode& exceptionOffsets = node.getChildNode("ExceptionOffsets");
        for (auto& offset : exceptionOffsets.getChildren())
            force->addExceptionChargeOffset(offset.getStringProperty("parameter"), offset.getIntProperty("exception"), offset.getDoubleProperty("q"));
        force->setExceptionsUsePeriodicBoundaryConditions(node.getIntProperty("exceptionsUsePeriodic"));
        const SerializationNode& particles = node.getChildNode("Particles");
        for (auto& particle : particles.getChildren())
            force->addParticle(particle.getDoubleProperty("q"), particle.getIntProperty("subset"));
        const SerializationNode& exceptions = node.getChildNode("Exceptions");
        for (auto& exception : exceptions.getChildren())
            force->addException(exception.getIntProperty("p1"), exception.getIntProperty("p2"), exception.getDoubleProperty("q"));
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
