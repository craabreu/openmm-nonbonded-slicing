/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "SlicedNonbondedForceProxy.h"
#include "SlicedNonbondedForce.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/Force.h"
#include <sstream>

using namespace NonbondedSlicing;
using namespace OpenMM;
using namespace std;

SlicedNonbondedForceProxy::SlicedNonbondedForceProxy() : SerializationProxy("SlicedNonbondedForce") {
}

void SlicedNonbondedForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const SlicedNonbondedForce& force = *reinterpret_cast<const SlicedNonbondedForce*>(object);
    node.setIntProperty("numSubsets", force.getNumSubsets());
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setIntProperty("method", (int) force.getNonbondedMethod());
    node.setDoubleProperty("cutoff", force.getCutoffDistance());
    node.setBoolProperty("useSwitchingFunction", force.getUseSwitchingFunction());
    node.setDoubleProperty("switchingDistance", force.getSwitchingDistance());
    node.setDoubleProperty("ewaldTolerance", force.getEwaldErrorTolerance());
    node.setDoubleProperty("rfDielectric", force.getReactionFieldDielectric());
    node.setIntProperty("dispersionCorrection", force.getUseDispersionCorrection());
    node.setIntProperty("exceptionsUsePeriodic", force.getExceptionsUsePeriodicBoundaryConditions());
    node.setBoolProperty("includeDirectSpace", force.getIncludeDirectSpace());
    double alpha;
    int nx, ny, nz;
    force.getPMEParameters(alpha, nx, ny, nz);
    node.setDoubleProperty("alpha", alpha);
    node.setIntProperty("nx", nx);
    node.setIntProperty("ny", ny);
    node.setIntProperty("nz", nz);
    force.getLJPMEParameters(alpha, nx, ny, nz);
    node.setDoubleProperty("ljAlpha", alpha);
    node.setIntProperty("ljnx", nx);
    node.setIntProperty("ljny", ny);
    node.setIntProperty("ljnz", nz);
    node.setIntProperty("recipForceGroup", force.getReciprocalSpaceForceGroup());
    SerializationNode& globalParams = node.createChildNode("GlobalParameters");
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalParams.createChildNode("Parameter").setStringProperty("name", force.getGlobalParameterName(i)).setDoubleProperty("default", force.getGlobalParameterDefaultValue(i));
    SerializationNode& particleOffsets = node.createChildNode("ParticleOffsets");
    for (int i = 0; i < force.getNumParticleParameterOffsets(); i++) {
        int particle;
        double chargeScale, sigmaScale, epsilonScale;
        string parameter;
        force.getParticleParameterOffset(i, parameter, particle, chargeScale, sigmaScale, epsilonScale);
        particleOffsets.createChildNode("Offset").setStringProperty("parameter", parameter).setIntProperty("particle", particle).setDoubleProperty("q", chargeScale).setDoubleProperty("sig", sigmaScale).setDoubleProperty("eps", epsilonScale);
    }
    SerializationNode& exceptionOffsets = node.createChildNode("ExceptionOffsets");
    for (int i = 0; i < force.getNumExceptionParameterOffsets(); i++) {
        int exception;
        double chargeProdScale, sigmaScale, epsilonScale;
        string parameter;
        force.getExceptionParameterOffset(i, parameter, exception, chargeProdScale, sigmaScale, epsilonScale);
        exceptionOffsets.createChildNode("Offset").setStringProperty("parameter", parameter).setIntProperty("exception", exception).setDoubleProperty("q", chargeProdScale).setDoubleProperty("sig", sigmaScale).setDoubleProperty("eps", epsilonScale);
    }
    SerializationNode& particles = node.createChildNode("Particles");
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, sigma, epsilon;
        force.getParticleParameters(i, charge, sigma, epsilon);
        particles.createChildNode("Particle").setDoubleProperty("q", charge).setDoubleProperty("sig", sigma).setDoubleProperty("eps", epsilon);
    }
    SerializationNode& exceptions = node.createChildNode("Exceptions");
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        exceptions.createChildNode("Exception").setIntProperty("p1", particle1).setIntProperty("p2", particle2).setDoubleProperty("q", chargeProd).setDoubleProperty("sig", sigma).setDoubleProperty("eps", epsilon);
    }
    SerializationNode& subsets = node.createChildNode("Subsets");
    for (int i = 0; i < force.getNumParticles(); i++) {
        int subset = force.getParticleSubset(i);
        if (subset != 0)
            subsets.createChildNode("Subset").setIntProperty("index", i).setIntProperty("subset", subset);
    }
    SerializationNode& scalingParameters = node.createChildNode("scalingParameters");
    for (int i = 0; i < force.getNumScalingParameters(); i++) {
        string parameter;
        int subset1, subset2;
        bool includeCoulomb, includeLJ;
        force.getScalingParameter(i, parameter, subset1, subset2, includeCoulomb, includeLJ);
        scalingParameters.createChildNode("scalingParameter").setStringProperty("parameter", parameter).setIntProperty("subset1", subset1).setIntProperty("subset2", subset2).setBoolProperty("includeCoulomb", includeCoulomb).setBoolProperty("includeLJ", includeLJ);
    }
    SerializationNode& scalingParameterDerivatives = node.createChildNode("scalingParameterDerivatives");
    for (int i = 0; i < force.getNumScalingParameterDerivatives(); i++)
        scalingParameterDerivatives.createChildNode("scalingParameterDerivative").setStringProperty("parameter", force.getScalingParameterDerivativeName(i));
}

void* SlicedNonbondedForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version != 1)
        throw OpenMMException("Unsupported version number");
    SlicedNonbondedForce* force = new SlicedNonbondedForce(node.getIntProperty("numSubsets"));
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setNonbondedMethod((SlicedNonbondedForce::NonbondedMethod) node.getIntProperty("method"));
        force->setCutoffDistance(node.getDoubleProperty("cutoff"));
        force->setUseSwitchingFunction(node.getBoolProperty("useSwitchingFunction", false));
        force->setSwitchingDistance(node.getDoubleProperty("switchingDistance", -1.0));
        force->setEwaldErrorTolerance(node.getDoubleProperty("ewaldTolerance"));
        force->setReactionFieldDielectric(node.getDoubleProperty("rfDielectric"));
        force->setUseDispersionCorrection(node.getIntProperty("dispersionCorrection"));
        if (node.hasProperty("includeDirectSpace"))
            force->setIncludeDirectSpace(node.getBoolProperty("includeDirectSpace"));
        double alpha = node.getDoubleProperty("alpha", 0.0);
        int nx = node.getIntProperty("nx", 0);
        int ny = node.getIntProperty("ny", 0);
        int nz = node.getIntProperty("nz", 0);
        force->setPMEParameters(alpha, nx, ny, nz);
        alpha = node.getDoubleProperty("ljAlpha", 0.0);
        nx = node.getIntProperty("ljnx", 0);
        ny = node.getIntProperty("ljny", 0);
        nz = node.getIntProperty("ljnz", 0);
        force->setLJPMEParameters(alpha, nx, ny, nz);
        force->setReciprocalSpaceForceGroup(node.getIntProperty("recipForceGroup", -1));
        const SerializationNode& globalParams = node.getChildNode("GlobalParameters");
        for (auto& parameter : globalParams.getChildren())
            force->addGlobalParameter(parameter.getStringProperty("name"), parameter.getDoubleProperty("default"));
        const SerializationNode& particleOffsets = node.getChildNode("ParticleOffsets");
        for (auto& offset : particleOffsets.getChildren())
            force->addParticleParameterOffset(offset.getStringProperty("parameter"), offset.getIntProperty("particle"), offset.getDoubleProperty("q"), offset.getDoubleProperty("sig"), offset.getDoubleProperty("eps"));
        const SerializationNode& exceptionOffsets = node.getChildNode("ExceptionOffsets");
        for (auto& offset : exceptionOffsets.getChildren())
            force->addExceptionParameterOffset(offset.getStringProperty("parameter"), offset.getIntProperty("exception"), offset.getDoubleProperty("q"), offset.getDoubleProperty("sig"), offset.getDoubleProperty("eps"));
        force->setExceptionsUsePeriodicBoundaryConditions(node.getIntProperty("exceptionsUsePeriodic"));
        const SerializationNode& particles = node.getChildNode("Particles");
        for (auto& particle : particles.getChildren())
            force->addParticle(particle.getDoubleProperty("q"), particle.getDoubleProperty("sig"), particle.getDoubleProperty("eps"));
        const SerializationNode& exceptions = node.getChildNode("Exceptions");
        for (auto& exception : exceptions.getChildren())
            force->addException(exception.getIntProperty("p1"), exception.getIntProperty("p2"), exception.getDoubleProperty("q"), exception.getDoubleProperty("sig"), exception.getDoubleProperty("eps"));
        const SerializationNode& subsets = node.getChildNode("Subsets");
        for (auto& subset : subsets.getChildren())
            force->setParticleSubset(subset.getIntProperty("index"), subset.getIntProperty("subset"));
        const SerializationNode& scalingParameters = node.getChildNode("scalingParameters");
        for (auto& param : scalingParameters.getChildren())
            force->addScalingParameter(param.getStringProperty("parameter"), param.getIntProperty("subset1"), param.getIntProperty("subset2"), param.getBoolProperty("includeCoulomb"), param.getBoolProperty("includeLJ"));
        const SerializationNode& scalingParameterDerivatives = node.getChildNode("scalingParameterDerivatives");
        for (auto& param : scalingParameterDerivatives.getChildren())
            force->addScalingParameterDerivative(param.getStringProperty("parameter"));
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
