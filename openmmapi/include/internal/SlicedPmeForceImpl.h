#ifndef OPENMM_SLICEDPMEFORCEIMPL_H_
#define OPENMM_SLICEDPMEFORCEIMPL_H_

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

#include "SlicedPmeForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include "openmm/System.h"
#include <utility>
#include <set>
#include <string>

using namespace OpenMM;

namespace NonbondedSlicing {

/**
 * This is the internal implementation of SlicedPmeForce.
 */

class OPENMM_EXPORT_NONBONDED_SLICING SlicedPmeForceImpl : public ForceImpl {
public:
    SlicedPmeForceImpl(const SlicedPmeForce& owner);
    ~SlicedPmeForceImpl();
    void initialize(ContextImpl& context);
    const SlicedPmeForce& getOwner() const {
        return owner;
    }
    void updateContextState(ContextImpl& context, bool& forcesInvalid) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters();
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(ContextImpl& context);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * This is a utility routine that calculates the values to use for alpha and kmax when using
     * Ewald summation.
     */
    static void calcEwaldParameters(const System& system, const SlicedPmeForce& force, double& alpha, int& kmaxx, int& kmaxy, int& kmaxz);
    /**
     * This is a utility routine that calculates the values to use for alpha and grid size when using
     * Particle Mesh Ewald.
     */
    static void calcPMEParameters(const System& system, const SlicedPmeForce& force, double& alpha, int& xsize, int& ysize, int& zsize, bool lj);
private:
    class ErrorFunction;
    class EwaldErrorFunction;
    static int findZero(const ErrorFunction& f, int initialGuess);
    const SlicedPmeForce& owner;
    Kernel kernel;
};

} // namespace OpenMM

#endif /*OPENMM_NONBONDEDFORCEIMPL_H_*/
