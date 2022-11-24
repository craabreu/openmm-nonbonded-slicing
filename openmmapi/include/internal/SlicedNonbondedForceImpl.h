#ifndef OPENMM_SLICEDNONBONDEDFORCEIMPL_H_
#define OPENMM_SLICEDNONBONDEDFORCEIMPL_H_

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
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <utility>
#include <set>
#include <string>

using namespace OpenMM;

namespace PmeSlicing {

/**
 * This is the internal implementation of SlicedNonbondedForce.
 */

class OPENMM_EXPORT_PMESLICING SlicedNonbondedForceImpl : public ForceImpl {
public:
    SlicedNonbondedForceImpl(const SlicedNonbondedForce& owner);
    ~SlicedNonbondedForceImpl();
    void initialize(ContextImpl& context);
    const SlicedNonbondedForce& getOwner() const {
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
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * This is a utility routine that calculates the values to use for alpha and kmax when using
     * Ewald summation.
     */
    static void calcEwaldParameters(const System& system, const SlicedNonbondedForce& force, double& alpha, int& kmaxx, int& kmaxy, int& kmaxz);
    /**
     * This is a utility routine that calculates the values to use for alpha and grid size when using
     * Particle Mesh Ewald.
     */
    static void calcPMEParameters(const System& system, const SlicedNonbondedForce& force, double& alpha, int& xsize, int& ysize, int& zsize, bool lj);
    /**
     * Compute the coefficient which, when divided by the periodic box volume, gives the
     * long range dispersion correction to the energy.
     */
    static double calcDispersionCorrection(const System& system, const SlicedNonbondedForce& force);
private:
    class ErrorFunction;
    class EwaldErrorFunction;
    static int findZero(const ErrorFunction& f, int initialGuess);
    static double evalIntegral(double r, double rs, double rc, double sigma);
    const SlicedNonbondedForce& owner;
    Kernel kernel;
};

} // namespace PmeSlicing

#endif /*OPENMM_SLICEDNONBONDEDFORCEIMPL_H_*/
