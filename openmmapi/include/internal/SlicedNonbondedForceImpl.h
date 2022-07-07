#ifndef OPENMM_SLICEDNONBONDEDFORCEIMPL_H_
#define OPENMM_SLICEDNONBONDEDFORCEIMPL_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2018 Stanford University and the Authors.      *
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
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include "openmm/System.h"
#include <utility>
#include <set>
#include <string>

using namespace OpenMM;

namespace NonbondedSlicing {

/**
 * This is the internal implementation of SlicedNonbondedForce.
 */

class OPENMM_EXPORT_NONBONDEDSLICING SlicedNonbondedForceImpl : public ForceImpl {
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
private:
    class ErrorFunction;
    class EwaldErrorFunction;
    static int findZero(const ErrorFunction& f, int initialGuess);
    const SlicedNonbondedForce& owner;
    Kernel kernel;
};

} // namespace OpenMM

#endif /*OPENMM_NONBONDEDFORCEIMPL_H_*/
