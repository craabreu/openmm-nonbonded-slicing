/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010-2021 Stanford University and the Authors.      *
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

#include "SlicedPmeForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace PmeSlicing;
using namespace OpenMM;
using namespace std;

void testSerialization() {
    // Create a Force.

    SlicedPmeForce force(2);
    force.setForceGroup(3);
    force.setName("custom name");
    force.setCutoffDistance(2.0);
    force.setEwaldErrorTolerance(1e-3);
    force.setExceptionsUsePeriodicBoundaryConditions(true);
    force.setIncludeDirectSpace(false);
    double alpha = 0.5;
    int nx = 3, ny = 5, nz = 7;
    force.setPMEParameters(alpha, nx, ny, nz);
    force.addParticle(1, 0);
    force.addParticle(0.5, 0);
    force.addParticle(-0.5, 1);
    force.addException(0, 1, 2);
    force.addException(1, 2, 0.2);
    force.addGlobalParameter("scale1", 1.0);
    force.addGlobalParameter("scale2", 2.0);
    force.addParticleChargeOffset("scale1", 2, 1.5);
    force.addExceptionChargeOffset("scale2", 1, -0.1);

    force.addGlobalParameter("lambda", 0.5);
    force.addCouplingParameter("lambda", 0, 1);
    force.addCouplingParameter("lambda", 1, 1);

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<SlicedPmeForce>(&force, "Force", buffer);
    SlicedPmeForce* copy = XmlSerializer::deserialize<SlicedPmeForce>(buffer);

    // Compare the two forces to see if they are identical.

    SlicedPmeForce& force2 = *copy;
    ASSERT_EQUAL(force.getNumSubsets(), force2.getNumSubsets());
    ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
    ASSERT_EQUAL(force.getName(), force2.getName());
    ASSERT_EQUAL(force.getCutoffDistance(), force2.getCutoffDistance());
    ASSERT_EQUAL(force.getEwaldErrorTolerance(), force2.getEwaldErrorTolerance());
    ASSERT_EQUAL(force.getExceptionsUsePeriodicBoundaryConditions(), force2.getExceptionsUsePeriodicBoundaryConditions());
    ASSERT_EQUAL(force.getNumParticles(), force2.getNumParticles());
    ASSERT_EQUAL(force.getNumExceptions(), force2.getNumExceptions());
    ASSERT_EQUAL(force.getNumGlobalParameters(), force2.getNumGlobalParameters());
    ASSERT_EQUAL(force.getNumCouplingParameters(), force2.getNumCouplingParameters());
    ASSERT_EQUAL(force.getNumParticleChargeOffsets(), force2.getNumParticleChargeOffsets());
    ASSERT_EQUAL(force.getNumExceptionChargeOffsets(), force2.getNumExceptionChargeOffsets());
    ASSERT_EQUAL(force.getIncludeDirectSpace(), force2.getIncludeDirectSpace());
    double alpha2;
    int nx2, ny2, nz2;
    force2.getPMEParameters(alpha2, nx2, ny2, nz2);
    ASSERT_EQUAL(alpha, alpha2);
    ASSERT_EQUAL(nx, nx2);
    ASSERT_EQUAL(ny, ny2);
    ASSERT_EQUAL(nz, nz2);
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        ASSERT_EQUAL(force.getGlobalParameterName(i), force2.getGlobalParameterName(i));
        ASSERT_EQUAL(force.getGlobalParameterDefaultValue(i), force2.getGlobalParameterDefaultValue(i));
    }
    for (int i = 0; i < force.getNumCouplingParameters(); i++) {
        string param1, param2;
        int m1, m2, n1, n2;
        force.getCouplingParameter(i, param1, m1, n1);
        force2.getCouplingParameter(i, param2, m2, n2);
        ASSERT_EQUAL(param1, param2);
        ASSERT_EQUAL(m1, m2);
        ASSERT_EQUAL(n1, n2);
    }
    for (int i = 0; i < force.getNumParticleChargeOffsets(); i++) {
        int index1, index2;
        string param1, param2;
        double charge1;
        double charge2;
        force.getParticleChargeOffset(i, param1, index1, charge1);
        force2.getParticleChargeOffset(i, param2, index2, charge2);
        ASSERT_EQUAL(index1, index1);
        ASSERT_EQUAL(param1, param2);
        ASSERT_EQUAL(charge1, charge2);
    }
    for (int i = 0; i < force.getNumExceptionChargeOffsets(); i++) {
        int index1, index2;
        string param1, param2;
        double charge1, charge2;
        force.getExceptionChargeOffset(i, param1, index1, charge1);
        force2.getExceptionChargeOffset(i, param2, index2, charge2);
        ASSERT_EQUAL(index1, index1);
        ASSERT_EQUAL(param1, param2);
        ASSERT_EQUAL(charge1, charge2);
    }
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge1 = force.getParticleCharge(i);
        double charge2 = force2.getParticleCharge(i);
        ASSERT_EQUAL(charge1, charge2);
        int subset1 = force.getParticleSubset(i);
        int subset2 = force2.getParticleSubset(i);
        ASSERT_EQUAL(subset1, subset2);
    }
    ASSERT_EQUAL(force.getNumExceptions(), force2.getNumExceptions());
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int a1, a2, b1, b2;
        double charge1, charge2;
        force.getExceptionParameters(i, a1, b1, charge1);
        force2.getExceptionParameters(i, a2, b2, charge2);
        ASSERT_EQUAL(a1, a2);
        ASSERT_EQUAL(b1, b2);
        ASSERT_EQUAL(charge1, charge2);
    }
}

int main() {
    try {
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
