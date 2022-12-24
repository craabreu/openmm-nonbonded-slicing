
/* Portions copyright (c) 2006-2016 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __ReferenceCoulomb14_H__
#define __ReferenceCoulomb14_H__

#include "openmm/reference/ReferenceBondIxn.h"
#include "internal/windowsExportPmeSlicing.h"

using namespace OpenMM;

namespace NonbondedSlicing {

class ReferenceCoulomb14 : public ReferenceBondIxn {

public:

    /**---------------------------------------------------------------------------------------

       Constructor

       --------------------------------------------------------------------------------------- */

     ReferenceCoulomb14();

    /**---------------------------------------------------------------------------------------

       Destructor

       --------------------------------------------------------------------------------------- */

     ~ReferenceCoulomb14();

     /**---------------------------------------------------------------------------------------

       Set the force to use periodic boundary conditions.

       @param vectors    the vectors defining the periodic box

       --------------------------------------------------------------------------------------- */

    void setPeriodic(OpenMM::Vec3* vectors);

    /**---------------------------------------------------------------------------------------

       Calculate nonbonded 1-4 interactinos

       @param atomIndices      atom indices of the atoms in each pair
       @param atomCoordinates  atom coordinates
       @param parameters       (sigma, 4*epsilon, charge product) for each pair
       @param forces           force array (forces added to current values)
       @param totalEnergy      if not null, the energy will be added to this

       --------------------------------------------------------------------------------------- */

    void calculateBondIxn(std::vector<int>& atomIndices, std::vector<OpenMM::Vec3>& atomCoordinates,
                          std::vector<double>& parameters, std::vector<OpenMM::Vec3>& forces,
                          double* totalEnergy, double* energyParamDerivs);

private:
    bool periodic;
    OpenMM::Vec3 periodicBoxVectors[3];
};

} // namespace NonbondedSlicing

#endif // __ReferenceCoulomb14_H__
