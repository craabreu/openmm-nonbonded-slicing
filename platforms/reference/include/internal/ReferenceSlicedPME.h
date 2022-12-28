#ifndef __REFERENCESLICEDPME_H__
#define __REFERENCESLICEDPME_H__

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022 Charlles Abreu                                          *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "openmm/Vec3.h"
#include "internal/windowsExportNonbondedSlicing.h"
#include <vector>

using namespace std;
using namespace OpenMM;

namespace NonbondedSlicing {

typedef double rvec[3];

typedef struct pme *
pme_t;

/*
 * Initialize a PME calculation and set up data structures
 *
 * Arguments:
 *
 * ppme        Pointer to an opaque pme_t object
 * ewaldcoeff  Coefficient derived from the beta factor to participate
 *             direct/reciprocal space. See gromacs code for documentation!
 *             We assume that you are using nm units...
 * natoms      Number of atoms to set up data structure sof
 * ngrid       Size of the full pme grid
 * pme_order   Interpolation order, almost always 4
 * epsilon_r   Dielectric coefficient, typically 1.0.
 */
int OPENMM_EXPORT_NONBONDED_SLICING
pme_init(pme_t* ppme,
         double ewaldcoeff,
         int natoms,
         int nsubsets,
         const int ngrid[3],
         int pme_order,
         double epsilon_r);

/*
 * Evaluate reciprocal space PME energy and forces.
 *
 * Args:
 *
 * pme         Opaque pme_t object, must have been initialized with pme_init()
 * x           Pointer to coordinate data array (nm)
 * f           Pointer to force data array (will be written as kJ/mol/nm)
 * charge      Array of charges (units of e)
 * box         Simulation cell dimensions (nm)
 * energy      Total energy (will be written in units of kJ/mol)
 */
int OPENMM_EXPORT_NONBONDED_SLICING
pme_exec(pme_t pme,
         const vector<OpenMM::Vec3>& atomCoordinates,
         const vector<int>& atomSubsets,
         const vector<vector<double>>& sliceLambdas,
         vector<OpenMM::Vec3>& forces,
         const vector<double>& charges,
         const OpenMM::Vec3 periodicBoxVectors[3],
         vector<vector<double>>& sliceEnergies);


/**
 * Evaluate reciprocal space PME dispersion energy and forces.
 *
 * Args:
 *
 * pme         Opaque pme_t object, must have been initialized with pme_init()
 * x           Pointer to coordinate data array (nm)
 * f           Pointer to force data array (will be written as kJ/mol/nm)
 * c6s         Array of c6 coefficients (units of sqrt(kJ/mol).nm^3 )
 * box         Simulation cell dimensions (nm)
 * energy      Total energy (will be written in units of kJ/mol)
 */
int OPENMM_EXPORT_NONBONDED_SLICING
pme_exec_dpme(pme_t pme,
         const vector<OpenMM::Vec3>& atomCoordinates,
         const vector<int>& atomSubsets,
         const vector<vector<double>>& sliceLambdas,
         vector<OpenMM::Vec3>& forces,
         const vector<double>& c6s,
         const OpenMM::Vec3 periodicBoxVectors[3],
         vector<vector<double>>& sliceEnergies);




/* Release all memory in pme structure */
int OPENMM_EXPORT_NONBONDED_SLICING
pme_destroy(pme_t    pme);

} // namespace OpenMM

#endif // __REFERENCESLICEDPME_H__
