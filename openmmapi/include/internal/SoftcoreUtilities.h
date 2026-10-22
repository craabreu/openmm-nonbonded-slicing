#ifndef OPENMM_SOFTCOREUTILITIES_H_
#define OPENMM_SOFTCOREUTILITIES_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2026 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include "SlicedNonbondedForce.h"
#include <algorithm>
#include <string>
#include <vector>

namespace NonbondedSlicing {

/**
 * Get the soft-core parameter of every slice of a SlicedNonbondedForce, indexed by slice
 * (0 for a hard-core slice).
 */
inline std::vector<double> getSliceSoftcoreAlphas(const SlicedNonbondedForce& force) {
    std::vector<double> alphas(force.getNumSlices(), 0.0);
    for (int index = 0; index < force.getNumScalingParameters(); index++) {
        double alpha = force.getScalingParameterSoftcoreAlpha(index);
        if (alpha > 0) {
            std::string parameter;
            int subset1, subset2;
            bool includeCoulomb, includeLJ;
            force.getScalingParameter(index, parameter, subset1, subset2, includeCoulomb, includeLJ);
            alphas[sliceIndex(subset1, subset2)] = alpha;
        }
    }
    return alphas;
}

/**
 * Whether any slice uses the soft-core potential, given the soft-core parameters per slice.
 */
inline bool anySoftcoreSlice(const std::vector<double>& alphas) {
    return std::any_of(alphas.begin(), alphas.end(), [](double alpha) { return alpha > 0; });
}

} // namespace NonbondedSlicing

#endif /*OPENMM_SOFTCOREUTILITIES_H_*/
