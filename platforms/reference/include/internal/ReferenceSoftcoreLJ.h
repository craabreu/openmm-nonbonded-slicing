#ifndef REFERENCE_SOFTCORE_LJ_H_
#define REFERENCE_SOFTCORE_LJ_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2026 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

namespace NonbondedSlicing {

/**
 * Lennard-Jones terms of one pair in the Beutler soft-core form, which reduces to the
 * hard-core form when alpha = 0 or lambda = 1.
 *
 * @param sig6             (sigma/r)^6
 * @param eps              4*epsilon
 * @param lambda           the LJ scaling parameter of the slice
 * @param alpha            the soft-core parameter of the slice (0 for hard-core)
 * @param energy           the unscaled energy, to be multiplied by lambda
 * @param rdUdr            the unscaled value of -r*dU/dr, to be multiplied by lambda
 * @param derivCorrection  the amount to add to energy to obtain d(lambda*energy)/d(lambda)
 */
inline void computeSoftcoreLJ(double sig6, double eps, double lambda, double alpha,
                              double& energy, double& rdUdr, double& derivCorrection) {
    double q = 1.0/(1.0 + alpha*(1.0 - lambda)*sig6);
    double s = sig6*q;
    energy = eps*(s - 1.0)*s;
    rdUdr = 6.0*eps*(2.0*s - 1.0)*s*q;
    derivCorrection = lambda*alpha*eps*s*s*(2.0*s - 1.0);
}

} // namespace NonbondedSlicing

#endif /*REFERENCE_SOFTCORE_LJ_H_*/
