#ifndef OPENMM_CAPPEDLJ_H_
#define OPENMM_CAPPEDLJ_H_

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
 * The capped Lennard-Jones potential of Correa et al. (J. Chem. Theory Comput. 18, 5876, 2022). For x = r/sigma < 1,
 * the bare potential 4 epsilon (x^-12 - x^-6) is replaced by 4 epsilon c(x), where c(x) = C0 + C4 x^4 + C5 x^5 + C6 x^6
 * matches the bare potential and its first two derivatives at x = 1 and has vanishing first three derivatives at
 * x = 0, where the capped potential is 119.2 epsilon. The decapping term is the bare potential minus the capped one,
 * which is zero for x >= 1. The GPU kernels take the same coefficients as defines.
 */
namespace CappedLJ {

const double C0 = 29.8, C4 = -360.0, C5 = 547.2, C6 = -217.0;

/**
 * The bound on (sigma/r)^6 in the bare part of the decapping term on the GPU platforms (x >= 1e-3), which keeps it
 * finite in single precision, so that a decapping parameter of zero gives zero rather than 0*inf.
 */
const double MaxDecapSig6 = 1e18;

/**
 * The capped potential and -r dU/dr at x = r/sigma < 1, in units of 4 epsilon.
 */
inline void evaluate(double x, double& energy, double& rForce) {
    double x4 = x*x*x*x;
    energy = C0 + x4*(C4 + x*(C5 + x*C6));
    rForce = -x4*(4*C4 + x*(5*C5 + x*6*C6));
}

/**
 * Split the Lennard-Jones terms of a pair at x = r/sigma < 1 into its capped and decapping parts. On input, energy and
 * rForce are the bare energy and -r dU/dr; on output, they are the capped ones, computed directly so as to remain
 * accurate as x goes to zero, and decapEnergy and decapRForce are the bare minus the capped ones.
 *
 * @param eps   4 epsilon
 */
inline void split(double x, double eps, double& energy, double& rForce, double& decapEnergy, double& decapRForce) {
    double capEnergy, capRForce;
    evaluate(x, capEnergy, capRForce);
    capEnergy *= eps;
    capRForce *= eps;
    decapEnergy = energy - capEnergy;
    decapRForce = rForce - capRForce;
    energy = capEnergy;
    rForce = capRForce;
}

} // namespace CappedLJ

} // namespace NonbondedSlicing

#endif /*OPENMM_CAPPEDLJ_H_*/
