#ifndef NONBONDED_SLICING_INNER_SPLIT_H_
#define NONBONDED_SLICING_INNER_SPLIT_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM Nonbonded Slicing                          *
 *                          ========================                          *
 *                                                                            *
 * An OpenMM plugin for slicing nonbonded potential energy calculations.      *
 *                                                                            *
 * Copyright (c) 2022-2026 Charlles Abreu                                     *
 * https://github.com/craabreu/openmm-nonbonded-slicing                       *
 * -------------------------------------------------------------------------- */

#include <cmath>

namespace NonbondedSlicing {

/**
 * Host-side constants of the inner force group split.  The inner force of a pair is S(r) times the
 * bare force, where S is a quintic switch that goes from 1 at rs to 0 at rc.  The inner energy is the
 * potential whose gradient is that force, shifted to vanish at rc.  Writing the bare pair potential as
 * V(r) = sum_n c_n r^-n for n in {12, 6, 1}, the inner energy is sum_n c_n energyTerm(n, r).
 */
class InnerSplit {
public:
    InnerSplit() : enabled(false), rs(0), rc(0) {
    }
    InnerSplit(double rs, double rc) : enabled(true), rs(rs), rc(rc) {
        double a = rs+rc, b = rs*rc;
        p[0] = b*b;
        p[1] = -2*a*b;
        p[2] = a*a+2*b;
        p[3] = -2*a;
        p[4] = 1.0;
        n[0] = 12;
        n[1] = 6;
        n[2] = 1;
        prefactor = 30.0/std::pow(rc-rs, 5);
        g0s = G0(rs);
        for (int i = 0; i < 3; i++) {
            gs[i] = G(n[i], rs);
            K[i] = std::pow(rc, -n[i]) + prefactor*(G(n[i], rc)-gs[i]-std::pow(rc, -n[i])*(G0(rc)-g0s));
        }
    }
    bool enabled;
    double rs, rc, p[5], prefactor, g0s, gs[3], K[3];
    int n[3];
    double switchValue(double r) const {
        if (r <= rs)
            return 1.0;
        if (r >= rc)
            return 0.0;
        double z = (r-rs)/(rc-rs);
        return 1+z*z*z*(-10+z*(15-z*6));
    }
    double G0(double r) const {
        double sum = 0, rk = r;
        for (int k = 0; k < 5; k++) {
            sum += p[k]*rk/(k+1);
            rk *= r;
        }
        return sum;
    }
    double G(int n, double r) const {
        double sum = 0;
        for (int k = 0; k < 5; k++) {
            int m = n-k-1;
            sum += p[k]*(m == 0 ? std::log(r) : -std::pow(r, -m)/m);
        }
        return sum;
    }
    double perturbation(int i, double r) const {
        if (r <= rs || r >= rc)
            return 0.0;
        return prefactor*(G(n[i], r)-gs[i]-std::pow(r, -n[i])*(G0(r)-g0s));
    }
    double energyTerm(int i, double r) const {
        if (r >= rc)
            return 0.0;
        return std::pow(r, -n[i]) + perturbation(i, r) - K[i];
    }
};

} // namespace NonbondedSlicing

#endif /*NONBONDED_SLICING_INNER_SPLIT_H_*/
