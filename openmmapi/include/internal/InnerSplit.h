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

#include <algorithm>
#include <cmath>

namespace NonbondedSlicing {

/**
 * Host-side definition of the inner force group split.  The inner force of a pair is S(r) times the
 * bare force, where S is a quintic switch that goes from 1 at rs to 0 at rc.  Writing the bare pair
 * potential as V(r) = sum_n c_n r^-n for n in {12, 6, 1}, the inner energy is sum_n c_n e_n(r), where
 *
 *     e_n(r) = n*integral(S(s)*s^-(n+1), s=r..rc)    for rs < r < rc,
 *     e_n(r) = r^-n - K_n                            for r <= rs,
 *     K_n = rc^-n + n*integral((1-S(s))*s^-(n+1), s=rs..rc),
 *
 * so that the inner energy is continuous and vanishes at rc.  All integrals have positive integrands
 * and are evaluated by Gauss-Legendre quadrature in log(s), which is free of the cancellation that the
 * closed form suffers for narrow switching windows or small switching distances.  The GPU kernels use
 * the same quadrature (with fewer panels) beyond rs and the constants K_n below it.
 */
class InnerSplit {
public:
    InnerSplit() : enabled(false), rs(0), rc(0) {
    }
    InnerSplit(double rs, double rc) : enabled(true), rs(rs), rc(rc) {
        double inner[3], outer[3];
        integrate(rs, inner, outer);
        for (int i = 0; i < 3; i++)
            K[i] = std::pow(rc, -power(i)) + outer[i];
    }
    bool enabled;
    double rs, rc, K[3];
    static int power(int i) {
        return i == 0 ? 12 : (i == 1 ? 6 : 1);
    }
    double switchValue(double r) const {
        if (r <= rs)
            return 1.0;
        if (r >= rc)
            return 0.0;
        double z = (r-rs)/(rc-rs);
        return 1+z*z*z*(-10+z*(15-z*6));
    }
    /**
     * Compute the shifted inner energy terms e_12(r), e_6(r) and e_1(r).
     */
    void energyTerms(double r, double e[3]) const {
        if (r >= rc) {
            for (int i = 0; i < 3; i++)
                e[i] = 0.0;
        }
        else if (r <= rs) {
            for (int i = 0; i < 3; i++)
                e[i] = std::pow(r, -power(i))-K[i];
        }
        else {
            double outer[3];
            integrate(r, e, outer);
        }
    }
private:
    /**
     * Compute n*integral(S(s)*s^-(n+1), s=a..rc) and n*integral((1-S(s))*s^-(n+1), s=a..rc) for the three
     * powers, by 6-point Gauss-Legendre quadrature in y = log(s) on panels with 12*width <= 1, whose
     * relative error is far below double precision rounding.
     */
    void integrate(double a, double inner[3], double outer[3]) const {
        static const double node[3] = {0.2386191860831969, 0.6612093864662645, 0.9324695142031521};
        static const double weight[3] = {0.4679139345726910, 0.3607615730481386, 0.1713244923791704};
        double range = std::log(rc/a);
        int panels = std::max(1, (int) std::ceil(12.0*range));
        double width = range/panels;
        for (int i = 0; i < 3; i++)
            inner[i] = outer[i] = 0.0;
        for (int p = 0; p < panels; p++) {
            double center = std::log(a)+(p+0.5)*width;
            for (int k = 0; k < 3; k++)
                for (int side = -1; side <= 1; side += 2) {
                    double s = std::exp(center+side*0.5*width*node[k]);
                    double S = switchValue(s);
                    for (int i = 0; i < 3; i++) {
                        double term = weight[k]*power(i)*std::pow(s, -power(i));
                        inner[i] += term*S;
                        outer[i] += term*(1.0-S);
                    }
                }
        }
        for (int i = 0; i < 3; i++) {
            inner[i] *= 0.5*width;
            outer[i] *= 0.5*width;
        }
    }
};

} // namespace NonbondedSlicing

#endif /*NONBONDED_SLICING_INNER_SPLIT_H_*/
