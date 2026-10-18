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
 * Host-side definition of the inner force group split.  The inner force of a pair is S(r) times the
 * bare force, where S is a quintic switch that goes from 1 at rs to 0 at rc.  Writing the bare pair
 * potential as V(r) = sum_n c_n r^-n for n in {12, 6, 1}, the inner energy is sum_n c_n e_n(r), where
 *
 *     e_n(r) = n*integral(S(s)*s^-(n+1), s=r..rc)    for rs < r < rc,
 *     e_n(r) = r^-n - K_n                            for r <= rs,
 *     K_n = rs^-n - e_n(rs),
 *
 * so that the inner energy is continuous and vanishes at rc.  In the variable y = 1 - s/rc the switch
 * is the smoothstep polynomial P(gamma*y) = 10(gamma*y)^3 - 15(gamma*y)^4 + 6(gamma*y)^5, with
 * gamma = rc/(rc-rs), and the integral becomes
 *
 *     e_n(r) = n*rc^-n*(10*gamma^3*J_3 - 15*gamma^4*J_4 + 6*gamma^5*J_5),
 *     J_k = integral(y^k*(1-y)^-(n+1), y=0..Y),    Y = 1 - r/rc.
 *
 * For n >= 6, J_k is a polynomial in x = Y/(1-Y) = (rc-r)/r with positive coefficients, so e_n(r) is
 * x^4 times a polynomial in x.  For n = 1 the three J_k share one transcendental function and
 *
 *     e_1(r) = gamma^3/rc*(Y^4*(10 - 15*gamma*Y + 6*gamma^2*Y^2)/(1-Y)
 *                          - 30*((gamma-1)^2*L_6(Y) + (1-2*gamma)*Y^5/5 + Y^4/4)),
 *
 * with the positive series L_6(Y) = sum(Y^j/j, j>=6) = -log(1-Y) - sum(Y^j/j, j<6).  Nothing is
 * expanded about a distant point and nothing is integrated from rs, which is what makes this form
 * accurate even in single precision (see docs/theory.rst).  The GPU kernels evaluate the same
 * expressions with constants computed here.  L_6(Y)/Y^6 = sum(Y^m/(m+6), m>=0) is the Lerch
 * transcendent Phi(Y, 1, 6) = 2F1(1, 6; 7; Y)/6, a universal function that the kernels evaluate from a
 * Chebyshev expansion on [0, 0.8] (accurate to 5e-10) and from the logarithm above it.
 */
class InnerSplit {
public:
    InnerSplit() : enabled(false), rs(0), rc(0), gamma(0) {
    }
    InnerSplit(double rs, double rc) : enabled(true), rs(rs), rc(rc), gamma(rc/(rc-rs)) {
        for (int m = 0; m <= 8; m++)
            a12[m] = ljCoefficient(12, m+4);
        for (int m = 0; m <= 2; m++)
            a6[m] = ljCoefficient(6, m+4);
        c1 = gamma*gamma*gamma/rc;
        for (int i = 0; i < 3; i++)
            K[i] = std::pow(rs, -power(i)) - term(i, rs);
    }
    bool enabled;
    double rs, rc, gamma;
    double a12[9], a6[3], c1, K[3];
    /**
     * Chebyshev coefficients of the Lerch transcendent Phi(Y, 1, 6) = sum(Y^m/(m+6), m>=0) on [0, 0.8],
     * in the variable u = 2.5*Y-1 (check: mpmath.lerchphi(Y, 1, 6)).
     */
    static const double* ellChebyshev() {
        static const double c[21] = {
        0.30681903032265678, 0.18357926827433432, 0.057331696979874676,
        0.018467109445561639, 0.0060876662519786223, 0.0020429525038043895,
        0.00069535973163290887, 0.00023940582996955279, 8.320724862272342e-05,
        2.9148923422378249e-05, 1.0280153692277236e-05, 3.6465376484839408e-06,
        1.2999726338738142e-06, 4.6546752601897949e-07, 1.6731015635892038e-07,
        6.034543970079298e-08, 2.1831621854894232e-08, 7.9181110167258509e-09,
        2.8737864621055545e-09, 1.0312006321595779e-09, 3.321143763440879e-10
        };
        return c;
    }
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
        for (int i = 0; i < 3; i++) {
            if (r >= rc)
                e[i] = 0.0;
            else if (r <= rs)
                e[i] = std::pow(r, -power(i))-K[i];
            else
                e[i] = term(i, r);
        }
    }
    /**
     * The positive series L_m(Y) = sum(Y^j/j, j>=m) = -log(1-Y) - sum(Y^j/j, j<m).
     */
    static double logRemainder(int m, double Y) {
        if (Y < 0.7) {
            double total = 0.0, term = std::pow(Y, m);
            for (int j = m; j < m+400; j++) {
                total += term/j;
                term *= Y;
                if (term/(j+1) < 1e-17*total)
                    break;
            }
            return total;
        }
        double total = -std::log1p(-Y), term = 1.0;
        for (int j = 1; j < m; j++) {
            term *= Y;
            total -= term/j;
        }
        return total;
    }
private:
    static double binomial(int n, int k) {
        if (k < 0 || k > n)
            return 0.0;
        double result = 1.0;
        for (int i = 1; i <= k; i++)
            result *= (double) (n-k+i)/i;
        return result;
    }
    /**
     * Coefficient of x^m in e_n(r) for n >= 6.
     */
    double ljCoefficient(int n, int m) const {
        double g3 = gamma*gamma*gamma;
        return n*std::pow(rc, -n)/m*(10*g3*binomial(n-4, m-4) - 15*g3*gamma*binomial(n-5, m-5) + 6*g3*gamma*gamma*binomial(n-6, m-6));
    }
    /**
     * e_n(r) for rs < r < rc.
     */
    double term(int i, double r) const {
        if (i < 2) {
            const double* a = (i == 0 ? a12 : a6);
            int degree = (i == 0 ? 8 : 2);
            double x = (rc-r)/r, poly = a[degree];
            for (int m = degree-1; m >= 0; m--)
                poly = poly*x+a[m];
            return poly*x*x*x*x;
        }
        double Y = (rc-r)/rc, Y2 = Y*Y, Y4 = Y2*Y2, Y5 = Y4*Y;
        double L6 = logRemainder(6, Y);
        return c1*(Y4*(10-15*gamma*Y+6*gamma*gamma*Y2)*(rc/r) - 30*((gamma-1)*(gamma-1)*L6+(1-2*gamma)*Y5/5+Y4/4));
    }
};

} // namespace NonbondedSlicing

#endif /*NONBONDED_SLICING_INNER_SPLIT_H_*/
