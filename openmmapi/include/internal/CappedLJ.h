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

#include <cmath>
#include <utility>
#include <vector>

namespace NonbondedSlicing {

/**
 * The bound on (sigma/r)^6 in the bare part of the decapping term on the GPU platforms (x >= 1e-3), which keeps it
 * finite in single precision, so that a decapping parameter of zero gives zero rather than 0*inf.
 */
const double MaxDecapSig6 = 1e18;

/**
 * The capped Lennard-Jones potential of Correa et al. (J. Chem. Theory Comput. 18, 5876, 2022). For x = r/sigma < 1,
 * the bare potential 4 epsilon (x^-12 - x^-6) is replaced by 4 epsilon c(x), where c(x) is the polynomial of degree
 * m0 + m1 whose first m0 derivatives vanish at x = 0 and whose value and first m1 derivatives match those of
 * x^-12 - x^-6 at x = 1. The first condition leaves the constant a_0 and the terms a_k x^k for k = m0+1, ..., m0+m1,
 * and the second determines these m1+1 coefficients. With m0 = m1 = 3, the capped potential is 119.2 epsilon at x = 0.
 * The decapping term is the bare potential minus the capped one, which is zero for x >= 1. The GPU kernels evaluate the
 * same polynomial, with these coefficients written into their source.
 */
class CappedLJ {
public:
    /** The largest order accepted for m0 and m1: beyond it, the coefficients grow and cancel near x = 1. */
    static const int MaxOrder = 6;
    CappedLJ(int orderAtZero = 3, int orderAtSigma = 3) : m0(orderAtZero), m1(orderAtSigma) {
        // Conditions n = 0, ..., m1 at x = 1 on the unknowns (a_0, a_{m0+1}, ..., a_{m0+m1}), with the n-th
        // derivative of x^p at x = 1 being the falling factorial p(p-1)...(p-n+1)
        int size = m1+1;
        std::vector<std::vector<double>> matrix(size, std::vector<double>(size+1, 0.0));
        for (int n = 0; n < size; n++) {
            matrix[n][0] = (n == 0 ? 1.0 : 0.0);
            for (int j = 1; j < size; j++)
                matrix[n][j] = fallingFactorial(m0+j, n);
            matrix[n][size] = fallingFactorial(-12, n) - fallingFactorial(-6, n);
        }
        // Gaussian elimination with partial pivoting
        for (int i = 0; i < size; i++) {
            int pivot = i;
            for (int r = i+1; r < size; r++)
                if (std::abs(matrix[r][i]) > std::abs(matrix[pivot][i]))
                    pivot = r;
            std::swap(matrix[i], matrix[pivot]);
            for (int r = 0; r < size; r++)
                if (r != i) {
                    double factor = matrix[r][i]/matrix[i][i];
                    for (int c = i; c <= size; c++)
                        matrix[r][c] -= factor*matrix[i][c];
                }
        }
        a0 = matrix[0][size]/matrix[0][0];
        for (int j = 1; j < size; j++)
            a.push_back(matrix[j][size]/matrix[j][j]);
    }
    int getOrderAtZero() const {
        return m0;
    }
    int getOrderAtSigma() const {
        return m1;
    }
    /** The constant term a_0, in units of 4 epsilon. */
    double getConstant() const {
        return a0;
    }
    /** The coefficients a_{m0+1}, ..., a_{m0+m1}, in units of 4 epsilon. */
    const std::vector<double>& getCoefficients() const {
        return a;
    }
    /**
     * The capped potential and -r dU/dr at x = r/sigma < 1, in units of 4 epsilon.
     */
    void evaluate(double x, double& energy, double& rForce) const {
        double poly = 0.0, rPoly = 0.0;
        for (int j = m1-1; j >= 0; j--) {
            poly = a[j] + x*poly;
            rPoly = (m0+1+j)*a[j] + x*rPoly;
        }
        double xp = std::pow(x, m0+1);
        energy = a0 + xp*poly;
        rForce = -xp*rPoly;
    }
    /**
     * Split the Lennard-Jones terms of a pair at x = r/sigma < 1 into its capped and decapping parts. On input, energy
     * and rForce are the bare energy and -r dU/dr; on output, they are the capped ones, computed directly so as to
     * remain accurate as x goes to zero, and decapEnergy and decapRForce are the bare minus the capped ones.
     *
     * @param eps   4 epsilon
     */
    void split(double x, double eps, double& energy, double& rForce, double& decapEnergy, double& decapRForce) const {
        double capEnergy, capRForce;
        evaluate(x, capEnergy, capRForce);
        capEnergy *= eps;
        capRForce *= eps;
        decapEnergy = energy - capEnergy;
        decapRForce = rForce - capRForce;
        energy = capEnergy;
        rForce = capRForce;
    }
private:
    static double fallingFactorial(int p, int n) {
        double result = 1.0;
        for (int k = 0; k < n; k++)
            result *= p-k;
        return result;
    }
    int m0, m1;
    double a0;
    std::vector<double> a;
};

} // namespace NonbondedSlicing

#endif /*OPENMM_CAPPEDLJ_H_*/
