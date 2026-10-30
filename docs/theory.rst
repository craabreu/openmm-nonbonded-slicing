======
Theory
======

Potential Energy Slicing
========================

This section contains the formulation used to implement the :class:`~nonbondedslicing.SlicedNonbondedForce`
class. For simplicity, the equations presented in here follow the conventional 1-based indexing, whereas
the class API follows Python's 0-based indexing.

Slicing the potential energy starts with partitioning all *N* particles of a system into *n* disjoint
subsets or "colors". This procedure results in :math:`n(n+1)/2` slices, distinguished by order-invariant
pairs of subset indices. Slice *I, J* comprises all pairs involving a particle in subset *I* and a
particle in another (or the same) subset *J*.

.. role:: subset1
.. role:: subset2
.. role:: subset3

.. figure:: _static/logo.png

    Example: three particle subsets (:subset1:`●`:subset2:`●`:subset3:`●`) form six energy slices

It is straightforward to slice pairwise Lennard-Jones interactions, as well as the direct-space,
self-energy, and exclusion/exception parts of the Ewald summation. Therefore, the present section
focuses on the reciprocal-space and background energy parts.

Sliced Background Energy
========================

In the case of systems with non-zero net charge, it is necessary to add the energy of
interaction with a neutralizing background charge. The background energy is given by
:cite:`Figueirido_1995`

.. math::

    E_{bg} = \frac{Q^2}{8\epsilon_0 V \alpha^2}.

where :math:`Q = \sum_{i=1}^N q_i` is the total charge of the system.
We can split the total charge into the sum of the charges in each subset,

.. math::

    Q = \sum_{I=1}^n Q_I,

where :math:`Q_I = \sum_{i \in I} q_i` is the charge of subset *I*.

The goal is to express the background energy as

.. math::

    E_{bg} = \sum_{I=1}^n \sum_{J=I}^n \lambda^{elec}_{I,J} E^{bg}_{I,J},

where :math:`E^{bg}_{I,J}` is the background contribution of Slice *I, J*.
For this, we rewrite the square of the total charge as

.. math::

    Q^2 = \sum_{I=1}^n \sum_{J=1}^n Q_I Q_J.

Therefore, the background energy of a slice *I, J* is given by

.. math::

    E^{bg}_{I,J} = (2 - \delta_{I,J}) \frac{Q_I Q_J}{8\epsilon_0 V \alpha^2},

where :math:`\delta_{I,J}` is the Kronecker delta. The prefactor :math:`2 - \delta_{I,J}` allows
us to run the summation over *J* only for :math:`J \geq I`.

Sliced Reciprocal-Space Energy
==============================

For a simulation box with edge matrix :math:`\mathbf L` containing *N* particles under periodic
boundary conditions, the standard reciprocal part of the electrostatic potential energy can be
expressed as

.. math::

    E_{rec} = \frac{2\pi}{\epsilon_0 V}
    \sum_{\mathbf n \neq \mathbf 0} \frac{e^{-\frac{k^2}{4\alpha^2}}}{k^2}
    \sum_{i=1}^N \sum_{j=1}^N q_i q_j
    e^{\text{ⅈ} {\mathbf k}\cdot({\mathbf r}_i - {\mathbf r}_j)},

where :math:`\epsilon_0` is the vacuum permittivity,
:math:`V` is the box volume,
:math:`\alpha` is the Ewald splitting parameter,
:math:`\mathbf n \in \mathbb Z^3` is an integer lattice vector,
:math:`\mathbf k = 2\pi \mathbf L^{-1}{\mathbf n}` is a reciprocal space wave vector,
:math:`k = \|\mathbf k\|` is the norm of :math:`\mathbf k`, and
:math:`\text{ⅈ} = \sqrt{-1}` is the imaginary unit.
As the summations over indices *i* and *j* run for all particles, we can write

.. math::

    E_{rec} = \frac{2\pi}{\epsilon_0 V}
    \sum_{\mathbf n \neq \mathbf 0} \frac{e^{-\frac{k^2}{4\alpha^2}}}{k^2}
    \Bigg(\sum_{i=1}^N q_i e^{\text{ⅈ} {\mathbf k}\cdot{\mathbf r}_i} \Bigg)
    \Bigg(\sum_{j=1}^N q_j e^{-\text{ⅈ} {\mathbf k}\cdot{\mathbf r}_j}\Bigg).

The first term inside parentheses is the charge structure factor

.. math::

    S(\mathbf k) = \sum_i q_i e^{\text{ⅈ} {\mathbf k}\cdot{\mathbf r}_i}

and the second one is its complex conjugate :math:`{\overline S}(\mathbf k)`,
so that we can write

.. math::

    E_{rec} = \frac{2\pi}{\epsilon_0 V}
    \sum_{\mathbf n \neq \mathbf 0}\frac{e^{-\frac{k^2}{4\alpha^2}}}{k^2} |S(\mathbf k)|^2.

The goal is to express the reciprocal-space energy as

.. math::

    E_{rec} = \sum_{I=1}^n \sum_{J=I}^n \lambda^{elec}_{I,J} E^{rec}_{I,J},

where :math:`E^{rec}_{I,J}` is the reciprocal-space contribution of Slice *I, J*.

The structure factor of a subset *I* can be computed as

.. math::
    S_I(\mathbf k) = \sum_{i \in I} q_i e^{\text{ⅈ} {\mathbf k}\cdot{\mathbf r}_i}.

Since we have disjoint subsets, this makes :math:`S(\mathbf k) = \sum_{I=1}^n S_I(\mathbf k)`.
Substituting into :math:`S(\mathbf k) {\overline S}(\mathbf k)` and expanding the product, all
imaginary terms cancel out, as expected. The reciprocal-space energy then becomes

.. math::

    E_{rec} = \frac{2\pi}{\epsilon_0 V}
    \sum_{\mathbf n \neq \mathbf 0} \frac{e^{-\frac{k^2}{4\alpha^2}}}{k^2}
    \sum_{I=1}^n \sum_{J=1}^n S_I(\mathbf k) \cdot S_J(\mathbf k),

where :math:`x \cdot y = \text{Re}(x)\text{Re}(y) + \text{Im}(x)\text{Im}(y)` for two complex
numbers *x* and *y*. It is now clear that the energy of a slice *I, J* should be calculated by

.. math::

    E^{rec}_{I,J} = (2-\delta_{I,J}) \frac{2\pi}{\epsilon_0 V}
    \sum_{\mathbf n \neq \mathbf 0} \frac{e^{-\frac{k^2}{4\alpha^2}}}{k^2}
    S_I(\mathbf k) \cdot S_J(\mathbf k).

Sliced PME Implementation
=========================

Each subset-specific structure factor :math:`S(\mathbf k)` can be calculated using the smooth
Particle Mesh Ewald method :cite:`Essmann_1995`. This requires *n* FFT calculations for evaluating
all slices. In the case of forces, *n* additional inverse FFT calculations are necessary.
Therefore, in a single processor and for a fixed number of particles, the computational effort
scales approximately linearly with the number of subsets (not the number of slices). In a GPU, as
with the OpenCL and CUDA instances of :OpenMM:`Platform`, we take advantage the capability of either
package vkFFT_ or cuFFT_ to speed-up parallel FFT calculations via batched transforms.

.. _vkFFT:                https://github.com/DTolm/VkFFT
.. _cuFFT:                https://docs.nvidia.com/cuda/cufft


Inner Force Group Energy
========================

When an inner force group is enabled, the inner force of a pair is :math:`S(r)` times the bare
Lennard-Jones plus Coulomb force, where :math:`S` is the quintic switch of the force-directed
switching scheme of :cite:`Abreu_2020`. Writing the bare potential as :math:`V(r) = \sum_n c_n r^{-n}`
with :math:`n \in \{12, 6, 1\}`, the energy whose gradient is that force, shifted to vanish at the
inner cutoff :math:`r_c`, is :math:`\hat V(r) = \sum_n c_n e_n(r)` with

.. math::
   e_n(r) = n \int_r^{r_c} S(s)\, s^{-(n+1)}\, ds \quad (r_s < r < r_c), \qquad
   e_n(r) = r^{-n} - K_n \quad (r \le r_s), \qquad K_n = r_s^{-n} - e_n(r_s),

where :math:`r_s` is the inner switching distance. Expanding :math:`S` in powers of :math:`s`
and integrating term by term gives a closed form whose terms are much larger than their sum:
the expansion coefficients grow like :math:`(r_s/h)^k` with :math:`h = r_c - r_s`, and the
integration constants involve integrals from :math:`r_s`. In single precision this loses most of
the significant digits even for a 0.5–0.8 nm switching range, and in double precision it fails for
narrow ranges or small :math:`r_s`.

The switch is the smoothstep polynomial :math:`P(u) = 10u^3 - 15u^4 + 6u^5` of
:math:`u = (r_c - s)/h`. In the variable :math:`y = 1 - s/r_c`, that is :math:`S = P(\gamma y)`
with :math:`\gamma = r_c/h`, and :math:`s^{-(n+1)} ds = -r_c^{-n} (1-y)^{-(n+1)} dy`, so that

.. math::
   e_n(r) = n\, r_c^{-n} \left( 10\gamma^3 J_3 - 15\gamma^4 J_4 + 6\gamma^5 J_5 \right), \qquad
   J_k = \int_0^Y y^k (1-y)^{-(n+1)}\, dy, \qquad Y = 1 - \frac{r}{r_c}.

The only cancellation left is the alternation of the three smoothstep coefficients, which is
bounded by a factor of about 30 because :math:`\gamma y \le 1` over the integration range. For the
Lennard-Jones powers, :math:`n \ge 6`, the substitution :math:`x = y/(1-y)` turns each
:math:`J_k` into a polynomial with positive coefficients in :math:`x = (r_c - r)/r`,

.. math::
   J_k = \int_0^x t^k (1+t)^{n-k-1}\, dt = \sum_{j=0}^{n-k-1} \binom{n-k-1}{j} \frac{x^{k+j+1}}{k+j+1},

so that :math:`e_{12}` and :math:`e_6` are :math:`x^4` times polynomials of degrees 8 and 2. For
the Coulomb power, :math:`n = 1`, the integral has a logarithm, and the three :math:`J_k` share it:

.. math::
   e_1(r) = \frac{\gamma^3}{r_c} \left[ \frac{Y^4 (10 - 15\gamma Y + 6\gamma^2 Y^2)}{1-Y}
   - 30 \left( (\gamma-1)^2 L_6(Y) + (1-2\gamma)\frac{Y^5}{5} + \frac{Y^4}{4} \right) \right],
   \qquad L_6(Y) = \sum_{j \ge 6} \frac{Y^j}{j} = -\ln(1-Y) - \sum_{j=1}^{5} \frac{Y^j}{j}.

The function :math:`L_6(Y)/Y^6 = \sum_{m \ge 0} Y^m/(m+6)` is the Lerch transcendent
:math:`\Phi(Y, 1, 6)`, equivalently :math:`{}_2F_1(1, 6; 7; Y)/6`, and :math:`L_6(Y)` itself is the
incomplete beta function :math:`B_Y(6, 0)`. None of these is available in the GPU math libraries,
and the logarithm, the only elementary function in the family, requires the subtraction above, which
cancels for small :math:`Y`. Since the function does not depend on the switching range, the GPU
kernels evaluate it from a fixed Chebyshev expansion on :math:`0 \le Y \le 0.8`, accurate to
:math:`5 \times 10^{-10}`, and from the logarithm above that, where the subtraction loses at most one
digit. The Reference platform sums the series for :math:`Y < 0.7` and uses :math:`-\ln(1-Y)` minus
the leading terms above that. With these expressions, :math:`e_{12}` and :math:`e_6` are accurate to
a few units in the last place and :math:`e_1` to about :math:`3 \times 10^{-6}` in single precision,
for any switching range. In double precision, :math:`e_{12}` and :math:`e_6` are exact to rounding
on all platforms and :math:`e_1` on the Reference platform, while the Chebyshev expansion limits
:math:`e_1` on the GPU platforms to an absolute error of about :math:`3 \times 10^{-10}` (about
:math:`10^{-9}` relative). These errors affect only the energies: the inner force is computed
directly as :math:`S(r)` times the bare force. The Reference platform and the constants of the GPU
kernels are computed by the same code, in ``InnerSplit.h``.

In a decapped slice (:doc:`guide/decapping`), the Lennard-Jones potential below :math:`\sigma` is not
the bare one. Since :math:`\sigma` cannot exceed :math:`r_s`, where :math:`S = 1`, the inner group takes
the whole interaction there: the capped force with energy :math:`u_{cap}(r) - K_{LJ}`, where
:math:`K_{LJ} = 4\epsilon(\sigma^{12} K_{12} - \sigma^6 K_6)` is the constant the remainder reports for
:math:`r \le r_s`, and all of the decapping term. The remainder is the same as for the bare potential.
Coulomb decapping, which caps the :math:`1/r` term of the pairs of a slice in the same way, is handled
likewise: the inner group takes the capped Coulomb force with energy :math:`q_i q_j [c(r/\sigma)/\sigma -
K_1]/(4 \pi \epsilon_0)` and all of the Coulomb decapping term. Below :math:`r_s` the remainder is
:math:`q_i q_j [K_1 - \mathrm{erf}(\alpha r)/r]/(4 \pi \epsilon_0)` with Ewald, PME and LJPME, evaluated
from the series of :math:`\mathrm{erf}(\alpha r)/r` at small :math:`\alpha r` so that it stays finite and
accurate as :math:`r \to 0`, and a constant plus the reaction-field term otherwise.
