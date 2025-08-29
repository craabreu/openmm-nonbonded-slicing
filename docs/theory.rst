=====================
Energy Slicing Theory
=====================

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

