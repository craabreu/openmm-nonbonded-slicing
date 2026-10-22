========
Overview
========

This OpenMM_ plugin implements a sliced variant of OpenMM_'s NonbondedForce class.
By partitioning all particles among $n$ disjoint subsets, the total potential energy becomes a linear
combination of contributions from pairs of subsets like

.. math::
   E = \sum_{I=0}^{n-1} \sum_{J=I}^{n-1} (\lambda^{vdW}_{I,J}E^{vdW}_{I,J}+\lambda^{elec}_{I,J}E^{elec}_{I,J}),

where each slice is defined by subsets *I* and *J*, superscripts *vdW* and *elec* denote van
der Waals and electrostatic contributions, :math:`E_{I,J}` is the potential energy of all particle pairs
formed by one particle in subset *I* and one in subset *J*, and :math:`\lambda_{I,J}` is a scaling parameter.

The nonbonded energy can be partitioned along a second, independent axis. Slicing by *subset*, above,
decides which particle pairs a scaling parameter applies to. Slicing by *distance*, through an optional
*inner force group* (see below), splits every pair interaction into a short-range part, switched off
smoothly between an inner switching distance and an inner cutoff, and a remainder, so that a multiple
time-step integrator can advance the two with different time steps. The two partitions compose exactly,

.. math::
   E = \sum_{I \le J} \sum_{t \in \{vdW, elec\}} \lambda^{t}_{I,J}
   \left( E^{t,\mathrm{inner}}_{I,J} + E^{t,\mathrm{outer}}_{I,J} \right),

because the inner part of a pair interaction is scaled by the same parameters as the rest of it, and
each force group reports an energy consistent with its force.

By default, all scaling parameters are constant and equal to 1. However, the user can turn selected
scaling parameters into variables and store their values in :OpenMM:`Context` global parameters. Derivatives
with respect to these variables can be requested and used, for instance, to report individual energy
slice contributions or sums thereof via :OpenMM:`Context`'s ``getState`` method with options
``getEnergy=True`` and ``getParameterDerivatives=True``. Derivatives are only computed along with
the energy: without ``getEnergy=True``, they are NaN.

A global parameter can scale the electrostatic contributions, the van der Waals contributions, or
both, of any number of slices. Unless a slice uses the soft-core potential (see below), the energy
is linear in the scaling parameters, and the derivative with respect to a parameter
:math:`\lambda` is the sum of the unscaled contributions it multiplies:

.. math::
   \frac{\partial E}{\partial \lambda} = \sum_{(I,J) \in S^{elec}_\lambda} E^{elec}_{I,J}
   + \sum_{(I,J) \in S^{vdW}_\lambda} E^{vdW}_{I,J},

where :math:`S^{elec}_\lambda` and :math:`S^{vdW}_\lambda` are the sets of slices whose electrostatic
and van der Waals contributions are scaled by :math:`\lambda`. Without soft-core slices, this
derivative does not depend on the values of the scaling parameters. For instance, a parameter that
scales both contributions of a single slice *I, J* yields its full energy
:math:`E^{vdW}_{I,J}+E^{elec}_{I,J}`, while one that scales the electrostatic contributions of
several slices yields their sum.

Soft-core Lennard-Jones Potential
=================================

Driving :math:`\lambda^{vdW}_{I,J}` to zero with the usual Lennard-Jones potential leaves a
singularity at :math:`r = 0` that makes free-energy calculations hard to converge. A scaling
parameter added with a positive ``softcoreAlpha`` switches its slice to the soft-core form of
:cite:`Beutler_1994`, in which each pair interacts through

.. math::
   U(r) = 4 \lambda_E \epsilon \left( s^2 - s \right), \qquad
   s = \frac{1}{\alpha (1 - \lambda_s) + (r/\sigma)^6},

where :math:`\lambda_E` is the scaling parameter that multiplies the Lennard-Jones contributions of
the slice (1 if there is none), :math:`\lambda_s` is the softening parameter and :math:`\alpha` is
``softcoreAlpha``, typically 0.5. The two parameters can be the same,

.. code-block:: python

    force.addScalingParameter("lambda", 0, 1, False, True, 0.5)

or different,

.. code-block:: python

    force.addScalingParameter("lambdaE", 0, 1, False, True)
    force.addScalingParameter("lambdaS", 0, 1, False, False, 0.5)

and a slice can be softened without scaling its energy at all (only the second line above). The
potential remains finite as :math:`r \to 0` whenever :math:`\lambda_s < 1` and :math:`\alpha > 0`,
and it reduces to the usual one when :math:`\lambda_s = 1` or :math:`\alpha = 0`. Keep
:math:`\lambda_s` between 0 and 1: for :math:`\lambda_s > 1`, the term :math:`\alpha (1 - \lambda_s)`
is negative and the potential becomes singular at
:math:`r = \left[ \alpha (\lambda_s - 1) \right]^{1/6} \sigma`. Since only the Lennard-Jones term is
softened, scale the Coulomb contribution of the slice with its own parameter
(``includeCoulomb=True, includeLJ=False``) and turn it off before softening the Lennard-Jones term.
Like the other properties of a scaling parameter, :math:`\alpha` is fixed when a :OpenMM:`Context`
is created, while :math:`\lambda_E` and :math:`\lambda_s` are its global parameters.

The derivatives with respect to the two parameters are

.. math::
   \frac{\partial U}{\partial \lambda_E} = 4 \epsilon (s^2 - s), \qquad
   \frac{\partial U}{\partial \lambda_s} = 4 \lambda_E \epsilon \alpha s^2 (2 s - 1),

and a parameter that plays both roles gets their sum. The soft-core form applies to the
direct-space Lennard-Jones term and to 1-4 exceptions, including the additive potential shift at
the cutoff with LJPME; the reciprocal-space part of LJPME and the long-range dispersion correction
are not softened, an approximation accurate to
:math:`O\left(\alpha (1-\lambda_s) (\sigma/r_c)^6\right)`. As with the usual potential, do not
combine LJPME with a switching function: the Reference platform ignores the switching function,
while the CUDA and OpenCL platforms apply it to the direct-space Lennard-Jones term, as OpenMM's
``NonbondedForce`` does on each platform.

.. warning::
   For a softened slice, :math:`\partial U / \partial \lambda_E` is the soft-core van der Waals
   energy of the slice, which differs from the usual one whenever :math:`\lambda_s < 1`, and
   :math:`\partial U / \partial \lambda_s` is not an energy. For a parameter that plays both roles,
   the reported derivative is the sum of the two, which is not an energy even at
   :math:`\lambda = 1`, where the soft-core and usual potentials coincide. To obtain the usual van der
   Waals energy of the slice, evaluate it in a separate force without the soft-core potential.

With Ewald, PME, or LJPME, the reciprocal-space part can be computed in a separate force group by
calling ``setReciprocalSpaceForceGroup``, as in multiple time stepping. As in OpenMM_'s
NonbondedForce, the terms that subtract the reciprocal-space interactions of excluded pairs and
exceptions are computed in the direct-space force group by default. Since they only cancel part of
the reciprocal-space sum, they should be integrated along with it whenever the two groups are
evaluated at different times :cite:`Procacci_1998,Morrone_2010`. Calling
``setReciprocalExclusions(True)`` moves them to the reciprocal-space force group, where they are
also kept when direct space is turned off with ``setIncludeDirectSpace(False)``.

Force splitting for multiple time-step integration
==================================================

A SlicedNonbondedForce can place a short-range, force-switched copy of its direct-space pair
interactions in a separate *inner* force group, so that a multiple time-step integrator can advance
it with a small step while the remainder (the rest of direct space, the reciprocal-space sum and its
exclusion corrections) is advanced with a large one. For a pair at distance :math:`r` with bare
potential :math:`V(r)` (Lennard-Jones plus Coulomb, scaled by the slice's parameters), the inner
force is

.. math::
   \mathbf{F}^\mathrm{inner}(r) = -S(r)\,V'(r)\,\hat{\mathbf{r}}, \qquad
   S = 1 - 10 z^3 + 15 z^4 - 6 z^5, \quad z = \frac{r - r_s}{r_c - r_s},

with :math:`S = 1` for :math:`r \le r_s` and :math:`S = 0` for :math:`r \ge r_c`, where
:math:`r_s` and :math:`r_c` are the inner switching and cutoff distances. The inner energy is the
potential :math:`\hat V` whose gradient is this force, shifted so that :math:`\hat V(r_c) = 0`
:cite:`Abreu_2020`; the direct-space group reports :math:`V - \hat V` and the corresponding force.
Every group therefore has an energy consistent with its force, and the groups add up to the
unsplit result. The closed form of :math:`\hat V` used by all platforms is derived in the theory
section. The same applies to 1-4 exceptions. Scaling parameters, their derivatives and
parameter offsets apply to both parts.

.. code-block:: python

   force = nonbondedslicing.SlicedNonbondedForce(nonbonded, 2)
   force.setForceGroup(2)               # direct-space remainder; reciprocal space goes there too
   force.setInnerForceGroup(1)
   force.setInnerSwitchingDistance(0.5*unit.nanometer)
   force.setInnerCutoffDistance(0.8*unit.nanometer)
   # bonded forces in group 0; steps of 90, 3 and 0.5 fs
   integrator = openmm.MTSIntegrator(90*unit.femtoseconds, [(2, 1), (1, 30), (0, 180)])

The inner force group must differ from the force group of the force and from its reciprocal-space
force group, and requires :math:`0 < r_s < r_c \le` cutoff distance. Changing these settings
requires a new :OpenMM:`Context`, as for any force group.

Installation
============

The plugin is distributed on conda-forge_:

.. code-block:: bash

    mamba install -c conda-forge openmm-nonbonded-slicing

Building from Source
====================

Requirements:

* OpenMM_ 8.4 or later
* CMake_ 3.17 or later and a C++ compiler
* SWIG_, the same version that built your OpenMM (OpenMM's conda-forge builds 8.4 and 8.5/8.6 used SWIG 4.4 and 4.5, respectively)
* Optional: OpenCL headers for the OpenCL platform, and a CUDA toolkit with NVRTC and cuFFT for the CUDA platform

In a conda environment where OpenMM is installed:

.. code-block:: bash

    mkdir build && cd build
    cmake ..
    make install
    make PythonInstall

Useful CMake options:

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``OPENMM_DIR``
     - ``$CONDA_PREFIX``
     - Where OpenMM is installed
   * - ``OPENMM_VERSION``
     - detected with ``python -c "import openmm"``
     - OpenMM version, if Python cannot import it
   * - ``CMAKE_INSTALL_PREFIX``
     - ``OPENMM_DIR``
     - Where to install the plugin
   * - ``PLUGIN_BUILD_OPENCL_LIB``
     - ``ON`` if OpenCL headers are found
     - Build the OpenCL platform
   * - ``PLUGIN_BUILD_CUDA_LIB``
     - ``ON`` if a CUDA toolkit is found
     - Build the CUDA platform (set ``CUDAToolkit_ROOT`` to pick a toolkit)
   * - ``VKFFT_INCLUDE_DIR``
     - downloaded at configure time
     - Directory containing ``vkFFT.h``

Test Cases
==========

From the build directory, run the C++ tests with ``make test`` (or ``ctest``) and the Python tests with ``make PythonTest``.


.. _CMake:                http://www.cmake.org
.. _OpenMM:               https://openmm.org
.. _SWIG:                 http://www.swig.org
.. _conda-forge:          https://anaconda.org/conda-forge/openmm-nonbonded-slicing
