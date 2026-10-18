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
both, of any number of slices. Since the energy is linear in the scaling parameters, the derivative
with respect to a parameter :math:`\lambda` is the sum of the unscaled contributions it multiplies:

.. math::
   \frac{\partial E}{\partial \lambda} = \sum_{(I,J) \in S^{elec}_\lambda} E^{elec}_{I,J}
   + \sum_{(I,J) \in S^{vdW}_\lambda} E^{vdW}_{I,J},

where :math:`S^{elec}_\lambda` and :math:`S^{vdW}_\lambda` are the sets of slices whose electrostatic
and van der Waals contributions are scaled by :math:`\lambda`. This derivative does not depend on the
values of the scaling parameters. For instance, a parameter that scales both contributions of a
single slice *I, J* yields its full energy :math:`E^{vdW}_{I,J}+E^{elec}_{I,J}`, while one that
scales the electrostatic contributions of several slices yields their sum.

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
