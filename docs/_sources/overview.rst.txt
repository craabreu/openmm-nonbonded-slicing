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
singularity at :math:`r = 0` that makes free-energy calculations hard to converge. Calling
``setUseSoftcorePotential(I, J, alpha)`` switches slice *I, J* to the soft-core form of
:cite:`Beutler_1994`, in which each pair interacts through

.. math::
   U(r) = 4 \lambda \epsilon \left( s^2 - s \right), \qquad
   s = \frac{1}{\alpha (1 - \lambda) + (r/\sigma)^6},

where :math:`\lambda` is the scaling parameter that multiplies the Lennard-Jones contributions of
the slice (which therefore must be defined) and :math:`\alpha` is typically 0.5. The potential remains
finite as :math:`r \to 0` whenever :math:`\lambda < 1` and :math:`\alpha > 0`, and it reduces to the usual
one when :math:`\lambda = 1` or :math:`\alpha = 0`. Keep :math:`\lambda` between 0 and 1: for
:math:`\lambda > 1`, the term :math:`\alpha (1 - \lambda)` is negative and the potential becomes singular
at :math:`r = \left[ \alpha (\lambda - 1) \right]^{1/6} \sigma`. Since only the Lennard-Jones term
is softened, scale the Coulomb contribution of the slice with its own parameter
(``includeCoulomb=True, includeLJ=False`` in ``addScalingParameter``) and turn it off before
softening the Lennard-Jones term.

Since the energy is no longer linear in :math:`\lambda`, its derivative gains a term,

.. math::
   \frac{\partial}{\partial \lambda} \left[ \lambda\, 4 \epsilon (s^2 - s) \right]
   = 4 \epsilon (s^2 - s) + 4 \lambda \epsilon \alpha s^2 (2 s - 1),

which the plugin includes in the derivatives it reports. The soft-core form applies to the
direct-space Lennard-Jones term and to 1-4 exceptions, including the additive potential shift
at the cutoff with LJPME; the reciprocal-space part of LJPME and the long-range dispersion
correction are not softened, an approximation accurate to
:math:`O\left(\alpha (1-\lambda) (\sigma/r_c)^6\right)`. As with the usual potential, do not combine
LJPME with a switching function: the Reference platform ignores the switching function, while the
CUDA and OpenCL platforms apply it to the direct-space Lennard-Jones term, as OpenMM's
``NonbondedForce`` does on each platform.

.. warning::
   For a soft-core slice, the derivative with respect to its Lennard-Jones scaling parameter is
   **not** the van der Waals energy of the slice, and it changes with :math:`\lambda`. This holds
   even at :math:`\lambda = 1`: the potentials coincide there, but the derivative keeps a term
   proportional to :math:`\alpha`. To obtain that energy, either evaluate the slice in a separate
   force without the soft-core potential or, when that scaling parameter scales nothing else, take
   the energy difference :math:`E(\lambda = 1) - E(\lambda = 0)`, since :math:`\lambda = 0` gives
   zero soft-core Lennard-Jones energy for the slice and :math:`\lambda = 1` gives the usual one.

With Ewald, PME, or LJPME, the reciprocal-space part can be computed in a separate force group by
calling ``setReciprocalSpaceForceGroup``, as in multiple time stepping. As in OpenMM_'s
NonbondedForce, the terms that subtract the reciprocal-space interactions of excluded pairs and
exceptions are computed in the direct-space force group by default. Since they only cancel part of
the reciprocal-space sum, they should be integrated along with it whenever the two groups are
evaluated at different times :cite:`Procacci_1998,Morrone_2010`. Calling
``setReciprocalExclusions(True)`` moves them to the reciprocal-space force group, where they are
also kept when direct space is turned off with ``setIncludeDirectSpace(False)``.

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
