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
slice contributions or sums thereof via :OpenMM:`Context`'s ``getState`` method with option
``getParameterDerivatives=True``.

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
