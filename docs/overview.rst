========
Overview
========

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

The `OpenMM PME Slicing plugin <https://github.com/craabreu/openmm-pme-slicing>`_ implements a
variant of the smooth Particle Mesh Ewald (PME) method that slices the total Coulomb potential
and applies a different switching constant to each slice.

By partitioning all particles among $n$ non-interesecting subsets, the total Coulomb potential
becomes

.. math::

    U = \sum_{I=0}^{n-1} \sum_{J=0}^{n-1} h_{I,J} U_{I,J}

where :math:`h_{I,J}` is a switching constant and :math:`U_{I,J}` is the sum over every pair interaction
involving a particle in subset I and a particle in subset J.

**Note**: In OpenMM_, Lennard-Jones and other pairwise potentials can undergo similar slicing
by means of the `addInteractionGroup` method of :OpenMM:`CustomNonbondedForce`. There are no built-in
alternatives for lattice-sum Coulomb interactions.

Building the Plugin
===================

This project uses CMake_ for its build system.  To build it, follow these steps:

#. Create a directory in which to build the plugin.
#. Run the CMake GUI or ccmake, specifying your new directory as the build directory and the top level directory of this project as the source directory.
#. Press "Configure".
#. Set OPENMM_DIR to point to the directory where OpenMM is installed.  This is needed to locate the OpenMM header files and libraries.
#. Set CMAKE_INSTALL_PREFIX to the directory where the plugin should be installed.  Usually, this will be the same as OPENMM_DIR, so the plugin will be added to your OpenMM installation.
#. If you plan to build the OpenCL platform, make sure that OPENCL_INCLUDE_DIR and OPENCL_LIBRARY are set correctly, and that PMESLICING_BUILD_OPENCL_LIB is selected.
#. If you plan to build the CUDA platform, make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly and that PMESLICING_BUILD_CUDA_LIB is selected.
#. Press "Configure" again if necessary, then press "Generate".
#. Use the build system you selected to build and install the plugin.  For example, if you selected Unix Makefiles, type `make install`.

Python Wrapper
==============

As OpenMM_, this project uses SWIG_ to generate its Python API.  SWIG takes an "interface
file", which is essentially a C++ header file with some extra annotations added, as its input.
It then generates a Python extension module exposing the C++ API in Python.

To build and install the Python API, build the `PythonInstall` target, for example by typing
`make PythonInstall` (if you are installing into the system Python, you may need to use sudo).

Test Cases
==========

To run the C++ test cases, build the "test" target, for example by typing `make test`.

To run the Python test cases, build the "PythonTest" target by typing `make PythonTest`.


.. _CMake:                http://www.cmake.org
.. _OpenMM:               https://openmm.org
.. _SWIG:                 http://www.swig.org
