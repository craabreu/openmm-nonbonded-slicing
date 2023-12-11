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

Building the Plugin
===================

This project uses CMake_ for its build system.  To build it, follow these steps:

#. Create a directory in which to build the plugin.
#. Run the CMake GUI or ccmake, specifying your new directory as the build directory and the top level directory of this project as the source directory.
#. Press "Configure".
#. Set OPENMM_DIR to point to the directory where OpenMM is installed.  This is needed to locate the OpenMM header files and libraries.
#. Set CMAKE_INSTALL_PREFIX to the directory where the plugin should be installed.  Usually, this will be the same as OPENMM_DIR, so the plugin will be added to your OpenMM installation.
#. If you plan to build the OpenCL platform, make sure that OPENCL_INCLUDE_DIR and OPENCL_LIBRARY are set correctly, and that PLUGIN_BUILD_OPENCL_LIB is selected.
#. If you plan to build the CUDA platform, make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly and that PLUGIN_BUILD_CUDA_LIB is selected.
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
