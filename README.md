WIP: OpenMM PME Slicing Plugin
==============================

[![GH Actions Status](https://github.com/craabreu/openmm-pme-slicing/workflows/Linux/badge.svg)](https://github.com/craabreu/openmm-pme-slicing/actions?query=branch%3Amain+workflow%3ALinux)
[![GH Actions Status](https://github.com/craabreu/openmm-pme-slicing/workflows/MacOS/badge.svg)](https://github.com/craabreu/openmm-pme-slicing/actions?query=branch%3Amain+workflow%3AMacOS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This [OpenMM] plugin implements a variant of the smooth Particle Mesh Ewald (PME) method that
slices the total Coulomb potential and applies a different coupling constant to each slice.

By partitioning all particles among $n$ non-interesecting subsets, the total Coulomb potential
becomes

![equation](https://latex.codecogs.com/svg.image?U&space;=&space;\sum_{I=0}^{n-1}&space;\sum_{J=0}^{n-1}&space;\lambda_{I,J}&space;U_{I,J})

where $\lambda_{I,J}$ is a coupling constant and $U_{I,J}$ is the sum over every pair interaction
involving a particle in subset I and a particle in subset J.

**Note**: In [OpenMM], Lennard-Jones and other pairwise potentials can undergo similar slicing
by means of the `addInteractionGroup` method of [CustomNonbondedForce]. There are no built-in
alternatives for the Coulomb potential.

Building the Plugin
===================

This project uses [CMake] for its build system.  To build it, follow these steps:

1. Create a directory in which to build the plugin.

2. Run the CMake GUI or ccmake, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

3. Press "Configure".

4. Set OPENMM_DIR to point to the directory where OpenMM is installed.  This is needed to locate
the OpenMM header files and libraries.

5. Set CMAKE_INSTALL_PREFIX to the directory where the plugin should be installed.  Usually,
this will be the same as OPENMM_DIR, so the plugin will be added to your OpenMM installation.

6. If you plan to build the OpenCL platform, make sure that OPENCL_INCLUDE_DIR and
OPENCL_LIBRARY are set correctly, and that PMESLICING_BUILD_OPENCL_LIB is selected.

7. If you plan to build the CUDA platform, make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly
and that PMESLICING_BUILD_CUDA_LIB is selected.

8. Press "Configure" again if necessary, then press "Generate".

9. Use the build system you selected to build and install the plugin.  For example, if you
selected Unix Makefiles, type `make install`.

Python API
==========

As [OpenMM], this project uses [SWIG] to generate its Python API.  SWIG takes an "interface
file", which is essentially a C++ header file with some extra annotations added, as its input.
It then generates a Python extension module exposing the C++ API in Python.

To build and install the Python API, build the `PythonInstall` target, for example by typing
`make PythonInstall` (if you are installing into the system Python, you may need to use sudo).
Once you do that, you can use the plugin from your Python scripts:

```py
    import openmm as mm
    import pmeslicing as plugin
    system = mm.System()
    force = plugin.SlicedPmeForce()
    system.addForce(force)
```

Test Cases
==========

To run the C++ test cases, build the "test" target, for example by typing `make test`.

To run the Python test cases, build the "PythonTest" target by typing `make PythonTest`.


[CMake]:                http://www.cmake.org
[CustomNonbondedForce]: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomNonbondedForce.html
[NonbondedForce]:       http://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
[OpenMM]:               https://openmm.org
[SWIG]:                 http://www.swig.org
