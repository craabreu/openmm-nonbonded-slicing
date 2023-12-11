OpenMM Nonbonded Slicing Plugin
===============================

[![Linux](https://github.com/craabreu/openmm-nonbonded-slicing/actions/workflows/Linux.yml/badge.svg)](https://github.com/craabreu/openmm-nonbonded-slicing/actions/workflows/Linux.yml)
[![MacOS](https://github.com/craabreu/openmm-nonbonded-slicing/actions/workflows/MacOS.yml/badge.svg)](https://github.com/craabreu/openmm-nonbonded-slicing/actions/workflows/MacOS.yml)
[![Doc](https://github.com/craabreu/openmm-nonbonded-slicing/actions/workflows/Doc.yml/badge.svg)](https://github.com/craabreu/openmm-nonbonded-slicing/actions/workflows/Doc.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This [OpenMM] plugin contains the **SlicedNonbondedForce** class, a variant of OpenMM's [NonbondedForce].
By partitioning all particles among $n$ disjoint subsets, the total potential energy becomes a linear
combination of contributions from pairs of subsets like

```math
E = \sum_{I=0}^{n-1} \sum_{J=I}^{n-1} \left( \lambda^{vdW}_{I,J}E^{vdW}_{I,J}+\lambda^{elec}_{I,J}E^{elec}_{I,J} \right)
```

where each slice is defined by subsets $I$ and $J$, superscripts _vdW_ and _elec_ denote van
der Waals and electrostatic contributions, $E_{I,J}$ is the potential energy of all particle pairs
formed by one particle in subset $I$ and one in subset $J$, and $\lambda_{I,J}$ is a scaling parameter.

By default, all scaling parameters are constant and equal to 1. However, the user can turn selected
scaling parameters into variables and store their values in [Context] global parameters. Derivatives
with respect to these variables can be requested and used, for instance, to report individual energy
slice contributions or sums thereof via [getState] with option `getParameterDerivatives=True`.

Documentation
=============

Documentation for this plugin is available at [Github Pages](https://craabreu.github.io/openmm-nonbonded-slicing/).
It includes the Python API and the theory for slicing lattice-sum energy contributions.

Installing from Source
======================

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
OPENCL_LIBRARY are set correctly, and that PLUGIN_BUILD_OPENCL_LIB is selected.

7. If you plan to build the CUDA platform, make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly
and that PLUGIN_BUILD_CUDA_LIB is selected.

8. Press "Configure" again if necessary, then press "Generate".

9. Use the build system you selected to build and install the plugin.  For example, if you
selected Unix Makefiles, type `make install`.

Python Wrapper and API
======================

As [OpenMM], this project uses [SWIG] to generate its Python API.  SWIG takes an "interface
file", which is essentially a C++ header file with some extra annotations added, as its input.
It then generates a Python extension module exposing the C++ API in Python.

To build and install the Python API, build the `PythonInstall` target, for example by typing
`make PythonInstall` (if you are installing into the system Python, you may need to use sudo).
Once you do that, you can use the plugin from your Python scripts:

```py
    import openmm as mm
    import nonbondedslicing as nbs
    system = mm.System()
    force = nbs.SlicedNonbondedForce(2)
    system.addForce(force)
```

Test Cases
==========

To run the C++ test cases, build the "test" target, for example by typing `make test`.

To run the Python test cases, build the "PythonTest" target by typing `make PythonTest`.


[CMake]:                http://www.cmake.org
[NonbondedForce]:       http://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
[Context]:              http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html
[getState]:             http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html#openmm.openmm.Context.getState
[OpenMM]:               https://openmm.org
[SWIG]:                 http://www.swig.org