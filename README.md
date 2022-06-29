OpenMM NativeNonbonded Plugin
=============================

[![GH Actions Status](https://github.com/craabreu/openmm-native-nonbonded-plugin/workflows/Linux/badge.svg)](https://github.com/craabreu/openmm-native-nonbonded-plugin/actions?query=branch%3Amain+workflow%3ALinux)
[![GH Actions Status](https://github.com/craabreu/openmm-native-nonbonded-plugin/workflows/MacOS/badge.svg)](https://github.com/craabreu/openmm-native-nonbonded-plugin/actions?query=branch%3Amain+workflow%3AMacOS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is an [OpenMM] plugin that simply reproduces the original [NonbondedForce] class.
The purpose is to allow developers to experiment with modifications to the class without having to
work directly with OpenMM's complete codebase.


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
OPENCL_LIBRARY are set correctly, and that NATIVENONBONDED_BUILD_OPENCL_LIB is selected.

7. If you plan to build the CUDA platform, make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly
and that NATIVENONBONDED_BUILD_CUDA_LIB is selected.

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
    import nativenonbondedplugin as plugin
    system = mm.System()
    force = plugin.NativeNonbondedForce()
    system.addForce(force)
```


Test Cases
==========

To run the C++ test cases, build the "test" target, for example by typing `make test`.

To run the Python test cases, build the "PythonTest" target, for example by typing `make PythonTest`.


[CMake]:            http://www.cmake.org
[NonbondedForce]:   http://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
[OpenMM]:           https://openmm.org
[SWIG]:             http://www.swig.org