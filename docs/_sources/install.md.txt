# Installation

## conda-forge

The plugin is distributed on [conda-forge](https://anaconda.org/conda-forge/openmm-nonbonded-slicing):

```bash
mamba install -c conda-forge openmm-nonbonded-slicing
```

## PyPI

Wheels are published on [PyPI](https://pypi.org/project/openmm-nonbonded-slicing) for Linux x86_64 and
macOS (Apple Silicon and Intel) with Python 3.10–3.14:

```bash
pip install openmm-nonbonded-slicing            # Reference and OpenCL platforms
pip install openmm-nonbonded-slicing[cuda12]    # plus the CUDA platform (Linux, CUDA 12)
pip install openmm-nonbonded-slicing[cuda13]    # plus the CUDA platform (Linux, CUDA 13)
```

Each PyPI release is published once per supported OpenMM version. pip installs the newest one, upgrading
OpenMM if needed; to keep an older OpenMM, pin it in the same command, for example
`pip install "openmm==8.5.*" openmm-nonbonded-slicing`:

| PyPI release  | OpenMM |
|---------------|--------|
| `X.Y.Z`       | 8.4    |
| `X.Y.Z.post1` | 8.5    |
| `X.Y.Z.post2` | 8.6    |

On Intel Macs, OpenMM 8.6 from PyPI only works with Python 3.10, so pip installs the OpenMM 8.5 release
there for Python 3.11 and later.

## Building from source

Requirements:

* [OpenMM](https://openmm.org) 8.4 or later
* [CMake](http://www.cmake.org) 3.17 or later and a C++ compiler
* [SWIG](http://www.swig.org), the same version that built your OpenMM (OpenMM's conda-forge builds 8.4
  and 8.5/8.6 used SWIG 4.4 and 4.5, respectively)
* Optional: OpenCL headers for the OpenCL platform, and a CUDA toolkit with NVRTC and cuFFT for the CUDA
  platform

In a conda environment where OpenMM is installed:

```bash
mkdir build && cd build
cmake ..
make install
make PythonInstall
```

Useful CMake options:

| Option | Default | Meaning |
|---|---|---|
| `OPENMM_DIR` | `$CONDA_PREFIX` | Where OpenMM is installed |
| `OPENMM_VERSION` | detected with `python -c "import openmm"` | OpenMM version, if Python cannot import it |
| `CMAKE_INSTALL_PREFIX` | `OPENMM_DIR` | Where to install the plugin |
| `PLUGIN_BUILD_OPENCL_LIB` | `ON` if OpenCL headers are found | Build the OpenCL platform |
| `PLUGIN_BUILD_CUDA_LIB` | `ON` if a CUDA toolkit is found | Build the CUDA platform (set `CUDAToolkit_ROOT` to pick a toolkit) |
| `VKFFT_INCLUDE_DIR` | downloaded at configure time | Directory containing `vkFFT.h` |

## Running the tests

From the build directory, run the C++ tests with `make test` (or `ctest`) and the Python tests with
`make PythonTest`.
