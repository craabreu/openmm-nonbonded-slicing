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

Installation
============

The plugin is distributed on [conda-forge]:

```bash
mamba install -c conda-forge openmm-nonbonded-slicing
```

and on [PyPI], for Linux x86_64 and macOS (Apple Silicon and Intel) with Python 3.10–3.14:

```bash
pip install openmm-nonbonded-slicing            # Reference and OpenCL platforms
pip install openmm-nonbonded-slicing[cuda12]    # plus the CUDA platform (Linux, CUDA 12)
pip install openmm-nonbonded-slicing[cuda13]    # plus the CUDA platform (Linux, CUDA 13)
```

Each PyPI release is published once per supported OpenMM version. pip installs the newest one,
upgrading OpenMM if needed; to keep an older OpenMM, pin it in the same command, for example
`pip install "openmm==8.5.*" openmm-nonbonded-slicing`:

| PyPI release  | OpenMM |
|---------------|--------|
| `X.Y.Z`       | 8.4    |
| `X.Y.Z.post1` | 8.5    |
| `X.Y.Z.post2` | 8.6    |

Instructions for building from source are in the [documentation](https://craabreu.github.io/openmm-nonbonded-slicing/).

Usage
=====

```py
import openmm as mm
import nonbondedslicing as nbs
system = mm.System()
force = nbs.SlicedNonbondedForce(2)
system.addForce(force)
```


[NonbondedForce]:       http://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
[Context]:              http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html
[getState]:             http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html#openmm.openmm.Context.getState
[OpenMM]:               https://openmm.org
[conda-forge]:          https://anaconda.org/conda-forge/openmm-nonbonded-slicing
[PyPI]:                 https://pypi.org/project/openmm-nonbonded-slicing
