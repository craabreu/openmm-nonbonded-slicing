OpenMM Nonbonded Slicing Plugin
===============================

[![Linux](https://github.com/craabreu/openmm-nonbonded-slicing/actions/workflows/Linux.yml/badge.svg)](https://github.com/craabreu/openmm-nonbonded-slicing/actions/workflows/Linux.yml)
[![MacOS](https://github.com/craabreu/openmm-nonbonded-slicing/actions/workflows/MacOS.yml/badge.svg)](https://github.com/craabreu/openmm-nonbonded-slicing/actions/workflows/MacOS.yml)
[![Doc](https://github.com/craabreu/openmm-nonbonded-slicing/actions/workflows/Doc.yml/badge.svg)](https://github.com/craabreu/openmm-nonbonded-slicing/actions/workflows/Doc.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This [OpenMM] plugin contains the **SlicedNonbondedForce** class, a variant of OpenMM's [NonbondedForce].
By partitioning all particles among $`n`$ disjoint subsets, the total potential energy becomes a linear
combination of contributions from pairs of subsets like

```math
E = \sum_{I=0}^{n-1} \sum_{J=I}^{n-1} \left( \lambda^{vdW}_{I,J}E^{vdW}_{I,J}+\lambda^{elec}_{I,J}E^{elec}_{I,J} \right)
```

where each slice is defined by subsets $`I`$ and $`J`$, superscripts _vdW_ and _elec_ denote van
der Waals and electrostatic contributions, $`E_{I,J}`$ is the potential energy of all particle pairs
formed by one particle in subset $`I`$ and one in subset $`J`$, and $`\lambda_{I,J}`$ is a scaling parameter.

The scaling parameters can be tied to [Context] global parameters, so that selected slices are scaled
while a simulation runs; a scaling parameter can multiply the decapping term of its slice, which caps the
core of its Lennard-Jones interactions, and optionally of its Coulomb interactions, for alchemical decoupling
without singularities; the derivative of the energy with respect to a scaling parameter can be requested
along with the energy, which is a way of reporting the energies of individual slices; and, for multiple
time-step integration, a short-range,
force-switched copy of the direct-space interactions can be computed in its own force group, as can the
reciprocal-space sum with Ewald, PME or LJPME. The [documentation] has a user guide for each of these
capabilities, the theory behind the sliced lattice sums, measured costs and the Python API.

Example
=======

```python
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import nonbondedslicing as nbs

forcefield = app.ForceField("tip3p.xml")
modeller = app.Modeller(app.Topology(), [])
modeller.addSolvent(forcefield, boxSize=mm.Vec3(2, 2, 2)*unit.nanometers)
system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME)
index, nonbonded = next((i, f) for i, f in enumerate(system.getForces()) if isinstance(f, mm.NonbondedForce))

force = nbs.SlicedNonbondedForce(nonbonded, 2)   # two subsets; every particle starts in subset 0
for atom in range(3):
    force.setParticleSubset(atom, 1)             # move the first water molecule to subset 1
force.addGlobalParameter("lambda", 1.0)
force.addScalingParameter("lambda", 0, 1, True, True)   # lambda scales slice (0, 1)
force.addEnergyParameterDerivative("lambda")            # dE/dlambda = energy of slice (0, 1)
system.removeForce(index)
system.addForce(force)

context = mm.Context(system, mm.VerletIntegrator(1*unit.femtosecond))
context.setPositions(modeller.positions)
state = context.getState(getEnergy=True, getParameterDerivatives=True)
print("Total energy:", state.getPotentialEnergy())
print("Energy of slice (0, 1):", state.getEnergyParameterDerivatives()["lambda"], "kJ/mol")
context.setParameter("lambda", 0.0)
print("Without slice (0, 1):", context.getState(getEnergy=True).getPotentialEnergy())
```

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

On Intel Macs, OpenMM 8.6 from PyPI only works with Python 3.10, so pip installs the OpenMM 8.5
release there for Python 3.11 and later.

Instructions for building from source are in the [documentation].

Documentation
=============

The [documentation] is hosted on GitHub Pages. It has a user guide covering subsets and scaling
parameters, energy parameter derivatives, the capped Lennard-Jones potential and force-group splitting
for multiple time-step integration, the theory for slicing lattice-sum energy contributions, measured
costs and the Python API.


[NonbondedForce]:       http://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
[Context]:              http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html
[OpenMM]:               https://openmm.org
[conda-forge]:          https://anaconda.org/conda-forge/openmm-nonbonded-slicing
[PyPI]:                 https://pypi.org/project/openmm-nonbonded-slicing
[documentation]:        https://craabreu.github.io/openmm-nonbonded-slicing/
