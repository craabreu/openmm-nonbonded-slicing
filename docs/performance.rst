===========
Performance
===========

The tables below compare :class:`~nonbondedslicing.SlicedNonbondedForce` with the
:OpenMM:`NonbondedForce` it replaces in two systems of about 50,000 atoms, using PME with a 1 nm
cutoff on the CUDA platform in mixed precision (NVIDIA RTX 4090, OpenMM 8.6.1 with the reduction of
parameter derivatives on the device from `openmm/openmm#5520`_). With *n* subsets,
the interactions between subset 0 and each other subset are scaled by two parameters, one for
Coulomb and one for Lennard-Jones, as one would do to decouple a ligand from its environment. A
*step* is a ``LangevinMiddleIntegrator`` step of 2 fs, which computes only forces. An *evaluation* is
a call to ``getState`` with ``getForces=True`` and ``getEnergy=True``. Columns marked *with
derivatives* request the derivatives with respect to all 2(*n*-1) parameters and, for evaluations,
pass ``getParameterDerivatives=True``. Times are in milliseconds and can be reproduced with
``devtools/benchmarks/benchmark.py``.

The first system is a box of 16,814 TIP3P water molecules (50,442 atoms, 8 nm) split into *n*
contiguous blocks of whole molecules.

=====================  =======  ===========  ======  ==================  ==========  ==================
Force                  Subsets  Derivatives  Step    Step with           Evaluation  Evaluation with
                                                     derivatives                     derivatives
=====================  =======  ===========  ======  ==================  ==========  ==================
NonbondedForce         --       --           0.26    --                  0.58        --
SlicedNonbondedForce   1        --           0.27    --                  0.59        --
SlicedNonbondedForce   2        2            0.31    0.31                0.62        0.85
SlicedNonbondedForce   4        6            0.59    0.60                0.93        1.47
SlicedNonbondedForce   8        14           0.92    1.00                1.31        2.44
=====================  =======  ===========  ======  ==================  ==========  ==================

The second system is the TYK2 kinase (Amber ff14SB) bound to ligand ejm_31 (OpenFF 2.3) from the
OpenFF protein-ligand benchmark, in TIP3P water with 0.15 M NaCl (51,405 atoms, 8.15 nm cubic box).
It is split into ligand (subset 0, 32 atoms), protein (4,652 atoms) and solvent.

=====================  =======  ===========  ======  ==================  ==========  ==================
Force                  Subsets  Derivatives  Step    Step with           Evaluation  Evaluation with
                                                     derivatives                     derivatives
=====================  =======  ===========  ======  ==================  ==========  ==================
NonbondedForce         --       --           0.27    --                  0.59        --
SlicedNonbondedForce   3        4            0.34    0.34                0.68        0.97
=====================  =======  ===========  ======  ==================  ==========  ==================

The cost of a step grows sublinearly with the number of subsets: going from 1 to 8 subsets makes a
step about 3.4 times as expensive. Since derivatives are only computed along with the energy,
requesting them barely affects steps (9% with 14 derivatives). The cost of an evaluation with
derivatives grows with their number, not with the number of slices: a parameter that scales several
slices adds a single derivative. Derivatives are best requested only for the parameters whose
energies are needed, and only in the steps where they are reported. With OpenMM releases that sum
the derivatives on the host, each evaluation with derivatives costs about 0.5 ms more, regardless of
the system size.

.. _openmm/openmm#5520: https://github.com/openmm/openmm/pull/5520
