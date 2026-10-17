"""Measure the cost of SlicedNonbondedForce against that of NonbondedForce.

Systems:
    water  TIP3P water box (8 nm, about 50k atoms) split into n = 1, 2, 4 and 8 subsets
    tyk2   TYK2 kinase (Amber ff14SB) with ligand ejm_31 (OpenFF 2.3) in TIP3P water and 0.15 M
           NaCl (about 51k atoms), split into ligand, protein and solvent

The interactions between subset 0 and each other subset are scaled by two parameters, one for
Coulomb and one for Lennard-Jones, making 2(n-1) parameters for n subsets. A step is a
LangevinMiddleIntegrator step of 2 fs, which computes only forces. An evaluation is a getState call
with forces and energy, plus parameter derivatives in the last column. Steps are measured without
and with the derivatives with respect to all parameters. Each time is the best of 3 batches.

The tyk2 system needs openmmforcefields, openff-toolkit and ambertools, and downloads its input
files from the OpenFF protein-ligand benchmark.
"""

import argparse
import time
import urllib.request
from pathlib import Path

import openmm as mm
from openmm import app, unit

import nonbondedslicing as nbs

PLB = ("https://raw.githubusercontent.com/openforcefield/protein-ligand-benchmark/"
       "fd88824f9114244f95a14b485e6d6c96c1de716d/data/tyk2")
LIGAND = "lig_ejm_31"


def water_box():
    forcefield = app.ForceField("amber14/tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(forcefield, boxSize=mm.Vec3(8, 8, 8) * unit.nanometer, model="tip3p")
    layouts = [(f"{n} blocks", subsets_in_blocks(modeller.topology, n)) for n in (1, 2, 4, 8)]
    return forcefield, modeller, layouts


def tyk2(cache):
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    cache.mkdir(parents=True, exist_ok=True)
    for name, path in [("protein.pdb", "01_protein/crd/protein.pdb"), ("ligands.sdf", "02_ligands/ligands.sdf")]:
        if not (cache / name).exists():
            urllib.request.urlretrieve(f"{PLB}/{path}", cache / name)
    molecules = Molecule.from_file(str(cache / "ligands.sdf"), allow_undefined_stereo=True)
    ligand = next(m for m in molecules if m.name == LIGAND)
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    generator = SMIRNOFFTemplateGenerator(molecules=ligand, forcefield="openff-2.3.0")
    forcefield.registerTemplateGenerator(generator.generator)
    protein = app.PDBFile(str(cache / "protein.pdb"))
    modeller = app.Modeller(protein.topology, protein.positions)
    modeller.delete([r for r in modeller.topology.residues() if r.name == "HOH"])
    ligand_topology = ligand.to_topology()
    modeller.add(ligand_topology.to_openmm(), ligand_topology.get_positions().to_openmm())
    protein_atoms = protein.topology.getNumAtoms() - sum(
        len(r) for r in protein.topology.residues() if r.name == "HOH")
    ligand_atoms = ligand.n_atoms
    modeller.addSolvent(forcefield, model="tip3p", padding=1 * unit.nanometer, boxShape="cube",
                        ionicStrength=0.15 * unit.molar)
    # Ligand (0), protein (1) and solvent with ions (2), in the order Modeller placed them
    subsets = [1] * protein_atoms + [0] * ligand_atoms
    subsets += [2] * (modeller.topology.getNumAtoms() - len(subsets))
    return forcefield, modeller, [("ligand/protein/solvent", subsets)]


def subsets_in_blocks(topology, n):
    """n contiguous blocks of whole molecules."""
    molecules = topology.getNumResidues()
    return [min(n - 1, atom.residue.index * n // molecules) for atom in topology.atoms()]


def sliced(system, subsets, derivatives):
    system = mm.XmlSerializer.clone(system)
    index = next(i for i, f in enumerate(system.getForces()) if isinstance(f, mm.NonbondedForce))
    n = max(subsets) + 1
    force = nbs.SlicedNonbondedForce(system.getForce(index), n)
    for atom, subset in enumerate(subsets):
        force.setParticleSubset(atom, subset)
    for j in range(1, n):
        for name, coulomb in [(f"lambda_coul_0_{j}", True), (f"lambda_lj_0_{j}", False)]:
            force.addGlobalParameter(name, 1.0)
            force.addScalingParameter(name, 0, j, coulomb, not coulomb)
            if derivatives:
                force.addEnergyParameterDerivative(name)
    system.removeForce(index)
    system.addForce(force)
    return system


def measure(system, positions, platform, properties, steps, derivatives=False):
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(positions)
    mm.LocalEnergyMinimizer.minimize(context, maxIterations=100)
    integrator.step(50)
    context.getState(getPositions=True)  # waits for the GPU without computing energies

    def best(run, count):
        """Shortest mean time over 3 batches, which is the least affected by other loads."""
        times = []
        for _ in range(3):
            start = time.perf_counter()
            run(count)
            context.getState(getPositions=True)  # waits for the GPU without computing energies
            times.append((time.perf_counter() - start) / count)
        return min(times)

    def evaluate(count):
        for _ in range(count):
            context.getState(getForces=True, getEnergy=True, getParameterDerivatives=derivatives)

    return best(integrator.step, steps), best(evaluate, steps // 10)


def inner_group(system, positions, platform, properties, steps):
    """Milliseconds per force evaluation of each part of the inner force group split (switching from 0.5 to
    0.8 nm), and of ufedmm's CustomNonbondedForce implementation of the same split if ufedmm is installed."""

    def times(system, groups):
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform, properties)
        context.setPositions(positions)
        context.getState(getForces=True)
        result = []
        for group in groups:
            batches = []
            for _ in range(3):
                start = time.perf_counter()
                for _ in range(steps):
                    context.getState(getForces=True, groups=group)
                batches.append((time.perf_counter() - start) / steps)
            result.append(min(batches))
        return result

    system = mm.XmlSerializer.clone(system)
    for force in system.getForces():
        force.setForceGroup(3)
    split = sliced(system, [0] * system.getNumParticles(), False)
    force = nbs.SlicedNonbondedForce.cast(next(f for f in split.getForces() if nbs.SlicedNonbondedForce.isinstance(f)))
    force.setForceGroup(0)
    force.setReciprocalSpaceForceGroup(2)
    force.setInnerForceGroup(1)
    force.setInnerSwitchingDistance(0.5)
    force.setInnerCutoffDistance(0.8)
    rows = list(zip(["inner", "direct remainder", "direct remainder + reciprocal"],
                    times(split, [{1}, {0}, {0, 2}])))
    try:
        from ufedmm.integrators import add_inner_nonbonded_force
    except ImportError:
        return rows
    custom = mm.XmlSerializer.clone(system)
    add_inner_nonbonded_force(custom, 0.5, 0.8, 1)
    rows.append(("ufedmm inner (Custom forces)", times(custom, [{1}])[0]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", choices=["water", "tyk2"])
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--cache", type=Path, default=Path("benchmark-inputs"), help="where tyk2 inputs are kept")
    parser.add_argument("--inner-group", action="store_true",
                        help="time the parts of the inner force group split instead of slicing and derivatives")
    args = parser.parse_args()

    forcefield, modeller, layouts = water_box() if args.system == "water" else tyk2(args.cache)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                     nonbondedCutoff=1 * unit.nanometer, constraints=app.HBonds, rigidWater=True)
    platform = mm.Platform.getPlatformByName(args.platform)
    properties = {"Precision": args.precision} if args.platform in ("CUDA", "OpenCL", "HIP") else {}

    print(f"{args.system}: {system.getNumParticles()} atoms, OpenMM {mm.__version__}, {args.platform} {properties}")
    if args.inner_group:
        print("ms per force evaluation")
        for label, seconds in inner_group(system, modeller.positions, platform, properties, args.steps):
            print(f"{label:>32} {1e3 * seconds:8.3f}", flush=True)
        return
    print("derivatives; times in ms: step, step with derivatives, evaluation, evaluation with derivatives")
    step, evaluation = measure(system, modeller.positions, platform, properties, args.steps)
    print(f"{'NonbondedForce':>24} {'':>6} {1e3 * step:8.3f} {'':>8} {1e3 * evaluation:8.3f}", flush=True)
    for label, subsets in layouts:
        derivatives = 2 * max(subsets)
        step, evaluation = measure(
            sliced(system, subsets, False), modeller.positions, platform, properties, args.steps)
        if derivatives == 0:
            print(f"{label:>24} {0:>6} {1e3 * step:8.3f} {'':>8} {1e3 * evaluation:8.3f}", flush=True)
            continue
        step2, evaluation2 = measure(
            sliced(system, subsets, True), modeller.positions, platform, properties, args.steps, derivatives=True)
        print(f"{label:>24} {derivatives:>6} {1e3 * step:8.3f} {1e3 * step2:8.3f} {1e3 * evaluation:8.3f} "
              f"{1e3 * evaluation2:8.3f}", flush=True)


if __name__ == "__main__":
    main()
