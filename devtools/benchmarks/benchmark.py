"""Measure the cost of SlicedNonbondedForce against that of NonbondedForce.

Systems:
    water  TIP3P water box (8 nm, about 50k atoms) split into n = 1, 2, 4 and 8 subsets
    tyk2   TYK2 kinase (Amber ff14SB) with ligand ejm_31 (OpenFF 2.3) in TIP3P water and 0.15 M
           NaCl (about 51k atoms), split into ligand, protein and solvent

Every slice has its own scaling parameter. A step is a LangevinMiddleIntegrator step of 2 fs, which
computes only forces. An evaluation is a getState call with forces and energy, plus parameter
derivatives in the last column. Steps are measured without and with one requested derivative per
slice. Each time is the best of 3 batches.

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
    for i in range(n):
        for j in range(i, n):
            name = f"lambda_{i}_{j}"
            force.addGlobalParameter(name, 1.0)
            force.addScalingParameter(name, i, j, True, True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", choices=["water", "tyk2"])
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--precision", default="mixed")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--cache", type=Path, default=Path("benchmark-inputs"), help="where tyk2 inputs are kept")
    args = parser.parse_args()

    forcefield, modeller, layouts = water_box() if args.system == "water" else tyk2(args.cache)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                     nonbondedCutoff=1 * unit.nanometer, constraints=app.HBonds, rigidWater=True)
    platform = mm.Platform.getPlatformByName(args.platform)
    properties = {"Precision": args.precision} if args.platform in ("CUDA", "OpenCL", "HIP") else {}

    print(f"{args.system}: {system.getNumParticles()} atoms, OpenMM {mm.__version__}, {args.platform} {properties}")
    print("slices; times in ms: step, step with derivatives, evaluation, evaluation with derivatives")
    step, evaluation = measure(system, modeller.positions, platform, properties, args.steps)
    print(f"{'NonbondedForce':>24} {'':>6} {1e3 * step:8.3f} {'':>8} {1e3 * evaluation:8.3f}", flush=True)
    for label, subsets in layouts:
        n = max(subsets) + 1
        step, evaluation = measure(
            sliced(system, subsets, False), modeller.positions, platform, properties, args.steps)
        step2, evaluation2 = measure(
            sliced(system, subsets, True), modeller.positions, platform, properties, args.steps, derivatives=True)
        print(f"{label:>24} {n * (n + 1) // 2:>6} {1e3 * step:8.3f} {1e3 * step2:8.3f} {1e3 * evaluation:8.3f} "
              f"{1e3 * evaluation2:8.3f}", flush=True)


if __name__ == "__main__":
    main()
