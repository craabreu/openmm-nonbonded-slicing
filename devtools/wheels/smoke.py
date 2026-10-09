"""Checks run in a clean venv after installing the wheels."""

import argparse
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package import missing_libraries  # noqa: E402


def check_base(expected):
    import openmm
    import nonbondedslicing

    failures = [f for f in openmm.Platform.getPluginLoadFailures()
                if "NonbondedSlicing" in f and "NonbondedSlicingCUDA" not in f]
    assert not failures, f"plugin load failures: {failures}"
    assert version("openmm-nonbonded-slicing") == expected, version("openmm-nonbonded-slicing")
    assert expected.split(".post")[0] == nonbondedslicing.__version__, nonbondedslicing.__version__

    system = openmm.System()
    for _ in range(2):
        system.addParticle(1.0)
    force = nonbondedslicing.SlicedNonbondedForce(2)
    force.addParticle(1.0, 1.0, 0.0)
    force.addParticle(-1.0, 1.0, 0.0)
    force.setParticleSubset(1, 1)
    system.addForce(force)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    context.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(1, 0, 0)])
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    print(f"Reference energy: {energy}")


def check_cuda(expected, cuda):
    import openmm.version

    assert version(f"openmm-nonbonded-slicing-cuda-{cuda}") == expected
    plugin = Path(openmm.version.openmm_library_path) / "plugins" / "libNonbondedSlicingCUDA.so"
    output = subprocess.run(["ldd", str(plugin)], check=True, capture_output=True, text=True).stdout
    missing = missing_libraries(output)
    # libcuda.so.1 comes from the NVIDIA driver, which CI runners do not have
    assert missing == {"libcuda.so.1"}, f"unresolved libraries: {sorted(missing)}\n{output}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["base", "cuda"])
    parser.add_argument("expected")
    parser.add_argument("cuda", nargs="?")
    args = parser.parse_args()
    if args.kind == "base":
        check_base(args.expected)
    else:
        check_cuda(args.expected, args.cuda)
    print("OK")


if __name__ == "__main__":
    main()
