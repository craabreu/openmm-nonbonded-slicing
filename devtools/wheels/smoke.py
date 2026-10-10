"""Checks run in a clean venv after installing the wheels."""

import argparse
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package import link_problems  # noqa: E402


def _energy(platform_name):
    import openmm
    import nonbondedslicing

    system = openmm.System()
    for _ in range(2):
        system.addParticle(1.0)
    force = nonbondedslicing.SlicedNonbondedForce(2)
    force.addParticle(1.0, 1.0, 0.0)
    force.addParticle(-1.0, 1.0, 0.0)
    force.setParticleSubset(1, 1)
    system.addForce(force)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    context.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(1, 0, 0)])
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    print(f"{platform_name} energy: {energy}")
    return energy._value


def check_base(expected, opencl):
    import openmm
    import nonbondedslicing

    failures = [f for f in openmm.Platform.getPluginLoadFailures()
                if "NonbondedSlicing" in f and "NonbondedSlicingCUDA" not in f]
    assert not failures, f"plugin load failures: {failures}"
    assert version("openmm-nonbonded-slicing") == expected, version("openmm-nonbonded-slicing")
    assert expected.split(".post")[0] == nonbondedslicing.__version__, nonbondedslicing.__version__

    reference = _energy("Reference")
    if opencl:
        names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
        assert "OpenCL" in names, f"no OpenCL platform in {names}; load failures: {openmm.Platform.getPluginLoadFailures()}"
        assert abs(_energy("OpenCL") - reference) <= 1e-4 * abs(reference), "OpenCL and Reference disagree"


def check_cuda(expected, cuda):
    import openmm.version

    assert version(f"openmm-nonbonded-slicing-cuda-{cuda}") == expected
    plugin = Path(openmm.version.openmm_library_path) / "plugins" / "libNonbondedSlicingCUDA.so"
    # Relocating every symbol catches a plugin built against newer CUDA libraries than pip installed
    result = subprocess.run(["ldd", "-r", str(plugin)], capture_output=True, text=True)
    output = result.stdout + result.stderr
    problems = link_problems(output)
    assert not problems, "\n".join(problems) + f"\n{output}"


def opencl_devices():
    import ctypes

    cl = ctypes.CDLL("libOpenCL.so.1")
    count = ctypes.c_uint()
    cl.clGetPlatformIDs(0, None, ctypes.byref(count))
    platforms = (ctypes.c_void_p * count.value)()
    cl.clGetPlatformIDs(count.value, platforms, None)
    for platform in platforms:
        devices = ctypes.c_uint()
        status = cl.clGetDeviceIDs(ctypes.c_void_p(platform), ctypes.c_uint64(0xFFFFFFFF), 0, None, ctypes.byref(devices))
        print(f"OpenCL platform {platform:#x}: status {status}, {devices.value} devices")
    assert count.value > 0, "no OpenCL platforms"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["base", "cuda", "opencl-devices"])
    parser.add_argument("expected", nargs="?")
    parser.add_argument("cuda", nargs="?")
    parser.add_argument("--opencl", action="store_true", help="also compare the OpenCL platform with Reference")
    args = parser.parse_args()
    if args.kind == "opencl-devices":
        opencl_devices()
    elif args.kind == "base":
        check_base(args.expected, args.opencl)
    else:
        check_cuda(args.expected, args.cuda)
    print("OK")


if __name__ == "__main__":
    main()
