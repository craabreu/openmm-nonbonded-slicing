"""Turn a CMake staging tree into wheels that install next to pip's OpenMM.

Our libraries go into site-packages/OpenMM.libs (owned by OpenMM's wheel), where
`import openmm` loads every plugin. Nothing from OpenMM, OpenCL or CUDA may be bundled.
"""

import argparse
import fnmatch
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

NVIDIA = "../../../nvidia"
CUDA_LIBRARY_DIRS = {
    12: ["cufft/lib", "cuda_nvrtc/lib", "cuda_runtime/lib"],
    13: ["cu13/lib"],
}
ORIGIN = {"Linux": "$ORIGIN", "Darwin": "@loader_path"}
WHEEL_NAME = re.compile(r"^(?P<dist>openmm_nonbonded_slicing(?:_cuda_\d+)?)-(?P<version>[^-]+)-")


def runtime_paths(relpath, system, cuda):
    origin = ORIGIN[system]
    parts = Path(relpath).parts
    if len(parts) == 1:
        return [f"{origin}/OpenMM.libs/lib"]
    if parts[-2] == "lib":
        return [origin]
    paths = [origin, f"{origin}/.."]
    if cuda is not None and "CUDA" in parts[-1]:
        paths += [f"{origin}/{NVIDIA}/{d}" for d in CUDA_LIBRARY_DIRS[cuda]]
    return paths


def _binaries(stage):
    for path in sorted(stage.rglob("*")):
        if path.is_file() and (path.suffix in (".so", ".dylib") or ".so." in path.name):
            yield path


def stage_module(site_packages, stage):
    stage.mkdir(parents=True, exist_ok=True)
    found = [site_packages / "nonbondedslicing.py", *site_packages.glob("_nonbondedslicing*.so")]
    if len(found) != 2 or not found[0].exists():
        raise SystemExit(f"nonbondedslicing module not found in {site_packages}: {found}")
    for path in found:
        shutil.copy2(path, stage / path.name)


def _macos_rpaths(binary):
    output = subprocess.run(["otool", "-l", str(binary)], check=True, capture_output=True, text=True).stdout
    return re.findall(r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (\S+)", output)


def fix_runtime_paths(stage, cuda):
    system = platform.system()
    for binary in _binaries(stage):
        wanted = runtime_paths(binary.relative_to(stage).as_posix(), system, cuda)
        if system == "Linux":
            subprocess.run(["patchelf", "--set-rpath", ":".join(wanted), str(binary)], check=True)
        else:
            for rpath in _macos_rpaths(binary):
                subprocess.run(["install_name_tool", "-delete_rpath", rpath, str(binary)], check=True)
            for rpath in wanted:
                subprocess.run(["install_name_tool", "-add_rpath", rpath, str(binary)], check=True)
            subprocess.run(["codesign", "--force", "--sign", "-", str(binary)], check=True)


def inject(wheel, stage):
    from delocate.wheeltools import InWheel

    wheel = Path(wheel).resolve()
    # The module is already in the wheel, and setuptools leaves its egg-info in the staging tree
    files = [
        p for p in sorted(Path(stage).rglob("*"))
        if p.is_file() and p.name != "nonbondedslicing.py"
        and not any(part.endswith(".egg-info") for part in p.relative_to(stage).parts)
    ]
    with InWheel(str(wheel), str(wheel)) as unpacked:
        for path in files:
            target = Path(unpacked) / path.relative_to(stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def check_wheel(wheel, required):
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
    problems = []
    for name in names:
        top = name.split("/")[0]
        if top.endswith(".dylibs") or (top.endswith(".libs") and top != "OpenMM.libs"):
            problems.append(f"bundled library: {name}")
        elif name.startswith("OpenMM.libs/lib/") and not Path(name).name.startswith("libNonbondedSlicing"):
            problems.append(f"foreign file in OpenMM.libs: {name}")
    for pattern in required:
        if not any(fnmatch.fnmatch(name, pattern) for name in names):
            problems.append(f"missing file matching {pattern}")
    if problems:
        raise SystemExit(f"{wheel}:\n  " + "\n  ".join(problems))


def missing_libraries(ldd_output):
    return set(re.findall(r"^\s*(\S+) => not found", ldd_output, re.MULTILINE))


def arrange(source, destination, expected):
    wheels = sorted(Path(source).rglob("*.whl"))
    if expected is not None and len(wheels) != expected:
        raise SystemExit(f"found {len(wheels)} wheels, expected {expected}")
    for wheel in wheels:
        match = WHEEL_NAME.match(wheel.name)
        if match is None:
            raise SystemExit(f"unexpected wheel {wheel.name}")
        post = re.search(r"\.post(\d+)$", match["version"])
        kind = "cuda" if "_cuda_" in match["dist"] else "base"
        target = Path(destination) / f"{post.group(1) if post else 0}-{kind}"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wheel, target / wheel.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("stage-module")
    command.add_argument("stage", type=Path)
    command = commands.add_parser("fix-runtime-paths")
    command.add_argument("stage", type=Path)
    command.add_argument("--cuda", type=int, choices=sorted(CUDA_LIBRARY_DIRS))
    command = commands.add_parser("inject")
    command.add_argument("wheel", type=Path)
    command.add_argument("stage", type=Path)
    command = commands.add_parser("check")
    command.add_argument("wheel", type=Path)
    command.add_argument("--require", action="append", default=[])
    command = commands.add_parser("arrange")
    command.add_argument("source", type=Path)
    command.add_argument("destination", type=Path)
    command.add_argument("--expected", type=int)
    args = parser.parse_args(argv)

    if args.command == "stage-module":
        import sysconfig
        stage_module(Path(sysconfig.get_paths()["platlib"]), args.stage)
    elif args.command == "fix-runtime-paths":
        fix_runtime_paths(args.stage, args.cuda)
    elif args.command == "inject":
        inject(args.wheel, args.stage)
    elif args.command == "check":
        check_wheel(args.wheel, args.require)
    elif args.command == "arrange":
        arrange(args.source, args.destination, args.expected)


if __name__ == "__main__":
    sys.exit(main())
