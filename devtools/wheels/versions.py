"""Release scheme of the PyPI wheels.

Each source release X.Y.Z is published once per supported OpenMM minor version, as X.Y.Z,
X.Y.Z.post1, X.Y.Z.post2, ... in increasing order of OpenMM version, so that pip picks the
release that matches the OpenMM in the environment.
"""

import argparse
import json
import re
from pathlib import Path

VERSIONS_FILE = Path(__file__).with_name("openmm-versions.json")
PYTHONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
REDUCED_PYTHON = "3.12"
CUDAS = [12, 13]
MACOS = [
    {"runner": "macos-15", "arch": "arm64", "target": "11.0"},
    {"runner": "macos-15-intel", "arch": "x86_64", "target": "10.13"},
]


def _key(version):
    return tuple(int(part) for part in re.findall(r"\d+", version))


def load_versions(path=VERSIONS_FILE):
    entries = json.loads(Path(path).read_text())
    posts = [entry["post"] for entry in entries.values()]
    if posts != list(range(len(posts))):
        raise ValueError(f"post numbers must be 0, 1, 2, ... in file order, got {posts}")
    minors = list(entries)
    if [_key(m) for m in minors] != sorted(_key(m) for m in minors):
        raise ValueError(f"post numbers must increase with the OpenMM version, got {minors}")
    for minor, entry in entries.items():
        if not entry["openmm"].startswith(minor + "."):
            raise ValueError(f"OpenMM {entry['openmm']} does not belong to minor version {minor}")
    return entries


def source_version(source_dir):
    text = (Path(source_dir) / "CMakeLists.txt").read_text()
    match = re.search(r"PROJECT\(OpenMMNonbondedSlicing\s+VERSION\s+([0-9.]+)\)", text, re.IGNORECASE)
    if match is None:
        raise ValueError(f"no project version in {source_dir}/CMakeLists.txt")
    return match.group(1)


def release_version(base, post):
    return base if post == 0 else f"{base}.post{post}"


def next_minor(minor):
    major, minor_number = minor.split(".")
    return f"{major}.{int(minor_number) + 1}"


def requirement(minor, entry, package="openmm"):
    return f"{package}>={entry['openmm']},<{next_minor(minor)}"


def build_matrix(entries, full):
    pythons = PYTHONS if full else [REDUCED_PYTHON]
    return {
        "linux": [{"openmm": m, "python": p} for m in entries for p in pythons],
        "cuda": [{"openmm": m, "cuda": c} for m in entries for c in CUDAS],
        "macos": [{"openmm": m, "python": p, **mac} for m in entries for p in pythons for mac in MACOS],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("version", help="print the release version of a build")
    command.add_argument("source_dir")
    command.add_argument("minor")
    command = commands.add_parser("matrix", help="print the GitHub Actions build matrix as JSON")
    command.add_argument("--full", action="store_true")
    command = commands.add_parser("field", help="print a field of an OpenMM entry")
    command.add_argument("minor")
    command.add_argument("name", choices=["post", "openmm", "swig"])
    command = commands.add_parser("pins", help="print '<openmm pin> <expected release>' lines")
    command.add_argument("source_dir")
    args = parser.parse_args(argv)

    entries = load_versions()
    if args.command == "version":
        print(release_version(source_version(args.source_dir), entries[args.minor]["post"]))
    elif args.command == "matrix":
        print(json.dumps(build_matrix(entries, args.full)))
    elif args.command == "field":
        print(entries[args.minor][args.name])
    elif args.command == "pins":
        base = source_version(args.source_dir)
        for entry in entries.values():
            print(f"openmm=={entry['openmm']} {release_version(base, entry['post'])}")


if __name__ == "__main__":
    main()
