"""Base wheel: Python module, extension, libNonbondedSlicing, Reference and OpenCL plugins."""

import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools.dist import Distribution

sys.path.insert(0, str(Path(__file__).resolve().parent))
from versions import load_versions, release_version, requirement, source_version  # noqa: E402

minor = os.environ["NBS_OPENMM_MINOR"]
entry = load_versions()[minor]
version = release_version(source_version(os.environ["NBS_SOURCE_DIR"]), entry["post"])
readme = Path(os.environ["NBS_SOURCE_DIR"]) / "README.md"


class BinaryDistribution(Distribution):
    # The extension and plugins are injected after the build; this makes the wheel platform-specific
    def has_ext_modules(self):
        return True


setup(
    name="openmm-nonbonded-slicing",
    version=version,
    description="An OpenMM plugin for slicing nonbonded interactions",
    long_description=readme.read_text() if readme.exists() else "",
    long_description_content_type="text/markdown",
    author="Charlles Abreu",
    url="https://github.com/craabreu/openmm-nonbonded-slicing",
    license="MIT",
    python_requires=">=3.10",
    package_dir={"": os.environ["NBS_STAGE_DIR"]},
    py_modules=["nonbondedslicing"],
    install_requires=[requirement(minor, entry)],
    extras_require={
        f"cuda{cuda}": [f'openmm-nonbonded-slicing-cuda-{cuda}=={version}; platform_system == "Linux"']
        for cuda in (12, 13)
    },
    distclass=BinaryDistribution,
)
