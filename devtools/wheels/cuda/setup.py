"""CUDA add-on wheel: only the CUDA plugin, installed into OpenMM.libs/lib/plugins."""

import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from versions import load_versions, release_version, requirement, source_version  # noqa: E402

minor = os.environ["NBS_OPENMM_MINOR"]
cuda = os.environ["NBS_CUDA"]
entry = load_versions()[minor]
version = release_version(source_version(os.environ["NBS_SOURCE_DIR"]), entry["post"])


class PlatformWheel(bdist_wheel):
    # The plugin does not depend on the Python version, only on the platform
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", os.environ["NBS_PLATFORM"]


setup(
    name=f"openmm-nonbonded-slicing-cuda-{cuda}",
    version=version,
    description=f"CUDA {cuda} platform for openmm-nonbonded-slicing",
    author="Charlles Abreu",
    url="https://github.com/craabreu/openmm-nonbonded-slicing",
    license="MIT",
    python_requires=">=3.10",
    py_modules=[],
    install_requires=[requirement(minor, entry, f"openmm-cuda-{cuda}"), f"openmm-nonbonded-slicing=={version}"],
    cmdclass={"bdist_wheel": PlatformWheel},
)
