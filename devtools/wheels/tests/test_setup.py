import email
import os
import subprocess
import sys
import zipfile
from pathlib import Path

WHEELS = Path(__file__).parents[1]


def _source(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "CMakeLists.txt").write_text("PROJECT(OpenMMNonbondedSlicing VERSION 0.3.0)\n")
    return source


def _build(project, out, env):
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(out), str(project)],
        check=True, env={**os.environ, **env}, capture_output=True)
    (wheel,) = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        (metadata,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        return wheel, email.message_from_bytes(archive.read(metadata))


def test_base_wheel_metadata(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "nonbondedslicing.py").write_text("")
    wheel, meta = _build(WHEELS, tmp_path / "out", {
        "NBS_SOURCE_DIR": str(_source(tmp_path)), "NBS_OPENMM_MINOR": "8.5", "NBS_STAGE_DIR": str(stage)})
    assert meta["Name"] == "openmm-nonbonded-slicing"
    assert meta["Version"] == "0.3.0.post1"
    requires = meta.get_all("Requires-Dist")
    assert "openmm<8.6,>=8.5.2" in requires
    assert "-none-any" not in wheel.name


def test_cuda_extras_are_linux_only(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "nonbondedslicing.py").write_text("")
    _, meta = _build(WHEELS, tmp_path / "out", {
        "NBS_SOURCE_DIR": str(_source(tmp_path)), "NBS_OPENMM_MINOR": "8.6", "NBS_STAGE_DIR": str(stage)})
    requires = meta.get_all("Requires-Dist")
    assert ('openmm-nonbonded-slicing-cuda-12==0.3.0.post2; platform_system == "Linux" and extra == "cuda12"'
            in requires)
    assert set(meta.get_all("Provides-Extra")) == {"cuda12", "cuda13"}


def test_cuda_wheel_metadata(tmp_path):
    wheel, meta = _build(WHEELS / "cuda", tmp_path / "out", {
        "NBS_SOURCE_DIR": str(_source(tmp_path)), "NBS_OPENMM_MINOR": "8.4", "NBS_CUDA": "13",
        "NBS_PLATFORM": "manylinux_2_34_x86_64"})
    assert meta["Name"] == "openmm-nonbonded-slicing-cuda-13"
    assert meta["Version"] == "0.3.0"
    assert set(meta.get_all("Requires-Dist")) == {
        "openmm-cuda-13<8.5,>=8.4.0.post2", "openmm-nonbonded-slicing==0.3.0"}
    assert wheel.name.endswith("-py3-none-manylinux_2_34_x86_64.whl")


def test_description_comes_from_the_built_source(tmp_path):
    # The project page must describe the release being built, not the tooling checkout
    source = _source(tmp_path)
    (source / "README.md").write_text("Release README\n")
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "nonbondedslicing.py").write_text("")
    _, meta = _build(WHEELS, tmp_path / "out", {
        "NBS_SOURCE_DIR": str(source), "NBS_OPENMM_MINOR": "8.4", "NBS_STAGE_DIR": str(stage)})
    assert meta.get_payload().strip() == "Release README"
