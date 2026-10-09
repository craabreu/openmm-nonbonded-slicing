import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import package  # noqa: E402

NVIDIA = "$ORIGIN/../../../nvidia"


def test_linux_runtime_paths():
    assert package.runtime_paths("_nonbondedslicing.cpython-312-x86_64-linux-gnu.so", "Linux", None) == [
        "$ORIGIN/OpenMM.libs/lib"]
    assert package.runtime_paths("OpenMM.libs/lib/libNonbondedSlicing.so", "Linux", None) == ["$ORIGIN"]
    assert package.runtime_paths("OpenMM.libs/lib/plugins/libNonbondedSlicingOpenCL.so", "Linux", None) == [
        "$ORIGIN", "$ORIGIN/.."]


def test_linux_cuda_runtime_paths():
    plugin = "OpenMM.libs/lib/plugins/libNonbondedSlicingCUDA.so"
    assert package.runtime_paths(plugin, "Linux", 12) == [
        "$ORIGIN", "$ORIGIN/..", f"{NVIDIA}/cufft/lib", f"{NVIDIA}/cuda_nvrtc/lib", f"{NVIDIA}/cuda_runtime/lib"]
    assert package.runtime_paths(plugin, "Linux", 13) == ["$ORIGIN", "$ORIGIN/..", f"{NVIDIA}/cu13/lib"]


def test_macos_runtime_paths():
    assert package.runtime_paths("_nonbondedslicing.cpython-312-darwin.so", "Darwin", None) == [
        "@loader_path/OpenMM.libs/lib"]
    assert package.runtime_paths("OpenMM.libs/lib/plugins/libNonbondedSlicingOpenCL.dylib", "Darwin", None) == [
        "@loader_path", "@loader_path/.."]


def _wheel(path, names):
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"")
    return path


BASE = [
    "nonbondedslicing.py",
    "_nonbondedslicing.cpython-312-x86_64-linux-gnu.so",
    "OpenMM.libs/lib/libNonbondedSlicing.so",
    "OpenMM.libs/lib/plugins/libNonbondedSlicingReference.so",
    "OpenMM.libs/lib/plugins/libNonbondedSlicingOpenCL.so",
    "OpenMM.libs/include/SlicedNonbondedForce.h",
]
REQUIRED = ["_nonbondedslicing*", "OpenMM.libs/lib/libNonbondedSlicing.*",
            "OpenMM.libs/lib/plugins/libNonbondedSlicingReference.*",
            "OpenMM.libs/lib/plugins/libNonbondedSlicingOpenCL.*"]


def test_check_wheel_accepts_our_files(tmp_path):
    package.check_wheel(_wheel(tmp_path / "ok.whl", BASE), REQUIRED)


def test_check_wheel_rejects_grafted_libraries(tmp_path):
    wheel = _wheel(tmp_path / "bad.whl", BASE + ["openmm_nonbonded_slicing.libs/libOpenMM-1a2b.so"])
    with pytest.raises(SystemExit, match="bundled"):
        package.check_wheel(wheel, REQUIRED)


def test_check_wheel_rejects_foreign_openmm_files(tmp_path):
    wheel = _wheel(tmp_path / "bad.whl", BASE + ["OpenMM.libs/lib/libOpenMM.so"])
    with pytest.raises(SystemExit, match="foreign"):
        package.check_wheel(wheel, REQUIRED)


def test_check_wheel_requires_opencl_plugin(tmp_path):
    wheel = _wheel(tmp_path / "noopencl.whl", [n for n in BASE if "OpenCL" not in n])
    with pytest.raises(SystemExit, match="missing"):
        package.check_wheel(wheel, REQUIRED)


def test_missing_libraries():
    output = (
        "\tlibOpenMM.so => /x/OpenMM.libs/lib/libOpenMM.so (0x1)\n"
        "\tlibcuda.so.1 => not found\n"
        "\tlibcufft.so.11 => /x/nvidia/cufft/lib/libcufft.so.11 (0x2)\n"
    )
    assert package.missing_libraries(output) == {"libcuda.so.1"}


def test_inject_adds_staged_files(tmp_path):
    stage = tmp_path / "stage"
    (stage / "OpenMM.libs/lib").mkdir(parents=True)
    (stage / "nonbondedslicing.py").write_text("")
    (stage / "OpenMM.libs/lib/libNonbondedSlicing.so").write_bytes(b"lib")
    wheel = tmp_path / "x-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("nonbondedslicing.py", "")
        archive.writestr("x-1.0.dist-info/RECORD", "nonbondedslicing.py,,\nx-1.0.dist-info/RECORD,,\n")
    package.inject(wheel, stage)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        record = archive.read("x-1.0.dist-info/RECORD").decode()
    assert "OpenMM.libs/lib/libNonbondedSlicing.so" in names
    assert names.count("nonbondedslicing.py") == 1
    assert "OpenMM.libs/lib/libNonbondedSlicing.so,sha256=" in record


def test_arrange_orders_by_post_and_kind(tmp_path):
    source, destination = tmp_path / "in", tmp_path / "out"
    (source / "a").mkdir(parents=True)
    names = [
        "openmm_nonbonded_slicing-0.3.0-cp312-cp312-manylinux_2_34_x86_64.whl",
        "openmm_nonbonded_slicing-0.3.0.post2-cp312-cp312-manylinux_2_34_x86_64.whl",
        "openmm_nonbonded_slicing_cuda_12-0.3.0.post2-py3-none-manylinux_2_34_x86_64.whl",
    ]
    for name in names:
        (source / "a" / name).write_bytes(b"")
    package.arrange(source, destination, expected=None)
    assert sorted(p.name for p in (destination / "0-base").iterdir()) == [names[0]]
    assert sorted(p.name for p in (destination / "2-base").iterdir()) == [names[1]]
    assert sorted(p.name for p in (destination / "2-cuda").iterdir()) == [names[2]]


def test_arrange_checks_count(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "openmm_nonbonded_slicing-0.3.0-cp312-cp312-manylinux_2_34_x86_64.whl").write_bytes(b"")
    with pytest.raises(SystemExit, match="expected 2"):
        package.arrange(tmp_path / "in", tmp_path / "out", expected=2)
