import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import versions  # noqa: E402


def test_shipped_versions_load():
    entries = versions.load_versions()
    assert list(entries) == ["8.4", "8.5", "8.6"]
    assert entries["8.6"] == {"post": 2, "openmm": "8.6.1", "swig": "4.5.0"}


def test_posts_increase_with_openmm(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({
        "8.5": {"post": 0, "openmm": "8.5.2", "swig": "4.4.1"},
        "8.4": {"post": 1, "openmm": "8.4.0", "swig": "4.4.0"},
    }))
    with pytest.raises(ValueError, match="increase"):
        versions.load_versions(path)


def test_posts_are_contiguous(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({
        "8.4": {"post": 0, "openmm": "8.4.0", "swig": "4.4.0"},
        "8.5": {"post": 2, "openmm": "8.5.2", "swig": "4.4.1"},
    }))
    with pytest.raises(ValueError, match="0, 1, 2"):
        versions.load_versions(path)


def test_build_version_matches_minor(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"8.4": {"post": 0, "openmm": "8.5.2", "swig": "4.4.1"}}))
    with pytest.raises(ValueError, match="8.4"):
        versions.load_versions(path)


def test_source_version(tmp_path):
    (tmp_path / "CMakeLists.txt").write_text("PROJECT(OpenMMNonbondedSlicing VERSION 0.3.0)\n")
    assert versions.source_version(tmp_path) == "0.3.0"


def test_release_version():
    assert versions.release_version("0.3.0", 0) == "0.3.0"
    assert versions.release_version("0.3.0", 2) == "0.3.0.post2"


def test_next_minor():
    assert versions.next_minor("8.6") == "8.7"
    assert versions.next_minor("8.10") == "8.11"


def test_requirement_bounds():
    entry = {"post": 0, "openmm": "8.4.0.post2", "swig": "4.4.0"}
    assert versions.requirement("8.4", entry) == "openmm>=8.4.0.post2,<8.5"
    assert versions.requirement("8.4", entry, "openmm-cuda-12") == "openmm-cuda-12>=8.4.0.post2,<8.5"


def test_full_matrix_sizes():
    matrix = versions.build_matrix(versions.load_versions(), full=True)
    assert len(matrix["linux"]) == 3 * 5
    assert len(matrix["cuda"]) == 3 * 2
    assert len(matrix["macos"]) == 3 * 5 * 2
    assert {m["runner"] for m in matrix["macos"]} == {"macos-15", "macos-15-intel"}


def test_reduced_matrix_has_one_python():
    matrix = versions.build_matrix(versions.load_versions(), full=False)
    assert {m["python"] for m in matrix["linux"] + matrix["macos"]} == {"3.12"}
    assert len(matrix["cuda"]) == 3 * 2
