"""Tests for provenance manifests: build, validate, hash."""

from __future__ import annotations

from pathlib import Path

from src.publishing.provenance import (
    build_provenance,
    compute_config_hash,
    compute_file_hash,
    validate_provenance,
)


def complete_provenance(**overrides):
    kwargs = {
        "model": "Qwen3.8-27B-akarui",
        "base_model": "Qwen/Qwen3.8-27B",
        "base_revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
        "training_method": "QLoRA",
        "training_date": "2026-11-18",
        "training_dataset_version": "v0",
        "training_dataset_hash": "a" * 64,
        "training_config_hash": "b" * 64,
        "source_repository": "https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui",
        "source_commit": "c" * 40,
        "training_environment": "Kaggle",
        "gpu_configuration": [{"model": "L4", "count": 1, "vram_gb": 24}],
        "evaluation_results": {"normalized_score": 0.55},
        "artifact_hashes": {"adapter.safetensors": "d" * 64},
        "release_version": "1",
    }
    kwargs.update(overrides)
    return build_provenance(**kwargs)


def test_complete_provenance_validates():
    manifest = complete_provenance()
    assert validate_provenance(manifest) == []
    assert set(manifest) == {
        "model",
        "base_model",
        "base_revision",
        "training_method",
        "training_date",
        "training_dataset_version",
        "training_dataset_hash",
        "training_config_hash",
        "source_repository",
        "source_commit",
        "training_environment",
        "gpu_configuration",
        "evaluation_results",
        "artifact_hashes",
        "release_version",
    }


def test_incomplete_provenance_rejected():
    manifest = complete_provenance(training_dataset_hash=None)
    errors = validate_provenance(manifest)
    assert any("training_dataset_hash" in e for e in errors)

    manifest = complete_provenance(release_version="")
    assert validate_provenance(manifest)

    assert validate_provenance({})  # empty manifest is invalid
    assert validate_provenance(None)  # type: ignore[arg-type]


def test_file_hash_is_stable(tmp_path: Path):
    path = tmp_path / "file.txt"
    path.write_text("hello akarui\n")
    assert compute_file_hash(path) == compute_file_hash(path)
    other = tmp_path / "other.txt"
    other.write_text("different\n")
    assert compute_file_hash(path) != compute_file_hash(other)


def test_config_hash_matches_file_hash(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("base_model: Qwen/Qwen3.8-27B\n")
    assert compute_config_hash(path) == compute_file_hash(path)
