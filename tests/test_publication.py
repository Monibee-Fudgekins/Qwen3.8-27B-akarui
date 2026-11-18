"""Tests for publication: Kaggle metadata, upload gates, releases, cards."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.publishing.kaggle_publisher import (
    build_model_metadata,
    build_variation_metadata,
    upload_variation,
)
from src.publishing.model_card import render_model_card
from src.publishing.release_manifest import (
    list_releases,
    load_release,
    write_release,
)
from tests.test_provenance import complete_provenance

ALLOWED_METADATA_KEYS = {
    "title",
    "subtitle",
    "description",
    "licenses",
    "tags",
    "instances",
}


def test_model_metadata_only_supported_fields():
    metadata = build_model_metadata(
        "Qwen3.8-27B-akarui",
        "qlora-sft",
        "Competition-solving fine-tune of Qwen3.8-27B.",
        ["qwen", "pytorch", "qlora"],
    )
    assert set(metadata) <= ALLOWED_METADATA_KEYS
    assert "Qwen3.8-27B-akarui" in metadata["title"]
    assert metadata["tags"] == ["qwen", "pytorch", "qlora"]
    instances = metadata["instances"]
    assert instances
    assert instances[0]["framework"] == "pytorch"


def test_variation_metadata_rejects_unknown():
    with pytest.raises(ValueError, match="unknown variation"):
        build_variation_metadata("not-a-variation")
    variation = build_variation_metadata("qlora-sft")
    assert variation["variationType"] == "qlora-sft"
    assert variation["fineTunable"] is True


def test_upload_refuses_incomplete_provenance(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="incomplete provenance"):
        upload_variation("qlora-sft", model_dir, {"model": "Qwen3.8-27B-akarui"})


def test_release_roundtrip_and_append_only(tmp_path: Path):
    provenance = complete_provenance()
    path = write_release(
        "0.1.0",
        provenance,
        {"normalized_score": None},
        "Initial infrastructure; no model released.",
        releases_dir=tmp_path,
    )
    assert path.exists()
    loaded = load_release("0.1.0", releases_dir=tmp_path)
    assert loaded["version"] == "0.1.0"
    assert loaded["provenance"]["base_revision"] == (
        "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
    )
    assert list_releases(releases_dir=tmp_path) == ["0.1.0"]
    with pytest.raises(FileExistsError):
        write_release(
            "0.1.0",
            provenance,
            {},
            "duplicate",
            releases_dir=tmp_path,
        )


def test_release_rejects_incomplete_provenance(tmp_path: Path):
    with pytest.raises(ValueError, match="cannot write release"):
        write_release(
            "0.2.0",
            {"model": "Qwen3.8-27B-akarui"},
            {},
            "bad provenance",
            releases_dir=tmp_path,
        )


def test_model_card_marks_unmeasured():
    card = render_model_card({"model_name": "Qwen3.8-27B-akarui"})
    assert "Qwen3.8-27B-akarui" in card
    assert "Not yet measured" in card
