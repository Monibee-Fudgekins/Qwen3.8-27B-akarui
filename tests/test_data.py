"""Tests for the dataset pipeline: trajectories, validation, dedup, leakage.

These tests run without torch, transformers, or network access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.data import dataset_builder, dataset_validation, deduplication
from src.data.competition_collector import (
    discover_kaggle_competitions,
    license_allows_training,
)
from src.data.trajectory_generator import (
    generate_trajectory,
    make_execution_environment,
)
from src.evaluation.leakage_detection import (
    detect_leakage,
    scan_record_for_leakage,
)


def valid_trajectory_kwargs(**overrides):
    """Keyword arguments for one schema-valid trajectory record."""
    kwargs = {
        "competition_id": "titanic",
        "source_url": "https://www.kaggle.com/competitions/titanic",
        "dataset_provenance": "Kaggle public competition data, train.csv",
        "license": "apache-2.0",
        "task_description": "Predict survival from passenger features.",
        "input_schema": {"features": ["Pclass", "Sex", "Age"], "rows": 891},
        "target_definition": "Binary target Survived (0/1).",
        "evaluation_metric": "Accuracy",
        "training_validation_strategy": "Stratified 5-fold cross-validation.",
        "reasoning_summary": "Gradient boosting with cross-validated features.",
        "generated_code": "import pandas as pd\nprint('train')\n",
        "execution_environment": make_execution_environment(python_version="3.12"),
        "execution_output": "accuracy: 0.81",
        "errors_and_debugging": [],
        "validation_score": 0.81,
        "final_outcome": "Valid submission produced.",
        "category": "feature_engineering_modeling",
        "dataset_version": "v0",
    }
    kwargs.update(overrides)
    return kwargs


def test_valid_trajectory_passes_validation():
    record = generate_trajectory(**valid_trajectory_kwargs())
    assert dataset_validation.validate_record(record) == []


def test_missing_field_fails_validation():
    record = generate_trajectory(**valid_trajectory_kwargs())
    del record["evaluation_metric"]
    errors = dataset_validation.validate_record(record)
    assert any("evaluation_metric" in e for e in errors)


def test_bad_category_fails_validation():
    with pytest.raises(ValueError, match="category"):
        generate_trajectory(**valid_trajectory_kwargs(category="not_a_category"))


def test_forbidden_content_rejected():
    record = generate_trajectory(**valid_trajectory_kwargs())
    record["reasoning_summary"] = "Used the hidden test labels to pick features."
    errors = dataset_validation.validate_record(record)
    assert errors, "hidden-test-label content must fail validation"
    assert any("hidden_test_labels" in e for e in errors)


def test_dataset_build_roundtrip(tmp_path: Path):
    records = [
        generate_trajectory(**valid_trajectory_kwargs()),
        generate_trajectory(
            **valid_trajectory_kwargs(
                competition_id="house-prices-advanced-regression-techniques",
                task_description="Predict house sale prices from 79 features.",
                category="competition_analysis",
            )
        ),
        {"competition_id": "broken"},  # invalid: missing required fields
    ]
    out_path = tmp_path / "train.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = dataset_builder.build_dataset(records, out_path, manifest_path)

    assert manifest["record_count"] == 2
    assert manifest["invalid_count"] == 1
    assert len(manifest["invalid_records"]) == 1
    assert manifest["category_counts"]["feature_engineering_modeling"] == 1
    assert manifest["category_counts"]["competition_analysis"] == 1
    assert manifest["licenses"] == ["apache-2.0"]
    # The manifest hash must match the file actually written.
    assert manifest["sha256"] == hashlib.sha256(out_path.read_bytes()).hexdigest()
    assert len(manifest["source_hashes"]) == 2

    # validate_file agrees with the build.
    summary = dataset_validation.validate_file(out_path)
    assert summary["total_records"] == 2
    assert summary["invalid_records"] == 0
    assert summary["valid_records"] == 2


def test_dedup_by_task_and_code():
    first = generate_trajectory(**valid_trajectory_kwargs())
    duplicate_task = generate_trajectory(
        **valid_trajectory_kwargs(validation_score=0.82)
    )
    duplicate_code = generate_trajectory(
        **valid_trajectory_kwargs(
            competition_id="digit-recognizer",
            task_description="Classify handwritten digits.",
        )
    )
    kept, dropped = deduplication.dedup_records(
        [first, duplicate_task, duplicate_code]
    )
    assert len(kept) == 1
    assert len(dropped) == 2
    assert all("reason" in entry for entry in dropped)


def test_leakage_overlap_detected():
    report = detect_leakage(
        ["titanic", "digit-recognizer"], ["digit-recognizer", "spaceship-titanic"]
    )
    assert report["overlap_ids"] == ["digit-recognizer"]
    assert report["clean"] is False
    assert detect_leakage(["titanic"], ["titanic"])["overlap_ids"] == ["titanic"]
    assert detect_leakage(["titanic"], ["digit-recognizer"])["clean"] is True


def test_leakage_scan_flags_hidden_labels():
    findings = scan_record_for_leakage(
        {"reasoning_summary": "Copied answers from the private test set."}
    )
    assert findings
    assert scan_record_for_leakage({"generated_code": "print('ok')"}) == []


def test_license_filter():
    assert license_allows_training("apache-2.0")
    assert license_allows_training("MIT")
    assert license_allows_training("CC-BY-4.0")
    assert not license_allows_training("unknown")
    assert not license_allows_training("")
    assert not license_allows_training(None)


def test_kaggle_discovery_degrades_without_client():
    # No kaggle client is installed in this environment; discovery must
    # return an empty list, not crash or touch the network.
    assert discover_kaggle_competitions() == []
