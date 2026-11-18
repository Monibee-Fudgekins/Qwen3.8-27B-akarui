"""Validation for Akarui trajectory records.

A trajectory record is one competition-solving run stored as a JSONL line.
Every field in :data:`REQUIRED_FIELDS` is required unless noted. Records
that fail :func:`validate_record` must never enter a training dataset.

Forbidden-content checks here are keyword / regex *heuristics* for
obviously disallowed material (hidden test labels, leaked solutions,
private data). They are a safety net, not a guarantee: provenance and
licensing must still be checked at collection time
(see :mod:`src.data.competition_collector`).
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

#: Fields every trajectory record must carry.
REQUIRED_FIELDS = (
    "competition_id",
    "source_url",
    "dataset_provenance",
    "license",
    "task_description",
    "input_schema",
    "target_definition",
    "evaluation_metric",
    "training_validation_strategy",
    "reasoning_summary",
    "generated_code",
    "execution_environment",
    "execution_output",
    "errors_and_debugging",
    "validation_score",
    "final_outcome",
    "dataset_version",
    "collected_at",
)

#: Allowed values for the ``category`` field.
CATEGORIES = (
    "competition_analysis",
    "programming_debugging",
    "feature_engineering_modeling",
    "validation_statistics",
    "experiment_optimization",
    "ensembling_postprocessing",
    "submission_workflows",
)

#: Initial sampling targets (guidance only; never forced at build time).
TARGET_PROPORTIONS = {
    "competition_analysis": 0.20,
    "programming_debugging": 0.20,
    "feature_engineering_modeling": 0.20,
    "validation_statistics": 0.15,
    "experiment_optimization": 0.10,
    "ensembling_postprocessing": 0.10,
    "submission_workflows": 0.05,
}

#: Heuristic patterns for disallowed content. Each entry is
#: (name, compiled regex). Matching text is rejected by validate_record.
#: These are heuristics: a match means "needs human review / reject",
#: never a legal judgement about the underlying material.
FORBIDDEN_PATTERNS = (
    ("hidden_test_labels", re.compile(r"hidden\s+test\s+label", re.IGNORECASE)),
    ("private_test_data", re.compile(r"private\s+test\s+(data|set|labels?)", re.IGNORECASE)),
    ("leaked_solution", re.compile(r"leak(ed|age)\s+(solution|test|answer)", re.IGNORECASE)),
    ("test_labels_file", re.compile(r"test\.csv[^ ]*label", re.IGNORECASE)),
    ("ground_truth_test", re.compile(r"ground[\s_-]?truth\s+(test|labels?)", re.IGNORECASE)),
    ("private_leaderboard", re.compile(r"private\s+leaderboard", re.IGNORECASE)),
    ("do_not_share", re.compile(r"do\s+not\s+(share|distribute|redistribute)", re.IGNORECASE)),
    ("kaggle_private_notebook", re.compile(r"private\s+(kaggle\s+)?notebook", re.IGNORECASE)),
    ("hidden_evaluation", re.compile(r"hidden\s+evaluation\s+(label|set|data)", re.IGNORECASE)),
)

#: Text fields scanned for forbidden patterns.
_TEXT_FIELDS = (
    "task_description",
    "reasoning_summary",
    "generated_code",
    "execution_output",
    "final_outcome",
    "dataset_provenance",
)


def _is_iso_timestamp(value: object) -> bool:
    """Return True when *value* is a parseable ISO-8601 timestamp string."""
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def _check_forbidden(record: dict) -> list[str]:
    """Scan text fields for forbidden-content heuristics."""
    errors: list[str] = []
    for field in _TEXT_FIELDS:
        value = record.get(field)
        if not isinstance(value, str):
            continue
        for name, pattern in FORBIDDEN_PATTERNS:
            match = pattern.search(value)
            if match:
                errors.append(
                    f"field '{field}' matches forbidden-content heuristic "
                    f"'{name}': {match.group(0)!r}"
                )
    return errors


def validate_record(record: object) -> list[str]:
    """Validate one trajectory record.

    Returns a list of error strings; an empty list means the record is
    valid. Never raises for malformed input -- bad records are reported,
    not crashed on.
    """
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record is not a JSON object"]

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"missing required field: {field}")

    category = record.get("category")
    if category is None:
        errors.append("missing required field: category")
    elif category not in CATEGORIES:
        errors.append(
            f"invalid category {category!r}; must be one of {sorted(CATEGORIES)}"
        )

    for field in (
        "competition_id",
        "source_url",
        "dataset_provenance",
        "license",
        "task_description",
        "target_definition",
        "evaluation_metric",
        "training_validation_strategy",
        "reasoning_summary",
        "generated_code",
        "final_outcome",
        "dataset_version",
    ):
        value = record.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"field '{field}' must be a non-empty string")

    for field in ("input_schema", "execution_environment"):
        value = record.get(field)
        if value is not None and not isinstance(value, (str, dict)):
            errors.append(f"field '{field}' must be a string or an object")

    errors_debugging = record.get("errors_and_debugging")
    if errors_debugging is not None:
        if not isinstance(errors_debugging, list):
            errors.append("field 'errors_and_debugging' must be a list (may be empty)")
        elif not all(isinstance(item, (str, dict)) for item in errors_debugging):
            errors.append("field 'errors_and_debugging' items must be strings or objects")

    score = record.get("validation_score")
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            errors.append("field 'validation_score' must be a number or null")
        elif not math.isfinite(score):
            errors.append("field 'validation_score' must be finite")

    collected_at = record.get("collected_at")
    if collected_at is not None and not _is_iso_timestamp(collected_at):
        errors.append("field 'collected_at' must be an ISO-8601 timestamp")

    errors.extend(_check_forbidden(record))
    return errors


def validate_file(path: str | Path) -> dict:
    """Validate every JSONL record in *path*.

    Returns a summary dict::

        {
            "path": str,
            "total_records": int,
            "valid_records": int,
            "invalid_records": int,
            "errors": [{"line": int, "errors": [str, ...]}, ...],
            "category_counts": {category: int, ...},
        }

    Lines that are not valid JSON are reported as errors, not skipped.
    """
    path = Path(path)
    summary: dict = {
        "path": str(path),
        "total_records": 0,
        "valid_records": 0,
        "invalid_records": 0,
        "errors": [],
        "category_counts": {},
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            summary["total_records"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                summary["invalid_records"] += 1
                summary["errors"].append(
                    {"line": line_number, "errors": [f"invalid JSON: {exc}"]}
                )
                continue
            record_errors = validate_record(record)
            category = record.get("category") if isinstance(record, dict) else None
            if isinstance(category, str):
                summary["category_counts"][category] = (
                    summary["category_counts"].get(category, 0) + 1
                )
            if record_errors:
                summary["invalid_records"] += 1
                summary["errors"].append({"line": line_number, "errors": record_errors})
            else:
                summary["valid_records"] += 1
    return summary
