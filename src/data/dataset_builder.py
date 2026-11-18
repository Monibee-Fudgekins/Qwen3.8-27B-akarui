"""Build a versioned Akarui training dataset from trajectory records.

:func:`build_dataset` validates every record against the trajectory
schema, writes the valid records as JSONL, and writes a manifest that
records the dataset version, record counts per category, source hashes,
licenses, and the SHA-256 of the output file. Invalid records are never
written; they are reported in the returned manifest summary.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .dataset_validation import CATEGORIES, validate_record


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash(record: dict) -> str:
    """Hash identifying the source of one record (URL + competition id)."""
    material = f"{record.get('source_url', '')}\n{record.get('competition_id', '')}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_dataset(
    records: list[dict],
    out_path: str | Path,
    manifest_path: str | Path,
    dataset_version: str = "v0",
) -> dict:
    """Validate *records*, write the valid ones to *out_path* (JSONL) and
    a manifest to *manifest_path*.

    Returns the manifest dict::

        {
            "dataset_version": str,
            "created_at": ISO timestamp,
            "record_count": int,          # valid records written
            "invalid_count": int,         # records rejected
            "invalid_records": [{"index": int, "errors": [str]}],
            "category_counts": {category: int},
            "source_hashes": [sha256, ...],  # one per written record
            "licenses": [str, ...],          # sorted unique licenses
            "sha256": str,                   # SHA-256 of the output file
        }
    """
    out_path = Path(out_path)
    manifest_path = Path(manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    valid: list[dict] = []
    invalid: list[dict] = []
    category_counts: dict[str, int] = {}
    source_hashes: list[str] = []
    licenses: set[str] = set()

    for index, record in enumerate(records):
        errors = validate_record(record)
        if errors:
            invalid.append({"index": index, "errors": errors})
            continue
        valid.append(record)
        category = record.get("category")
        if isinstance(category, str) and category in CATEGORIES:
            category_counts[category] = category_counts.get(category, 0) + 1
        source_hashes.append(_source_hash(record))
        license_value = record.get("license")
        if isinstance(license_value, str) and license_value.strip():
            licenses.add(license_value.strip())

    with out_path.open("w", encoding="utf-8") as handle:
        for record in valid:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "record_count": len(valid),
        "invalid_count": len(invalid),
        "invalid_records": invalid,
        "category_counts": category_counts,
        "source_hashes": source_hashes,
        "licenses": sorted(licenses),
        "sha256": _sha256_file(out_path),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest
