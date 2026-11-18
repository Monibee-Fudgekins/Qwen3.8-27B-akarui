"""Provenance manifests for Qwen3.8-27B-akarui releases.

Every published version must be traceable to the exact base model
revision, dataset version, training configuration, code commit, GPU
configuration, evaluation results, and artifact hashes that produced
it. No provenance field is ever filled with a placeholder in a
published manifest: :func:`validate_provenance` treats any required
field that is ``None`` or empty as an error, and the publisher refuses
to upload an incomplete manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# The provenance contract. These are the only fields a manifest may
# carry; adding a field requires updating the schema and the docs.
PROVENANCE_FIELDS = (
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
)

# Fields that must hold a non-empty value. ``gpu_configuration`` may
# legitimately be an empty list only when training ran on CPU-only
# infrastructure, which is not a supported configuration for this
# project, so it is required to be a non-empty list as well.
_REQUIRED_NON_EMPTY = (
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
    "artifact_hashes",
    "release_version",
)


def compute_file_hash(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes.

    Parameters
    ----------
    path:
        Path to the file to hash.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_config_hash(config_path: str | Path) -> str:
    """Return the SHA-256 hex digest of a training config file.

    This is a convenience wrapper around :func:`compute_file_hash` so
    call sites read as intent ("hash the config") rather than
    mechanics. The config file is hashed byte-for-byte, so a config
    regenerated with identical content hashes identically.
    """
    return compute_file_hash(config_path)


def build_provenance(
    *,
    model: str,
    base_model: str,
    base_revision: str,
    training_method: str,
    training_date: str,
    training_dataset_version: str,
    training_dataset_hash: str,
    training_config_hash: str,
    source_repository: str,
    source_commit: str,
    training_environment: str,
    gpu_configuration: list[dict],
    evaluation_results: dict,
    artifact_hashes: dict[str, str],
    release_version: str,
) -> dict:
    """Build a provenance manifest with exactly the contract fields.

    All arguments are keyword-only so a caller cannot silently swap
    two similar-looking hashes. Values are stored as given; this
    function does not invent or default any value. Call
    :func:`validate_provenance` before publishing.
    """
    return {
        "model": model,
        "base_model": base_model,
        "base_revision": base_revision,
        "training_method": training_method,
        "training_date": training_date,
        "training_dataset_version": training_dataset_version,
        "training_dataset_hash": training_dataset_hash,
        "training_config_hash": training_config_hash,
        "source_repository": source_repository,
        "source_commit": source_commit,
        "training_environment": training_environment,
        "gpu_configuration": gpu_configuration,
        "evaluation_results": evaluation_results,
        "artifact_hashes": artifact_hashes,
        "release_version": release_version,
    }


def validate_provenance(manifest: dict) -> list[str]:
    """Validate a provenance manifest.

    Returns a list of error strings; an empty list means the manifest
    is complete. A required field that is ``None``, an empty string,
    an empty list, or an empty dict is an error. Unknown extra fields
    are also errors, so a manifest can never silently carry a field
    the schema does not define.
    """
    errors: list[str] = []

    if not isinstance(manifest, dict):
        return ["provenance manifest must be a JSON object, not "
                f"{type(manifest).__name__}"]

    for field in _REQUIRED_NON_EMPTY:
        value = manifest.get(field)
        if value is None or value == "" or value == [] or value == {}:
            errors.append(f"provenance field '{field}' is missing or empty")

    if not isinstance(manifest.get("evaluation_results"), dict):
        errors.append("provenance field 'evaluation_results' must be an object")
    if not isinstance(manifest.get("gpu_configuration"), list):
        errors.append("provenance field 'gpu_configuration' must be a list")
    if not isinstance(manifest.get("artifact_hashes"), dict):
        errors.append("provenance field 'artifact_hashes' must be an object")

    for field in manifest:
        if field not in PROVENANCE_FIELDS:
            errors.append(f"provenance field '{field}' is not part of the contract")

    return errors


def provenance_to_json(manifest: dict, path: str | Path) -> Path:
    """Write a validated provenance manifest to ``path`` as JSON.

    Raises
    ------
    ValueError
        If :func:`validate_provenance` reports any error.
    """
    errors = validate_provenance(manifest)
    if errors:
        raise ValueError("incomplete provenance manifest: " + "; ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
