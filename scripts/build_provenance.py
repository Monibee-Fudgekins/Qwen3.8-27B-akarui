"""Build a release provenance manifest from a training run.

Usage:
    python scripts/build_provenance.py \\
        --adapter-dir output/akarui-sft-v0 \\
        --evaluation output/evaluation-results.json \\
        --release-version 1 --output output/provenance.json

The training kernel already writes ``training_manifest.json`` into
the adapter directory (base model + pinned revision, dataset version
and hash, config hash, training date/method, GPU info). This script
completes it -- it does not invent any of those fields:

* ``source_commit`` from ``--source-commit`` (defaults to the
  ``GITHUB_SHA`` environment variable, which the weekly workflow
  sets); refused when unknown, never defaulted to a placeholder.
* ``evaluation_results`` from the candidate evaluation JSON.
* ``artifact_hashes`` from SHA-256 hashes of the adapter files.
* ``release_version`` from ``--release-version``.

GPU configuration: the training manifest stores the device dict
from ``get_device_info()``; the provenance contract requires a
list, so the per-GPU entries (``gpus``) are used. A run with no GPU
recorded fails here -- provenance is incomplete and the release
cannot be published.

No credentials are read, written, or printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.publishing.provenance import (  # noqa: E402
    build_provenance,
    compute_file_hash,
    provenance_to_json,
    validate_provenance,
)

SOURCE_REPOSITORY = "https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui"


def _gpu_configuration(training_manifest: dict) -> list[dict]:
    """Normalize the training manifest's GPU info to the contract list.

    ``get_device_info()`` returns a dict with a ``gpus`` list; the
    provenance contract requires a non-empty list of per-GPU dicts.
    Raises ValueError when no GPU was recorded (CPU-only run).
    """
    gpu_info = training_manifest.get("gpu_configuration")
    if isinstance(gpu_info, list) and gpu_info:
        return gpu_info
    if isinstance(gpu_info, dict):
        gpus = gpu_info.get("gpus")
        if isinstance(gpus, list) and gpus:
            return gpus
    raise ValueError(
        "training manifest records no GPU configuration; a release "
        "requires a measured GPU configuration (see "
        "src/training/trainer_utils.get_device_info)"
    )


def build_provenance_manifest(
    *,
    adapter_dir: Path,
    evaluation: dict,
    source_commit: str,
    release_version: str,
) -> dict:
    """Build the provenance manifest. Raises ValueError when incomplete."""
    manifest_path = adapter_dir / "training_manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            f"training manifest not found in adapter dir: {manifest_path}. "
            "The training kernel writes it (src/training/sft.py)."
        )
    training = json.loads(manifest_path.read_text())

    artifact_hashes: dict[str, str] = {}
    for item in sorted(adapter_dir.iterdir()):
        if item.is_file() and item.name != "training_manifest.json":
            artifact_hashes[item.name] = compute_file_hash(item)
    if not artifact_hashes:
        raise ValueError(f"adapter directory is empty: {adapter_dir}")

    manifest = build_provenance(
        model="Qwen3.8-27B-akarui",
        base_model=training.get("base_model"),
        base_revision=training.get("base_revision"),
        training_method=training.get("training_method"),
        training_date=training.get("training_date"),
        training_dataset_version=training.get("training_dataset_version"),
        training_dataset_hash=training.get("training_dataset_hash"),
        training_config_hash=training.get("training_config_hash"),
        source_repository=SOURCE_REPOSITORY,
        source_commit=source_commit,
        training_environment=training.get("training_environment") or "Kaggle",
        gpu_configuration=_gpu_configuration(training),
        evaluation_results=evaluation,
        artifact_hashes=artifact_hashes,
        release_version=release_version,
    )
    errors = validate_provenance(manifest)
    if errors:
        raise ValueError(
            "provenance manifest incomplete: " + "; ".join(errors)
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, else 1."""
    parser = argparse.ArgumentParser(
        description="Build a release provenance manifest from a training run."
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Adapter directory containing training_manifest.json",
    )
    parser.add_argument(
        "--evaluation",
        required=True,
        help="Candidate evaluation results JSON",
    )
    parser.add_argument(
        "--source-commit",
        default=os.environ.get("GITHUB_SHA"),
        help="Source commit SHA (default: $GITHUB_SHA)",
    )
    parser.add_argument(
        "--release-version", required=True, help="Release version, e.g. 1"
    )
    parser.add_argument(
        "--output",
        default="output/provenance.json",
        help="Where to write the provenance manifest",
    )
    args = parser.parse_args(argv)

    if not args.source_commit:
        print(
            "error: source commit unknown: pass --source-commit or set "
            "$GITHUB_SHA",
            file=sys.stderr,
        )
        return 1
    evaluation_path = Path(args.evaluation)
    if not evaluation_path.exists():
        print(f"error: evaluation file not found: {evaluation_path}", file=sys.stderr)
        return 1
    try:
        evaluation = json.loads(evaluation_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"error: cannot parse evaluation JSON: {exc}", file=sys.stderr)
        return 1

    try:
        manifest = build_provenance_manifest(
            adapter_dir=Path(args.adapter_dir),
            evaluation=evaluation,
            source_commit=args.source_commit,
            release_version=args.release_version,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    provenance_to_json(manifest, Path(args.output))
    print(f"Provenance manifest written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
