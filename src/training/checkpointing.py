"""Checkpoint management and training-manifest writing.

Checkpoints live under the run's output directory as
``checkpoint-<step>/``. Manifests are machine-readable provenance:
config hash, dataset hash, seed, GPU info, and artifact hashes, so
every adapter can be traced back to the code and data that made it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_checkpoint(model, tokenizer, output_dir: str | Path, step: int) -> Path:
    """Save *model* and *tokenizer* to ``<output_dir>/checkpoint-<step>``.

    Returns the checkpoint path. The caller is responsible for the
    training loop; this only persists the artifacts.
    """
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / f"checkpoint-{int(step)}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the highest-numbered ``checkpoint-<step>`` dir, or None."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for entry in output_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("checkpoint-"):
            continue
        suffix = entry.name[len("checkpoint-"):]
        if not suffix.isdigit():
            continue
        step = int(suffix)
        if best is None or step > best[0]:
            best = (step, entry)
    return best[1] if best else None


def prune_checkpoints(output_dir: str | Path, keep: int = 3) -> list[Path]:
    """Delete all but the *keep* newest checkpoints.

    Returns the list of removed checkpoint paths. Never deletes the
    newest checkpoint, and does nothing when ``keep`` covers all of them.
    """
    output_dir = Path(output_dir)
    checkpoints: list[tuple[int, Path]] = []
    if output_dir.is_dir():
        for entry in output_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("checkpoint-"):
                suffix = entry.name[len("checkpoint-"):]
                if suffix.isdigit():
                    checkpoints.append((int(suffix), entry))
    checkpoints.sort(key=lambda item: item[0], reverse=True)
    removed: list[Path] = []
    for _, path in checkpoints[keep:]:
        shutil.rmtree(path)
        removed.append(path)
    return removed


def write_training_manifest(
    output_dir: str | Path,
    *,
    config: dict | None = None,
    config_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    seed: int | None = None,
    gpu_info: dict | None = None,
    extra: dict | None = None,
) -> tuple[Path, dict]:
    """Write ``training_manifest.json`` into *output_dir*.

    ``config`` is hashed canonically (or from ``config_path`` when the
    file is available); ``dataset_path`` is hashed from the file when
    given. No placeholders are written: fields with no known value are
    ``None``. Returns ``(manifest_path, manifest)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extra = dict(extra or {})

    config_hash = None
    if config_path is not None and Path(config_path).is_file():
        config_hash = _sha256_file(Path(config_path))
    elif config is not None:
        config_hash = _sha256_json(config)

    dataset_hash = None
    if dataset_path is not None and Path(dataset_path).is_file():
        dataset_hash = _sha256_file(Path(dataset_path))

    manifest: dict = {
        "model": "Qwen3.8-27B-akarui",
        "base_model": (config or {}).get("base_model"),
        "base_revision": (config or {}).get("base_revision"),
        "training_method": extra.pop("training_method", None),
        "training_date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_dataset_version": (config or {}).get("dataset", {}).get("version"),
        "training_dataset_hash": dataset_hash,
        "training_config_hash": config_hash,
        "source_repository": "https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui",
        "source_commit": extra.pop("source_commit", None),
        "training_environment": "Kaggle",
        "gpu_configuration": gpu_info,
        "seed": seed,
        "evaluation_results": None,
        "artifact_hashes": {},
        "release_version": None,
    }
    if extra:
        manifest.update(extra)
    manifest_path = output_dir / "training_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest_path, manifest
