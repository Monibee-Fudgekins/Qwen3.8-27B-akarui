"""Held-out competition benchmark runner.

Loads a benchmark manifest (held-out competitions with task type,
metric and baseline score), runs each task through a solver, and writes
versioned results JSON.

The runner never fabricates scores: if no solver is attached, every
competition is recorded with ``status: "not_run"`` and null scores.
A solver is any callable taking one competition dict and returning a
result dict (see :mod:`src.evaluation.competition_evaluator`).
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_manifest(manifest: dict) -> dict:
    """Normalize a manifest to the canonical internal shape.

    Accepts both ``competitions`` and ``held_out_competitions`` as the
    competition list key, ``competition_id`` and ``id`` as the identifier,
    and ``benchmark_version`` and ``version`` as the version key. The
    shipped manifest (evaluation/benchmark_manifest.json) uses the
    ``held_out_competitions`` / ``id`` / ``version`` spellings.
    """
    competitions = manifest.get("competitions")
    if competitions is None:
        competitions = manifest.get("held_out_competitions", [])
    normalized = []
    for competition in competitions:
        entry = dict(competition)
        entry["competition_id"] = entry.get("competition_id") or entry.get("id")
        normalized.append(entry)
    return {
        "benchmark_version": manifest.get("benchmark_version")
        or manifest.get("version"),
        "competitions": normalized,
    }


def load_benchmark_manifest(path: str | Path) -> dict:
    """Load a benchmark manifest JSON file and normalize it.

    Expected shape (either spelling accepted)::

        {
            "benchmark_version": "v0",
            "competitions": [
                {
                    "competition_id": "...",
                    "task_type": "tabular_classification",
                    "metric": "roc_auc",
                    "metric_direction": "higher_is_better",
                    "baseline_score": 0.80
                },
                ...
            ]
        }

    ``held_out_competitions`` / ``id`` / ``version`` are accepted as
    aliases and normalized to the shape above.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or (
        "competitions" not in manifest and "held_out_competitions" not in manifest
    ):
        raise ValueError(
            f"benchmark manifest {path} must be a JSON object with a "
            "'competitions' or 'held_out_competitions' list"
        )
    return _normalize_manifest(manifest)


def run_benchmark(
    model_ref: str,
    benchmark_manifest: str | Path | dict,
    output_path: str | Path,
    solver=None,
) -> dict:
    """Run the benchmark for *model_ref* and write versioned results.

    ``benchmark_manifest`` is a manifest path or a loaded manifest
    dict. ``solver`` is an optional callable ``solver(competition) ->
    dict``; when it is None every competition is recorded as
    ``"not_run"`` with null scores -- scores are never invented.

    Returns the results document (also written to *output_path*).
    """
    if isinstance(benchmark_manifest, dict):
        manifest = _normalize_manifest(benchmark_manifest)
    else:
        manifest = load_benchmark_manifest(benchmark_manifest)
    benchmark_version = manifest.get("benchmark_version")

    results: list[dict] = []
    for competition in manifest.get("competitions", []):
        competition_id = competition.get("competition_id")
        if solver is None:
            results.append(
                {
                    "competition_id": competition_id,
                    "task_type": competition.get("task_type"),
                    "metric": competition.get("metric"),
                    "baseline_score": competition.get("baseline_score"),
                    "status": "not_run",
                    "score": None,
                    "normalized_score": None,
                    "submission_valid": None,
                }
            )
            continue
        try:
            result = solver(competition)
            if not isinstance(result, dict):
                raise TypeError("solver must return a dict")
            entry = {
                "competition_id": competition_id,
                "task_type": competition.get("task_type"),
                "metric": competition.get("metric"),
                "baseline_score": competition.get("baseline_score"),
                "status": result.get("status", "run"),
            }
            entry.update(
                {k: v for k, v in result.items() if k not in entry}
            )
            results.append(entry)
        except Exception as exc:  # noqa: BLE001 - one bad task must not kill the run
            results.append(
                {
                    "competition_id": competition_id,
                    "task_type": competition.get("task_type"),
                    "metric": competition.get("metric"),
                    "baseline_score": competition.get("baseline_score"),
                    "status": "error",
                    "score": None,
                    "normalized_score": None,
                    "submission_valid": None,
                    "error": str(exc),
                }
            )

    document = {
        "model": model_ref,
        "benchmark_version": benchmark_version,
        "created_at": _now_iso(),
        "results": results,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return document


def run_evaluation(config_path: str | Path, model: str) -> dict:
    """Run the full evaluation described by an evaluation config.

    ``config_path`` is a YAML file like ``configs/evaluation.yaml`` with
    a ``benchmark_manifest`` entry (repo-relative path). ``model`` is
    ``"base"`` for the pinned base model or a LoRA adapter directory.

    Returns a summary dict with the aggregate ``normalized_score``
    (mean of measured per-competition normalized scores, or None when
    nothing was measured), ``submission_validity`` (fraction of measured
    competitions with a valid submission, or None), and the full
    per-competition ``results`` list. The caller (scripts/evaluate.py)
    writes this dict to its ``--output`` path. No solver is attached
    here, so competitions are recorded ``"not_run"`` with null scores
    unless a solver is wired in by the caller.
    """
    import yaml

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    manifest_ref = config.get("benchmark_manifest", "evaluation/benchmark_manifest.json")
    manifest_path = Path(manifest_ref)
    if not manifest_path.exists():
        # Fall back to resolving relative to the repository root
        # (the config file lives in configs/).
        candidate = config_path.resolve().parent.parent / manifest_ref
        if candidate.exists():
            manifest_path = candidate
    if not manifest_path.exists():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_ref}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        document = run_benchmark(
            model_ref=model,
            benchmark_manifest=manifest_path,
            output_path=Path(tmp_dir) / "benchmark_results.json",
        )
    results = document.get("results", [])
    scores = [
        entry["normalized_score"]
        for entry in results
        if entry.get("normalized_score") is not None
    ]
    validity = [
        entry["submission_valid"]
        for entry in results
        if entry.get("submission_valid") is not None
    ]
    return {
        "model": model,
        "benchmark_version": document.get("benchmark_version"),
        "created_at": document.get("created_at"),
        "normalized_score": (sum(scores) / len(scores)) if scores else None,
        "submission_validity": (
            sum(1 for value in validity if value) / len(validity)
            if validity
            else None
        ),
        "results": results,
    }
