"""Held-out competition benchmark runner.

Loads a benchmark manifest (held-out competitions with task type,
metric and baseline score), runs each task through a solver, and writes
versioned results JSON.

The runner never fabricates scores: if no solver is attached, every
competition is recorded with ``status: "not_run"`` and null scores.
A solver is any callable taking one competition dict and returning a
result dict (see :mod:`src.evaluation.competition_evaluator`).
The real solver is :mod:`src.evaluation.solver`, wired in by
:func:`run_evaluation` (see its docstring).
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
            # The solver computes the reference baseline in-kernel
            # (the manifest carries none); let it fill the slot when
            # the manifest left it empty.
            if entry.get("baseline_score") is None and result.get(
                "baseline_score"
            ) is not None:
                entry["baseline_score"] = result["baseline_score"]
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


def run_evaluation(
    config_path: str | Path,
    model: str,
    *,
    solver=None,
    solver_factory=None,
    generate=None,
    data_root: str | Path | None = None,
    train_file: str | Path | None = None,
) -> dict:
    """Run the full evaluation described by an evaluation config.

    ``config_path`` is a YAML file like ``configs/evaluation.yaml`` with
    a ``benchmark_manifest`` entry (repo-relative path). ``model`` is
    ``"base"`` for the pinned base model or a LoRA adapter directory.

    The solver is resolved in this order: an injected ``solver``
    callable (tests), a ``solver_factory`` callable returning one,
    an injected ``generate`` callable wrapped in a
    :class:`src.evaluation.solver.CompetitionSolver`, and finally
    the real solver built from the loaded model (GPU, in-kernel
    only) via :func:`src.evaluation.solver.build_solver`. The
    leakage guard runs before the benchmark whenever a training
    file is available, and the regression probes
    (``evaluation/regression_probes.json``) run with the same
    generation callable; their per-category pass fractions are
    returned as ``regression_probes`` and the full summary as
    ``regression``.

    Returns a summary dict with the aggregate ``normalized_score``
    (mean of measured per-competition normalized scores, or None when
    nothing was measured), ``submission_validity`` (fraction of measured
    competitions with a valid submission, or None), and the full
    per-competition ``results`` list. The caller (scripts/evaluate.py)
    writes this dict to its ``--output`` path.
    """
    import yaml

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    repo_root = config_path.resolve().parent.parent

    def _resolve(path_value: str | Path, default: str) -> Path:
        candidate = Path(path_value if path_value else default)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if (repo_root / candidate).exists():
            return repo_root / candidate
        if candidate.exists():
            return candidate
        return repo_root / default

    manifest_ref = config.get("benchmark_manifest", "evaluation/benchmark_manifest.json")
    manifest_path = _resolve(manifest_ref, "evaluation/benchmark_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_ref}")
    manifest = load_benchmark_manifest(manifest_path)
    held_out_ids = [
        str(entry.get("competition_id"))
        for entry in manifest.get("competitions", [])
    ]
    if not held_out_ids:
        held_out_ids = [
            str(entry) for entry in config.get("held_out_competition_ids", [])
        ]

    solver_config = config.get("solver", {}) or {}
    if train_file is None:
        train_file = _resolve(
            solver_config.get("train_file", "data/train.jsonl"),
            "data/train.jsonl",
        )
    if data_root is None:
        data_root = solver_config.get("data_root", "evaluation-data")

    # Leakage guard first: a contaminated training file refuses
    # before any model is loaded or any competition attempted.
    from src.evaluation.solver import (  # noqa: PLC0415 - lazy: keeps import light
        CompetitionSolver,
        build_solver,
        check_training_leakage,
    )

    check_training_leakage(train_file, held_out_ids)

    resolved_solver = solver
    resolved_generate = generate
    if resolved_solver is None and solver_factory is not None:
        resolved_solver = solver_factory()
    if resolved_solver is None and resolved_generate is not None:
        resolved_solver = CompetitionSolver(
            resolved_generate,
            data_root=data_root,
            seed=int(solver_config.get("seed", 42)),
            val_fraction=float(solver_config.get("val_fraction", 0.2)),
            execution_timeout_s=int(
                solver_config.get("execution_timeout_s", 600)
            ),
            execution_memory_mb=int(
                solver_config.get("execution_memory_mb", 4096)
            ),
        )
    if resolved_solver is None:
        # Real solver: loads the pinned base model (+ adapter).
        # Heavy and GPU-only; only reached in-kernel / on demand.
        base_config_path = repo_root / "configs" / "base.yaml"
        with base_config_path.open("r", encoding="utf-8") as handle:
            model_config = yaml.safe_load(handle) or {}
        resolved_solver = build_solver(
            model,
            config=model_config,
            data_root=data_root,
            train_file=train_file,
            held_out_ids=held_out_ids,
            seed=int(solver_config.get("seed", 42)),
            val_fraction=float(solver_config.get("val_fraction", 0.2)),
            max_new_tokens=int(solver_config.get("max_new_tokens", 4096)),
            execution_timeout_s=int(
                solver_config.get("execution_timeout_s", 600)
            ),
            execution_memory_mb=int(
                solver_config.get("execution_memory_mb", 4096)
            ),
        )
        resolved_generate = resolved_solver.generate

    with tempfile.TemporaryDirectory() as tmp_dir:
        document = run_benchmark(
            model_ref=model,
            benchmark_manifest=manifest_path,
            output_path=Path(tmp_dir) / "benchmark_results.json",
            solver=resolved_solver,
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

    # Regression probes with the same generation callable.
    regression_summary: dict = {
        "total": 0, "run": 0, "passed": 0, "failed": 0, "not_run": 0,
        "regressions": [], "results": [],
    }
    regression_probes: dict[str, float] = {}
    probe_path = _resolve(
        solver_config.get(
            "regression_probes_file", "evaluation/regression_probes.json"
        ),
        "evaluation/regression_probes.json",
    )
    baseline_path = _resolve(
        "evaluation/baseline_results.json", "evaluation/baseline_results.json"
    )
    if probe_path.exists():
        from src.evaluation.regression_tests import (  # noqa: PLC0415
            run_regression,
        )

        regression_summary = run_regression(
            resolved_generate,
            probe_path,
            baseline_path if baseline_path.is_file() else None,
        )
        by_category: dict[str, list[float]] = {}
        for entry in regression_summary.get("results", []):
            if entry.get("score") is None or entry.get("category") is None:
                continue
            by_category.setdefault(str(entry["category"]), []).append(
                float(entry["score"])
            )
        regression_probes = {
            category: sum(values) / len(values)
            for category, values in by_category.items()
        }

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
        "regression_probes": regression_probes,
        "regression": regression_summary,
        "results": results,
    }
