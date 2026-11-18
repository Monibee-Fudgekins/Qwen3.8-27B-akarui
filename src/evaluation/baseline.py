"""Measured baseline / champion result records for Qwen3.8-27B-akarui.

Two shapes circulate in this project and this module is the single
place that tells them apart:

* An **evaluation summary** (what ``scripts/evaluate.py`` writes):
  ``model``, ``benchmark_version``, ``normalized_score``,
  ``submission_validity`` and a per-competition ``results`` list.
  No ``status`` field.
* A **canonical measured record** (what lives in
  ``evaluation/baseline_results.json`` once the base model has been
  measured, and in ``evaluation/champion_results.json`` once a
  version has been promoted): the summary fields plus ``status`` =
  ``"measured"``, ``sample_size`` (competitions actually measured),
  ``measured_at``, and the pinned base model / revision.

The shipped ``evaluation/baseline_results.json`` is an honest
not-measured template (``status: "not_measured"``, every score null).
:func:`is_measured` returns False for it, so no gate can ever pass
against it. A score is never invented here: a record counts as
measured only when a benchmark actually ran and produced a
non-null ``normalized_score`` over at least one competition.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
BENCHMARK = "akarui-held-out-benchmark-v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def measured_sample_size(evaluation: dict) -> int:
    """Competitions with a measured normalized score.

    Counts the per-competition ``results`` list when present; falls
    back to an explicit ``sample_size`` field. Zero means nothing
    was measured, however the file is labelled.
    """
    results = evaluation.get("results")
    if isinstance(results, list):
        return sum(
            1
            for entry in results
            if isinstance(entry, dict)
            and entry.get("normalized_score") is not None
        )
    sample_size = evaluation.get("sample_size")
    return int(sample_size) if isinstance(sample_size, int) else 0


def is_measured(payload: object) -> bool:
    """True only when *payload* holds a genuinely measured score.

    The not-measured template, a summary with a null score, and a
    record with a ``results`` list but zero measured competitions
    all return False.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("status") == "not_measured":
        return False
    if payload.get("normalized_score") is None:
        return False
    if isinstance(payload.get("results"), list):
        return measured_sample_size(payload) > 0
    return True


def load_measured(path: str | Path) -> dict | None:
    """Load *path* and return it only when it holds a measured score.

    Returns None when the file is missing, unparseable, or
    unmeasured (including the shipped not-measured template). Never
    raises for those cases: callers treat None as "no reference".
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if is_measured(payload) else None


def canonical_baseline_record(
    evaluation: dict, *, measured_at: str | None = None
) -> dict:
    """Build the canonical baseline record from a base evaluation.

    ``evaluation`` must be the output of ``scripts/evaluate.py
    --model base``: ``model`` is ``"base"``, ``normalized_score``
    is non-null, and at least one competition was measured.
    Raises ValueError otherwise -- a baseline score is never
    written from an unmeasured or candidate evaluation.
    """
    if evaluation.get("model") != "base":
        raise ValueError(
            "baseline record requires an evaluation of the base model "
            f"(model='base'), got model={evaluation.get('model')!r}"
        )
    if not is_measured(evaluation):
        raise ValueError(
            "baseline record requires a measured evaluation "
            "(non-null normalized_score over at least one competition); "
            "refusing to record an unmeasured baseline"
        )
    return {
        "model": "base",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "benchmark": BENCHMARK,
        "benchmark_version": evaluation.get("benchmark_version"),
        "measured_at": measured_at or _now_iso(),
        "status": "measured",
        "normalized_score": evaluation.get("normalized_score"),
        "submission_validity": evaluation.get("submission_validity"),
        "sample_size": measured_sample_size(evaluation),
        "regression_probes": evaluation.get("regression_probes", {}) or {},
        "results": evaluation.get("results", []),
        "note": (
            "Base-model benchmark score, measured by scripts/evaluate.py "
            "on the pinned base model. The first promotion must beat "
            "this score by at least "
            "configs/evaluation.yaml promotion_gates."
            "min_normalized_score_improvement; until this file is "
            "measured, no promotion can pass."
        ),
    }
