"""Evaluate one held-out competition against its baseline.

:func:`evaluate_competition` takes a ``model`` and a competition dict
and returns a structured result. The model may be:

* ``None`` -- nothing is measured; every score field is null and the
  status is ``"not_run"``.
* a callable ``model(competition) -> dict`` -- the callable runs the
  competition (or wraps the real model) and returns at least ``score``
  and optionally ``submission_valid`` and ``fold_scores``.

Scores are never invented by this module: unmeasured fields stay null.
"""

from __future__ import annotations

import statistics


def _normalized_score(score, baseline, direction: str) -> float | None:
    """Score relative to the baseline.

    For ``higher_is_better`` metrics the normalized score is
    ``score / baseline``; for ``lower_is_better`` it is
    ``baseline / score``. Returns None when either value is missing
    or the division is undefined. A value of 1.0 means "matches the
    baseline"; above 1.0 means "beats the baseline".
    """
    if score is None or baseline is None:
        return None
    try:
        score = float(score)
        baseline = float(baseline)
    except (TypeError, ValueError):
        return None
    if direction == "higher_is_better":
        if baseline == 0:
            return None
        return score / baseline
    if direction == "lower_is_better":
        if score == 0:
            return None
        return baseline / score
    return None


def evaluate_competition(model, competition: dict) -> dict:
    """Evaluate *model* on one competition.

    ``competition`` carries ``competition_id``, ``metric``,
    ``metric_direction`` (``"higher_is_better"`` or
    ``"lower_is_better"``, default ``"higher_is_better"``) and
    ``baseline_score``. Returns::

        {
            "competition_id": ...,
            "metric": ...,
            "baseline_score": ...,
            "status": "not_run" | "run" | "error",
            "score": float | None,
            "normalized_score": float | None,
            "submission_valid": bool | None,
            "fold_scores": [...] | None,
            "fold_score_std": float | None,
            "error": str | None,
        }
    """
    competition_id = competition.get("competition_id")
    baseline = competition.get("baseline_score")
    direction = competition.get("metric_direction", "higher_is_better")
    result: dict = {
        "competition_id": competition_id,
        "metric": competition.get("metric"),
        "baseline_score": baseline,
        "status": "not_run",
        "score": None,
        "normalized_score": None,
        "submission_valid": None,
        "fold_scores": None,
        "fold_score_std": None,
        "error": None,
    }
    if model is None:
        return result
    try:
        outcome = model(competition)
    except Exception as exc:  # noqa: BLE001 - evaluation must report, not crash
        result["status"] = "error"
        result["error"] = str(exc)
        return result
    if not isinstance(outcome, dict):
        result["status"] = "error"
        result["error"] = "model callable must return a dict"
        return result
    result["status"] = outcome.get("status", "run")
    result["score"] = outcome.get("score")
    result["submission_valid"] = outcome.get("submission_valid")
    fold_scores = outcome.get("fold_scores")
    if isinstance(fold_scores, list) and fold_scores:
        try:
            numeric = [float(value) for value in fold_scores]
            result["fold_scores"] = numeric
            result["fold_score_std"] = (
                statistics.pstdev(numeric) if len(numeric) > 1 else 0.0
            )
        except (TypeError, ValueError):
            result["fold_scores"] = fold_scores
    result["normalized_score"] = _normalized_score(
        result["score"], baseline, direction
    )
    return result
