"""Gate decisions and the champion record for Qwen3.8-27B-akarui.

The promotion lifecycle has exactly one reference at any time:

* the **champion** -- ``evaluation/champion_results.json``, the last
  promoted version's measured evaluation, when one exists;
* otherwise the **baseline** -- ``evaluation/baseline_results.json``,
  the base model's measured score, once it has been measured;
* otherwise **no reference**, and no promotion can pass.

:func:`resolve_reference` implements that order. A gate decision is
an append-only JSON record in ``releases/gate-decisions/`` carrying
the decision (``promote`` | ``stay_private`` | ``blocked`` |
``skipped``), both scores, the sample size, the champion version,
the benchmark version, the gate thresholds in force, and the
timestamp. :func:`update_champion` writes the new champion file
after a successful publication; it refuses an unmeasured
evaluation, so the champion can never be an unmeasured record.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.baseline import is_measured, load_measured

DEFAULT_DECISIONS_DIR = Path(__file__).resolve().parents[2] / "releases" / "gate-decisions"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "candidate"
    return slug[:80]


def resolve_reference(
    champion_path: str | Path, baseline_path: str | Path
) -> tuple[dict | None, str | None, str | None]:
    """Resolve the promotion reference.

    Returns ``(reference, kind, version)`` where ``kind`` is
    ``"champion"`` (a promoted version exists), ``"baseline"`` (no
    champion yet, but the base model has been measured), or None
    (neither measured -- no promotion can pass). ``version`` is the
    champion version label, or ``"base"`` for the baseline.
    """
    champion = load_measured(champion_path)
    if champion is not None:
        return (
            champion,
            "champion",
            str(
                champion.get("champion_version")
                or champion.get("release_version")
                or "unknown"
            ),
        )
    baseline = load_measured(baseline_path)
    if baseline is not None:
        return baseline, "baseline", "base"
    return None, None, None


def build_gate_decision(
    *,
    decision: str,
    reasons: list[str],
    candidate: dict,
    reference: dict | None,
    reference_kind: str | None,
    reference_version: str | None,
    baseline_score: float | None,
    candidate_ref: str,
    gates: dict | None = None,
    release_version: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Build a gate-decision record.

    ``decision`` is ``promote``, ``stay_private``, ``blocked`` or
    ``skipped``. ``candidate`` is the candidate evaluation summary;
    ``reference`` is the resolved champion/baseline record (or None).
    ``baseline_score`` is the base model's measured score when known
    (the reference score when the reference is the baseline, else
    the candidate's ``baseline_normalized_score`` or None).
    """
    from src.evaluation.baseline import measured_sample_size

    return {
        "decision": decision,
        "decided_at": _now_iso(),
        "candidate_ref": candidate_ref,
        "candidate_score": candidate.get("normalized_score"),
        "candidate_sample_size": measured_sample_size(candidate),
        "benchmark_version": candidate.get("benchmark_version"),
        "reference_kind": reference_kind,
        "reference_version": reference_version,
        "reference_score": (
            reference.get("normalized_score") if reference else None
        ),
        "baseline_score": baseline_score,
        "release_version": release_version,
        "dry_run": dry_run,
        "gates": gates or {},
        "reasons": list(reasons),
    }


def record_gate_decision(
    decision: dict, *, decisions_dir: str | Path | None = None
) -> Path:
    """Write *decision* as an append-only JSON record. Returns its path.

    The filename is the decision timestamp plus the candidate ref;
    a collision appends a counter instead of overwriting.
    """
    directory = Path(decisions_dir) if decisions_dir is not None else DEFAULT_DECISIONS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = re.sub(r"[^0-9T]", "", decision.get("decided_at", _now_iso()))[:15] or "unknown"
    base = f"{stamp}_{_slug(str(decision.get('candidate_ref', 'candidate')))}"
    path = directory / f"{base}.json"
    counter = 2
    while path.exists():
        path = directory / f"{base}-{counter}.json"
        counter += 1
    path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def update_champion(
    evaluation: dict,
    *,
    champion_path: str | Path,
    champion_version: str,
    promoted_at: str | None = None,
    kaggle_model_url: str | None = None,
    gate_decision_path: str | Path | None = None,
) -> Path:
    """Write the champion file for a newly promoted version.

    ``evaluation`` must be measured; raises ValueError otherwise.
    The previous champion file (if any) is overwritten -- its score
    survives in the gate-decision record that justified this
    promotion, which is append-only.
    """
    if not is_measured(evaluation):
        raise ValueError(
            "refusing to record an unmeasured evaluation as champion"
        )
    record = dict(evaluation)
    record["status"] = "measured"
    record["champion_version"] = champion_version
    record["promoted_at"] = promoted_at or _now_iso()
    if kaggle_model_url:
        record["kaggle_model_url"] = kaggle_model_url
    if gate_decision_path is not None:
        record["gate_decision"] = Path(gate_decision_path).name
    path = Path(champion_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
