"""Verify a release candidate before publication.

Checks, in order:
1. The provenance manifest exists and every required field is present
   and non-null. Explicit ``None`` values are rejected, never published.
2. The evaluation results exist and contain a normalized competition
   score.
3. The promotion gates in the evaluation config hold against the
   champion's results (or, when no champion exists yet, against the
   recorded base-model baseline): the candidate must improve the
   normalized score by at least ``min_normalized_score_improvement``
   and no regression probe may drop by more than
   ``max_allowed_regression``. A candidate with no measured baseline
   to beat can never pass -- the base model must be measured first.

Exit codes:
    0  Release is verified and may be published.
    1  Verification failed; the reason is printed in plain language.

Also importable: :func:`verify_release` returns the list of failure
reasons (empty when the release is verified). No network calls are made.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None  # type: ignore[assignment]

# Provenance fields every published version must carry, per the project
# contract. A missing or null value blocks publication.
REQUIRED_PROVENANCE_FIELDS = [
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
]


def load_json(path: Path) -> dict:
    """Load a JSON file, raising a clear error if it is missing/invalid."""
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return json.loads(path.read_text())


def check_provenance(manifest: dict) -> list[str]:
    """Return failure reasons for an incomplete provenance manifest."""
    reasons: list[str] = []
    for field in REQUIRED_PROVENANCE_FIELDS:
        if field not in manifest:
            reasons.append(f"provenance field missing: {field}")
        elif manifest[field] is None:
            reasons.append(f"provenance field is null: {field}")
        elif isinstance(manifest[field], str) and not manifest[field].strip():
            reasons.append(f"provenance field is empty: {field}")
        elif isinstance(manifest[field], (list, dict)) and not manifest[field]:
            reasons.append(f"provenance field is empty: {field}")
    return reasons


def check_gates(
    evaluation: dict, champion: dict | None, gates: dict
) -> list[str]:
    """Return failure reasons when the promotion gates do not hold.

    The reference is the champion when one is given, otherwise the
    baseline recorded in the evaluation
    (``baseline_normalized_score``). Either way the candidate must
    beat the reference by at least
    ``min_normalized_score_improvement`` -- the first promotion is
    gated on the measured base model, not waved through for having
    a baseline field present. Submission validity and benchmark
    completeness are enforced in both branches.
    """
    reasons: list[str] = []
    score = evaluation.get("normalized_score")
    if score is None:
        reasons.append("evaluation results have no normalized_score")
        return reasons

    reference: float | None = None
    reference_label = ""
    reference_probes: dict = {}
    if champion is not None:
        reference = champion.get("normalized_score")
        reference_label = "champion"
        reference_probes = champion.get("regression_probes", {}) or {}
        if reference is None:
            reasons.append("champion results have no normalized_score")
            return reasons
    elif evaluation.get("baseline_normalized_score") is not None:
        reference = evaluation.get("baseline_normalized_score")
        reference_label = "baseline"
        reference_probes = evaluation.get("baseline_regression_probes", {}) or {}
    if reference is None:
        reasons.append(
            "no champion results provided and evaluation has no "
            "baseline_normalized_score to compare against; the base "
            "model must be measured before the first promotion can pass"
        )
        return reasons

    min_improvement = float(gates.get("min_normalized_score_improvement", 0.01))
    improvement = float(score) - float(reference)
    if improvement < min_improvement:
        reasons.append(
            f"candidate normalized score {score} improves on {reference_label} "
            f"{reference} by {improvement:.4f}, below the required "
            f"{min_improvement}"
        )

    max_regression = float(gates.get("max_allowed_regression", 0.02))
    candidate_probes = evaluation.get("regression_probes", {}) or {}
    for name, candidate_value in candidate_probes.items():
        baseline_value = reference_probes.get(name)
        if baseline_value is None:
            continue
        drop = float(baseline_value) - float(candidate_value)
        if drop > max_regression:
            reasons.append(
                f"regression probe {name!r} dropped by {drop:.4f} "
                f"(max allowed {max_regression})"
            )

    if gates.get("require_submission_validity"):
        min_validity = float(gates.get("min_submission_validity", 0.9))
        validity = evaluation.get("submission_validity")
        if validity is None:
            reasons.append("evaluation results have no submission_validity")
        elif float(validity) < min_validity:
            reasons.append(
                f"submission validity {validity} is below the required "
                f"{min_validity}"
            )

    results = evaluation.get("results")
    if isinstance(results, list):
        if not results:
            reasons.append("evaluation results list is empty")
        else:
            unmeasured = [
                entry.get("competition_id")
                for entry in results
                if isinstance(entry, dict)
                and entry.get("normalized_score") is None
            ]
            if unmeasured:
                reasons.append(
                    "benchmark incomplete: "
                    f"{len(unmeasured)} of {len(results)} competition(s) "
                    "unmeasured: " + ", ".join(str(c) for c in unmeasured[:5])
                    + (" (+ more)" if len(unmeasured) > 5 else "")
                )
    return reasons


def verify_release(
    provenance_path: Path,
    evaluation_path: Path,
    champion_path: Path | None,
    config_path: Path,
) -> list[str]:
    """Verify a release candidate. Returns failure reasons (empty = pass)."""
    reasons: list[str] = []
    try:
        manifest = load_json(provenance_path)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return [f"cannot read provenance manifest: {exc}"]
    reasons.extend(check_provenance(manifest))

    try:
        evaluation = load_json(evaluation_path)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return reasons + [f"cannot read evaluation results: {exc}"]

    champion = None
    if champion_path is not None and champion_path.exists():
        try:
            champion = load_json(champion_path)
        except json.JSONDecodeError as exc:
            return reasons + [f"cannot read champion results: {exc}"]

    gates: dict = {}
    if config_path.exists():
        if yaml is None:
            reasons.append("PyYAML is required to read the evaluation config")
        else:
            config = yaml.safe_load(config_path.read_text()) or {}
            gates = config.get("promotion_gates", {}) or {}
            if not config.get("held_out_competition_ids"):
                reasons.append(
                    "no held-out competitions are registered in the "
                    "evaluation config; a candidate cannot be promoted "
                    "without a held-out benchmark"
                )
    reasons.extend(check_gates(evaluation, champion, gates))
    return reasons


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 when the release is verified, else 1."""
    parser = argparse.ArgumentParser(
        description="Verify a release candidate against the promotion gates."
    )
    parser.add_argument("--provenance", required=True, help="Provenance manifest JSON")
    parser.add_argument("--evaluation", required=True, help="Evaluation results JSON")
    parser.add_argument("--champion", help="Champion results JSON (optional)")
    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
        help="Evaluation config YAML with promotion gates",
    )
    args = parser.parse_args(argv)

    reasons = verify_release(
        Path(args.provenance),
        Path(args.evaluation),
        Path(args.champion) if args.champion else None,
        Path(args.config),
    )
    if reasons:
        print("Release verification FAILED:")
        for reason in reasons:
            print(f"  - {reason}")
        return 1
    print("Release verified: provenance complete and promotion gates pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
