"""Automated evaluate-and-promote step for the weekly lifecycle.

Usage (run on the GitHub runner AFTER the eval kernel completes and
its output has been downloaded):
    python scripts/evaluate_and_promote.py \\
        --evaluation output/eval/evaluation-results.json \\
        --provenance output/provenance.json \\
        --adapter output/adapter

Flow:
1. A training run record with ``status: "skipped_quota"`` records a
   ``skipped`` decision and exits 0 -- nothing was trained, so there
   is nothing to evaluate.
2. The promotion reference is resolved: the measured champion
   (``evaluation/champion_results.json``) when one exists, else the
   measured baseline (``evaluation/baseline_results.json``). When
   neither is measured the decision is ``blocked`` -- "baseline not
   measured" -- and the adapter stays private. No promotion can pass
   until the baseline is measured (see ``scripts/record_baseline.py``).
3. ``scripts/verify_release.py`` enforces the gates
   (``configs/evaluation.yaml``): the candidate must beat the
   reference by at least ``min_normalized_score_improvement``, no
   regression probe may drop by more than ``max_allowed_regression``,
   submission validity must clear ``min_submission_validity``, and
   every benchmark competition must be measured. On failure the
   decision is ``stay_private``, the adapter is NOT published, and
   the workflow exits 0 -- a failing gate is a normal outcome, not
   a broken build.
4. On a pass, ``scripts/publish_kaggle.py`` uploads the adapter to
   the public Kaggle Model (adapters first; the base model is never
   re-uploaded). Only after a successful upload is the champion
   file updated and a release record written.
5. Every decision is recorded in ``releases/gate-decisions/`` with
   the candidate and reference scores, sample size, champion
   version, benchmark version, and timestamp.

Exit codes: 0 = done (promoted, stayed private, blocked, or
skipped, with the decision recorded); 1 = infrastructure error
(missing files, unparseable results, publication failure after the
gates passed). ``--dry-run`` records a ``promote`` decision without
uploading or updating the champion.

This script never publishes on its own authority: publication goes
through ``publish_kaggle.py``, which re-runs the same gate. The
manual publish path (``.github/workflows/publish-model.yml``, with
its ``confirm_gates_passed`` approval) is unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_release import verify_release  # noqa: E402

from src.evaluation.baseline import load_measured  # noqa: E402
from src.publishing.gate_decision import (  # noqa: E402
    build_gate_decision,
    record_gate_decision,
    resolve_reference,
    update_champion,
)
from src.publishing.release_manifest import write_release  # noqa: E402

KAGGLE_MODEL_URL = "https://www.kaggle.com/models/monibeefudgekins/qwen3-8-27b-akarui"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_gates(config_path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    if not config_path.exists():
        return {}
    config = yaml.safe_load(config_path.read_text()) or {}
    return config.get("promotion_gates", {}) or {}


def _next_versions(reference_version: str | None, reference_kind: str | None) -> tuple[str, str]:
    """Derive the next release/model versions from the reference.

    First promotion (baseline reference): release ``"1"``,
    model ``"qlora-sft-v1"``. Later promotions increment the
    champion's trailing ``-vN`` (``qlora-sft-v1`` -> ``"2"`` /
    ``"qlora-sft-v2"``). Raises ValueError when the champion
    version cannot be parsed -- pass explicit versions then.
    """
    if reference_kind != "champion" or not reference_version:
        return "1", "qlora-sft-v1"
    match = re.match(r"^(.*-v)(\d+)$", reference_version)
    if not match:
        raise ValueError(
            f"cannot derive the next version from champion version "
            f"{reference_version!r}; pass --release-version and "
            "--model-version explicitly"
        )
    number = int(match.group(2)) + 1
    return str(number), f"{match.group(1)}{number}"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 when handled, else 1."""
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate against the champion and promote only if the gates pass."
    )
    parser.add_argument("--evaluation", required=True, help="Candidate evaluation results JSON")
    parser.add_argument("--provenance", required=True, help="Provenance manifest JSON")
    parser.add_argument("--adapter", required=True, help="Adapter directory to publish")
    parser.add_argument("--model-card", default="model_cards/MODEL_CARD.md")
    parser.add_argument("--champion", default="evaluation/champion_results.json")
    parser.add_argument("--baseline", default="evaluation/baseline_results.json")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--run-record", help="Training run record JSON (quota skips)")
    parser.add_argument("--candidate-ref", help="Candidate identifier for the decision record")
    parser.add_argument("--release-version", default="auto", help="Release version, or 'auto'")
    parser.add_argument("--model-version", default="auto", help="Variation label, or 'auto'")
    parser.add_argument("--decisions-dir", default="releases/gate-decisions")
    parser.add_argument("--releases-dir", default="releases")
    parser.add_argument("--staging-dir", default="output/kaggle-publish")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    decisions_dir = Path(args.decisions_dir)
    gates = _load_gates(Path(args.config))

    # 1. Quota skip: nothing was trained, nothing to evaluate.
    if args.run_record:
        run_record_path = Path(args.run_record)
        if run_record_path.exists():
            record = _load_json(run_record_path)
            if record.get("status") == "skipped_quota":
                decision = build_gate_decision(
                    decision="skipped",
                    reasons=["training run skipped: weekly GPU quota exhausted"],
                    candidate={},
                    reference=None,
                    reference_kind=None,
                    reference_version=None,
                    baseline_score=None,
                    candidate_ref=args.candidate_ref or "weekly-candidate",
                    gates=gates,
                )
                path = record_gate_decision(decision, decisions_dir=decisions_dir)
                print(f"Training skipped (quota); decision recorded at {path}")
                return 0

    evaluation_path = Path(args.evaluation)
    if not evaluation_path.exists():
        print(f"error: evaluation file not found: {evaluation_path}", file=sys.stderr)
        return 1
    try:
        candidate = _load_json(evaluation_path)
    except json.JSONDecodeError as exc:
        print(f"error: cannot parse evaluation JSON: {exc}", file=sys.stderr)
        return 1
    provenance_path = Path(args.provenance)
    if not provenance_path.exists():
        print(f"error: provenance manifest not found: {provenance_path}", file=sys.stderr)
        return 1

    # 2. Resolve the reference: measured champion, else measured baseline.
    reference, reference_kind, reference_version = resolve_reference(
        Path(args.champion), Path(args.baseline)
    )
    baseline_record = load_measured(Path(args.baseline))
    baseline_score = None
    if reference_kind == "baseline" and reference is not None:
        baseline_score = reference.get("normalized_score")
    elif baseline_record is not None:
        baseline_score = baseline_record.get("normalized_score")
    elif candidate.get("baseline_normalized_score") is not None:
        baseline_score = candidate.get("baseline_normalized_score")

    if reference is None:
        decision = build_gate_decision(
            decision="blocked",
            reasons=[
                "no measured champion and no measured baseline: the base "
                "model must be measured before the first promotion can "
                "pass (run the eval kernel in base mode, then "
                "scripts/record_baseline.py)"
            ],
            candidate=candidate,
            reference=None,
            reference_kind=None,
            reference_version=None,
            baseline_score=None,
            candidate_ref=args.candidate_ref or "weekly-candidate",
            gates=gates,
        )
        path = record_gate_decision(decision, decisions_dir=decisions_dir)
        print(f"Promotion BLOCKED (baseline not measured); decision recorded at {path}")
        return 0

    # 3. Enforce the gates.
    reference_path = Path(args.champion) if reference_kind == "champion" else Path(args.baseline)
    reasons = verify_release(
        provenance_path, evaluation_path, reference_path, Path(args.config)
    )
    candidate_ref = args.candidate_ref or "weekly-candidate"

    if reasons:
        decision = build_gate_decision(
            decision="stay_private",
            reasons=reasons,
            candidate=candidate,
            reference=reference,
            reference_kind=reference_kind,
            reference_version=reference_version,
            baseline_score=baseline_score,
            candidate_ref=candidate_ref,
            gates=gates,
        )
        path = record_gate_decision(decision, decisions_dir=decisions_dir)
        print("Candidate stays PRIVATE -- promotion gates failed:")
        for reason in reasons:
            print(f"  - {reason}")
        print(f"Decision recorded at {path}")
        return 0

    # Gates passed: resolve the version labels for this promotion.
    try:
        release_version, model_version = _next_versions(reference_version, reference_kind)
        if args.release_version != "auto":
            release_version = args.release_version
        if args.model_version != "auto":
            model_version = args.model_version
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    candidate_ref = args.candidate_ref or model_version

    # 4. Gates passed: publish (unless dry-run), then record.
    if args.dry_run:
        decision = build_gate_decision(
            decision="promote",
            reasons=["promotion gates passed (dry run: nothing uploaded)"],
            candidate=candidate,
            reference=reference,
            reference_kind=reference_kind,
            reference_version=reference_version,
            baseline_score=baseline_score,
            candidate_ref=candidate_ref,
            gates=gates,
            release_version=release_version,
            dry_run=True,
        )
        path = record_gate_decision(decision, decisions_dir=decisions_dir)
        print(f"Dry run: gates passed, would publish {model_version}; decision at {path}")
        return 0

    from publish_kaggle import main as publish_main  # noqa: PLC0415

    publish_rc = publish_main(
        [
            "--provenance", str(provenance_path),
            "--evaluation", str(evaluation_path),
            "--champion", str(Path(args.champion)),
            "--baseline", str(Path(args.baseline)),
            "--config", str(args.config),
            "--adapter", str(args.adapter),
            "--model-card", str(args.model_card),
            "--release-version", release_version,
            "--model-version", model_version,
            "--staging-dir", str(args.staging_dir),
        ]
    )
    if publish_rc != 0:
        decision = build_gate_decision(
            decision="blocked",
            reasons=["promotion gates passed but publication failed"],
            candidate=candidate,
            reference=reference,
            reference_kind=reference_kind,
            reference_version=reference_version,
            baseline_score=baseline_score,
            candidate_ref=candidate_ref,
            gates=gates,
            release_version=release_version,
        )
        path = record_gate_decision(decision, decisions_dir=decisions_dir)
        print(f"error: publication failed after gates passed; decision at {path}", file=sys.stderr)
        return 1

    decision = build_gate_decision(
        decision="promote",
        reasons=["promotion gates passed; adapter published to Kaggle Models"],
        candidate=candidate,
        reference=reference,
        reference_kind=reference_kind,
        reference_version=reference_version,
        baseline_score=baseline_score,
        candidate_ref=candidate_ref,
        gates=gates,
        release_version=release_version,
    )
    decision_path = record_gate_decision(decision, decisions_dir=decisions_dir)
    try:
        champion_path = update_champion(
            candidate,
            champion_path=Path(args.champion),
            champion_version=model_version,
            kaggle_model_url=KAGGLE_MODEL_URL,
            gate_decision_path=decision_path,
        )
        manifest = _load_json(provenance_path)
        release_path = write_release(
            release_version,
            manifest,
            candidate,
            f"Promoted {model_version}: candidate normalized score "
            f"{candidate.get('normalized_score')} vs {reference_kind} "
            f"{reference.get('normalized_score')} "
            f"(gate decision {decision_path.name}).",
            releases_dir=Path(args.releases_dir),
        )
    except (ValueError, FileExistsError, OSError) as exc:
        print(
            "error: published to Kaggle but failed to record the release "
            f"locally: {exc}",
            file=sys.stderr,
        )
        return 1
    print(f"Promoted {model_version} to {KAGGLE_MODEL_URL}")
    print(f"  champion updated: {champion_path}")
    print(f"  release record: {release_path}")
    print(f"  gate decision: {decision_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
