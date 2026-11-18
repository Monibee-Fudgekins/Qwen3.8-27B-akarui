"""Record a measured base-model baseline.

Usage:
    python scripts/evaluate.py --config configs/evaluation.yaml \\
        --model base --output output/evaluation-base.json
    python scripts/record_baseline.py --evaluation output/evaluation-base.json

Reads the base-model evaluation summary and writes the canonical
measured record to ``evaluation/baseline_results.json``, replacing
the shipped not-measured template. Refuses when the evaluation is
not of the base model, has no measured score, or measured zero
competitions -- a baseline score is never invented. Refuses to
overwrite an already-measured baseline unless ``--force`` is given,
so a measured baseline cannot be silently replaced.

Until this file is measured, no promotion can pass: the gate in
``scripts/verify_release.py`` has no reference to compare the first
candidate against, and ``scripts/evaluate_and_promote.py`` records
the decision as ``blocked``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.baseline import (  # noqa: E402
    canonical_baseline_record,
    is_measured,
)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, else 1."""
    parser = argparse.ArgumentParser(
        description="Record a measured base-model baseline."
    )
    parser.add_argument(
        "--evaluation",
        required=True,
        help="Base-model evaluation summary JSON (scripts/evaluate.py --model base)",
    )
    parser.add_argument(
        "--output",
        default="evaluation/baseline_results.json",
        help="Where to write the canonical baseline record",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an already-measured baseline (default: refuse)",
    )
    args = parser.parse_args(argv)

    evaluation_path = Path(args.evaluation)
    if not evaluation_path.exists():
        print(f"error: evaluation file not found: {evaluation_path}", file=sys.stderr)
        return 1
    try:
        evaluation = json.loads(evaluation_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"error: cannot parse evaluation JSON: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        try:
            existing = json.loads(output_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = None
        if is_measured(existing):
            print(
                "error: baseline is already measured at "
                f"{output_path} (normalized_score="
                f"{existing.get('normalized_score')}); refusing to "
                "overwrite without --force",
                file=sys.stderr,
            )
            return 1

    try:
        record = canonical_baseline_record(evaluation)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Baseline recorded at {output_path}")
    print(f"  normalized_score: {record['normalized_score']}")
    print(f"  submission_validity: {record['submission_validity']}")
    print(f"  sample_size: {record['sample_size']} competition(s) measured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
