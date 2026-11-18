"""Evaluation entry point for Qwen3.8-27B-akarui.

Usage:
    python scripts/evaluate.py --config configs/evaluation.yaml --model base --output evaluation/results.json
    python scripts/evaluate.py --config configs/evaluation.yaml --model output/akarui-sft-v0 --output evaluation/results.json

``--model`` is ``base`` for the pinned Qwen3.8-27B base model, or the path
to a LoRA adapter directory. Results are written as JSON with at least
``normalized_score`` and ``submission_validity``; the benchmark runner in
``src.evaluation`` produces the full result record.

Evaluation libraries are imported lazily, so ``--help`` works without
them. No benchmark result is ever claimed before the benchmark runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The evaluation modules live in src/ at the repository root; the script
# directory alone is not enough when run as `python scripts/evaluate.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument(
        "--config", required=True, help="Evaluation config YAML"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="'base' for the pinned base model, or a LoRA adapter directory",
    )
    parser.add_argument(
        "--output", required=True, help="Where to write the results JSON"
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        return 1
    if args.model != "base" and not Path(args.model).exists():
        print(f"error: model path not found: {args.model}", file=sys.stderr)
        return 1

    try:
        from src.evaluation.benchmark_runner import run_evaluation  # noqa: PLC0415
    except ImportError as exc:
        print(
            f"error: cannot import the benchmark runner: {exc}. "
            "Install the evaluation dependencies (requirements.txt) and "
            "ensure src/evaluation is implemented.",
            file=sys.stderr,
        )
        return 1

    results = run_evaluation(
        config_path=config_path, model=args.model
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Evaluation results written to {output_path}")
    print(f"  normalized_score: {results.get('normalized_score')}")
    print(f"  submission_validity: {results.get('submission_validity')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
