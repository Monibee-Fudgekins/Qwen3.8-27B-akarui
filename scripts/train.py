"""Training entry point for Qwen3.8-27B-akarui.

Usage:
    python scripts/train.py --config configs/sft.yaml --stage sft
    python scripts/train.py --config configs/dpo.yaml --stage dpo --resume
    python scripts/train.py --config configs/grpo.yaml --stage grpo --force

Training libraries (torch, transformers, peft, trl) are imported lazily
inside the stage runners, so ``--help`` and config validation work on a
machine without them installed. Resuming reuses the latest checkpoint in
the output directory unless ``--resume-from-checkpoint`` names one.

GRPO is experimental and disabled by default in configs/grpo.yaml; it
refuses to run unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The training modules live in src/ at the repository root; the script
# directory alone is not enough when run as `python scripts/train.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None  # type: ignore[assignment]


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the newest ``checkpoint-*`` directory, or None.

    Args:
        output_dir: Training output directory.
    """
    if not output_dir.exists():
        return None
    checkpoints = sorted(
        (child for child in output_dir.iterdir() if child.is_dir()),
        key=lambda child: child.stat().st_mtime,
    )
    numbered = [c for c in checkpoints if c.name.startswith("checkpoint-")]
    return numbered[-1] if numbered else None


def load_stage_config(config_path: Path) -> dict:
    """Load a stage config with base inheritance (same merge as prepare)."""
    if yaml is None:
        raise RuntimeError("PyYAML is required: pip install pyyaml")
    from prepare_training import load_config  # noqa: PLC0415

    return load_config(config_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Train Qwen3.8-27B-akarui.")
    parser.add_argument("--config", required=True, help="Stage config YAML")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["sft", "dpo", "grpo"],
        help="Training stage to run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the output directory",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Explicit checkpoint directory to resume from",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow experimental stages (GRPO) even when disabled in config",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        return 1
    try:
        config = load_stage_config(config_path)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    config_stage = config.get("stage")
    if config_stage and config_stage != args.stage:
        print(
            f"error: config stage is {config_stage!r} but --stage is "
            f"{args.stage!r}; use the matching config file.",
            file=sys.stderr,
        )
        return 1

    if args.stage == "grpo" and not config.get("grpo", {}).get("enabled", False):
        if not args.force:
            print(
                "error: GRPO is experimental and disabled in the config. "
                "Enable it in configs/grpo.yaml or pass --force.",
                file=sys.stderr,
            )
            return 1
        print("warning: running experimental GRPO with --force")

    resume_from = None
    if args.resume_from_checkpoint:
        resume_from = Path(args.resume_from_checkpoint)
        if not resume_from.exists():
            print(f"error: checkpoint not found: {resume_from}", file=sys.stderr)
            return 1
    elif args.resume:
        resume_from = find_latest_checkpoint(Path(config.get("output_dir", "output")))
        if resume_from is None:
            print("warning: --resume given but no checkpoint found; starting fresh")
        else:
            print(f"Resuming from checkpoint: {resume_from}")

    train_file = config.get("dataset", {}).get("train_file")
    output_dir = config.get("output_dir", "output")
    if not train_file:
        print(
            "error: config has no dataset.train_file; refusing to train "
            "without an explicit training file.",
            file=sys.stderr,
        )
        return 1

    # Lazy imports: training libraries are only needed on the training host.
    try:
        if args.stage == "sft":
            from src.training.sft import run_sft  # noqa: PLC0415

            run_sft(config, train_file, output_dir, resume_from=resume_from)
        elif args.stage == "dpo":
            from src.training.dpo import run_dpo  # noqa: PLC0415

            run_dpo(config, train_file, output_dir, resume_from=resume_from)
        else:
            from src.training.grpo import run_grpo  # noqa: PLC0415

            run_grpo(config, train_file, output_dir, resume_from=resume_from)
    except ImportError as exc:
        print(
            f"error: cannot import the {args.stage} trainer: {exc}. "
            "Install the training dependencies (requirements.txt) and "
            "ensure src/training is implemented.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
