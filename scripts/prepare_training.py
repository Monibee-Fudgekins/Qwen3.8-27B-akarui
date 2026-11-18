"""Resolve a training configuration into an executable training plan.

Loads a stage config (``configs/sft.yaml`` etc.), deep-merges it over its
``base`` config, runs GPU detection, selects a distributed strategy, and
writes a resolved training plan JSON.

Rules:
* Never assumes 4 GPUs or 96 GB of aggregate memory. The strategy is
  chosen from the hardware actually detected.
* DDP is used only when at least 2 GPUs each meet ``hardware.min_gpu_memory_gb``.
* Gradient accumulation is adjusted so the effective batch size stays 4:
  ``effective_batch = per_device_batch * grad_accum * num_gpus``.
* The base-model revision must be pinned (a non-empty value that is not
  ``main`` / ``master`` / ``latest``). An unpinned revision is an error.
* The 11h runtime guard is resolved and validated
  (``src/training/runtime_guard.py``): ``runtime.max_runtime_hours``
  must be in (0, 11] and the ``training.max_steps`` cap must fit the
  hour budget at the worst-case step rate. An inconsistent config is
  an error, raised before training starts.
* With ``--allow-no-gpu`` the plan is written for validation purposes
  without detected hardware; the strategy is recorded as ``unresolved``
  and must be re-resolved on the training machine.

No network calls are made at import time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None  # type: ignore[assignment]

from detect_gpus import detect  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None  # type: ignore[assignment]

from detect_gpus import detect  # noqa: E402
from src.training.runtime_guard import resolve_runtime_guard  # noqa: E402

TARGET_EFFECTIVE_BATCH_SIZE = 4
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$|^[0-9]+\.[0-9]+\.[0-9]+$")


def load_config(path: Path) -> dict:
    """Load a YAML config, deep-merging its ``base`` file if declared.

    Args:
        path: Path to the stage config file.

    Returns:
        The merged config dict.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required: pip install pyyaml")
    config = yaml.safe_load(path.read_text()) or {}
    base_name = config.get("base")
    if base_name:
        base_path = path.parent / base_name
        base_config = yaml.safe_load(base_path.read_text()) or {}
        config = _deep_merge(base_config, config)
        config.pop("base", None)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` over ``base`` (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_base_revision(config: dict) -> str:
    """Return the pinned base revision, or raise if it is not pinned.

    Args:
        config: Merged training config.

    Raises:
        ValueError: If the revision is missing, a branch name, or a
            placeholder.
    """
    revision = str(config.get("base_revision") or "").strip()
    if not revision or revision.lower() in {"main", "master", "latest"}:
        raise ValueError(
            "base_revision is not pinned: set it to the exact base-model "
            "commit hash (40 hex characters) in the config."
        )
    if not REVISION_PATTERN.match(revision):
        raise ValueError(
            f"base_revision {revision!r} does not look like a pinned "
            "commit hash or release tag."
        )
    return revision


def select_strategy(config: dict, hardware: dict) -> dict:
    """Select the distributed strategy from detected hardware.

    Args:
        config: Merged training config.
        hardware: Report from :func:`detect_gpus.detect`.

    Returns:
        Dict with ``strategy``, ``num_gpus``, ``per_device_train_batch_size``
        and ``gradient_accumulation_steps``.
    """
    training = config.get("training", {})
    min_memory = float(config.get("hardware", {}).get("min_gpu_memory_gb", 20))
    per_device_batch = int(training.get("per_device_train_batch_size", 1))

    capable = [
        gpu
        for gpu in hardware.get("gpus", [])
        if gpu.get("memory_total_gb", 0) >= min_memory
    ]
    num_gpus = len(capable)
    if num_gpus >= 2:
        strategy = "ddp"
    elif num_gpus == 1:
        strategy = "single_gpu"
    else:
        strategy = "none"

    grad_accum = max(
        1,
        TARGET_EFFECTIVE_BATCH_SIZE
        // max(1, per_device_batch * max(1, num_gpus)),
    )
    return {
        "strategy": strategy,
        "num_gpus": num_gpus,
        "capable_gpus": capable,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": per_device_batch * grad_accum * max(1, num_gpus),
        "target_effective_batch_size": TARGET_EFFECTIVE_BATCH_SIZE,
    }


def build_plan(config_path: Path, allow_no_gpu: bool = False) -> dict:
    """Build the resolved training plan for a config file.

    Args:
        config_path: Path to the stage config.
        allow_no_gpu: If True, write the plan even with no GPU detected,
            with the strategy recorded as ``unresolved``.

    Raises:
        ValueError: If the base revision is not pinned.
        RuntimeError: If no usable GPU is detected and ``allow_no_gpu``
            is False.
    """
    config = load_config(config_path)
    revision = validate_base_revision(config)
    # Validates the 11h guard (hours in (0, 11], step cap fits the
    # budget at the worst-case step rate); raises ValueError if the
    # config is inconsistent, before any hardware is touched.
    runtime_guard = resolve_runtime_guard(config)
    hardware = detect()
    plan_strategy = select_strategy(config, hardware)

    if plan_strategy["strategy"] == "none" and not allow_no_gpu:
        raise RuntimeError(
            "No GPU meeting the minimum memory requirement "
            f"({config.get('hardware', {}).get('min_gpu_memory_gb', 20)} GB) "
            "was detected. Refusing to plan a run on unsuitable hardware."
        )
    if plan_strategy["strategy"] == "none":
        plan_strategy["strategy"] = "unresolved"
        plan_strategy["num_gpus"] = 0

    config_text = json.dumps(config, sort_keys=True, default=str)
    return {
        "config_file": str(config_path),
        "config_hash": hashlib.sha256(config_text.encode()).hexdigest(),
        "base_model": config.get("base_model"),
        "base_revision": revision,
        "stage": config.get("stage"),
        "training_method": config.get("training_method"),
        "hardware": hardware,
        "strategy": plan_strategy,
        "training": config.get("training", {}),
        "runtime_guard": runtime_guard,
        "quantization": config.get("quantization", {}),
        "lora": config.get("lora", {}),
        "dataset": config.get("dataset", {}),
        "output_dir": config.get("output_dir"),
        # Full merged config, so stage-specific sections (dpo, grpo)
        # are never silently dropped from the plan.
        "config": config,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Resolve a training config into a training plan JSON."
    )
    parser.add_argument("--config", required=True, help="Stage config YAML")
    parser.add_argument(
        "--output",
        default="output/training-plan.json",
        help="Where to write the resolved plan JSON",
    )
    parser.add_argument(
        "--allow-no-gpu",
        action="store_true",
        help="Write the plan even with no GPU (strategy recorded unresolved)",
    )
    args = parser.parse_args(argv)

    try:
        plan = build_plan(Path(args.config), allow_no_gpu=args.allow_no_gpu)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2))
    print(
        f"Plan written to {output_path}: "
        f"strategy={plan['strategy']['strategy']} "
        f"gpus={plan['strategy']['num_gpus']} "
        f"effective_batch={plan['strategy']['effective_batch_size']} "
        f"max_runtime_hours={plan['runtime_guard']['max_runtime_hours']} "
        f"max_steps={plan['runtime_guard']['max_steps']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
