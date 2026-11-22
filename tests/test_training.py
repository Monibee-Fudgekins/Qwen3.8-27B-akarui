"""Tests for training helpers, config loading, checkpointing, and gates.

Runs without torch/transformers/GPU. Anything that needs the training
libraries must fail cleanly with an informative error, never a bare crash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training import checkpointing
from src.training.trainer_utils import (
    compute_effective_batch_size,
    format_messages_for_training,
    get_device_info,
    load_config,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent


def test_config_merge_sft():
    import prepare_training

    config = prepare_training.load_config(ROOT / "configs" / "sft.yaml")
    assert config["base_model"] == "Qwen/Qwen3.8-27B"
    assert (
        config["base_revision"]
        == "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
    )
    assert config["stage"] == "sft"
    assert config["lora"]["r"] == 32
    assert config["lora"]["alpha"] == 64
    assert config["training"]["learning_rate"] == 0.0001
    assert config["dataset"]["train_file"] == "data/train.jsonl"


def test_config_merge_dpo_and_grpo():
    import prepare_training

    dpo = prepare_training.load_config(ROOT / "configs" / "dpo.yaml")
    assert dpo["dpo"]["beta"] == 0.1
    assert dpo["dataset"]["train_file"] == "data/dpo.jsonl"
    assert dpo["output_dir"] == "output/akarui-dpo-v0"

    grpo = prepare_training.load_config(ROOT / "configs" / "grpo.yaml")
    assert grpo["grpo"]["enabled"] is False
    assert grpo["grpo"]["num_generations"] == 4
    # GRPO inherits the base dataset file through the merge.
    assert grpo["dataset"]["train_file"] == "data/train.jsonl"


def test_load_config_plain_yaml():
    config = load_config(ROOT / "configs" / "base.yaml")
    assert config["base_model"] == "Qwen/Qwen3.8-27B"
    assert config["quantization"]["bnb_4bit_quant_type"] == "nf4"


def test_effective_batch_size():
    assert compute_effective_batch_size(1, 4) == 4
    assert compute_effective_batch_size(1, 4, num_devices=4) == 16
    with pytest.raises(ValueError):
        compute_effective_batch_size(0, 4)
    with pytest.raises(ValueError):
        compute_effective_batch_size(1, 0)


def test_device_info_without_gpu():
    info = get_device_info()
    assert info["cuda_available"] is False


def test_set_seed_is_deterministic():
    set_seed(42)
    set_seed(42)  # must not raise; determinism is the point


class _StubTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )


def test_format_messages_keeps_reasoning():
    messages = [
        {"role": "user", "content": "Solve this."},
        {
            "role": "assistant",
            "content": "<think>reasoning stays</think>Answer.",
        },
    ]
    rendered = format_messages_for_training(_StubTokenizer(), messages)
    assert "reasoning stays" in rendered


def test_format_messages_requires_template():
    class _NoTemplate:
        chat_template = None

    with pytest.raises(ValueError, match="chat template"):
        format_messages_for_training(_NoTemplate(), [{"role": "user", "content": "x"}])


def test_grpo_disabled_gate_raises_before_heavy_import():
    from src.training.grpo import run_grpo

    with pytest.raises(RuntimeError, match="EXPERIMENTAL"):
        run_grpo(
            {"grpo": {"enabled": False}},
            "nonexistent-tasks.jsonl",
            "output",
        )


def test_dpo_and_grpo_configs_carry_step_caps():
    import prepare_training

    dpo = prepare_training.load_config(ROOT / "configs" / "dpo.yaml")
    assert dpo["training"]["max_steps"] == 600
    assert dpo["runtime"]["max_runtime_hours"] == 10.0
    assert dpo["runtime"]["setup_allowance_hours"] == 1.0

    grpo = prepare_training.load_config(ROOT / "configs" / "grpo.yaml")
    assert grpo["training"]["max_steps"] == 600
    assert grpo["runtime"]["max_runtime_hours"] == 10.0
    assert grpo["runtime"]["setup_allowance_hours"] == 1.0


def test_runtime_guard_callback_stops_training_at_deadline(monkeypatch):
    import sys
    import time
    import types

    from src.training.runtime_guard import RuntimeGuard, runtime_guard_callback

    # The training stack is not installed in the test env; stub just
    # enough of transformers for the callback factory to build.
    fake_transformers = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    fake_transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class _State:
        global_step = 42

    class _Control:
        should_training_stop = False

    # Fresh guard: deadline 10h away, training continues.
    guard = RuntimeGuard(10.0)
    callback = runtime_guard_callback(guard)
    control = _Control()
    callback.on_step_end(None, _State(), control)
    assert control.should_training_stop is False
    assert callback.triggered is False

    # Expired guard: deadline already passed, training stops.
    expired = RuntimeGuard(10.0, start=time.monotonic() - 10 * 3600 - 60)
    callback = runtime_guard_callback(expired)
    control = _Control()
    callback.on_step_end(None, _State(), control)
    assert control.should_training_stop is True
    assert callback.triggered is True


def test_run_dpo_rejects_over_cap_guard_before_heavy_import(tmp_path: Path):
    from src.training.dpo import run_dpo

    config = {
        "runtime": {"max_runtime_hours": 12.0, "setup_allowance_hours": 1.0},
        "training": {"max_steps": 600},
        "dpo": {"beta": 0.1},
    }
    # The preference file need not exist: the guard is resolved
    # before any pair is loaded or any training library is imported.
    with pytest.raises(ValueError, match="max_runtime_hours"):
        run_dpo(config, tmp_path / "pairs.jsonl", tmp_path / "out")


def test_run_dpo_rejects_step_cap_over_budget(tmp_path: Path):
    from src.training.dpo import run_dpo

    config = {
        "runtime": {"max_runtime_hours": 10.0, "setup_allowance_hours": 1.0},
        "training": {"max_steps": 700},
        "dpo": {"beta": 0.1},
    }
    with pytest.raises(ValueError, match="max_steps"):
        run_dpo(config, tmp_path / "pairs.jsonl", tmp_path / "out")


def test_run_dpo_valid_guard_reaches_heavy_import(tmp_path: Path):
    from src.training.dpo import run_dpo

    config = {
        "runtime": {"max_runtime_hours": 10.0, "setup_allowance_hours": 1.0},
        "training": {"max_steps": 600},
        "dpo": {"beta": 0.1},
    }
    # A valid guard passes validation; the run then fails on the
    # missing training stack (datasets/trl are not installed in the
    # test env), proving the guard was resolved first.
    with pytest.raises(ImportError, match="datasets"):
        run_dpo(config, tmp_path / "pairs.jsonl", tmp_path / "out")


def test_run_grpo_rejects_over_cap_guard_before_heavy_import(tmp_path: Path):
    from src.training.grpo import run_grpo

    config = {
        "grpo": {"enabled": True},
        "runtime": {"max_runtime_hours": 12.0, "setup_allowance_hours": 1.0},
        "training": {"max_steps": 600},
    }
    # The tasks file need not exist: the enabled check passes, then
    # the guard is resolved before any training library is imported.
    with pytest.raises(ValueError, match="max_runtime_hours"):
        run_grpo(config, tmp_path / "tasks.jsonl", tmp_path / "out")


def test_run_grpo_valid_guard_reaches_heavy_import(tmp_path: Path):
    from src.training.grpo import run_grpo

    config = {
        "grpo": {"enabled": True},
        "runtime": {"max_runtime_hours": 10.0, "setup_allowance_hours": 1.0},
        "training": {"max_steps": 600},
    }
    # A valid guard passes validation; the run then fails on the
    # missing training stack, proving the guard was resolved first.
    with pytest.raises(ImportError, match="datasets"):
        run_grpo(config, tmp_path / "tasks.jsonl", tmp_path / "out")


def test_checkpoint_find_and_prune(tmp_path: Path):
    output_dir = tmp_path / "output"
    for step in (100, 200, 300, 400, 500):
        (output_dir / f"checkpoint-{step}").mkdir(parents=True)
    assert (
        checkpointing.find_latest_checkpoint(output_dir).name == "checkpoint-500"
    )
    removed = checkpointing.prune_checkpoints(output_dir, keep=3)
    assert len(removed) == 2
    remaining = sorted(p.name for p in output_dir.iterdir())
    assert remaining == ["checkpoint-300", "checkpoint-400", "checkpoint-500"]
    assert checkpointing.find_latest_checkpoint(tmp_path / "empty") is None


def test_training_manifest_roundtrip(tmp_path: Path):
    output_dir = tmp_path / "run"
    path, manifest = checkpointing.write_training_manifest(
        output_dir,
        config={"base_model": "Qwen/Qwen3.8-27B"},
        seed=42,
        extra={"training_method": "QLoRA-SFT"},
    )
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["seed"] == 42
    assert loaded["training_method"] == "QLoRA-SFT"
    assert loaded["training_config_hash"] is not None
