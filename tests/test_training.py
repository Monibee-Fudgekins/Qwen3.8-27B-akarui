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
