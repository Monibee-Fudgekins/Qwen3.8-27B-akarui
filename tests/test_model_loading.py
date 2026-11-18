"""Tests for model loading: pinned revision, memory estimates, LoRA config.

Runs without torch/transformers/GPU. Heavy-library functions must either
work from pure-Python estimates or fail with a clear, actionable error.
"""

from __future__ import annotations

import pytest

from src.training import model_loader

PINNED_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"


def test_pinned_revision_constant():
    assert model_loader.BASE_MODEL == "Qwen/Qwen3.8-27B"
    assert model_loader.BASE_REVISION == PINNED_REVISION
    assert model_loader.EXPECTED_MODEL_TYPE == "qwen3_5"


def test_estimate_memory_single_l4():
    config = {
        "lora": {"r": 32},
        "training": {"max_seq_length": 4096},
    }
    estimate = model_loader.estimate_memory(config)
    # 27B params at 4-bit: ~13.5 GB of weights.
    assert estimate["weights_gb"] == pytest.approx(13.5, abs=0.5)
    assert estimate["trainable_params_estimate"] > 100_000_000
    assert estimate["fits_single_l4_24gb"] is True
    assert estimate["reasoning"]


def test_lora_config_or_clear_error():
    config = {
        "lora": {
            "r": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "task_type": "CAUSAL_LM",
        }
    }
    try:
        import peft  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="peft"):
            model_loader.get_lora_config(config)
    else:
        lora_config = model_loader.get_lora_config(config)
        assert lora_config.r == 32
        assert lora_config.lora_alpha == 64
        assert set(lora_config.target_modules) == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }


def test_load_base_model_fails_cleanly_without_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            model_loader.load_base_model({"base_model": "Qwen/Qwen3.8-27B"})
