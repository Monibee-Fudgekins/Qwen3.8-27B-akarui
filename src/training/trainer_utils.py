"""Shared training utilities: config loading, seeding, device info,
batch-size math, and chat-template formatting.

Heavy libraries (torch, transformers, ...) are imported lazily inside
the functions that need them, so importing this module never requires a
GPU stack.
"""

from __future__ import annotations

import random
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Load a YAML config file with :func:`yaml.safe_load`.

    Raises :class:`FileNotFoundError` when the file does not exist and
    :class:`ValueError` when it does not parse into a mapping.
    """
    import yaml  # lazy: PyYAML is only needed when a config is read

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config file {path} did not parse into a mapping")
    return config


def set_seed(seed: int) -> None:
    """Seed Python, NumPy (if installed) and PyTorch (if installed)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device_info() -> dict:
    """Describe the available compute device(s).

    Returns a dict that is always safe to record in a manifest. When
    torch is unavailable or CUDA is absent, the result is a CPU-only
    dict with ``cuda_available: False`` -- never an exception.
    """
    info: dict = {"cuda_available": False, "device": "cpu", "gpu_count": 0}
    try:
        import torch
    except ImportError:
        info["error"] = "torch not installed"
        return info
    if not torch.cuda.is_available():
        info["error"] = "CUDA not available"
        return info
    count = torch.cuda.device_count()
    gpus = []
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1e9, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
        )
    info.update(
        {
            "cuda_available": True,
            "device": "cuda",
            "gpu_count": count,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpus": gpus,
        }
    )
    return info


def compute_effective_batch_size(
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_devices: int = 1,
) -> int:
    """Effective batch size = per-device batch x grad-accum x devices."""
    if per_device_train_batch_size < 1:
        raise ValueError("per_device_train_batch_size must be >= 1")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if num_devices < 1:
        raise ValueError("num_devices must be >= 1")
    return per_device_train_batch_size * gradient_accumulation_steps * num_devices


def format_messages_for_training(tokenizer, messages: list[dict]) -> str:
    """Render chat messages with the model's own chat template.

    The original tokenizer and chat template are used exactly as shipped
    -- native reasoning / thinking blocks are kept intact and are never
    stripped or rewritten. Raises :class:`ValueError` when the tokenizer
    has no chat template.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "tokenizer has no chat template; refusing to invent one -- "
            "the original Qwen chat template must be preserved"
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
