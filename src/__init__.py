"""Qwen3.8-27B-akarui: a competition-solving fine-tune of Qwen/Qwen3.8-27B.

Subpackages:
    src.data: competition discovery, trajectory datasets, validation.
    src.training: model loading, SFT / DPO / GRPO training, checkpointing.
    src.evaluation: benchmarks, leakage detection, regression tests.
    src.publishing: Kaggle publication (implemented separately).
    src.agent: researcher / executor agent components (implemented separately).

This package __init__ is intentionally light: it imports nothing heavy so
that ``import src`` works on a machine without torch / transformers / trl.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
