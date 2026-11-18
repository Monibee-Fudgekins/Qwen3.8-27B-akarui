"""Model-card rendering for Qwen3.8-27B-akarui.

:func:`render_model_card` fills the model-card template from a
context dict. Only verified values are rendered as facts; anything
not yet measured is rendered as "Not yet measured" — the template
never invents a score, a benchmark result, or a capability claim.
"""

from __future__ import annotations

_NOT_MEASURED = "Not yet measured"

_TEMPLATE = """# Qwen3.8-27B-akarui

## Model identity

- **Model name:** {model_name}
- **Base model:** {base_model} (revision `{base_revision}`)
- **Base model license:** Apache 2.0
- **Base model page:** https://huggingface.co/Qwen/Qwen3.8-27B
- **Base model repository:** https://github.com/QwenLM/Qwen3.8
- **Fine-tune name:** Akarui
- **Architecture:** Dense vision-language transformer (Qwen3_5ForConditionalGeneration)
- **Model size:** 27B parameters
- **Context length:** 262,144 native tokens
- **Framework:** PyTorch
- **Project repository:** https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui
- **Published on:** Kaggle Models (owner: monibeefudgekins)
- **Release version:** {release_version}

## Intended use

Qwen3.8-27B-akarui is a competition-solving fine-tune of Qwen3.8-27B, developed to
improve autonomous data science workflows: analysing unfamiliar Kaggle competitions,
developing machine-learning strategies, writing executable solutions, evaluating
experiments, debugging, and improving competition performance.

It is designed to operate as a research-oriented competition agent, learning from
verified competition-solving trajectories, executable experiments, validation results
and preference optimisation.

**Out of scope:** general-purpose chat, medical or legal advice, and any use that
requires private competition data or hidden test labels. The model must not be used
to train on private competition data, leaked solutions, or restricted datasets.

## Training method

- **Stage 1 — Supervised fine-tuning:** QLoRA, 4-bit NF4, double quantisation,
  BF16 compute, LoRA rank 32, alpha 64, dropout 0.05, learning rate 1e-4 (initial
  experimental value), cosine schedule, 3% warmup, gradient checkpointing.
- **Stage 2 — Direct Preference Optimisation:** preference pairs built only from
  verified execution results; never from another LLM's opinion alone.
- **Stage 3 — GRPO (experimental):** candidate solutions executed in isolation and
  rewarded on measured performance. Not required for the initial public release.

Training starts from the pinned Qwen3.8-27B base revision on every cycle; weekly
candidates are trained from the pinned base on the current approved dataset, never
by blindly fine-tuning the previous week's adapter.

## Training data

{training_data_summary}

Full dataset documentation: `model_cards/TRAINING_DATA.md`. The dataset is a
curated, versioned corpus of competition-solving trajectories (JSONL), built only
from legally accessible, appropriately licensed sources. Private competition data,
hidden test labels, private notebooks, leaked solutions and restricted datasets are
excluded.

## Evaluation

- **Held-out benchmark:** {benchmark_summary}
- **Baseline (original Qwen3.8-27B):** {baseline_summary}
- **Fine-tuned model results:** {evaluation_summary}
- **Regression checks (base-model capabilities):** {regression_summary}

A candidate is promoted only when it improves on the current champion under the
evaluation protocol in `evaluation/evaluation_protocol.md` without unacceptable
regressions. A candidate that does not improve stays unpublished or is marked
experimental.

## Adapter vs merged release

- **Adapter-only release (e.g. `qlora-sft`):** LoRA adapter weights plus
  configuration and tokenizer references. Requires the Qwen/Qwen3.8-27B base
  model to run; it is not a standalone model.
- **Merged release (e.g. `merged-bf16`):** full inference-ready weights, published
  only when hardware and storage budget support it.
- **Training checkpoint:** an intermediate training state, not a release.
- **Inference-ready release:** a merged release, or an adapter release paired
  with the pinned base model and the loading instructions in the README.

## Limitations

{limitations_summary}

See `model_cards/LIMITATIONS.md` for the full list.

## Provenance

Every release ships a machine-readable provenance manifest
(`model_cards/PROVENANCE.md` and `provenance.json` in the uploaded variation)
recording the base revision, dataset version and hash, config hash, source
commit, GPU configuration, evaluation results and artifact hashes.

## Citation

See `CITATION.cff`. Credit the Qwen development team for the original model;
this project is the fine-tuning work built on top of it.
"""


def render_model_card(context: dict) -> str:
    """Render the model card markdown from ``context``.

    Any missing or ``None`` value is rendered as "Not yet measured",
    so the card can be generated before the first training run without
    inventing results. Recognised keys: ``model_name``,
    ``base_model``, ``base_revision``, ``release_version``,
    ``training_data_summary``, ``benchmark_summary``,
    ``baseline_summary``, ``evaluation_summary``,
    ``regression_summary``, ``limitations_summary``.
    """
    values = {
        "model_name": context.get("model_name") or "Qwen3.8-27B-akarui",
        "base_model": context.get("base_model") or "Qwen/Qwen3.8-27B",
        "base_revision": context.get("base_revision") or _NOT_MEASURED,
        "release_version": context.get("release_version") or _NOT_MEASURED,
        "training_data_summary": context.get("training_data_summary") or _NOT_MEASURED,
        "benchmark_summary": context.get("benchmark_summary") or _NOT_MEASURED,
        "baseline_summary": context.get("baseline_summary") or _NOT_MEASURED,
        "evaluation_summary": context.get("evaluation_summary") or _NOT_MEASURED,
        "regression_summary": context.get("regression_summary") or _NOT_MEASURED,
        "limitations_summary": context.get("limitations_summary") or _NOT_MEASURED,
    }
    return _TEMPLATE.format(**values)
