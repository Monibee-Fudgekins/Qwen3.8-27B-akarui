# Changelog

All notable changes to Qwen3.8-27B-akarui are documented here. The
format follows Keep a Changelog; the repository uses semantic
versioning and Kaggle model versions increment per published
variation version.

## [Unreleased]

- Nothing pending beyond the 0.1.0 infrastructure release.

## [0.1.0] — Initial infrastructure

- Repository created with the full project structure: `src/data`,
  `src/training`, `src/evaluation`, `src/publishing`, `src/agent`,
  `configs/`, `notebooks/`, `scripts/`, `tests/`, `data/`,
  `evaluation/`, `model_cards/`.
- Training configurations committed (base, SFT, DPO, GRPO,
  evaluation) for QLoRA fine-tuning of the pinned Qwen3.8-27B base
  (revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`).
- Dataset schemas and manifest rules defined; no dataset version
  assembled yet.
- Held-out benchmark defined (12 competitions, initial proposal —
  verify IDs before the first run); baseline results template
  recorded as `not_measured`.
- Publishing, provenance, model-card, release-manifest, researcher,
  executor, experiment-memory, and tool-registry modules
  implemented.
- Weekly GitHub Actions workflow configured (orchestration only;
  training runs on Kaggle via the `KAGGLE_TOKEN` secret).
- **No model has been trained or published yet. No benchmark has
  run. No performance claim is made.**
