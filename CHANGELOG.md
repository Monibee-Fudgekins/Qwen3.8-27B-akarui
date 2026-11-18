# Changelog

All notable changes to Qwen3.8-27B-akarui are documented here. The
format follows Keep a Changelog; the repository uses semantic
versioning and Kaggle model versions increment per published
variation version.

## [Unreleased]

- Kernel packaging: `scripts/package_kernel.py` assembles a
  self-contained Kaggle kernel directory (`kernels/sft/`: notebook 05
  + `src/` + `scripts/` + `configs/` + `data/train.jsonl` +
  `kernel-metadata.json` for `monibeefudgekins/akarui-sft-training`);
  it refuses to package when the training file is missing, empty,
  malformed, unusable, or names a held-out benchmark competition.
  Base weights are loaded from Hugging Face in-kernel at the pinned
  revision and quantized to 4-bit NF4 there (no pre-quantized repo;
  a vLLM-only MXFP4 variant was checked and not adopted).
- Trigger: `scripts/trigger_kaggle.py` now defaults to the packaged
  kernel dir, auto-packages when it is missing, and exits 0 with a
  `skipped_quota` run record when the weekly GPU quota refuses the
  push (other push failures still exit 1).
- 11h guard: SFT runs are bounded by `runtime.max_runtime_hours`
  (default 10h) and `training.max_steps` (600); on wall-clock
  expiry training stops gracefully with a checkpoint and a
  `partial_complete` record, exit 0. `prepare_training.py` resolves
  and validates the guard into the training plan.
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
