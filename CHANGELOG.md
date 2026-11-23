# Changelog

All notable changes to Qwen3.8-27B-akarui are documented here. The
format follows Keep a Changelog; the repository uses semantic
versioning and Kaggle model versions increment per published
variation version.

## [Unreleased]

- torchvision ~=0.22.0 added (both requirements files): pairs with the
  pinned torch 2.7 (torchvision 0.22.0 requires torch==2.7.0). The
  base-mode eval kernel v7 failed at model load with
  `RuntimeError: operator torchvision::nms does not exist` — transformers
  5.2's qwen3_5 modeling imports torchvision and its nms op must be
  registered, but torchvision was unpinned and mismatched the pinned
  torch, so its C++ ops never loaded and
  `Qwen3_5ForConditionalGeneration` was unimportable.
- transformers 4.57.0 -> 5.2.0 (both requirements files): 5.2.0 is the
  first release supporting model_type `qwen3_5` (verified against the
  5.1.0/5.2.0 tags: 5.1.0 has no qwen3_5 model, 5.2.0 does). The
  base-mode eval kernel v6 failed at model load with
  `KeyError: 'qwen3_5'` because 4.57 predates the architecture. The
  other pins (trl 0.23, peft 0.17, accelerate 1.10, torch 2.7,
  bitsandbytes 0.48, datasets 4.4) place no upper bound on
  transformers, so no companion bump was needed.
- Publish-path hardening (first live publish is Wed 2026-11-25 08:00 UTC):
  the version-create handle now uses the BARE variation
  (``monibeefudgekins/qwen3-8-27b-akarui/pytorch/qlora-sft``), never
  the release label (``qlora-sft-v1``) — Kaggle treats the 4th handle
  component as the instance slug, so the label would have created a
  new instance per release. The kagglehub fallback now passes the
  4-part handle kagglehub 0.3.12 `parse_model_handle` requires (the
  2-part `owner/model` form raises ValueError). The CLI path ensures
  the instance exists before creating a version (`kaggle models
  instances get`, then `kaggle models instances create -p <dir>`
  from a `model-instance-metadata.json` only when missing), so the
  first publish of a variation works; after `kaggle models create`
  (new model shell only) the same ensure + version create follows.
  kagglehub needs no separate instance step — its `model_upload`
  creates a missing model and instance itself.
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
