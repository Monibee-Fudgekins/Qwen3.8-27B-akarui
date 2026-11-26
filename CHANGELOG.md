# Changelog

All notable changes to Qwen3.8-27B-akarui are documented here. The
format follows Keep a Changelog; the repository uses semantic
versioning and Kaggle model versions increment per published
variation version.

## [0.2.0](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/compare/v0.1.0...v0.2.0) (2026-11-26)


### Features

* automated evaluate-and-promote lifecycle with gated Kaggle publication ([b1ce3de](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/b1ce3de09ce39e0dfbc1138df6e19c021b31d1e7))
* check workflow advances the akarui cycle every 6h ([20ed878](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/20ed878009d3b2673953273b22976d0f7e3a9f42))
* runtime guard for DPO and GRPO (10h + 600-step cap) ([2dd945b](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/2dd945b38fe7b64308d3a2e8fc8f07776e770a0b))
* submit-only trigger + cycle check workflow support ([15954e5](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/15954e5ed412d44f3537f77f169927e0dd65e5b9))


### Bug Fixes

* add six — kaggle 1.7 imports it but does not declare it (CLI fails to start without it) ([23d68f0](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/23d68f0cb8ea73ed5569baeb1a3042e1d7db1d18))
* ASCII-only eval kernel titles — Kaggle SaveKernel 400'd on the em-dash ([3d2b5b7](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/3d2b5b7b73cf46b68ee16adaca8376d604e45519))
* auto-merge condition checks PR author, not who pushed ([646f935](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/646f93519747c0ec637871950a261f5a1dd42107))
* bump transformers 4.57 -&gt; 5.2.0 (first release with qwen3_5 support) ([a017f11](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/a017f1117753acd7ee25f42a134b9421f6189923))
* kernel titles must slugify to their ids (Kaggle creates kernel under title-derived slug) + resilient status polling ([85e4583](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/85e45835ec245d024d2e517a8ace29219a98ca0c))
* kernels install requirements-kaggle.txt at runtime (bitsandbytes missing on Kaggle) ([53d10e2](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/53d10e24aca3c511872005538a5c6a6ad6bb74b5))
* kernels self-bootstrap from the public repo; eval mode baked at package time ([d8773b8](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/d8773b89be02ab0411e58fb1db12b0104763ff21))
* pin torchvision 0.22 to match torch 2.7 (qwen3_5 needs torchvision::nms) ([babd6bf](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/babd6bf473c4000c9351a12ad31059336afc3d5a))
* quota exhaustion at push or kernel start is a graceful skip, not a failure; diagnostic reports API error field ([7f6fc37](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/7f6fc3772b9ef79d510542533a408c607eb2ec63))
* resolve confirm window at call time so tests can shorten it ([4995fd4](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/4995fd46167289c3a6034f0f02d24d9a60194b4a))
* restore required shell escaping in auto-merge jq filter ([e826a42](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/e826a42344bd4c0250c31678e026d81e38a5068e))
* shorten eval kernel titles to &lt;=50 chars (Kaggle SaveKernel 400) + guard both packagers + tests ([d8e5693](https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui/commit/d8e5693ff4746f20bdaebbe24a1bedb11ddb413c))

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
