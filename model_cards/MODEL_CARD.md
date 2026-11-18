# Qwen3.8-27B-akarui

## Model identity

- **Model name:** Qwen3.8-27B-akarui
- **Base model:** Qwen/Qwen3.8-27B (revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0`)
- **Base model license:** Apache 2.0
- **Base model page:** https://huggingface.co/Qwen/Qwen3.8-27B
- **Base model repository:** https://github.com/QwenLM/Qwen3.8
- **Fine-tune name:** Akarui
- **Architecture:** Dense vision-language transformer
  (`Qwen3_5ForConditionalGeneration`, model type `qwen3_5`)
- **Model size:** 27B parameters
- **Context length:** 262,144 native tokens
- **Framework:** PyTorch
- **Project repository:** https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui
- **Published on:** Kaggle Models (owner: monibeefudgekins)
- **Release version:** Not yet measured — no model version has been published yet.

## Intended use

Qwen3.8-27B-akarui is a competition-solving fine-tune of Qwen3.8-27B,
developed to improve autonomous data science workflows: analysing
unfamiliar Kaggle competitions, developing machine-learning strategies,
writing executable solutions, evaluating experiments, debugging, and
improving competition performance.

It is designed to operate as a research-oriented competition agent,
learning from verified competition-solving trajectories, executable
experiments, validation results and preference optimisation.

**Out of scope:** general-purpose chat, medical or legal advice, and
any use that requires private competition data or hidden test labels.
The model must not be used to train on private competition data,
leaked solutions, or restricted datasets.

## Training method

- **Stage 1 — Supervised fine-tuning:** QLoRA, 4-bit NF4, double
  quantisation, BF16 compute, LoRA rank 32, alpha 64, dropout 0.05,
  learning rate 1e-4 (initial experimental value), cosine schedule,
  3% warmup, gradient checkpointing.
- **Stage 2 — Direct Preference Optimisation:** preference pairs built
  only from verified execution results; never from another LLM's
  opinion alone.
- **Stage 3 — GRPO (experimental):** candidate solutions executed in
  isolation and rewarded on measured performance. Not required for
  the initial public release.

Training starts from the pinned Qwen3.8-27B base revision on every
cycle; weekly candidates are trained from the pinned base on the
current approved dataset, never by blindly fine-tuning the previous
week's adapter.

## Architecture (base model, verified)

- Text transformer: hidden size 5120, 64 layers, 24 attention heads,
  4 key-value heads, intermediate size 17408, vocabulary 248320,
  max position embeddings 262144.
- Vision tower: hidden size 1152, patch size 16.
- The original tokenizer and chat template are preserved; the
  architecture is not modified. Vision-language and native
  reasoning/thinking functionality are retained where the training
  implementation supports them.

## Training data

Not yet measured — no training dataset version has been assembled
yet. The dataset will be a curated, versioned corpus of
competition-solving trajectories (JSONL), built only from legally
accessible, appropriately licensed sources. Full documentation:
`model_cards/TRAINING_DATA.md`. Private competition data, hidden test
labels, private notebooks, leaked solutions and restricted datasets
are excluded.

## Evaluation

- **Held-out benchmark:** Not yet measured. The benchmark is defined
  in `evaluation/benchmark_manifest.json` (12 held-out competitions;
  initial proposal — verify IDs before the first run).
- **Baseline (original Qwen3.8-27B):** Not yet measured.
- **Fine-tuned model results:** Not yet measured.
- **Regression checks (base-model capabilities):** Not yet measured.

A candidate is promoted only when it improves on the current champion
under the evaluation protocol in `evaluation/evaluation_protocol.md`
without unacceptable regressions. A candidate that does not improve
stays unpublished or is marked experimental.

## Adapter vs merged release

- **Adapter-only release (e.g. `qlora-sft`):** LoRA adapter weights
  plus configuration and tokenizer references. Requires the
  Qwen/Qwen3.8-27B base model to run; it is not a standalone model.
- **Merged release (e.g. `merged-bf16`):** full inference-ready
  weights, published only when hardware and storage budget support it.
- **Training checkpoint:** an intermediate training state, not a release.
- **Inference-ready release:** a merged release, or an adapter release
  paired with the pinned base model and the loading instructions in
  the README.

## Limitations

See `model_cards/LIMITATIONS.md`. In short: no model has been
published yet, no benchmark has run, and no capability-retention
claim can be made until regression testing confirms it.

## Provenance

Every release ships a machine-readable provenance manifest
(`model_cards/PROVENANCE.md` and `provenance.json` in the uploaded
variation) recording the base revision, dataset version and hash,
config hash, source commit, GPU configuration, evaluation results
and artifact hashes.

## Citation

See `CITATION.cff`. Credit the Qwen development team for the
original model; this project is the fine-tuning work built on top
of it.
