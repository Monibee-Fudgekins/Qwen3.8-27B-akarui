# Qwen3.8-27B-akarui

A competition-solving fine-tune of **Qwen/Qwen3.8-27B**, trained on
Kaggle and published to Kaggle Models. Akarui is designed to operate
as a research-oriented competition agent: it analyses unfamiliar
Kaggle competitions, develops machine-learning strategies, writes
executable solutions, evaluates experiments, debugs, and improves
competition performance.

This is **not** a general-purpose chatbot fine-tune.

> **Status: 0.1.0 — initial infrastructure.** No model has been
> trained or published yet, no baseline has been measured, and no
> performance claim is made anywhere in this repository. Everything
> under "Not yet measured" stays that way until a benchmark has
> actually run.

## Model characteristics

| Attribute | Value |
| --- | --- |
| Model family | Qwen3.8 |
| Model size | 27B |
| Fine-tune name | Akarui |
| Architecture | Dense vision-language transformer (`Qwen3_5ForConditionalGeneration`) |
| Primary specialisation | Data science competitions |
| Secondary specialisation | Coding and reasoning |
| Base model | Qwen/Qwen3.8-27B, revision `37561816e66b5b9153b1b64c10968c18c1e6f4a0` |
| Initial method | QLoRA |
| Preference training | DPO |
| Advanced training | GRPO (experimental) |
| Training framework | PyTorch / Transformers / PEFT / TRL |
| Training environment | Kaggle |
| Intended publication | Kaggle Models (owner: monibeefudgekins) |
| License | Apache 2.0 |

Base-model facts (verified against the Qwen3.8-27B configuration):
text transformer hidden size 5120, 64 layers, 24 attention heads,
4 key-value heads, intermediate size 17408, vocabulary 248320,
max position embeddings 262144 (262,144 native context tokens);
vision tower hidden size 1152, patch size 16. The original
tokenizer and chat template are preserved; the architecture is not
modified.

## Intended use

- Autonomous analysis of Kaggle competitions: task type, metric,
  schema, validation strategy.
- Feature engineering, model selection, experiment planning, and
  ensembling.
- Writing and debugging executable Python solutions in a sandbox.
- Learning from verified execution trajectories, not from memorised
  notebooks.

Out of scope: general-purpose chat, medical or legal advice, and any
use involving private competition data or hidden test labels.

## Training methodology

A progressive, evidence-driven system. Each weekly cycle trains a
**candidate from the pinned base model** on the current approved
dataset — never by blindly fine-tuning the previous week's adapter —
and promotes it only if it beats the current champion under the
evaluation protocol.

1. **Stage 1 — SFT (QLoRA).** 4-bit NF4, double quantisation, BF16
   compute, LoRA rank 32, alpha 64, dropout 0.05, learning rate
   1e-4 (initial experimental value), cosine schedule, 3% warmup,
   gradient checkpointing, 8-bit AdamW where compatible. Sequence
   length 2,048–4,096, microbatch 1, gradient accumulation 4
   initially. FSDP-QLoRA only if demonstrably compatible with the
   Qwen architecture and quantisation; otherwise a supported
   distributed configuration is used and its limits documented.
2. **Stage 2 — DPO.** Preference pairs built only from verified
   execution results. Preferred: correct validation, reproducible
   experiments, better feature engineering, effective model
   selection, correct debugging, proper leakage handling, reliable
   submissions. Rejected: demonstrably weaker or invalid approaches.
   Never labelled from another LLM's opinion alone.
3. **Stage 3 — GRPO (experimental).** Candidate solutions executed
   in isolation and rewarded on measured performance. Not required
   for the initial public release.

See `configs/` for the training, DPO, GRPO, and evaluation
configurations.

## Architecture

```
Qwen3.8-27B-akarui/
├── README.md / LICENSE / CITATION.cff / CONTRIBUTING.md / SECURITY.md / CHANGELOG.md
├── .github/workflows/      # weekly training, evaluation, publication, validation
├── configs/                # base, SFT, DPO, GRPO, evaluation configs
├── notebooks/              # 01 environment → 08 Kaggle publication
├── src/
│   ├── data/               # competition collection, dataset building, trajectories,
│   │                       # deduplication, validation
│   ├── training/           # model loading, SFT, DPO, GRPO, checkpointing
│   ├── evaluation/         # benchmark runner, competition evaluator,
│   │                       # leakage detection, regression tests
│   ├── publishing/         # Kaggle publisher, model card, provenance, releases
│   └── agent/              # researcher, executor, experiment memory, tool registry
├── data/                   # dataset README, schemas, manifests
├── evaluation/             # benchmark manifest, baseline results, protocol
├── model_cards/            # model card, provenance, training data, limitations,
│                           # release history
├── scripts/                # GPU detection, training, evaluation, merge, publish
└── tests/                  # data, training, model loading, provenance, publication
```

## Dataset strategy

The model learns from competition-solving trajectories (JSONL records
conforming to `data/schemas/trajectory.schema.json`): competition id,
source URL, dataset provenance, license, task description, input
schema, target definition, evaluation metric, validation strategy,
reasoning summary, generated code, execution environment and output,
errors and debugging attempts, validation score, final outcome,
dataset version, and collection timestamp.

Sources: Kaggle public competition descriptions, public competition
datasets where redistribution and training use are permitted, public
notebooks and discussions where licensing permits, public GitHub ML
repositories with compatible licenses, Hugging Face datasets with
compatible terms, internally generated synthetic tasks, and Akarui's
own execution trajectories.

**Never included:** private competition data, hidden test labels,
private notebooks, leaked solutions, restricted datasets, or future
leaderboard information. The 12 held-out benchmark competitions are
always excluded from training data. Every dataset version ships a
manifest (`data/manifests/`) with source hashes, licenses,
collection dates, and filtering decisions. See `data/README.md` and
`model_cards/TRAINING_DATA.md`.

## Kaggle hardware requirements

Target hardware is **4 × NVIDIA L4 (24GB each, 96GB aggregate)**,
but this is a target, not a guarantee. Two honest notes:

- **Kaggle notebooks typically provide a single GPU**, and the
  weekly GPU quota is 30 hours. Four GPUs are not assumed: the
  training code detects actual hardware (`scripts/detect_gpus.py`,
  `nvidia-smi` plus programmatic detection of GPU count, model,
  compute capability, VRAM, CUDA and PyTorch versions) and selects
  a reduced configuration or fails gracefully when fewer GPUs are
  available.
- **Do not assume four GPUs give 96GB of usable memory to one
  model.** Memory is per-device; a 27B model in 4-bit QLoRA must be
  validated against the hardware actually obtained before a full
  run.

Checkpoints and manifests persist outside the temporary session
filesystem using Kaggle's persistent output and model publication
facilities. Training is resumable after session interruptions.

## Installation

```bash
git clone https://github.com/Monibee-Fudgekins/Qwen3.8-27B-akarui.git
cd Qwen3.8-27B-akarui
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pinned, tested versions — see requirements.lock
```

Python 3.10+. Dependency versions are pinned and locked; the weekly
workflow never blindly installs latest versions.

## Training

```bash
# 1. Detect hardware and validate the environment
python scripts/detect_gpus.py

# 2. Prepare the dataset (validates trajectories + manifest)
python scripts/prepare_training.py --config configs/sft.yaml

# 3. Smoke test first (small run, verifies loading + a training step)
python scripts/train.py --config configs/sft.yaml --smoke-test

# 4. Full SFT run (on Kaggle, via the weekly workflow or notebook 05)
python scripts/train.py --config configs/sft.yaml
```

The GitHub Actions weekly workflow (`weekly-training.yml`,
Mondays 08:00 UTC) only orchestrates: it triggers the Kaggle
training notebook via `scripts/trigger_kaggle.py` using the
`KAGGLE_TOKEN` repository secret, then waits for the run to complete
before evaluation and publication. GitHub-hosted runners never train
the 27B model.

### Kaggle kernel packaging

The training notebook runs on Kaggle from a self-contained kernel
directory built by `scripts/package_kernel.py`:

```bash
# Build kernels/sft/ (notebook + src/ + scripts/ + configs/ + data/train.jsonl)
python scripts/package_kernel.py --out kernels/sft

# Submit and poll (auto-packages when kernels/sft is missing)
python scripts/trigger_kaggle.py --run-record output/kaggle-run.json
```

`package_kernel.py` **refuses** to package when
`data/train.jsonl` is missing, empty, malformed, has no usable
records, or names any of the 12 held-out benchmark competitions.
The packaged directory is a build artifact (git-ignored); see
`kernels/sft/README.md` for its layout and for how the kernel gets
the base weights.

**Base weights:** the kernel downloads the full
`Qwen/Qwen3.8-27B` base from Hugging Face in-kernel at the pinned
revision and quantizes it to 4-bit NF4 there (BitsAndBytes,
`src/training/model_loader.py`); no pre-quantized weights are
bundled. A pre-quantized HF variant
(`RedHatAI/Qwen3.8-27B-MXFP4xFP8`) was checked and not adopted --
it targets vLLM inference, not Transformers training. The kernel
metadata therefore sets `enable_internet: true`.

**Weekly GPU quota:** if `kaggle kernels push` is refused because
the weekly GPU quota is exhausted, the trigger writes a run record
with `status: "skipped_quota"` and exits 0 -- the run never started,
so there is nothing to retry until the quota resets. Any other push
failure exits 1. Only a kernel Kaggle itself reports `complete`
counts as a successful run.

**11h guard:** Kaggle hard-kills notebooks at 12h. SFT runs are
bounded by `runtime.max_runtime_hours` (default 10h, in
`configs/base.yaml`) and `training.max_steps` (600, in
`configs/sft.yaml`); see `src/training/runtime_guard.py` for the
arithmetic (600 steps x worst-case 50 s/step = 8.33h training +
1h setup = 9.33h, 1.67h of margin under the 11h effective cap).
When the wall clock runs out, training stops gracefully, a
checkpoint is saved, `output/partial_complete.json` is written, and
the run exits 0 with `status="partial_complete"` -- resume it with
`--resume`. The 50 s/step planning rate is borrowed from the
sibling Glimmer kernel's worst measured rate, not an Akarui
measurement; the first real Akarui run must measure its own rate
and the cap must be revised from it.

## Inference examples

Loading an adapter release (adapter weights are **not** a standalone
model — the pinned base model is required):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen3.8-27B"
revision = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"

tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    base, revision=revision, torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(model, "monibeefudgekins/Qwen3.8-27B-akarui")

messages = [
    {"role": "user", "content": "Plan a validation strategy for a time-series forecasting competition scored by RMSLE."}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

On Kaggle, notebook `08_kaggle_publication.ipynb` shows the same
flow end to end, and the model page carries usage instructions for
each published variation.

## Kaggle publication

Variations: `qlora-sft` (first), then `qlora-dpo`, `qlora-grpo`,
`merged-bf16` (merged only when hardware and storage allow).
Adapter-only releases are labelled as adapters, never as complete
standalone models. Previous stable versions are never overwritten
or deleted.

```bash
# Build metadata and upload a validated variation version
python scripts/publish_kaggle.py --variation qlora-sft --model-dir ./output/qlora-sft
```

Publication requires a complete provenance manifest; the publisher
refuses to upload without one. Metadata fields follow Kaggle's
supported model metadata format (title, subtitle, description,
licenses, tags, instances). Proposed tags — verify against the
Kaggle model UI before publishing: `qwen`, `qwen3`,
`language-model`, `vision-language-model`, `fine-tuned`, `pytorch`,
`machine-learning`, `data-science`, `kaggle-competition`, `coding`,
`reasoning`, `qlora`, `autonomous-agent`.

## Model provenance

Every release ships a machine-readable provenance manifest
(`provenance.json`) with the base model and pinned revision, training
method and date, dataset version and SHA-256, config hash, source
repository and commit, training environment, GPU configuration,
evaluation results, artifact hashes, and release version. See
`model_cards/PROVENANCE.md` and `src/publishing/provenance.py`.

## Evaluation

The held-out benchmark (`evaluation/benchmark_manifest.json`) is 12
public Kaggle competitions covering tabular classification and
regression, computer vision, NLP, and time series. Three
configurations are compared: **A** original Qwen3.8-27B, **B**
Qwen3.8-27B-akarui, **C** Akarui with execution tools and experiment
memory. Metrics: normalised competition performance, submission
validity, cross-validation quality, experiment efficiency,
generalisation, code reliability, rule compliance, and base-capability
regression.

Promotion gates (`evaluation/evaluation_protocol.md`): a candidate
replaces the champion only when it measurably improves on the
benchmark without unacceptable regressions, with leakage checks
passing and provenance complete. A failing candidate stays
unpublished or is marked experimental.

**Current status: not yet measured.** No baseline, no fine-tuned
results, no regression results exist yet.

## Known limitations

See `model_cards/LIMITATIONS.md`. In short: no model has been
published, no benchmark has run, capability retention is unproven
until regression testing confirms it, GRPO is experimental, and the
4× L4 target is subject to Kaggle runtime availability.

## License

Apache License 2.0 — see `LICENSE`. The base model Qwen3.8-27B is
also Apache 2.0; this project retains full attribution to the Qwen
team.

## Authors

- **Project author:** Monibee-Fudgekins (GitHub) /
  monibeefudgekins (Kaggle).
- **Base model:** the Qwen development team (Qwen / Alibaba) —
  https://huggingface.co/Qwen/Qwen3.8-27B.

## Citation

See `CITATION.cff`. If you use Qwen3.8-27B-akarui, cite this project
and the original Qwen3.8 model.

## Release history

See `CHANGELOG.md` and `model_cards/RELEASE_HISTORY.md`. Current
version: **0.1.0 — initial infrastructure; no model released yet.**
