# Packaged SFT kernel — `kernels/sft/`

Build artifact, not a commit. Rebuilt by `scripts/package_kernel.py`
(and by `scripts/trigger_kaggle.py` when it is missing) from the repo
at push time. Do not hand-edit files here; edit the repo sources and
re-package. Exception: this README is committed documentation of the
directory and is preserved across rebuilds by the packager.

## Layout

```
kernels/sft/
├── 05_sft_training.ipynb   # code_file: the SFT notebook
├── kernel-metadata.json    # id monibeefudgekins/akarui-sft-training
├── packaging-manifest.json # train-file SHA, row count, base revision
├── src/                    # training, data, evaluation, agent code
├── scripts/                # prepare_training.py, train.py, ...
├── configs/                # base.yaml, sft.yaml, ...
├── data/train.jsonl        # training file (copy; pushed separately)
└── requirements-kaggle.txt # reference for the Kaggle environment
```

The notebook runs from this directory as its working directory, so
its relative imports (`sys.path.insert(0, ".")`,
`scripts/prepare_training.py --config configs/sft.yaml`,
`config["dataset"]["train_file"]` = `data/train.jsonl`) resolve here
exactly as they do at the repo root.

## How the kernel gets the 27B base weights

The kernel downloads the **full base model from Hugging Face
in-kernel** and quantizes it there. No pre-quantized weights are
bundled, attached, or fetched from anywhere else.

* Base: `Qwen/Qwen3.8-27B`, pinned revision
  `37561816e66b5b9153b1b64c10968c18c1e6f4a0` (verified to exist on
  Hugging Face; Apache 2.0, no access approval needed).
* `src/training/model_loader.py` loads it with
  `AutoModelForImageTextToText.from_pretrained(..., revision=<pinned>)`
  plus a BitsAndBytes NF4 double-quantization config (BF16 compute),
  i.e. the 4-bit weights are produced in-kernel from the full BF16
  weights (~54 GB download, ~13.5 GB in memory). The tokenizer is
  loaded from the same pinned revision, unmodified.
* `kernel-metadata.json` therefore sets `enable_internet: true` --
  without it the download fails. No dataset or model attachment is
  needed for the base weights.

This follows the sibling Glimmer kernel's pattern
(`monibeefudgekins/fine-tune-muse-glimmer-30b-for-stock-trading`,
`scripts/prepare_kernel.py` in `glimmer-stock-trader`): the kernel
loads its base from Hugging Face in-kernel at run time; the only
difference is that Glimmer attaches its training data as a Kaggle
dataset while Akarui ships `data/train.jsonl` inside the kernel
directory (small enough to bundle; the dataset is versioned in the
repo, not in a separate Kaggle dataset).

### Quantized variants considered and not used

A pre-quantized HF variant was checked and deliberately **not**
adopted: `RedHatAI/Qwen3.8-27B-MXFP4xFP8` exists on Hugging Face
(mixed MXFP4/FP8 weights, quantized from `Qwen/Qwen3.8-27B`) but its
card targets vLLM inference (requires vLLM 0.32.0+); it is not a
Transformers training load and its mixed-precision format is not
what the QLoRA path (BitsAndBytes NF4 + PEFT) expects. Using it
would trade a verified in-kernel quantization for an unverified
third-party weight format. If a future quantized variant is ever
adopted, it must be verified to load under `transformers` +
`bitsandbytes` and to match the pinned base revision before use --
never assumed from its name.

## Running it

```bash
# 1. Package (refuses when data/train.jsonl is missing/invalid)
python scripts/package_kernel.py --out kernels/sft

# 2. Trigger (auto-packages when kernels/sft is missing; polls until
#    the kernel reports complete; quota skip exits 0 with a record)
python scripts/trigger_kaggle.py --run-record output/kaggle-run.json
```

## 11h guard

The SFT run is bounded by `runtime.max_runtime_hours` (default 10h)
and `training.max_steps` (600) -- see
`src/training/runtime_guard.py` for the arithmetic. If the wall
clock runs out, training stops gracefully, a checkpoint is saved,
`output/partial_complete.json` is written, and the run exits 0 with
`status="partial_complete"`. Resume with
`python scripts/train.py --config configs/sft.yaml --stage sft --resume`.
