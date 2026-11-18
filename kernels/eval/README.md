# Packaged eval kernel — `kernels/eval/`

Build artifact, not a commit. Rebuilt by `scripts/package_eval_kernel.py`
(and by the weekly workflow at trigger time) from the repo. Do not
hand-edit files here; edit the repo sources and re-package. Exception:
this README is committed documentation of the directory and is
preserved across rebuilds by the packager.

## Layout

```
kernels/eval/
├── 07_evaluation.ipynb   # code_file: the evaluation notebook
├── kernel-metadata.json  # id monibeefudgekins/akarui-eval-candidate (or -baseline)
├── eval-mode.json        # {"mode": "candidate"|"base"}
├── packaging-manifest.json
├── src/                  # training, data, evaluation, agent code
├── scripts/              # evaluate.py, verify_release.py, ...
├── configs/              # evaluation.yaml, ...
├── evaluation/           # benchmark manifest + protocol
├── data/train.jsonl      # training file copy (leakage check only)
└── requirements-kaggle.txt
```

## Why a separate eval kernel

Evaluation needs the 27B model loaded on a GPU. The GitHub-hosted
runner has no GPU, and the training kernel is already budgeted to
~10h of Kaggle's 12h kill limit with no room for a 12-competition
eval. So evaluation runs in its own Kaggle kernel, triggered and
polled by the same `scripts/trigger_kaggle.py` pattern as training:

```bash
# Candidate mode: evaluate the trained adapter
python scripts/package_eval_kernel.py --mode candidate --out kernels/eval
python scripts/trigger_kaggle.py --notebook-dir kernels/eval \
    --kernel monibeefudgekins/akarui-eval-candidate \
    --run-record output/kaggle-eval-run.json

# Base mode: measure the base model (how the baseline gets measured)
python scripts/package_eval_kernel.py --mode base --out kernels/eval
python scripts/trigger_kaggle.py --notebook-dir kernels/eval \
    --kernel monibeefudgekins/akarui-eval-baseline \
    --run-record output/kaggle-eval-run.json
```

## How the candidate adapter reaches the eval kernel

Candidate mode sets `kernel_sources` to the training kernel
(`monibeefudgekins/akarui-sft-training`), so the training kernel's
output (the adapter) is mounted under `/kaggle/input/`. Notebook 07
resolves the adapter by searching `/kaggle/input/` for
`adapter_config.json` (falling back to `$ADAPTER_DIR`, then
`output/akarui-sft-v0`). The exact mount path is verified on the
first live run (Mon 2026-11-23); if Kaggle mounts it elsewhere, the
glob in notebook 07 is the one place to fix.

## What the kernel writes

`output/evaluation-results.json` — the candidate (or base) evaluation
summary the promotion gate reads. Download it after the kernel
completes (`kaggle kernels output <slug> -p output/eval`), then:

```bash
# Baseline: record the measured base score (first time only)
python scripts/record_baseline.py --evaluation output/eval/evaluation-results.json

# Candidate: gate, and promote only on a pass
python scripts/evaluate_and_promote.py \
    --evaluation output/eval/evaluation-results.json \
    --provenance output/provenance.json --adapter output/adapter
```

## Quota

The eval kernel draws from the same 30h/week Kaggle GPU quota as
training. Its runtime is unmeasured until the first live run; if the
eval kernel is killed or errors, no results file exists, the gate
fails closed, and the candidate stays private.
