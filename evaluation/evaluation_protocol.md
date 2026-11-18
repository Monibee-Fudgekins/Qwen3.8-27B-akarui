# Evaluation protocol

How Qwen3.8-27B-akarui candidates are evaluated, compared against the
champion, and promoted — or not.

## Held-out benchmark

The benchmark is defined in `evaluation/benchmark_manifest.json`
(`akarui-held-out-benchmark-v1`): 12 held-out public Kaggle
competitions covering tabular classification, tabular regression,
computer vision, NLP, and time series. **Training must exclude every
competition in the manifest.** Leakage checks
(`src/evaluation/leakage_detection.py`) verify this before any
evaluation run is accepted.

> The manifest is an initial proposal — verify every competition ID
> against Kaggle before the first run. Each entry now also carries
> `metric_direction` (derived from the metric name, not measured).
> No entry carries a `baseline_score`: baselines are computed
> in-kernel by the solver (below), never invented.

## Solver: how scores are measured

`src/evaluation/solver.py` is the solver; `run_evaluation`
(`src/evaluation/benchmark_runner.py`) builds it from the loaded
model and passes it to `run_benchmark`, and runs the regression
probes with the same generation callable.

- **Generation.** One fixed prompt per competition (task, file
  listing, column schema, target/id columns, metric, output
  contract), greedy decoding (`do_sample=False`), fixed seed 42,
  `max_new_tokens=4096`. Base and candidate get identical prompts.
- **Local scoring, no leaderboard.** Kaggle test labels are hidden
  and the eval kernel never submits to the leaderboard. Each
  competition's *train* file is split 80/20 (seed 42) into a local
  train/val split; the generated solution fits on the local train
  part and predicts the local val part. Time series
  (`store-sales-...`, `bike-sharing-demand`) split
  chronologically (last 20% is val); the rest use a deterministic
  hash-based split, stratified by target for classification.
- **Execution.** Generated code runs only through
  `src/agent/executor.py` (600 s wall-clock timeout, 4 GB
  address-space limit, restricted working directory, environment
  scrubbed of `KAGGLE_*` / `HF_TOKEN` / `AWS_*`). Timeout, crash,
  missing or invalid `predictions.csv` → unmeasured (null scores),
  never a fake score. Limitation, stated honestly: the Executor
  does **not** block network access (the kernel needs the internet
  for the base weights). The prompt forbids network/Kaggle-API use
  and credentials are scrubbed, but network isolation is not
  guaranteed by the sandbox.
- **Metrics.** Accuracy, RMSLE, RMSE, F1, ROC AUC implemented in
  pure stdlib in the solver. `predictions.csv` must carry
  `id,prediction` for exactly the val rows (digit-recognizer has
  no id column: row position is the id).
- **Normalization / baselines.** No baseline numbers are invented.
  The solver computes a fixed deterministic reference on the same
  val split (no model involved) and reports
  `normalized_score = candidate / reference` (higher-is-better)
  or `reference / candidate` (lower-is-better), matching
  `competition_evaluator._normalized_score`.

### Per-competition scoring table

| Competition | Metric | Target | Split | Reference (deterministic) |
| --- | --- | --- | --- | --- |
| titanic | Accuracy | Survived | stratified | Sex rule (female→1, else 0) |
| house-prices-advanced-regression-techniques | RMSLE | SalePrice | random | median SalePrice |
| digit-recognizer | Accuracy | label | stratified | majority digit |
| spaceship-titanic | Accuracy | Transported | stratified | CryoSleep rule (true→1, else 0) |
| playground-series-s3e11 | RMSLE | *(auto-detected: last column)* | random | median target |
| store-sales-time-series-forecasting | RMSLE | sales | chronological (date) | median sales |
| nlp-getting-started | F1 | target | stratified | disaster-keyword rule |
| petfinder-pawpularity-score | RMSE | Pawpularity | random | median Pawpularity |
| tabular-playground-series-jan-2024 | ROC AUC | *(auto-detected: last column)* | stratified | constant 0.5 (AUC 0.5) |
| forest-cover-type-prediction | Accuracy | Cover_Type | stratified | majority class |
| bike-sharing-demand | RMSLE | count | chronological (datetime) | median count |
| commonlitreadabilityprize | RMSE | target | random | median target |

Two target columns are not known from the repo and are
auto-detected at runtime as the last column of `train.csv`
(`playground-series-s3e11`, `tabular-playground-series-jan-2024`);
the detected column is recorded per competition
(`target_column`, `target_detected: true`) in the results — check
it in the first live results before citing those two scores.

### What cannot be scored as designed

Nothing categorically. Three competitions are at elevated risk
on the first live run and may come back unmeasured (gate fails
closed, candidate stays private): **store-sales-...** (3M+ train
rows; a generated solution fitting the full local train part may
hit the 600 s execution timeout), **digit-recognizer** (42k×785
pixels; a slow fit may time out), and **petfinder-...** (image
files in the download; the reference and prompt use the tabular
metadata only, so image signal is unscored). A download failure
(auth, or a competition whose rules have not been accepted on
kaggle.com — that cannot be fixed from the kernel) is likewise
recorded `not_run` with null scores.

### Runtime estimate (first live run, Wed 2026-11-25 08:00 UTC)

Budget for the eval kernel, from the same 30 h/week Kaggle GPU
quota as training (training budgets ~10 h; the weekly trainings
are staggered — akarui Wed, gemma Mon, Glimmer Sat — so the eval
run must fit the remaining quota that week):

| Step | Estimate |
| --- | --- |
| Model load (27B NF4 + optional adapter) | ~10 min |
| Competition downloads (12, incl. petfinder images) | ~10–20 min |
| Generation (12 × ~3 min greedy, 4096 tokens) | ~35 min |
| Reference baselines + split (streaming CSV) | ~10 min |
| Solution execution (12 × ≤10 min cap) | ~20 min typical, ≤2 h worst case |
| **Total** | **~1.5 h typical, ~3.5 h worst case** |

The worst case assumes every solution runs to its timeout; the
typical case assumes most finish in 1–2 min. If the kernel is
killed or errors, no results file exists, the gate fails closed,
and the candidate stays private. The first live run measures the
real numbers; update this table with them.

## Configurations compared

Every evaluation run scores three configurations on the same
competitions:

- **A — Original Qwen3.8-27B.** The pinned base model, unmodified.
- **B — Qwen3.8-27B-akarui.** The candidate adapter on the pinned base.
- **C — Qwen3.8-27B-akarui with tools.** The candidate plus the agent
  executor and experiment memory.

## Metrics

| Metric | Measurement |
| --- | --- |
| Competition performance | Normalised score relative to the baseline |
| Submission validity | Successful submission percentage |
| Cross-validation quality | Fold and seed consistency |
| Experiment efficiency | Useful experiments per GPU-hour |
| Generalisation | Performance on unseen competitions |
| Code reliability | Successful execution rate |
| Rule compliance | Leakage and prohibited-data checks passed |
| Regression | Retention of base-model capabilities |

Results are stored as versioned artifacts in `evaluation/`
(`baseline_results.json` is the honest template: every measured
field is `null` and the status is `not_measured` until the benchmark
has actually run).

## Promotion gates

A candidate replaces the champion **only if all of the following
hold**:

1. The benchmark ran to completion on all 12 held-out competitions
   for configurations A, B, and C.
2. Leakage checks pass: no held-out competition, hidden test label,
   or private data appears in the candidate's training data.
3. Configuration B beats the current champion on competition
   performance (normalised score) by more than the measurement
   noise of the benchmark. A tie is not an improvement.
4. No unacceptable regression in base-model capabilities: the
   regression suite (`src/evaluation/regression_tests.py`) must
   pass its thresholds for reasoning, coding, instruction following,
   and (where supported) vision-language behaviour.
5. Submission validity and code reliability do not drop below the
   champion's values.
6. Provenance is complete: the release manifest validates with no
   errors (see `src/publishing/provenance.py`).

## What counts as improvement

Improvement means a measured, reproducible gain on the held-out
benchmark under this protocol — not a single competition, not a
training-set score, and not an LLM's opinion of the output. The
comparison is always candidate vs the current champion on the same
benchmark version, same metrics, same seeds.

## What happens to a failing candidate

A candidate that does not pass the gates **stays unpublished** or is
marked experimental. It is never promoted because training
completed, and it never overwrites the previous stable release. The
evaluation results are still recorded in the release history so the
failure is visible and the experiment is not silently repeated.
Every gate decision is recorded in `releases/gate-decisions/`
(`promote` | `stay_private` | `blocked` | `skipped`, with both
scores, sample size, champion version, and timestamp); the weekly
workflow commits these records back to the repo.

## Where evaluation runs

Evaluation needs the 27B model loaded on a GPU, so it runs in a
dedicated Kaggle eval kernel — not on the GitHub-hosted runner
(no GPU) and not in the training kernel's tail (the training run is
already budgeted to ~10h of Kaggle's 12h kill limit, with no room
for a 12-competition eval). The eval kernel is packaged by
`scripts/package_eval_kernel.py` (notebook 07; candidate mode mounts
the training kernel's output as a kernel input, base mode evaluates
the pinned base model) and triggered/polled with the same
`scripts/trigger_kaggle.py` pattern as training. After it
completes, the runner downloads `output/evaluation-results.json`
and `scripts/evaluate_and_promote.py` runs the gate; only a pass
publishes. The eval kernel draws from the same 30h/week Kaggle GPU
quota as training, and its runtime is estimated in the solver
section above (~1.5 h typical, ~3.5 h worst case) — unmeasured
until the first live run (Wed 2026-11-25 08:00 UTC). If it is
killed or errors, no results file exists, the gate fails closed,
and the candidate stays private.

## Baseline measurement

The champion is the last promoted version's score
(`evaluation/champion_results.json`). Before any promotion exists,
the reference is the base model's measured score
(`evaluation/baseline_results.json`). That file ships as an honest
not-measured template; **no promotion can pass until it is
measured**. Measurement: run the eval kernel in base mode
(`package_eval_kernel.py --mode base`), download its
`output/evaluation-results.json`, and run
`scripts/record_baseline.py --evaluation <that file>`, which writes
the canonical measured record and refuses unmeasured input or an
already-measured baseline (without `--force`). The first candidate
must then beat that measured baseline score by at least
`min_normalized_score_improvement` — the gate compares against the
baseline, it does not wave the first promotion through.

## Regression policy

Base-model capabilities are part of the release gates, not an
afterthought. If a candidate improves competition scores but
materially degrades general reasoning, coding, or
instruction-following, it does not ship. The regression thresholds
live in `configs/evaluation.yaml`; a candidate that fails them is
treated the same as a candidate that fails the benchmark.
