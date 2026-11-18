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
> against Kaggle before the first run.

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

## Regression policy

Base-model capabilities are part of the release gates, not an
afterthought. If a candidate improves competition scores but
materially degrades general reasoning, coding, or
instruction-following, it does not ship. The regression thresholds
live in `configs/evaluation.yaml`; a candidate that fails them is
treated the same as a candidate that fails the benchmark.
