# Release history

Versioned history of Qwen3.8-27B-akarui releases. Kaggle model
versions increment with each published variation version; repository
versions use semantic versioning.

## 0.1.0 — Initial infrastructure (unreleased model)

- Repository created; project structure, configurations, and CI
  implemented.
- No model has been trained or published yet.
- No dataset version has been assembled yet.
- No baseline or benchmark results have been measured.

## Future releases

- **0.2.0:** First validated SFT adapter (planned; requires a
  completed training run, a passing evaluation, and a complete
  provenance manifest).
- **0.3.0:** First DPO-enhanced candidate (planned).
- **1.0.0:** First stable competition-solving release (planned).

A release is added here only when it has actually shipped. Weekly
candidates that fail the promotion gates are not listed as releases;
their gate decisions are recorded in `releases/gate-decisions/` and
the champion (once one exists) in `evaluation/champion_results.json`.
