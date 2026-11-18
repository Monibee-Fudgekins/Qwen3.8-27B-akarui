# Training data

## What the model learns from

Qwen3.8-27B-akarui learns from competition-solving trajectories
rather than memorised notebooks: structured records of a competition
task, the reasoning behind an approach, the code that was written,
what happened when it ran, how it was debugged, and the measured
validation score.

## Dataset versions

No dataset version has been assembled yet. When one is, it is
recorded here with its version, record counts, source list, and
SHA-256 hash, and its manifest is stored under `data/manifests/`.

## Sources

Only legally accessible, appropriately licensed material:

- Kaggle public competition descriptions.
- Public competition datasets where redistribution and training use
  are permitted.
- Public notebooks and solution discussions where licensing permits.
- Public GitHub machine-learning repositories with compatible licenses.
- Hugging Face datasets with compatible terms.
- Internally generated synthetic competition tasks.
- Original execution trajectories produced by Akarui.

Each source is recorded in the dataset manifest with its URL,
license, SHA-256, collection date, and filtering decisions.

## Categories

Initial sampling targets (not hard quotas):

| Category | Target |
| --- | --- |
| Competition analysis and planning | 20% |
| Programming and debugging | 20% |
| Feature engineering and modeling | 20% |
| Validation and statistics | 15% |
| Experiment optimization | 10% |
| Ensembling and post-processing | 10% |
| Competition submission workflows | 5% |

## Exclusions

The following are **never** included in training data:

- Private competition data or hidden test labels.
- Private notebooks or leaked solutions.
- Restricted datasets.
- Future leaderboard information.
- The 12 held-out benchmark competitions listed in
  `evaluation/benchmark_manifest.json`.

Leakage checks (`src/evaluation/leakage_detection.py`) verify the
held-out competitions are absent before any evaluation run is
accepted.

## Format

One JSON object per line (JSONL), conforming to
`data/schemas/trajectory.schema.json`. See `data/README.md` for the
field list, manifest rules, and licensing rules.
