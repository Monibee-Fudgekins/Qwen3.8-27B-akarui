# Contributing to Qwen3.8-27B-akarui

## Scope

Contributions are welcome for training infrastructure, dataset
tooling, evaluation, agent components, documentation, and tests.
Contributions that change training behaviour must include the
evidence for the change (a measured result or a reproducible
failure), not just an opinion.

## How to contribute

1. Fork the repository and create a branch from `main`.
2. Keep changes focused; one concern per pull request.
3. Add or update tests for any code change (`tests/`).
4. Run the test suite and the repository validator before opening
   the pull request.
5. Update `CHANGELOG.md` under the Unreleased section.
6. Describe what you changed, why, and what you measured.

## Data contributions

- Only material whose license permits redistribution and training
  use. Record the exact license in the dataset manifest.
- **Never contribute** private competition data, hidden test labels,
  private notebooks, leaked solutions, or restricted datasets.
- Never add any of the 12 held-out benchmark competitions
  (`evaluation/benchmark_manifest.json`) to training data.
- New trajectories must validate against
  `data/schemas/trajectory.schema.json` and be deduplicated against
  previous dataset versions.

## Code standards

- Python 3.10+, with docstrings on public functions and classes.
- No credentials in code, tests, logs, or fixtures. Secrets are
  read from the environment only.
- Generated code execution stays sandboxed
  (`src/agent/executor.py`); do not weaken the sandbox.
- Do not claim benchmark improvements that have not been measured.

## Reporting issues

Open an issue with a clear reproduction, the exact command run, and
the relevant log excerpt (with any credentials redacted). Security
issues follow `SECURITY.md` — do not open them as public issues.
