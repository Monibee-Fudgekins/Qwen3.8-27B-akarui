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
5. Do not hand-edit `CHANGELOG.md` — releases are automated (see
   "Commit messages" below); release-please generates the changelog
   from commit messages.
6. Describe what you changed, why, and what you measured.

## Commit messages

All commits use [Conventional Commits](https://www.conventionalcommits.org/),
because releases are automated with release-please: the version bump
and the changelog entry for a release are derived from the commit
messages since the previous release.

- `feat:` — a new feature; triggers a **minor** version bump.
- `fix:` and `perf:` — a bug fix or a performance improvement;
  triggers a **patch** version bump.
- `feat!:` or `fix!:` (or a `BREAKING CHANGE:` footer) — a breaking
  change; triggers a **major** version bump.
- `docs:`, `chore:`, `ci:`, `test:`, `refactor:` — no version bump
  on their own and no changelog entry.
- A commit that does **not** follow this format produces no release
  and no changelog entry, so keep the prefix on every commit,
  including documentation and housekeeping changes.

Examples:

```
feat: add 11h runtime guard to SFT training
fix: refuse kernel packaging when train.jsonl is missing
docs: document conventional commits for release-please
```

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
