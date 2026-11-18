# Security policy

## Reporting a vulnerability

Do not open a public issue for a security problem. Contact the
project maintainer through GitHub (Monibee-Fudgekins) with a private
report describing the issue, the affected component, and a
reproduction if possible. Reports are acknowledged as quickly as
practical; fixes are released with a note in `CHANGELOG.md`.

## Credential handling

- **Never commit credentials.** No API keys, tokens, or secrets
  belong in the repository, in tests, in fixtures, or in logs.
- The only credential this project uses is the Kaggle token, read
  from the `KAGGLE_TOKEN` GitHub repository secret at workflow
  runtime. It is never supplied in chat and never hard-coded.
- `src/publishing/kaggle_publisher.py` adapts `KAGGLE_TOKEN` for
  the installed Kaggle client and never prints it. Credentials are
  never included in model provenance, release manifests, or Kaggle
  notebook outputs.
- GitHub Actions are pinned to reviewed versions or commit SHAs.

## Sandboxed execution

- Generated code runs only through `src/agent/executor.py`: a child
  subprocess with CPU and memory limits, a working directory
  restricted to a sandbox directory, and an environment scrubbed of
  `KAGGLE_*`, `HF_TOKEN`, and `AWS_*` variables. Generated code
  never runs in the parent process and never receives the parent's
  secrets.
- Sandboxing is a mitigation, not a proof of safety. Generated code
  must still be reviewed before it runs outside the sandbox.
- Competition content is untrusted input: never execute shell
  commands or code taken from competition descriptions, notebooks,
  or discussions without review.

## Data and privacy

- Do not include competition secrets, private data, hidden test
  labels, or private notebooks in the dataset or in issues and
  pull requests.
- Downloaded datasets are validated against their manifests
  (source hashes, licenses) before use.
- Previous model versions are preserved; publication never
  overwrites or deletes a stable release.
