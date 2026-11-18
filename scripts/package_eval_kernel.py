"""Assemble a Kaggle eval kernel directory for Akarui candidates.

Copies everything notebook ``07_evaluation.ipynb`` needs into one
self-contained directory (default ``kernels/eval/``):

* ``07_evaluation.ipynb`` (the ``code_file`` in kernel-metadata.json)
* ``kernel-metadata.json`` -- kernel slug
  ``monibeefudgekins/akarui-eval-candidate`` (``--mode candidate``)
  or ``monibeefudgekins/akarui-eval-baseline`` (``--mode base``),
  notebook type, GPU on, internet on (the base weights are
  downloaded from Hugging Face in-kernel at the pinned revision)
* copies of ``src/``, ``scripts/``, ``configs/``, ``evaluation/``
  (benchmark manifest + protocol) and ``data/train.jsonl`` (the
  training file, so the notebook's leakage check can run in-kernel).
  ``src/evaluation/solver.py`` (copied with ``src/``) is the
  solver that produces measured scores; ``scripts/kaggle_auth.py``
  is required because the solver downloads competition data with
  the Kaggle CLI in-kernel.
* ``eval-mode.json`` -- ``{"mode": "candidate"|"base"}``, read by
  the notebook to decide whether to evaluate the trained adapter
  or the pinned base model
* ``requirements-kaggle.txt``

Why a separate eval kernel (see evaluation/evaluation_protocol.md):
the training kernel is already budgeted to ~10h of Kaggle's 12h
kill limit, the GitHub-hosted runner has no GPU and cannot load a
27B model, and the eval kernel reuses the trigger/poll pattern of
``scripts/trigger_kaggle.py`` (pass ``--notebook-dir kernels/eval``
and ``--kernel <slug>``). The candidate-mode kernel takes the
training kernel as a ``kernel_sources`` input, so the trained
adapter is mounted at ``/kaggle/input/akarui-sft-training/``; the
notebook resolves the adapter directory from there.

Refusal rules (exit 1, no partial directory left behind):

* notebook ``07_evaluation.ipynb`` is missing.
* the benchmark manifest is missing.
* ``--mode`` is not ``candidate`` or ``base``.

The packaged directory is a build artifact, not a commit: it is
ignored by git (see ``.gitignore``) and rebuilt by the weekly
workflow at trigger time. Never include this directory's contents
in a repo push. The README in ``kernels/eval/`` is the exception:
it is committed documentation and preserved across rebuilds.

No credentials are read, written, or printed by this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

NOTEBOOK_FILE = "07_evaluation.ipynb"
NOTEBOOK_SRC = Path("notebooks") / NOTEBOOK_FILE

KERNEL_IDS = {
    "candidate": "monibeefudgekins/akarui-eval-candidate",
    "base": "monibeefudgekins/akarui-eval-baseline",
}
KERNEL_TITLES = {
    # ASCII only: the Kaggle SaveKernel API rejected the push (HTTP 400)
    # when these titles contained an em-dash. The sibling Glimmer and
    # gemma kernels use ASCII titles and push fine.
    "candidate": "Akarui evaluation - candidate adapter (held-out benchmark)",
    "base": "Akarui evaluation - base model baseline (held-out benchmark)",
}
#: The training kernel whose output (the adapter) the candidate-mode
#: eval kernel mounts as an input.
TRAINING_KERNEL_ID = "monibeefudgekins/akarui-sft-training"

REQUIRED_FILES = [
    NOTEBOOK_SRC,
    Path("scripts/evaluate.py"),
    Path("scripts/verify_release.py"),
    Path("scripts/kaggle_auth.py"),
    Path("src/evaluation/solver.py"),
    Path("configs/evaluation.yaml"),
    Path("evaluation/benchmark_manifest.json"),
    Path("data/train.jsonl"),
    Path("requirements-kaggle.txt"),
]


def build_kernel(repo_root: Path, out_dir: Path, mode: str) -> dict:
    """Assemble the eval kernel directory. Returns a summary dict.

    Raises:
        ValueError: On any refusal (unknown mode, missing required
            files, unsafe output location).
    """
    if mode not in KERNEL_IDS:
        raise ValueError(
            f"refusing to package: unknown eval mode {mode!r}; "
            "expected 'candidate' or 'base'"
        )
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    if out_dir == repo_root:
        raise ValueError("refusing to package: output dir is the repo root")
    if repo_root in out_dir.parents:
        rel = out_dir.relative_to(repo_root)
        if rel.parts[0] != "kernels":
            raise ValueError(
                f"refusing to package: unexpected output location {out_dir}"
            )

    missing = [str(p) for p in REQUIRED_FILES if not (repo_root / p).is_file()]
    if not (repo_root / "src" / "__init__.py").is_file():
        missing.append("src/__init__.py")
    if missing:
        raise ValueError(
            "refusing to package: required file(s) missing from the repo: "
            + ", ".join(missing)
        )

    preserved_readme = None
    readme_path = out_dir / "README.md"
    if readme_path.is_file():
        preserved_readme = readme_path.read_bytes()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copy2(repo_root / NOTEBOOK_SRC, out_dir / NOTEBOOK_FILE)
    for dirname in ("src", "scripts", "configs", "evaluation"):
        shutil.copytree(
            repo_root / dirname,
            out_dir / dirname,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copy2(
        repo_root / "requirements-kaggle.txt",
        out_dir / "requirements-kaggle.txt",
    )
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo_root / "data" / "train.jsonl", out_dir / "data" / "train.jsonl")

    kernel_id = KERNEL_IDS[mode]
    metadata = {
        "id": kernel_id,
        "title": KERNEL_TITLES[mode],
        "code_file": NOTEBOOK_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        # Candidate mode mounts the training kernel's output (the
        # adapter) as an input; base mode needs no kernel input.
        "kernel_sources": [TRAINING_KERNEL_ID] if mode == "candidate" else [],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (out_dir / "eval-mode.json").write_text(
        json.dumps(
            {"mode": mode, "model": "candidate" if mode == "candidate" else "base"},
            indent=2,
        )
        + "\n"
    )

    manifest = json.loads(
        (repo_root / "evaluation" / "benchmark_manifest.json").read_text()
    )
    packaging_manifest = {
        "kernel_id": kernel_id,
        "mode": mode,
        "packaged_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_model": "Qwen/Qwen3.8-27B",
        "base_revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
        "benchmark": manifest.get("name"),
        "benchmark_version": manifest.get("version"),
        "notebook": NOTEBOOK_FILE,
    }
    (out_dir / "packaging-manifest.json").write_text(
        json.dumps(packaging_manifest, indent=2) + "\n"
    )
    if preserved_readme is not None:
        (out_dir / "README.md").write_bytes(preserved_readme)
    return {
        "out_dir": str(out_dir),
        "kernel_id": kernel_id,
        "mode": mode,
        "notebook": NOTEBOOK_FILE,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on refusal."""
    parser = argparse.ArgumentParser(
        description="Assemble the Kaggle eval kernel directory."
    )
    parser.add_argument(
        "--mode",
        default="candidate",
        choices=["candidate", "base"],
        help="Evaluate the trained adapter ('candidate') or the pinned "
        "base model ('base', for measuring the baseline)",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--out",
        default="kernels/eval",
        help="Kernel directory to build (default: kernels/eval)",
    )
    args = parser.parse_args(argv)

    try:
        summary = build_kernel(Path(args.repo_root), Path(args.out), args.mode)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Eval kernel packaged in {summary['out_dir']}: "
        f"kernel={summary['kernel_id']} mode={summary['mode']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
