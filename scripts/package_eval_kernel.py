"""Assemble a Kaggle eval kernel directory for Akarui candidates.

Assembles the build inputs for notebook ``07_evaluation.ipynb`` in
one directory (default ``kernels/eval/``). Only the notebook file
itself is uploaded to Kaggle -- see the runtime contract below -- so
the directory is a complete local record of what the kernel runs,
not a bundle that travels with it:

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
* ``eval-mode.json`` -- ``{"mode": "candidate"|"base"}``, written
  for local tooling. Nothing in-kernel may rely on it (see the
  runtime contract below).
* ``requirements-kaggle.txt``

Runtime contract (run #7 failure, 2026-11-20): ``kaggle kernels
push`` uploads ONLY the notebook file named as ``code_file`` in
``kernel-metadata.json``. The copied ``src/``, ``scripts/``,
``configs/``, ``evaluation/``, ``data/`` and ``eval-mode.json`` are
build-time inputs only -- they never reach Kaggle, so the packaged
notebook must be self-sufficient: its first code cell clones the
public repo at runtime and runs from the clone, and
:func:`_bake_mode` rewrites the ``mode = "candidate"`` default in
the packaged notebook to the actual ``--mode`` value, since the
notebook's ``eval-mode.json`` fallback cannot fire on Kaggle.

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
import re
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
    # Titles MUST slugify to their kernel ids (see _slugify below):
    # Kaggle's SaveKernel creates the kernel under the TITLE-derived
    # slug when they differ, so the push "succeeds" but lands at the
    # wrong URL as an unrun draft and the status poll of the intended
    # slug finds nothing (seen live 2026-11-18, run #5). The previous
    # titles ("Akarui evaluation - base model baseline", 39 chars)
    # slugified to "akarui-evaluation-base-model-baseline", not
    # "akarui-eval-baseline". Keep titles short (<= 50 chars, the
    # SaveKernel limit) and matching their ids.
    "candidate": "Akarui Eval Candidate",
    "base": "Akarui Eval Baseline",
}
#: Kaggle kernel titles are capped at 50 characters (SaveKernel API).
MAX_TITLE_LENGTH = 50


def _slugify(text: str) -> str:
    """Slugify an ASCII title the way Kaggle's CLI does (python-slugify).

    Lowercase, collapse every run of non-alphanumeric characters to a
    single hyphen, strip leading/trailing hyphens. Titles here are
    ASCII-only, so no transliteration is needed.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
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

#: Marker comment for the self-bootstrap cell that must be the first
#: code cell of the packaged notebook (see the runtime contract in
#: the module docstring). The packager refuses a notebook without it.
BOOTSTRAP_MARKER = "AKARUI_KERNEL_SELF_BOOTSTRAP"

#: Marker comment for the dependency-install cell. The packager refuses
#: a notebook without it: the Kaggle image lacks bitsandbytes, so a
#: 4-bit model load fails unless the kernel installs
#: requirements-kaggle.txt itself (run #8, 2026-11-21).
INSTALL_MARKER = "AKARUI_KERNEL_INSTALL_REQUIREMENTS"

#: The mode default line in the source notebook. The packager rewrites
#: it in the packaged copy to the actual --mode value; the line must
#: occur exactly once in a code cell.
MODE_DEFAULT_LINE = 'mode = "candidate"'


def _notebook_cells(notebook_path: Path) -> dict:
    """Load a notebook JSON file. Raises ValueError when unparseable."""
    try:
        return json.loads(notebook_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"refusing to package: cannot read notebook {notebook_path}: {exc}"
        ) from exc


def _write_notebook(notebook_path: Path, notebook: dict) -> None:
    """Write a notebook JSON file in nbformat's 1-space indent."""
    notebook_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    )


def _verify_bootstrap_cell(notebook_path: Path) -> None:
    """Refuse a packaged notebook without the self-bootstrap cell.

    Without it the kernel would have no src/, scripts/, configs/,
    evaluation/ or data/ at runtime (only the notebook file is
    uploaded), and would fail at the first import or subprocess call.
    """
    notebook = _notebook_cells(notebook_path)
    count = sum(
        1
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
        and BOOTSTRAP_MARKER in "".join(cell.get("source", []))
    )
    if count != 1:
        raise ValueError(
            f"refusing to package: expected exactly one self-bootstrap "
            f"cell ({BOOTSTRAP_MARKER}) in {notebook_path}, found {count}"
        )


def _verify_install_cell(notebook_path: Path) -> None:
    """Refuse a packaged notebook without the dependency-install cell.

    Without it the kernel would fail at the first 4-bit model load:
    the Kaggle image ships torch/transformers but not bitsandbytes
    (run #8, 2026-11-21).
    """
    notebook = _notebook_cells(notebook_path)
    count = sum(
        1
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
        and INSTALL_MARKER in "".join(cell.get("source", []))
    )
    if count != 1:
        raise ValueError(
            f"refusing to package: expected exactly one dependency-install "
            f"cell ({INSTALL_MARKER}) in {notebook_path}, found {count}"
        )


def _bake_mode(notebook_path: Path, mode: str) -> None:
    """Rewrite the mode default in the packaged notebook to ``mode``.

    The notebook reads ``eval-mode.json`` only when the file exists,
    and that file never reaches Kaggle, so the packaged notebook must
    carry the mode itself. Exactly one ``mode = "candidate"`` line
    must exist in a code cell; anything else is a refusal, so a
    silently wrong mode can never be packaged.
    """
    notebook = _notebook_cells(notebook_path)
    replaced = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        lines = [source] if isinstance(source, str) else list(source)
        for index, line in enumerate(lines):
            if line.strip() == MODE_DEFAULT_LINE:
                newline = "\n" if line.endswith("\n") else ""
                lines[index] = f'mode = "{mode}"{newline}'
                replaced += 1
        cell["source"] = lines
    if replaced != 1:
        raise ValueError(
            f"refusing to package: expected exactly one "
            f"{MODE_DEFAULT_LINE!r} line in {notebook_path}, found {replaced}"
        )
    _write_notebook(notebook_path, notebook)


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
    # The packaged notebook must be self-sufficient on Kaggle (only
    # this file is uploaded): refuse a notebook without the bootstrap
    # cell or the dependency-install cell, then bake the dispatched
    # mode into it.
    _verify_bootstrap_cell(out_dir / NOTEBOOK_FILE)
    _verify_install_cell(out_dir / NOTEBOOK_FILE)
    _bake_mode(out_dir / NOTEBOOK_FILE, mode)
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
    title = KERNEL_TITLES[mode]
    if len(title) > MAX_TITLE_LENGTH:
        raise ValueError(
            f"refusing to package: eval kernel title is {len(title)} chars, "
            f"over Kaggle's {MAX_TITLE_LENGTH}-char limit: {title!r}"
        )
    slug = kernel_id.split("/", 1)[1]
    if _slugify(title) != slug:
        raise ValueError(
            f"refusing to package: eval kernel title {title!r} slugifies to "
            f"{_slugify(title)!r}, not the kernel id {kernel_id!r}; Kaggle "
            "would create the kernel under the title-derived slug, so the "
            "push would land at the wrong URL and the status poll would fail"
        )
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
