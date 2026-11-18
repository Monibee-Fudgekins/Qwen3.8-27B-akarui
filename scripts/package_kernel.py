"""Assemble a Kaggle kernel directory for SFT training.

Copies everything notebook ``05_sft_training.ipynb`` needs into one
self-contained directory (default ``kernels/sft/``):

* ``05_sft_training.ipynb`` (the ``code_file`` in kernel-metadata.json)
* ``kernel-metadata.json`` -- kernel slug
  ``monibeefudgekins/akarui-sft-training``, notebook type, GPU on,
  internet on (the base weights are downloaded from Hugging Face
  in-kernel; see ``kernels/sft/README.md``)
* copies of ``src/``, ``scripts/``, ``configs/`` (everything the
  notebook imports or shells out to)
* ``data/train.jsonl`` (the training file the configs point at)
* ``requirements-kaggle.txt`` (reference for the Kaggle environment)

Refusal rules (exit 1, no partial directory left behind):

* ``data/train.jsonl`` is missing or empty.
* Any line is not a JSON object, or has no usable training content
  (a non-empty ``messages`` list, non-empty ``text``, or
  ``task_description`` + ``generated_code`` -- the same shapes
  ``src/training/sft.py`` accepts).
* Any record names one of the 12 held-out benchmark competitions in
  ``evaluation/benchmark_manifest.json`` -- training on those is
  never allowed.

The packaged directory is a build artifact, not a commit: it is
ignored by git (see ``.gitignore``) and rebuilt by
``scripts/trigger_kaggle.py`` when it is missing, so the weekly
workflow needs no change. Never include this directory's contents in
a repo push -- it carries a copy of ``data/train.jsonl``, which is
pushed separately.

No credentials are read, written, or printed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

KERNEL_ID = "monibeefudgekins/akarui-sft-training"
KERNEL_TITLE = "Akarui SFT training (QLoRA, Qwen3.8-27B)"
#: Kaggle kernel titles are capped at 50 characters (SaveKernel API
#: rejects longer titles with HTTP 400). Guarded in build_kernel so a
#: too-long title fails at packaging time, not at push time.
MAX_TITLE_LENGTH = 50
NOTEBOOK_FILE = "05_sft_training.ipynb"
NOTEBOOK_SRC = Path("notebooks") / NOTEBOOK_FILE

#: Fallback held-out list, used only when
#: ``evaluation/benchmark_manifest.json`` cannot be read. It must match
#: that manifest; the manifest is the source of truth.
HELD_OUT_FALLBACK = [
    "titanic",
    "house-prices-advanced-regression-techniques",
    "digit-recognizer",
    "spaceship-titanic",
    "playground-series-s3e11",
    "store-sales-time-series-forecasting",
    "nlp-getting-started",
    "petfinder-pawpularity-score",
    "tabular-playground-series-jan-2024",
    "forest-cover-type-prediction",
    "bike-sharing-demand",
    "commonlitreadabilityprize",
]

#: Files that must exist in the repo for the kernel to run.
REQUIRED_FILES = [
    NOTEBOOK_SRC,
    Path("scripts/prepare_training.py"),
    Path("scripts/train.py"),
    Path("configs/base.yaml"),
    Path("configs/sft.yaml"),
    Path("requirements-kaggle.txt"),
]


def load_held_out(repo_root: Path) -> set[str]:
    """Held-out competition ids from the benchmark manifest.

    Falls back to the embedded list when the manifest is unreadable,
    so packaging never silently trains on a benchmark competition.
    """
    manifest_path = repo_root / "evaluation" / "benchmark_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        ids = {
            str(item["id"])
            for item in manifest.get("held_out_competitions", [])
            if item.get("id")
        }
        if ids:
            return ids
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    return set(HELD_OUT_FALLBACK)


def _record_is_usable(record: dict) -> bool:
    """True when SFT can render this record into training text."""
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        return True
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return True
    return bool(
        record.get("task_description") and record.get("generated_code")
    )


def validate_train_file(train_path: Path, held_out: set[str]) -> dict:
    """Validate ``data/train.jsonl``. Returns row/sha stats.

    Raises:
        ValueError: If the file is missing, empty, malformed, has no
            usable records, or names a held-out competition.
    """
    if not train_path.is_file():
        raise ValueError(
            f"refusing to package: training file not found: {train_path}. "
            "Build data/train.jsonl first (see data/README.md)."
        )
    digest = hashlib.sha256()
    rows = 0
    usable = 0
    offenders: list[tuple[int, str]] = []
    with train_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            line = raw.decode("utf-8", errors="strict").strip()
            if not line:
                continue
            rows += 1
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(
                    f"refusing to package: {train_path} line {line_number} "
                    f"is not valid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"refusing to package: {train_path} line {line_number} "
                    "is not a JSON object"
                )
            competition_id = str(record.get("competition_id") or "").strip()
            if competition_id in held_out:
                offenders.append((line_number, competition_id))
            if _record_is_usable(record):
                usable += 1
    if offenders:
        detail = ", ".join(
            f"line {n} ({comp})" for n, comp in offenders[:5]
        )
        raise ValueError(
            "refusing to package: training file names held-out benchmark "
            f"competition(s): {detail}"
            + (f" (+{len(offenders) - 5} more)" if len(offenders) > 5 else "")
            + ". Held-out competitions must never appear in training data."
        )
    if rows == 0:
        raise ValueError(
            f"refusing to package: training file is empty: {train_path}"
        )
    if usable == 0:
        raise ValueError(
            f"refusing to package: no usable training records in {train_path} "
            "(each line needs messages, text, or task_description + "
            "generated_code)"
        )
    return {
        "rows": rows,
        "usable_rows": usable,
        "sha256": digest.hexdigest(),
        "train_file": str(train_path),
    }


def build_kernel(repo_root: Path, out_dir: Path) -> dict:
    """Assemble the kernel directory. Returns a summary dict.

    Raises:
        ValueError: On any refusal (missing/invalid training data,
            missing required files, unsafe output location).
    """
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    if out_dir == repo_root:
        raise ValueError("refusing to package: output dir is the repo root")
    if repo_root in out_dir.parents:
        # Inside the repo, only the kernels/ build-artifact tree is
        # allowed -- never nest the output inside src/, scripts/,
        # data/ or another source tree.
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

    held_out = load_held_out(repo_root)
    stats = validate_train_file(repo_root / "data" / "train.jsonl", held_out)

    # Rebuild from scratch so a stale file can never survive -- except
    # README.md, which is committed documentation of this directory
    # (see .gitignore) and is preserved across rebuilds.
    preserved_readme = None
    readme_path = out_dir / "README.md"
    if readme_path.is_file():
        preserved_readme = readme_path.read_bytes()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data").mkdir(parents=True)

    shutil.copy2(repo_root / NOTEBOOK_SRC, out_dir / NOTEBOOK_FILE)
    for dirname in ("src", "scripts", "configs"):
        shutil.copytree(
            repo_root / dirname,
            out_dir / dirname,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copy2(
        repo_root / "data" / "train.jsonl", out_dir / "data" / "train.jsonl"
    )
    shutil.copy2(
        repo_root / "requirements-kaggle.txt",
        out_dir / "requirements-kaggle.txt",
    )

    if len(KERNEL_TITLE) > MAX_TITLE_LENGTH:
        raise ValueError(
            f"refusing to package: kernel title is {len(KERNEL_TITLE)} chars, "
            f"over Kaggle's {MAX_TITLE_LENGTH}-char limit: {KERNEL_TITLE!r}"
        )
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": NOTEBOOK_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    packaging_manifest = {
        "kernel_id": KERNEL_ID,
        "packaged_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_model": "Qwen/Qwen3.8-27B",
        "base_revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
        "train_file_sha256": stats["sha256"],
        "train_rows": stats["rows"],
        "train_usable_rows": stats["usable_rows"],
        "held_out_competitions_excluded": sorted(held_out),
        "notebook": NOTEBOOK_FILE,
    }
    (out_dir / "packaging-manifest.json").write_text(
        json.dumps(packaging_manifest, indent=2) + "\n"
    )
    if preserved_readme is not None:
        (out_dir / "README.md").write_bytes(preserved_readme)
    return {
        "out_dir": str(out_dir),
        "kernel_id": KERNEL_ID,
        "notebook": NOTEBOOK_FILE,
        **stats,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on refusal."""
    parser = argparse.ArgumentParser(
        description="Assemble the Kaggle SFT kernel directory."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--out",
        default="kernels/sft",
        help="Kernel directory to build (default: kernels/sft)",
    )
    args = parser.parse_args(argv)

    try:
        summary = build_kernel(Path(args.repo_root), Path(args.out))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Kernel packaged in {summary['out_dir']}: "
        f"kernel={summary['kernel_id']} "
        f"train_rows={summary['rows']} usable={summary['usable_rows']} "
        f"train_sha256={summary['sha256'][:16]}..."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
