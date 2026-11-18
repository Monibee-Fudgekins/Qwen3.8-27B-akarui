"""Submit the Kaggle training notebook and wait for it to finish.

Flow:
1. Configure Kaggle auth from ``KAGGLE_TOKEN`` via :mod:`kaggle_auth`.
2. Submit the training notebook with ``kaggle kernels push``.
3. Poll ``kaggle kernels status`` until the kernel reports ``complete``
   (exit 0), ``error`` / ``cancelled`` (exit 1), or the timeout elapses
   (exit 1).

The run is only reported as successful when Kaggle itself reports the
kernel complete. A timeout or a failed kernel is a failure, never a
silent success. A run record JSON is written for provenance.

Two exceptions to failure:

* Weekly GPU quota: if ``kaggle kernels push`` fails because the
  weekly GPU quota is exhausted (Kaggle kills nothing -- the push is
  refused), a run record with ``status: "skipped_quota"`` is written
  and the script exits 0. This is a skip, not a success and not a
  failure: the run never started, so there is nothing to retry yet
  and no red build to chase. Any other push failure exits 1.
* Missing packaged kernel: if the notebook directory has no
  ``kernel-metadata.json``, ``scripts/package_kernel.py`` is run to
  build it from the repo first (it refuses when
  ``data/train.jsonl`` is missing or invalid). Pass ``--no-package``
  to disable this and fail instead.

The Kaggle CLI (``kaggle``) must be installed and the notebook directory
must contain the notebook plus its ``kernel-metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_auth import configure_kaggle_auth  # noqa: E402

DEFAULT_NOTEBOOK_DIR = "kernels/sft"
DEFAULT_KERNEL = "monibeefudgekins/akarui-sft-training"
DEFAULT_TIMEOUT_MINUTES = 720
POLL_INTERVAL_SECONDS = 300

#: Substrings (lowercased) in push output that mean the weekly GPU
#: quota is exhausted. Kaggle's message is "Maximum weekly GPU quota
#: of 30.00 hours reached" (seen on the sibling Glimmer kernel when
#: its push was refused while GitHub stayed green).
QUOTA_MARKERS = ("weekly gpu quota", "weekly quota", "gpu quota")

#: How many times to retry a status call whose output cannot be
#: parsed at all, and how long to wait between attempts. The first
#: poll can land seconds after the push, before the API serves the
#: new kernel; an unparseable response then is propagation delay, not
#: a dead kernel. (A response that does parse -- including one whose
#: text contains "error" -- is returned immediately, not retried.)
STATUS_PARSE_ATTEMPTS = 3
STATUS_PARSE_RETRY_SECONDS = 30


class QuotaExceededError(RuntimeError):
    """``kaggle kernels push`` was refused: weekly GPU quota exhausted."""


def is_quota_failure(output: str) -> bool:
    """True when CLI output reports the weekly GPU quota is exhausted."""
    lowered = output.lower()
    return any(marker in lowered for marker in QUOTA_MARKERS)


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Run a kaggle CLI command and return the completed process."""
    return subprocess.run(
        ["kaggle", *args], capture_output=True, text=True, timeout=120
    )


def push_kernel(notebook_dir: Path) -> None:
    """Submit the notebook directory to Kaggle.

    Raises:
        RuntimeError: If the CLI is missing, the notebook is missing, or
            the push fails.
    """
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "kaggle CLI not found on PATH. Install it (pip install kaggle) "
            "and configure auth first."
        )
    notebooks = sorted(notebook_dir.glob("*.ipynb"))
    if not notebooks:
        raise RuntimeError(f"no .ipynb notebook found in {notebook_dir}")
    if not (notebook_dir / "kernel-metadata.json").exists():
        raise RuntimeError(
            f"kernel-metadata.json not found in {notebook_dir}; the Kaggle "
            "CLI needs it to submit the notebook."
        )
    result = _run_cli(["kernels", "push", "-p", str(notebook_dir)])
    if result.returncode != 0:
        output = result.stderr.strip() or result.stdout.strip()
        if is_quota_failure(result.stdout + result.stderr):
            raise QuotaExceededError(
                "kaggle kernels push refused: weekly GPU quota exhausted: "
                f"{output}"
            )
        raise RuntimeError(f"kaggle kernels push failed: {output}")
    print(f"Kernel submitted from {notebook_dir}")


def ensure_packaged_kernel(notebook_dir: Path, repo_root: Path) -> None:
    """Build the packaged kernel dir if it is missing its metadata.

    Runs ``scripts/package_kernel.py`` against the repo root. Raises
    RuntimeError when packaging refuses (e.g. no training data yet).
    Does nothing when the directory already has kernel-metadata.json.
    """
    if (notebook_dir / "kernel-metadata.json").exists():
        return
    script = repo_root / "scripts" / "package_kernel.py"
    print(f"Packaged kernel not found in {notebook_dir}; building it...")
    result = subprocess.run(
        [sys.executable, str(script), "--repo-root", str(repo_root),
         "--out", str(notebook_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "could not package the kernel: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    print(result.stdout.strip())


def kernel_status(kernel: str) -> str:
    """Return the Kaggle kernel status (e.g. ``complete``), lowercased.

    Retries a few times when the status call's output cannot be parsed
    at all (propagation delay right after a push). After the retries
    it raises with the raw CLI output -- stdout and stderr -- so the
    log shows what the status call actually returned instead of an
    empty message.
    """
    last_output = ""
    for attempt in range(STATUS_PARSE_ATTEMPTS):
        result = _run_cli(["kernels", "status", kernel])
        output = (result.stdout + result.stderr).lower()
        last_output = (
            f"stdout={result.stdout.strip()!r} "
            f"stderr={result.stderr.strip()!r}"
        )
        for status in ("complete", "error", "cancelled", "running", "queued"):
            if status in output:
                return status
        if attempt < STATUS_PARSE_ATTEMPTS - 1:
            print(
                f"Kernel status not parseable yet for {kernel} "
                f"(attempt {attempt + 1}/{STATUS_PARSE_ATTEMPTS}); "
                f"retrying in {STATUS_PARSE_RETRY_SECONDS}s"
            )
            time.sleep(STATUS_PARSE_RETRY_SECONDS)
    raise RuntimeError(
        f"could not parse kernel status for {kernel}: {last_output}"
    )


def wait_for_completion(kernel: str, timeout_minutes: int) -> str:
    """Poll until the kernel finishes. Returns the final status string."""
    deadline = time.monotonic() + timeout_minutes * 60
    while True:
        status = kernel_status(kernel)
        print(f"Kernel {kernel}: {status}")
        if status in ("complete", "error", "cancelled"):
            return status
        if time.monotonic() >= deadline:
            return "timeout"
        time.sleep(min(POLL_INTERVAL_SECONDS, max(0, deadline - time.monotonic())))


def _write_run_record(path: Path, record: dict) -> None:
    """Write a run record JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on completion or a quota skip."""
    parser = argparse.ArgumentParser(
        description="Submit the Kaggle training notebook and wait for it."
    )
    parser.add_argument(
        "--notebook-dir",
        default=DEFAULT_NOTEBOOK_DIR,
        help="Directory with the training notebook + kernel-metadata.json "
        "(default: the packaged kernel dir kernels/sft)",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Repository root (used to build the packaged kernel)",
    )
    parser.add_argument(
        "--no-package",
        action="store_true",
        help="Do not build the packaged kernel when it is missing; "
        "fail instead",
    )
    parser.add_argument(
        "--kernel", default=DEFAULT_KERNEL, help="Kaggle kernel slug"
    )
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=DEFAULT_TIMEOUT_MINUTES,
        help="Max minutes to wait for completion",
    )
    parser.add_argument(
        "--run-record",
        default="output/kaggle-run.json",
        help="Where to write the run record JSON",
    )
    args = parser.parse_args(argv)

    started = datetime.now(timezone.utc).isoformat()
    record_path = Path(args.run_record)
    try:
        if not args.no_package:
            ensure_packaged_kernel(Path(args.notebook_dir), Path(args.repo_root))
        auth = configure_kaggle_auth()
        print(f"Kaggle auth configured via {auth['method']}")
        push_kernel(Path(args.notebook_dir))
        final_status = wait_for_completion(args.kernel, args.timeout_minutes)
    except QuotaExceededError as exc:
        # The run never started: record the skip and exit 0. This is
        # not a success -- no model was produced -- and not a failure
        # to retry immediately; it waits for the quota to reset.
        print(f"skipped: {exc}", file=sys.stderr)
        _write_run_record(
            record_path,
            {
                "kernel": args.kernel,
                "notebook_dir": args.notebook_dir,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "skipped_quota",
                "reason": str(exc),
            },
        )
        print("Run skipped (weekly GPU quota exhausted); recorded, exit 0.")
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    record = {
        "kernel": args.kernel,
        "notebook_dir": args.notebook_dir,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": final_status,
    }
    _write_run_record(record_path, record)

    if final_status == "complete":
        print(f"Training kernel completed: {args.kernel}")
        return 0
    print(
        f"Training kernel did NOT complete (status: {final_status}). "
        "No model was produced by this run.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
