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

DEFAULT_NOTEBOOK_DIR = "notebooks"
DEFAULT_KERNEL = "monibeefudgekins/akarui-sft-training"
DEFAULT_TIMEOUT_MINUTES = 720
POLL_INTERVAL_SECONDS = 300


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
        raise RuntimeError(
            f"kaggle kernels push failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    print(f"Kernel submitted from {notebook_dir}")


def kernel_status(kernel: str) -> str:
    """Return the Kaggle kernel status (e.g. ``complete``), lowercased."""
    result = _run_cli(["kernels", "status", kernel])
    output = (result.stdout + result.stderr).lower()
    for status in ("complete", "error", "cancelled", "running", "queued"):
        if status in output:
            return status
    raise RuntimeError(
        f"could not parse kernel status for {kernel}: {result.stdout.strip()}"
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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 only when the kernel completes."""
    parser = argparse.ArgumentParser(
        description="Submit the Kaggle training notebook and wait for it."
    )
    parser.add_argument(
        "--notebook-dir",
        default=DEFAULT_NOTEBOOK_DIR,
        help="Directory with the training notebook + kernel-metadata.json",
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
    try:
        auth = configure_kaggle_auth()
        print(f"Kaggle auth configured via {auth['method']}")
        push_kernel(Path(args.notebook_dir))
        final_status = wait_for_completion(args.kernel, args.timeout_minutes)
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
    record_path = Path(args.run_record)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2))

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
