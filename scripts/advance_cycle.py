"""Advance the Akarui weekly cycle by one step, from its committed state.

The weekly workflow (``.github/workflows/weekly-training.yml``) only
STARTS a Kaggle cycle and records it in ``evaluation/akarui-cycle.json``:

    {"cycle": "2026-11-25", "mode": "candidate", "phase": "training",
     "train_kernel": "monibeefudgekins/akarui-sft-training",
     "eval_kernel": "monibeefudgekins/akarui-eval-candidate", ...}

This script is the check workflow's state machine. It checks the
kernel status ONCE and either advances the cycle or exits quietly:

    phase=training + training COMPLETE  -> package + trigger the eval
        kernel (submit-only), commit state phase=eval.
    phase=training + training ERROR     -> commit state phase=failed,
        exit 1 (loud: the kernel failed, no re-dispatch).
    phase=eval + eval COMPLETE         -> action=completed; the workflow
        then downloads outputs and runs the gate/promote (candidate) or
        records the baseline (base) with the existing step bodies.
    phase=eval + eval ERROR             -> commit state phase=failed,
        exit 1.
    phase=training/eval + still running -> action=none, exit 0.
    no state / phase idle|done|failed   -> action=none, exit 0.

It never waits on a kernel and never re-dispatches a cycle. A failed
eval trigger (push refused for a non-quota reason, packaging refused)
leaves the state untouched and exits 1, so the next check retries the
trigger -- the trigger itself is the only non-idempotent step, and a
duplicate eval push is harmless (the eval is deterministic; the
completion path always downloads the latest eval output).

The script commits the state file itself on transitions (eval
triggered, skipped, failed) so the next check run sees the new phase
even if this run dies afterwards. The completion path (action=
completed) is committed by the workflow's final step together with
the gate decision records.

Exit codes: 0 = advanced, still running, or nothing to do; 1 = a
kernel failed, the state is corrupt, or an infrastructure step
(git, packaging, trigger) failed. The ``action`` and ``reason``
outputs are written to $GITHUB_OUTPUT even on failure, so the
workflow can scope its Diagnose step to push failures only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trigger_kaggle import kernel_failure_message, kernel_status  # noqa: E402

DEFAULT_STATE_PATH = "evaluation/akarui-cycle.json"
TRAIN_KERNEL = "monibeefudgekins/akarui-sft-training"
EVAL_KERNELS = {
    "candidate": "monibeefudgekins/akarui-eval-candidate",
    "base": "monibeefudgekins/akarui-eval-baseline",
}
TERMINAL_PHASES = ("idle", "done", "failed")
ACTIVE_PHASES = ("training", "eval")

#: Reasons written to $GITHUB_OUTPUT on failure, so the workflow can
#: tell a push failure (worth diagnosing) from a kernel error (the
#: cycle is failed; re-pushing would start a stray kernel version).
REASON_PUSH_FAILED = "push_failed"
REASON_KERNEL_ERROR = "kernel_error"
REASON_INFRA = "infra"


def load_state(path: Path) -> dict | None:
    """Load the cycle state file. None when the file does not exist.

    Raises:
        ValueError: When the file exists but is not valid JSON.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"cycle state file {path} is not valid JSON: {exc}")


def save_state(path: Path, state: dict) -> None:
    """Write the cycle state file, stamping updated_at."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2) + "\n")


def git_commit_file(repo_root: Path, path: Path, message: str) -> None:
    """Stage one file, commit (only when it changed), and push.

    Raises:
        RuntimeError: When any git step fails.
    """
    rel = str(path.relative_to(repo_root)) if path.is_absolute() else str(path)

    def _run(args: list[str]) -> None:
        result = subprocess.run(
            args, cwd=repo_root, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args[1:])} failed: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

    _run(["git", "add", rel])
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=repo_root
    )
    if diff.returncode == 0:
        print(f"no state change to commit for {rel}")
        return
    _run(["git", "commit", "-m", message])
    _run(["git", "push"])
    print(f"committed state: {message}")


def _trigger_eval(repo_root: Path, mode: str, eval_kernel: str) -> str:
    """Package + submit-only trigger the eval kernel.

    Returns the run-record status (``"submitted"``, ``"complete"``,
    ``"skipped_quota"``, ...). Raises RuntimeError when packaging or
    the trigger itself fails (the caller leaves the state untouched
    so the next check retries).
    """
    package = subprocess.run(
        [
            sys.executable,
            "scripts/package_eval_kernel.py",
            "--mode",
            mode,
            "--out",
            "kernels/eval",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if package.returncode != 0:
        raise RuntimeError(
            "packaging the eval kernel failed: "
            f"{package.stderr.strip() or package.stdout.strip()}"
        )
    if package.stdout.strip():
        print(package.stdout.strip())
    trigger = subprocess.run(
        [
            sys.executable,
            "scripts/trigger_kaggle.py",
            "--notebook-dir",
            "kernels/eval",
            "--kernel",
            eval_kernel,
            "--run-record",
            "output/kaggle-eval-run.json",
            "--submit-only",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if trigger.stdout.strip():
        print(trigger.stdout.strip())
    if trigger.stderr.strip():
        print(trigger.stderr.strip(), file=sys.stderr)
    if trigger.returncode != 0:
        raise RuntimeError(
            f"eval kernel trigger failed (exit {trigger.returncode}); "
            "state left at phase=training so the next check retries"
        )
    record_path = repo_root / "output" / "kaggle-eval-run.json"
    if not record_path.exists():
        raise RuntimeError(
            "eval kernel trigger exited 0 but wrote no run record at "
            f"{record_path}"
        )
    return json.loads(record_path.read_text()).get("status", "")


def _fail_kernel(
    state_path: Path,
    repo_root: Path,
    state: dict,
    kernel: str,
    status: str,
) -> tuple[str, dict]:
    """Record a kernel error in the state, commit it, and report it."""
    message = kernel_failure_message(kernel)
    state["phase"] = "failed"
    state["failure"] = message or f"{kernel} {status}"
    save_state(state_path, state)
    git_commit_file(
        repo_root, state_path, f"chore: akarui cycle failed ({kernel} {status})"
    )
    print(f"kernel {status}: {kernel}: {state['failure']}")
    return "failed", state


def advance(state_path: Path, repo_root: Path) -> tuple[str, dict, str]:
    """Advance one step of the cycle.

    Returns ``(action, state, reason)``. Actions: ``"none"`` (still
    running, or no active cycle), ``"eval_triggered"``,
    ``"completed"``, ``"skipped"`` (eval trigger hit the quota),
    ``"failed"`` (a kernel errored). The reason is "" on success and
    one of the REASON_* constants on failure.
    """
    state = load_state(state_path)
    if state is None:
        print("no cycle state file; nothing to do")
        return "none", {}, ""
    phase = state.get("phase")
    mode = state.get("mode", "candidate")
    if phase in TERMINAL_PHASES:
        print(f"cycle phase is {phase!r}; nothing to do")
        return "none", state, ""
    if phase not in ACTIVE_PHASES:
        raise RuntimeError(f"unknown cycle phase {phase!r} in {state_path}")
    if mode not in EVAL_KERNELS:
        raise RuntimeError(f"unknown cycle mode {mode!r} in {state_path}")

    if phase == "training":
        train_kernel = state.get("train_kernel", TRAIN_KERNEL)
        status = kernel_status(train_kernel)
        print(f"training kernel {train_kernel}: {status}")
        if status in ("running", "queued"):
            return "none", state, ""
        if status in ("error", "cancelled"):
            action, state = _fail_kernel(
                state_path, repo_root, state, train_kernel, status
            )
            return action, state, REASON_KERNEL_ERROR
        # status == "complete": trigger the eval kernel.
        eval_kernel = EVAL_KERNELS[mode]
        eval_status = _trigger_eval(repo_root, mode, eval_kernel)
        if eval_status == "skipped_quota":
            state["phase"] = "idle"
            state["reason"] = "skipped_quota"
            save_state(state_path, state)
            git_commit_file(
                repo_root,
                state_path,
                "chore: akarui cycle skipped (Kaggle GPU quota exhausted at eval trigger)",
            )
            print("eval trigger skipped (weekly GPU quota exhausted)")
            return "skipped", state, ""
        if eval_status not in ("submitted", "complete"):
            raise RuntimeError(
                f"eval kernel trigger ended with status {eval_status!r}; "
                "state left at phase=training so the next check retries"
            )
        state["phase"] = "eval"
        state["eval_kernel"] = eval_kernel
        state["eval_triggered_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state_path, state)
        git_commit_file(
            repo_root, state_path, f"chore: akarui eval triggered ({mode})"
        )
        return "eval_triggered", state, ""

    # phase == "eval"
    eval_kernel = state.get("eval_kernel") or EVAL_KERNELS[mode]
    status = kernel_status(eval_kernel)
    print(f"eval kernel {eval_kernel}: {status}")
    if status in ("running", "queued"):
        return "none", state, ""
    if status in ("error", "cancelled"):
        action, state = _fail_kernel(
            state_path, repo_root, state, eval_kernel, status
        )
        return action, state, REASON_KERNEL_ERROR
    # status == "complete": the workflow downloads outputs and promotes.
    return "completed", state, ""


def _emit_outputs(action: str, reason: str, state: dict) -> None:
    """Write action/mode/phase/reason to $GITHUB_OUTPUT (when set)."""
    lines = [
        f"action={action}",
        f"reason={reason}",
        f"mode={state.get('mode', 'candidate')}",
        f"phase={state.get('phase', '')}",
        f"train_kernel={state.get('train_kernel', TRAIN_KERNEL)}",
        f"eval_kernel={state.get('eval_kernel', '')}",
    ]
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 when handled, else 1."""
    parser = argparse.ArgumentParser(
        description="Advance the Akarui weekly cycle by one step."
    )
    parser.add_argument(
        "--state",
        default=DEFAULT_STATE_PATH,
        help="Cycle state file (default: evaluation/akarui-cycle.json)",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current directory)",
    )
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    state_arg = Path(args.state)
    state_path = state_arg if state_arg.is_absolute() else repo_root / state_arg
    action, state, reason = "failed", {}, REASON_INFRA
    try:
        action, state, reason = advance(state_path, repo_root)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        # Distinguish a push/trigger failure (worth diagnosing) from
        # anything else. A packaging or trigger RuntimeError from the
        # eval transition carries its own message.
        if "trigger failed" in str(exc) or "packaging" in str(exc):
            reason = REASON_PUSH_FAILED
        else:
            reason = REASON_INFRA
    _emit_outputs(action, reason, state)
    return 1 if action == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
