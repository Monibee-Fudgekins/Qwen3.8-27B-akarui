"""Tests for the submit-only trigger mode (``--submit-only``).

Runs without torch/transformers/GPU and without network. The Kaggle
CLI is never invoked: tests monkeypatch the CLI runner, following the
pattern in test_kernel_packaging.py.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import trigger_kaggle


def _fake_notebook_dir(tmp_path: Path) -> Path:
    notebook_dir = tmp_path / "kernels" / "sft"
    notebook_dir.mkdir(parents=True)
    (notebook_dir / "05_sft_training.ipynb").write_text("{}")
    (notebook_dir / "kernel-metadata.json").write_text(
        json.dumps({"id": "monibeefudgekins/akarui-sft-training"})
    )
    return notebook_dir


def _setup(monkeypatch, statuses: list[str]):
    """Stub auth, the kaggle binary, and the CLI.

    ``statuses`` is the scripted sequence of `kernels status` outputs,
    popped one per call; the push always succeeds.
    """
    monkeypatch.setattr(
        trigger_kaggle, "configure_kaggle_auth",
        lambda: {"method": "test"},
    )
    monkeypatch.setattr(trigger_kaggle.shutil, "which", lambda _: "/usr/bin/kaggle")
    monkeypatch.setattr(trigger_kaggle.time, "sleep", lambda _: None)
    # Confirm after a single status poll: the weekly trigger must not
    # sit in the confirm window while the test suite runs.
    monkeypatch.setattr(trigger_kaggle, "DEFAULT_CONFIRM_MINUTES", 0)
    queue = list(statuses)
    last = {"text": "running"}

    def _fake_cli(args):
        if args[:2] == ["kernels", "push"]:
            return subprocess.CompletedProcess(
                args, returncode=0, stdout="ok", stderr=""
            )
        # The failure-message read is a second status call: return the
        # same output so the message is found.
        text = queue.pop(0) if queue else last["text"]
        last["text"] = text
        return subprocess.CompletedProcess(
            args, returncode=0, stdout=text, stderr=""
        )

    monkeypatch.setattr(trigger_kaggle, "_run_cli", _fake_cli)


def _run_submit_only(tmp_path: Path, monkeypatch, statuses: list[str]):
    _setup(monkeypatch, statuses)
    notebook_dir = _fake_notebook_dir(tmp_path)
    record_path = tmp_path / "run.json"
    rc = trigger_kaggle.main([
        "--notebook-dir", str(notebook_dir),
        "--no-package",
        "--run-record", str(record_path),
        "--submit-only",
    ])
    return rc, json.loads(record_path.read_text())


def test_submit_only_running_records_submitted(tmp_path: Path, monkeypatch):
    rc, record = _run_submit_only(tmp_path, monkeypatch, ["running"])
    assert rc == 0
    assert record["status"] == "submitted"


def test_submit_only_queued_records_submitted(tmp_path: Path, monkeypatch):
    rc, record = _run_submit_only(tmp_path, monkeypatch, ["queued"])
    assert rc == 0
    assert record["status"] == "submitted"


def test_submit_only_complete_records_complete(tmp_path: Path, monkeypatch):
    rc, record = _run_submit_only(tmp_path, monkeypatch, ["complete"])
    assert rc == 0
    assert record["status"] == "complete"


def test_submit_only_kernel_error_exits_one(tmp_path: Path, monkeypatch):
    rc, record = _run_submit_only(
        tmp_path, monkeypatch,
        ['monibeefudgekins/akarui-sft-training has status "error"\n'
         'Failure message: "OOM killed"'],
    )
    assert rc == 1
    assert record["status"] == "error"


def test_submit_only_kernel_error_on_quota_is_skip(tmp_path: Path, monkeypatch):
    """A kernel that errors immediately on quota never really started
    -- skip, don't fail (same rule as the wait path, run #6)."""
    rc, record = _run_submit_only(
        tmp_path, monkeypatch,
        ['monibeefudgekins/akarui-sft-training has status "error"\n'
         'Failure message: "Maximum weekly GPU quota of 30.00 hours reached."'],
    )
    assert rc == 0
    assert record["status"] == "skipped_quota"
    assert "quota" in record["reason"].lower()


def test_submit_only_push_quota_is_skip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        trigger_kaggle, "configure_kaggle_auth",
        lambda: {"method": "test"},
    )
    monkeypatch.setattr(trigger_kaggle.shutil, "which", lambda _: "/usr/bin/kaggle")
    monkeypatch.setattr(
        trigger_kaggle, "_run_cli",
        lambda args: subprocess.CompletedProcess(
            args, returncode=1, stdout="",
            stderr="Maximum weekly GPU quota of 30.00 hours reached",
        ),
    )
    notebook_dir = _fake_notebook_dir(tmp_path)
    record_path = tmp_path / "run.json"
    rc = trigger_kaggle.main([
        "--notebook-dir", str(notebook_dir),
        "--no-package",
        "--run-record", str(record_path),
        "--submit-only",
    ])
    assert rc == 0
    assert json.loads(record_path.read_text())["status"] == "skipped_quota"


def test_submit_only_push_failure_exits_one(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        trigger_kaggle, "configure_kaggle_auth",
        lambda: {"method": "test"},
    )
    monkeypatch.setattr(trigger_kaggle.shutil, "which", lambda _: "/usr/bin/kaggle")
    monkeypatch.setattr(
        trigger_kaggle, "_run_cli",
        lambda args: subprocess.CompletedProcess(
            args, returncode=1, stdout="",
            stderr="403 Forbidden: bad credentials",
        ),
    )
    notebook_dir = _fake_notebook_dir(tmp_path)
    rc = trigger_kaggle.main([
        "--notebook-dir", str(notebook_dir),
        "--no-package",
        "--run-record", str(tmp_path / "run.json"),
        "--submit-only",
    ])
    assert rc == 1


def test_confirm_started_waits_out_window_then_submits(monkeypatch):
    """Queued, then still running when the confirm window elapses:
    the start is confirmed ("submitted"), not a failure."""
    monkeypatch.setattr(trigger_kaggle.time, "sleep", lambda _: None)
    calls = {"n": 0}

    def _fake_cli(args):
        calls["n"] += 1
        return subprocess.CompletedProcess(
            args, returncode=0, stdout="running", stderr=""
        )

    monkeypatch.setattr(trigger_kaggle, "_run_cli", _fake_cli)
    # Fake clock: first call sets the deadline (t=0), the second is
    # inside the window (t=100), the third is past it (t=601 > 600s
    # window), so the loop polls twice and then confirms the start.
    clock = iter([0.0, 100.0, 601.0])
    monkeypatch.setattr(
        trigger_kaggle.time, "monotonic", lambda: next(clock, 601.0)
    )
    assert trigger_kaggle.confirm_started("k", confirm_minutes=10) == "submitted"
    assert calls["n"] == 2


def test_confirm_started_returns_error_status(monkeypatch):
    """A kernel that failed outright is reported, not confirmed."""
    monkeypatch.setattr(trigger_kaggle.time, "sleep", lambda _: None)

    def _fake_cli(args):
        return subprocess.CompletedProcess(
            args, returncode=0,
            stdout='kernel has status "error"\nFailure message: "OOM killed"',
            stderr="",
        )

    monkeypatch.setattr(trigger_kaggle, "_run_cli", _fake_cli)
    assert trigger_kaggle.confirm_started("k") == "error"


def test_wait_path_unchanged_without_submit_only(tmp_path: Path, monkeypatch):
    """Regression guard: the default (wait) path still waits for
    completion and exits 0, as before."""
    monkeypatch.setattr(
        trigger_kaggle, "configure_kaggle_auth",
        lambda: {"method": "test"},
    )
    monkeypatch.setattr(trigger_kaggle.shutil, "which", lambda _: "/usr/bin/kaggle")
    monkeypatch.setattr(trigger_kaggle.time, "sleep", lambda _: None)

    def _fake_cli(args):
        if args[:2] == ["kernels", "push"]:
            return subprocess.CompletedProcess(args, returncode=0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(args, returncode=0, stdout="complete", stderr="")

    monkeypatch.setattr(trigger_kaggle, "_run_cli", _fake_cli)
    notebook_dir = _fake_notebook_dir(tmp_path)
    record_path = tmp_path / "run.json"
    rc = trigger_kaggle.main([
        "--notebook-dir", str(notebook_dir),
        "--no-package",
        "--run-record", str(record_path),
        "--timeout-minutes", "5",
    ])
    assert rc == 0
    assert json.loads(record_path.read_text())["status"] == "complete"
