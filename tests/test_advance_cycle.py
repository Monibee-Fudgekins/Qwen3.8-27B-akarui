"""Tests for the cycle state machine (scripts/advance_cycle.py).

Runs without network and without Kaggle: kernel status, the eval
trigger, and git are all stubbed. The workflow contract (the
$GITHUB_OUTPUT action/mode/phase outputs) is pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import advance_cycle


def _write_state(path: Path, state: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    return path


def _base_state(**overrides) -> dict:
    state = {
        "cycle": "2026-11-25",
        "mode": "candidate",
        "phase": "training",
        "train_kernel": "monibeefudgekins/akarui-sft-training",
        "eval_kernel": "monibeefudgekins/akarui-eval-candidate",
        "dispatched_at": "2026-11-25T08:00:00Z",
    }
    state.update(overrides)
    return state


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch):
    """A repo root with a state file and every external call stubbed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    state_path = repo / "evaluation" / "akarui-cycle.json"
    calls = {"status": [], "trigger": [], "commits": []}
    monkeypatch.setattr(
        advance_cycle, "kernel_status",
        lambda kernel: calls["status"].append(kernel) or "running",
    )
    monkeypatch.setattr(
        advance_cycle, "kernel_failure_message", lambda kernel: ""
    )
    monkeypatch.setattr(
        advance_cycle, "_trigger_eval",
        lambda root, mode, kernel: calls["trigger"].append((mode, kernel))
        or "submitted",
    )
    monkeypatch.setattr(
        advance_cycle, "git_commit_file",
        lambda root, path, message: calls["commits"].append(message),
    )
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    return {"repo": repo, "state_path": state_path, "calls": calls}


def _advance(ctx, **overrides):
    _write_state(ctx["state_path"], _base_state(**overrides))
    return advance_cycle.advance(ctx["state_path"], ctx["repo"])


def test_no_state_file_is_nothing_to_do(ctx):
    action, state, reason = advance_cycle.advance(
        ctx["state_path"], ctx["repo"]
    )
    assert (action, state, reason) == ("none", {}, "")
    assert ctx["calls"]["status"] == []


def test_terminal_phases_are_nothing_to_do(ctx):
    for phase in ("idle", "done", "failed"):
        action, state, reason = _advance(ctx, phase=phase)
        assert action == "none"
        assert state["phase"] == phase
    assert ctx["calls"]["status"] == []
    assert ctx["calls"]["trigger"] == []


def test_unknown_phase_raises(ctx):
    _write_state(ctx["state_path"], _base_state(phase="bogus"))
    with pytest.raises(RuntimeError, match="unknown cycle phase"):
        advance_cycle.advance(ctx["state_path"], ctx["repo"])


def test_invalid_state_json_raises(ctx):
    ctx["state_path"].parent.mkdir(parents=True, exist_ok=True)
    ctx["state_path"].write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        advance_cycle.advance(ctx["state_path"], ctx["repo"])


def test_training_running_is_nothing_to_do(ctx):
    action, state, reason = _advance(ctx, phase="training")
    assert (action, reason) == ("none", "")
    assert state["phase"] == "training"
    assert ctx["calls"]["trigger"] == []
    assert ctx["calls"]["commits"] == []


def test_training_complete_triggers_eval(ctx, monkeypatch):
    monkeypatch.setattr(advance_cycle, "kernel_status", lambda kernel: "complete")
    action, state, reason = _advance(ctx, phase="training")
    assert (action, reason) == ("eval_triggered", "")
    assert state["phase"] == "eval"
    assert state["eval_kernel"] == "monibeefudgekins/akarui-eval-candidate"
    assert state["eval_triggered_at"]
    assert ctx["calls"]["trigger"] == [
        ("candidate", "monibeefudgekins/akarui-eval-candidate")
    ]
    assert len(ctx["calls"]["commits"]) == 1


def test_training_complete_eval_trigger_quota_is_skip(ctx, monkeypatch):
    monkeypatch.setattr(advance_cycle, "kernel_status", lambda kernel: "complete")
    monkeypatch.setattr(
        advance_cycle, "_trigger_eval", lambda root, mode, kernel: "skipped_quota"
    )
    action, state, reason = _advance(ctx, phase="training")
    assert (action, reason) == ("skipped", "")
    assert state["phase"] == "idle"
    assert state["reason"] == "skipped_quota"
    assert len(ctx["calls"]["commits"]) == 1


def test_training_error_marks_failed(ctx, monkeypatch):
    monkeypatch.setattr(advance_cycle, "kernel_status", lambda kernel: "error")
    monkeypatch.setattr(
        advance_cycle, "kernel_failure_message", lambda kernel: "OOM killed"
    )
    action, state, reason = _advance(ctx, phase="training")
    assert (action, reason) == ("failed", advance_cycle.REASON_KERNEL_ERROR)
    assert state["phase"] == "failed"
    assert state["failure"] == "OOM killed"
    assert len(ctx["calls"]["commits"]) == 1


def test_eval_running_is_nothing_to_do(ctx):
    action, state, reason = _advance(ctx, phase="eval")
    assert (action, reason) == ("none", "")
    assert state["phase"] == "eval"


def test_eval_complete_is_completed(ctx, monkeypatch):
    monkeypatch.setattr(advance_cycle, "kernel_status", lambda kernel: "complete")
    action, state, reason = _advance(ctx, phase="eval")
    assert (action, reason) == ("completed", "")
    assert state["phase"] == "eval"  # the workflow marks done on commit
    assert ctx["calls"]["trigger"] == []


def test_eval_error_marks_failed(ctx, monkeypatch):
    monkeypatch.setattr(advance_cycle, "kernel_status", lambda kernel: "cancelled")
    action, state, reason = _advance(ctx, phase="eval")
    assert (action, reason) == ("failed", advance_cycle.REASON_KERNEL_ERROR)
    assert state["phase"] == "failed"


def test_main_exits_one_on_failed_kernel(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_path = _write_state(
        repo / "evaluation" / "akarui-cycle.json",
        _base_state(phase="eval"),
    )
    monkeypatch.setattr(advance_cycle, "kernel_status", lambda kernel: "error")
    monkeypatch.setattr(
        advance_cycle, "kernel_failure_message", lambda kernel: "boom"
    )
    monkeypatch.setattr(
        advance_cycle, "git_commit_file", lambda root, path, message: None
    )
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    rc = advance_cycle.main([
        "--state", str(state_path), "--repo-root", str(repo),
    ])
    assert rc == 1


def test_main_writes_github_outputs(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_path = _write_state(
        repo / "evaluation" / "akarui-cycle.json",
        _base_state(phase="eval"),
    )
    monkeypatch.setattr(advance_cycle, "kernel_status", lambda kernel: "complete")
    out_path = tmp_path / "github-output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_path))
    rc = advance_cycle.main([
        "--state", str(state_path), "--repo-root", str(repo),
    ])
    assert rc == 0
    outputs = dict(
        line.split("=", 1) for line in out_path.read_text().splitlines()
    )
    assert outputs["action"] == "completed"
    assert outputs["mode"] == "candidate"
    assert outputs["phase"] == "eval"
    assert outputs["eval_kernel"] == "monibeefudgekins/akarui-eval-candidate"


def test_main_trigger_failure_is_push_failed_reason(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_path = _write_state(
        repo / "evaluation" / "akarui-cycle.json",
        _base_state(phase="training"),
    )
    monkeypatch.setattr(advance_cycle, "kernel_status", lambda kernel: "complete")
    monkeypatch.setattr(
        advance_cycle, "_trigger_eval",
        lambda root, mode, kernel: (_ for _ in ()).throw(
            RuntimeError("eval kernel trigger failed (exit 1)")
        ),
    )
    out_path = tmp_path / "github-output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_path))
    rc = advance_cycle.main([
        "--state", str(state_path), "--repo-root", str(repo),
    ])
    assert rc == 1
    outputs = dict(
        line.split("=", 1) for line in out_path.read_text().splitlines()
    )
    assert outputs["action"] == "failed"
    assert outputs["reason"] == advance_cycle.REASON_PUSH_FAILED
    # State untouched: the next check retries the trigger.
    assert json.loads(state_path.read_text())["phase"] == "training"


def test_save_and_load_state_round_trip(tmp_path: Path):
    path = tmp_path / "cycle.json"
    state = _base_state(phase="eval")
    advance_cycle.save_state(path, state)
    loaded = advance_cycle.load_state(path)
    assert loaded["phase"] == "eval"
    assert loaded["mode"] == "candidate"
    assert loaded["updated_at"]
    assert advance_cycle.load_state(tmp_path / "missing.json") is None
