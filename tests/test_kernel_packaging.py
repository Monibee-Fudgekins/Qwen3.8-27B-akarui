"""Tests for kernel packaging, the trigger quota-skip, and the 11h guard.

Runs without torch/transformers/GPU and without network. The Kaggle
CLI is never invoked: trigger tests monkeypatch the CLI runner.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import package_kernel
import trigger_kaggle
from src.training.runtime_guard import RuntimeGuard, resolve_runtime_guard

ROOT = Path(__file__).resolve().parent.parent

VALID_RECORDS = [
    {
        "competition_id": "some-other-competition",
        "task_description": "Predict churn.",
        "generated_code": "print('hello')",
        "reasoning_summary": "Try logistic regression.",
    },
    {
        "messages": [
            {"role": "user", "content": "Plan a validation strategy."},
            {"role": "assistant", "content": "Use time-based splits."},
        ],
    },
]


def _minimal_repo(tmp_path: Path, train_lines: list[str] | None) -> Path:
    """Build a minimal repo tree with the files package_kernel needs."""
    repo = tmp_path / "repo"
    (repo / "notebooks").mkdir(parents=True)
    shutil.copy2(ROOT / "notebooks" / "05_sft_training.ipynb",
                 repo / "notebooks" / "05_sft_training.ipynb")
    shutil.copytree(ROOT / "src", repo / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (repo / "scripts").mkdir()
    for name in ("prepare_training.py", "train.py"):
        shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "configs").mkdir()
    for name in ("base.yaml", "sft.yaml"):
        shutil.copy2(ROOT / "configs" / name, repo / "configs" / name)
    (repo / "evaluation").mkdir()
    shutil.copy2(ROOT / "evaluation" / "benchmark_manifest.json",
                 repo / "evaluation" / "benchmark_manifest.json")
    shutil.copy2(ROOT / "requirements-kaggle.txt",
                 repo / "requirements-kaggle.txt")
    if train_lines is not None:
        (repo / "data").mkdir()
        (repo / "data" / "train.jsonl").write_text("\n".join(train_lines) + "\n")
    return repo


def _jsonl(records: list[dict]) -> list[str]:
    return [json.dumps(r) for r in records]


# --- package_kernel refusal paths --------------------------------------


def test_package_refuses_missing_train_file(tmp_path: Path):
    repo = _minimal_repo(tmp_path, train_lines=None)
    with pytest.raises(ValueError, match="not found"):
        package_kernel.build_kernel(repo, tmp_path / "out")
    assert package_kernel.main(
        ["--repo-root", str(repo), "--out", str(tmp_path / "out")]
    ) == 1


def test_package_refuses_invalid_jsonl(tmp_path: Path):
    repo = _minimal_repo(tmp_path, train_lines=["{not json"])
    with pytest.raises(ValueError, match="not valid JSON"):
        package_kernel.build_kernel(repo, tmp_path / "out")


def test_package_refuses_held_out_competition(tmp_path: Path):
    bad = dict(VALID_RECORDS[0], competition_id="titanic")
    repo = _minimal_repo(tmp_path, train_lines=_jsonl([bad]))
    with pytest.raises(ValueError, match="held-out"):
        package_kernel.build_kernel(repo, tmp_path / "out")


def test_package_refuses_no_usable_records(tmp_path: Path):
    repo = _minimal_repo(tmp_path, train_lines=_jsonl([{"foo": 1}]))
    with pytest.raises(ValueError, match="no usable training records"):
        package_kernel.build_kernel(repo, tmp_path / "out")


def test_package_refuses_unsafe_output_location(tmp_path: Path):
    repo = _minimal_repo(tmp_path, train_lines=_jsonl(VALID_RECORDS))
    with pytest.raises(ValueError, match="unexpected output location"):
        package_kernel.build_kernel(repo, repo / "src" / "kernel-out")


# --- package_kernel success path ----------------------------------------


def test_package_success_builds_tree(tmp_path: Path):
    repo = _minimal_repo(tmp_path, train_lines=_jsonl(VALID_RECORDS))
    out = tmp_path / "kernel"
    summary = package_kernel.build_kernel(repo, out)
    assert summary["rows"] == 2
    assert summary["usable_rows"] == 2
    assert (out / "05_sft_training.ipynb").is_file()
    assert (out / "src" / "training" / "sft.py").is_file()
    assert (out / "scripts" / "prepare_training.py").is_file()
    assert (out / "configs" / "sft.yaml").is_file()
    assert (out / "data" / "train.jsonl").is_file()
    assert (out / "requirements-kaggle.txt").is_file()
    assert not list(out.rglob("__pycache__"))

    metadata = json.loads((out / "kernel-metadata.json").read_text())
    assert metadata["id"] == "monibeefudgekins/akarui-sft-training"
    assert metadata["code_file"] == "05_sft_training.ipynb"
    assert metadata["kernel_type"] == "notebook"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["is_private"] is True

    manifest = json.loads((out / "packaging-manifest.json").read_text())
    assert manifest["train_file_sha256"] == summary["sha256"]
    assert manifest["train_rows"] == 2
    assert manifest["base_revision"] == (
        "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
    )
    assert "titanic" in manifest["held_out_competitions_excluded"]

    # The packaged train copy is byte-identical to the source file.
    assert (out / "data" / "train.jsonl").read_bytes() == (
        repo / "data" / "train.jsonl"
    ).read_bytes()


def test_package_main_success_exit_zero(tmp_path: Path):
    repo = _minimal_repo(tmp_path, train_lines=_jsonl(VALID_RECORDS))
    assert package_kernel.main(
        ["--repo-root", str(repo), "--out", str(tmp_path / "kernel")]
    ) == 0


def test_package_preserves_kernel_readme_across_rebuilds(tmp_path: Path):
    repo = _minimal_repo(tmp_path, train_lines=_jsonl(VALID_RECORDS))
    out = tmp_path / "kernel"
    package_kernel.build_kernel(repo, out)
    (out / "README.md").write_text("# kernel docs\n")
    (out / "stale.txt").write_text("stale")
    package_kernel.build_kernel(repo, out)
    assert (out / "README.md").read_text() == "# kernel docs\n"
    assert not (out / "stale.txt").exists()


# --- trigger_kaggle quota-skip ------------------------------------------


def _fake_notebook_dir(tmp_path: Path) -> Path:
    notebook_dir = tmp_path / "kernels" / "sft"
    notebook_dir.mkdir(parents=True)
    (notebook_dir / "05_sft_training.ipynb").write_text("{}")
    (notebook_dir / "kernel-metadata.json").write_text(
        json.dumps({"id": "monibeefudgekins/akarui-sft-training"})
    )
    return notebook_dir


def test_trigger_quota_skip_exits_zero(tmp_path: Path, monkeypatch):
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
    ])
    assert rc == 0
    record = json.loads(record_path.read_text())
    assert record["status"] == "skipped_quota"
    assert "quota" in record["reason"].lower()


def test_trigger_other_push_failure_exits_one(tmp_path: Path, monkeypatch):
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
    ])
    assert rc == 1


def test_trigger_complete_still_exits_zero(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        trigger_kaggle, "configure_kaggle_auth",
        lambda: {"method": "test"},
    )
    monkeypatch.setattr(trigger_kaggle.shutil, "which", lambda _: "/usr/bin/kaggle")

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
    ])
    assert rc == 0
    assert json.loads(record_path.read_text())["status"] == "complete"


# --- 11h runtime guard ---------------------------------------------------


def test_runtime_guard_defaults_match_sft_config():
    import prepare_training

    config = prepare_training.load_config(ROOT / "configs" / "sft.yaml")
    guard = resolve_runtime_guard(config)
    assert guard["max_runtime_hours"] == 10.0
    assert guard["max_steps"] == 600
    # 600 steps x 50 s/step = 8.33h training + 1h setup = 9.33h total,
    # inside the 10h budget with 1.67h margin under the 11h cap.
    assert guard["estimated_total_hours"] == pytest.approx(9.33, abs=0.01)
    assert guard["margin_hours_under_cap"] == pytest.approx(1.67, abs=0.01)
    assert guard["steps_at_worst_rate"] == 648


def test_runtime_guard_rejects_over_cap_hours():
    with pytest.raises(ValueError, match="max_runtime_hours"):
        resolve_runtime_guard({"runtime": {"max_runtime_hours": 12.0}})


def test_runtime_guard_rejects_step_cap_over_budget():
    with pytest.raises(ValueError, match="max_steps"):
        resolve_runtime_guard({
            "runtime": {"max_runtime_hours": 10.0,
                        "setup_allowance_hours": 1.0},
            "training": {"max_steps": 700},
        })


def test_runtime_guard_deadline_math():
    guard = RuntimeGuard(10.0, start=1000.0)
    assert guard.exceeded(now=1000.0 + 10 * 3600 - 1) is False
    assert guard.exceeded(now=1000.0 + 10 * 3600) is True
    assert guard.remaining_seconds(now=1000.0) == pytest.approx(36000.0)


def test_prepare_plan_includes_runtime_guard():
    import prepare_training

    plan = prepare_training.build_plan(
        ROOT / "configs" / "sft.yaml", allow_no_gpu=True
    )
    assert plan["runtime_guard"]["max_runtime_hours"] == 10.0
    assert plan["runtime_guard"]["max_steps"] == 600
    assert plan["strategy"]["strategy"] == "unresolved"
