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

import package_eval_kernel
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


def test_trigger_push_quota_in_success_output_is_skip(tmp_path: Path, monkeypatch):
    """The CLI exits 0 even when the API refuses the push in the
    response body (200 with an error field) -- e.g. quota. That is a
    skip, not a submitted kernel (seen live 2026-11-18, run #6)."""
    monkeypatch.setattr(
        trigger_kaggle, "configure_kaggle_auth",
        lambda: {"method": "test"},
    )
    monkeypatch.setattr(trigger_kaggle.shutil, "which", lambda _: "/usr/bin/kaggle")
    monkeypatch.setattr(
        trigger_kaggle, "_run_cli",
        lambda args: subprocess.CompletedProcess(
            args, returncode=0, stdout="",
            stderr="Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.",
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


def test_trigger_kernel_error_on_quota_is_skip(tmp_path: Path, monkeypatch):
    """A kernel that errors immediately because the weekly GPU quota
    is exhausted never really started -- skip, don't fail (run #6)."""
    monkeypatch.setattr(
        trigger_kaggle, "configure_kaggle_auth",
        lambda: {"method": "test"},
    )
    monkeypatch.setattr(trigger_kaggle.shutil, "which", lambda _: "/usr/bin/kaggle")

    def _fake_cli(args):
        if args[:2] == ["kernels", "push"]:
            return subprocess.CompletedProcess(
                args, returncode=0, stdout="Kernel submitted", stderr=""
            )
        return subprocess.CompletedProcess(
            args, returncode=0,
            stdout=(
                'monibeefudgekins/akarui-eval-baseline has status "error"\n'
                'Failure message: "Maximum weekly GPU quota of 30.00 hours reached."'
            ),
            stderr="",
        )

    monkeypatch.setattr(trigger_kaggle, "_run_cli", _fake_cli)
    monkeypatch.setattr(trigger_kaggle.time, "sleep", lambda _: None)
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


def test_diagnose_reports_api_error_field_as_failure(tmp_path: Path, monkeypatch):
    """The diagnostic must not call a push 'SUCCEEDED' when the API
    refused it in the response body (200 with an error field)."""
    import diagnose_kernel_push

    notebook_dir = tmp_path / "eval"
    notebook_dir.mkdir()
    (notebook_dir / "07_evaluation.ipynb").write_text("{}")
    (notebook_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": "monibeefudgekins/akarui-eval-baseline",
        "title": "Akarui Eval Baseline",
        "kernel_type": "notebook",
        "code_file": "07_evaluation.ipynb",
    }))

    class _FakeApi:
        def authenticate(self):
            return None

        def kernels_push(self, folder):
            from types import SimpleNamespace

            return SimpleNamespace(
                error="Maximum weekly GPU quota of 30.00 hours reached.",
                url="", versionNumber=0,
            )

    monkeypatch.setattr(
        diagnose_kernel_push, "configure_kaggle_auth",
        lambda: {"method": "test"},
    )
    # The test venv has no `kaggle` package; stub the module the
    # diagnostic imports KaggleApi from at call time.
    import sys
    import types

    fake_api_module = types.ModuleType("kaggle.api.kaggle_api_extended")
    fake_api_module.KaggleApi = _FakeApi
    monkeypatch.setitem(sys.modules, "kaggle", types.ModuleType("kaggle"))
    monkeypatch.setitem(sys.modules, "kaggle.api", types.ModuleType("kaggle.api"))
    monkeypatch.setitem(
        sys.modules, "kaggle.api.kaggle_api_extended", fake_api_module
    )
    rc = diagnose_kernel_push.main(["--notebook-dir", str(notebook_dir)])
    assert rc == 1


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


# --- Kaggle kernel title length -----------------------------------------
# The SaveKernel API rejects titles over 50 chars with HTTP 400
# ("The title cannot exceed 50 characters."). Seen live 2026-11-18:
# the 60-char base eval title failed three pushes before the
# diagnostic printed the API's message. These tests pin the limit.


def test_all_kernel_titles_within_kaggle_limit():
    assert len(package_kernel.KERNEL_TITLE) <= package_kernel.MAX_TITLE_LENGTH
    for mode, title in package_eval_kernel.KERNEL_TITLES.items():
        assert len(title) <= package_eval_kernel.MAX_TITLE_LENGTH, mode


def test_all_kernel_titles_slugify_to_their_ids():
    """Kaggle creates a kernel under its TITLE-derived slug when the
    title does not slugify to the id, so the push "succeeds" at the
    wrong URL as an unrun draft and the status poll finds nothing
    (live failure 2026-11-18, run #5). Glimmer and gemma match."""
    assert (
        package_kernel._slugify(package_kernel.KERNEL_TITLE)
        == package_kernel.KERNEL_ID.split("/", 1)[1]
    )
    for mode, title in package_eval_kernel.KERNEL_TITLES.items():
        slug = package_eval_kernel.KERNEL_IDS[mode].split("/", 1)[1]
        assert package_eval_kernel._slugify(title) == slug, mode


def test_package_refuses_overlong_title(tmp_path: Path, monkeypatch):
    repo = _minimal_repo(tmp_path, train_lines=_jsonl(VALID_RECORDS))
    monkeypatch.setattr(
        package_kernel, "KERNEL_TITLE", "x" * (package_kernel.MAX_TITLE_LENGTH + 1)
    )
    with pytest.raises(ValueError, match="over Kaggle"):
        package_kernel.build_kernel(repo, tmp_path / "kernel")


def test_package_refuses_title_slug_mismatch(tmp_path: Path, monkeypatch):
    repo = _minimal_repo(tmp_path, train_lines=_jsonl(VALID_RECORDS))
    monkeypatch.setattr(package_kernel, "KERNEL_TITLE", "A Different Title")
    with pytest.raises(ValueError, match="slugifies"):
        package_kernel.build_kernel(repo, tmp_path / "kernel")


def _minimal_eval_repo(tmp_path: Path) -> Path:
    """Build a minimal repo tree with the files package_eval_kernel needs."""
    repo = tmp_path / "eval-repo"
    (repo / "notebooks").mkdir(parents=True)
    shutil.copy2(ROOT / "notebooks" / "07_evaluation.ipynb",
                 repo / "notebooks" / "07_evaluation.ipynb")
    shutil.copytree(ROOT / "src", repo / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (repo / "scripts").mkdir()
    for name in ("evaluate.py", "verify_release.py", "kaggle_auth.py"):
        shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "configs").mkdir()
    shutil.copy2(ROOT / "configs" / "evaluation.yaml",
                 repo / "configs" / "evaluation.yaml")
    (repo / "evaluation").mkdir()
    shutil.copy2(ROOT / "evaluation" / "benchmark_manifest.json",
                 repo / "evaluation" / "benchmark_manifest.json")
    shutil.copy2(ROOT / "requirements-kaggle.txt",
                 repo / "requirements-kaggle.txt")
    (repo / "data").mkdir()
    (repo / "data" / "train.jsonl").write_text(
        "\n".join(_jsonl(VALID_RECORDS)) + "\n"
    )
    return repo


def test_eval_package_writes_short_titles(tmp_path: Path):
    repo = _minimal_eval_repo(tmp_path)
    for mode in ("candidate", "base"):
        out = tmp_path / f"eval-{mode}"
        package_eval_kernel.build_kernel(repo, out, mode)
        metadata = json.loads((out / "kernel-metadata.json").read_text())
        assert len(metadata["title"]) <= package_eval_kernel.MAX_TITLE_LENGTH
        assert metadata["id"] == package_eval_kernel.KERNEL_IDS[mode]


def test_eval_package_refuses_overlong_title(tmp_path: Path, monkeypatch):
    repo = _minimal_eval_repo(tmp_path)
    monkeypatch.setitem(
        package_eval_kernel.KERNEL_TITLES,
        "base",
        "x" * (package_eval_kernel.MAX_TITLE_LENGTH + 1),
    )
    with pytest.raises(ValueError, match="over Kaggle"):
        package_eval_kernel.build_kernel(repo, tmp_path / "eval", "base")


def test_eval_package_refuses_title_slug_mismatch(tmp_path: Path, monkeypatch):
    repo = _minimal_eval_repo(tmp_path)
    monkeypatch.setitem(
        package_eval_kernel.KERNEL_TITLES, "base", "A Different Title"
    )
    with pytest.raises(ValueError, match="slugifies"):
        package_eval_kernel.build_kernel(repo, tmp_path / "eval", "base")


# --- trigger_kaggle status polling --------------------------------------


def _cli_result(stdout: str = "", stderr: str = ""):
    from types import SimpleNamespace

    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=0)


def test_kernel_status_retries_unparseable_output(tmp_path: Path, monkeypatch):
    calls = {"n": 0}

    def fake_cli(args):
        calls["n"] += 1
        if calls["n"] < 3:
            return _cli_result(stdout="", stderr="")
        return _cli_result(
            stdout='monibeefudgekins/akarui-eval-baseline has status "running"'
        )

    monkeypatch.setattr(trigger_kaggle, "_run_cli", fake_cli)
    monkeypatch.setattr(trigger_kaggle.time, "sleep", lambda _: None)
    assert (
        trigger_kaggle.kernel_status("monibeefudgekins/akarui-eval-baseline")
        == "running"
    )
    assert calls["n"] == 3


def test_kernel_status_error_names_the_raw_output(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        trigger_kaggle,
        "_run_cli",
        lambda args: _cli_result(
            stdout="", stderr="unexpected response (no status field)"
        ),
    )
    monkeypatch.setattr(trigger_kaggle.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="unexpected response"):
        trigger_kaggle.kernel_status("monibeefudgekins/akarui-eval-baseline")


# --- self-bootstrap + baked mode (run #7 failure, 2026-11-20) ---------
# `kaggle kernels push` uploads ONLY the notebook file named as
# code_file, so the packaged notebook must clone the public repo at
# runtime and carry its mode baked in. These tests pin that.


def _packaged_notebook(out: Path, filename: str) -> dict:
    return json.loads((out / filename).read_text())


def _mode_lines(notebook: dict) -> list[str]:
    return [
        line.strip()
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
        for line in cell.get("source", [])
        if line.strip().startswith('mode = "')
    ]


def _bootstrap_cells(notebook: dict, marker: str) -> list[dict]:
    return [
        cell
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
        and marker in "".join(cell.get("source", []))
    ]


def test_eval_package_bakes_mode_base(tmp_path: Path):
    repo = _minimal_eval_repo(tmp_path)
    out = tmp_path / "eval-base"
    package_eval_kernel.build_kernel(repo, out, "base")
    notebook = _packaged_notebook(out, "07_evaluation.ipynb")
    assert _mode_lines(notebook) == ['mode = "base"']
    assert len(_bootstrap_cells(
        notebook, package_eval_kernel.BOOTSTRAP_MARKER)) == 1
    assert json.loads((out / "eval-mode.json").read_text())["mode"] == "base"


def test_eval_package_bakes_mode_candidate(tmp_path: Path):
    repo = _minimal_eval_repo(tmp_path)
    out = tmp_path / "eval-candidate"
    package_eval_kernel.build_kernel(repo, out, "candidate")
    notebook = _packaged_notebook(out, "07_evaluation.ipynb")
    assert _mode_lines(notebook) == ['mode = "candidate"']
    assert len(_bootstrap_cells(
        notebook, package_eval_kernel.BOOTSTRAP_MARKER)) == 1
    assert (
        json.loads((out / "eval-mode.json").read_text())["mode"] == "candidate"
    )


def test_eval_package_refuses_notebook_without_mode_line(tmp_path: Path):
    repo = _minimal_eval_repo(tmp_path)
    notebook_path = repo / "notebooks" / "07_evaluation.ipynb"
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cell["source"] = [
            line for line in cell.get("source", [])
            if line.strip() != 'mode = "candidate"'
        ]
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    with pytest.raises(ValueError, match="exactly one"):
        package_eval_kernel.build_kernel(repo, tmp_path / "eval", "base")


def test_eval_package_refuses_notebook_with_duplicate_mode_line(tmp_path: Path):
    repo = _minimal_eval_repo(tmp_path)
    notebook_path = repo / "notebooks" / "07_evaluation.ipynb"
    notebook = json.loads(notebook_path.read_text())
    done = False
    for cell in notebook["cells"]:
        if done or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        for index, line in enumerate(source):
            if line.strip() == 'mode = "candidate"':
                source.insert(index + 1, 'mode = "candidate"\n')
                done = True
                break
    assert done
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    with pytest.raises(ValueError, match="exactly one"):
        package_eval_kernel.build_kernel(repo, tmp_path / "eval", "base")


def test_sft_package_contains_bootstrap_cell(tmp_path: Path):
    repo = _minimal_repo(tmp_path, train_lines=_jsonl(VALID_RECORDS))
    out = tmp_path / "kernel"
    package_kernel.build_kernel(repo, out)
    notebook = _packaged_notebook(out, "05_sft_training.ipynb")
    assert len(_bootstrap_cells(
        notebook, package_kernel.BOOTSTRAP_MARKER)) == 1
    # The training notebook has no mode: nothing to bake, and the
    # packager must not invent one.
    assert _mode_lines(notebook) == []


def test_sft_package_refuses_notebook_without_bootstrap_cell(tmp_path: Path):
    repo = _minimal_repo(tmp_path, train_lines=_jsonl(VALID_RECORDS))
    notebook_path = repo / "notebooks" / "05_sft_training.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"] = [
        cell for cell in notebook["cells"]
        if package_kernel.BOOTSTRAP_MARKER
        not in "".join(cell.get("source", []))
    ]
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    with pytest.raises(ValueError, match="self-bootstrap"):
        package_kernel.build_kernel(repo, tmp_path / "kernel")


def test_eval_package_refuses_notebook_without_bootstrap_cell(tmp_path: Path):
    repo = _minimal_eval_repo(tmp_path)
    notebook_path = repo / "notebooks" / "07_evaluation.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"] = [
        cell for cell in notebook["cells"]
        if package_eval_kernel.BOOTSTRAP_MARKER
        not in "".join(cell.get("source", []))
    ]
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    with pytest.raises(ValueError, match="self-bootstrap"):
        package_eval_kernel.build_kernel(repo, tmp_path / "eval", "base")


# --- dependency-install cell (run #8 failure, 2026-11-21) ---------------
# The Kaggle image lacks bitsandbytes, so a 4-bit model load fails with
# ImportError unless the kernel installs requirements-kaggle.txt
# itself. These tests pin that cell in both packaged notebooks.


def test_sft_package_contains_install_cell(tmp_path: Path):
    repo = _minimal_repo(tmp_path, train_lines=_jsonl(VALID_RECORDS))
    out = tmp_path / "kernel"
    package_kernel.build_kernel(repo, out)
    notebook = _packaged_notebook(out, "05_sft_training.ipynb")
    assert len(_bootstrap_cells(
        notebook, package_kernel.INSTALL_MARKER)) == 1


def test_eval_package_contains_install_cell(tmp_path: Path):
    repo = _minimal_eval_repo(tmp_path)
    out = tmp_path / "eval-base"
    package_eval_kernel.build_kernel(repo, out, "base")
    notebook = _packaged_notebook(out, "07_evaluation.ipynb")
    assert len(_bootstrap_cells(
        notebook, package_eval_kernel.INSTALL_MARKER)) == 1


def test_sft_package_refuses_notebook_without_install_cell(tmp_path: Path):
    repo = _minimal_repo(tmp_path, train_lines=_jsonl(VALID_RECORDS))
    notebook_path = repo / "notebooks" / "05_sft_training.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"] = [
        cell for cell in notebook["cells"]
        if package_kernel.INSTALL_MARKER
        not in "".join(cell.get("source", []))
    ]
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    with pytest.raises(ValueError, match="dependency-install"):
        package_kernel.build_kernel(repo, tmp_path / "kernel")


def test_eval_package_refuses_notebook_without_install_cell(tmp_path: Path):
    repo = _minimal_eval_repo(tmp_path)
    notebook_path = repo / "notebooks" / "07_evaluation.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"] = [
        cell for cell in notebook["cells"]
        if package_eval_kernel.INSTALL_MARKER
        not in "".join(cell.get("source", []))
    ]
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    with pytest.raises(ValueError, match="dependency-install"):
        package_eval_kernel.build_kernel(repo, tmp_path / "eval", "base")
