"""Tests for the agent components: researcher, executor, memory, tools."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.agent.executor import Executor
from src.agent.experiment_memory import ExperimentMemory
from src.agent.researcher import Researcher
from src.agent.tool_registry import get_available_tools


def test_researcher_plan_structure():
    researcher = Researcher(generate=lambda prompt: "unused")
    plan = researcher.plan(
        "Predict survival from passenger features.",
        {"features": ["Pclass", "Sex", "Age"]},
        "Accuracy",
    )
    assert plan["task_type"]
    assert plan["validation_strategy"]
    assert plan["hypotheses"]
    assert plan["experiments"]


def test_researcher_time_series_validation():
    researcher = Researcher(generate=lambda prompt: "unused")
    plan = researcher.plan(
        "Forecast daily store sales.",
        {"features": ["date", "store", "sales"]},
        "RMSLE",
    )
    assert plan["validation_strategy"]["method"] == "time_based_split"


def test_executor_runs_code_and_captures_output(tmp_path: Path):
    executor = Executor(sandbox_dir=tmp_path)
    result = executor.run_code("print('hello akarui')")
    assert result.succeeded
    assert "hello akarui" in result.stdout
    assert result.returncode == 0


def test_executor_reports_errors(tmp_path: Path):
    executor = Executor(sandbox_dir=tmp_path)
    result = executor.run_code("raise ValueError('boom')")
    assert not result.succeeded
    assert "boom" in result.stderr


def test_executor_scrubs_credentials(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KAGGLE_TOKEN", "secret-value-that-must-not-leak")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    executor = Executor(sandbox_dir=tmp_path)
    result = executor.run_code(
        "import os; print(os.environ.get('KAGGLE_TOKEN')); "
        "print(os.environ.get('HF_TOKEN'))"
    )
    assert "secret-value-that-must-not-leak" not in result.stdout
    assert "hf-secret" not in result.stdout


def test_executor_timeout(tmp_path: Path):
    executor = Executor(sandbox_dir=tmp_path)
    result = executor.run_code("import time; time.sleep(30)", timeout_s=1)
    assert not result.succeeded
    assert result.timed_out


def test_experiment_memory_best(tmp_path: Path):
    memory = ExperimentMemory(tmp_path / "experiments.jsonl")
    memory.append("e1", "titanic", {"model": "logreg"}, 0.80, "ok")
    memory.append("e2", "titanic", {"model": "xgboost"}, 0.83, "ok")
    memory.append("e3", "titanic", {"model": "broken"}, None, "failed")
    best = memory.best_experiment("titanic")
    assert best is not None
    assert best["experiment_id"] == "e2"
    assert memory.query("e1")["validation_score"] == 0.80
    assert memory.query("missing") is None
    assert len(memory.list("titanic")) == 3


def test_tool_registry_lists_all_tools():
    tools = get_available_tools()
    assert len(tools) == 12
    names = {tool["name"] for tool in tools}
    assert {"pandas", "numpy", "scikit-learn", "lightgbm", "optuna"} <= names
    assert all("available" in tool for tool in tools)
