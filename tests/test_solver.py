"""Tests for the competition solver (src/evaluation/solver.py).

Covers the metric implementations, reference baselines and
normalization, the mocked-model end-to-end solve, timeout/crash
paths (unmeasured, never a fake score), the leakage guard,
run_evaluation aggregation with a mocked solver, and eval-kernel
packaging in both modes. No test touches the network, Kaggle, or
a GPU: downloads and generation are mocked, execution uses the
real sandboxed Executor on tiny generated programs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.evaluation.competition_evaluator import _normalized_score
from src.evaluation.solver import (
    COMPETITION_SPECS,
    CompetitionSolver,
    DataUnavailableError,
    build_solver,
    canonical_metric,
    check_training_leakage,
    compute_metric,
    extract_code,
)

ROOT = Path(__file__).resolve().parent.parent

# A generate mock that returns code containing every regression
# probe's expected substring (used for run_evaluation tests).
PROBE_GENERATE_TEXT = (
    "yes 60 Mixed 42 36 def read_csv for i in range(5): BLUE Red"
)

TITANIC_ROWS = [
    ["PassengerId", "Survived", "Pclass", "Sex", "Age"],
    ["1", "0", "3", "male", "22"],
    ["2", "1", "1", "female", "38"],
    ["3", "1", "3", "female", "26"],
    ["4", "1", "1", "female", "35"],
    ["5", "0", "3", "male", "35"],
    ["6", "0", "3", "male", "28"],
    ["7", "0", "1", "male", "54"],
    ["8", "0", "3", "male", "2"],
    ["9", "1", "3", "female", "27"],
    ["10", "1", "2", "female", "14"],
]

HOUSE_ROWS = [
    ["Id", "GrLivArea", "SalePrice"],
    ["1", "1000", "100000"],
    ["2", "1500", "150000"],
    ["3", "800", "80000"],
    ["4", "2000", "200000"],
    ["5", "1200", "120000"],
    ["6", "900", "90000"],
    ["7", "1800", "180000"],
    ["8", "1100", "110000"],
    ["9", "1300", "130000"],
    ["10", "1600", "160000"],
]


def _write_train_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _downloader_for(rows: list[list[str]]):
    """A downloader mock that writes *rows* as train.csv."""

    def _download(competition_id: str, dest_dir: Path) -> Path:
        _write_train_csv(Path(dest_dir) / "train.csv", rows)
        return Path(dest_dir)

    return _download


def _failing_downloader(competition_id: str, dest_dir: Path) -> Path:
    raise DataUnavailableError(f"no such competition: {competition_id}")


def _titanic_solution() -> str:
    """Generated code: the Sex rule, written as a solution would be."""
    return (
        "import csv\n"
        "with open('val.csv', newline='') as f:\n"
        "    rows = list(csv.DictReader(f))\n"
        "with open('predictions.csv', 'w', newline='') as f:\n"
        "    w = csv.writer(f)\n"
        "    w.writerow(['id', 'prediction'])\n"
        "    for r in rows:\n"
        "        w.writerow([r['PassengerId'], 1 if r['Sex'] == 'female' else 0])\n"
    )


def _median_solution(target: str, id_col: str) -> str:
    return (
        "import csv, statistics\n"
        "with open('train_local.csv', newline='') as f:\n"
        f"    targets = [float(r['{target}']) for r in csv.DictReader(f)]\n"
        "med = statistics.median(targets)\n"
        "with open('val.csv', newline='') as f:\n"
        "    rows = list(csv.DictReader(f))\n"
        "with open('predictions.csv', 'w', newline='') as f:\n"
        "    w = csv.writer(f)\n"
        "    w.writerow(['id', 'prediction'])\n"
        "    for r in rows:\n"
        f"        w.writerow([r['{id_col}'], med])\n"
    )


def _solver(tmp_path: Path, generate, *, rows, timeout_s: int = 60) -> CompetitionSolver:
    return CompetitionSolver(
        generate,
        data_root=tmp_path / "data",
        execution_timeout_s=timeout_s,
        downloader=_downloader_for(rows),
    )


# --- metrics ---------------------------------------------------------------


def test_metric_accuracy_rmse_rmsle():
    assert compute_metric("accuracy", [1, 0, 1], [1, 0, 0]) == pytest.approx(2 / 3)
    assert compute_metric("rmse", [3.0, 5.0], [3.0, 5.0]) == pytest.approx(0.0)
    assert compute_metric("rmsle", [100.0, 100.0], [100.0, 100.0]) == pytest.approx(0.0)
    # RMSLE clips negative predictions at 0 instead of crashing.
    assert compute_metric("rmsle", [10.0], [-5.0]) == pytest.approx(
        compute_metric("rmsle", [10.0], [0.0])
    )


def test_metric_f1_and_roc_auc():
    assert compute_metric("f1", [1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(2 / 3)
    assert compute_metric("f1", [1, 1], [0, 0]) == 0.0
    assert compute_metric("roc_auc", [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert compute_metric("roc_auc", [0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_canonical_metric_aliases():
    assert canonical_metric("Accuracy") == "accuracy"
    assert canonical_metric("RMSLE (log RMSE)") == "rmsle"
    assert canonical_metric("ROC AUC") == "roc_auc"
    with pytest.raises(ValueError, match="unknown metric"):
        canonical_metric("MAPE")


def test_normalization_both_directions():
    # Higher-is-better: score / baseline.
    assert _normalized_score(0.9, 0.8, "higher_is_better") == pytest.approx(1.125)
    # Lower-is-better: baseline / score.
    assert _normalized_score(0.4, 0.8, "lower_is_better") == pytest.approx(2.0)
    assert _normalized_score(None, 0.8, "higher_is_better") is None
    assert _normalized_score(0.0, 0.8, "lower_is_better") is None


# --- code extraction ---------------------------------------------------------


def test_extract_code_prefers_fenced_block():
    text = "Here is the solution:\n```python\nprint('hi')\n```\nDone."
    assert extract_code(text) == "print('hi')\n"
    assert extract_code("print('raw')") == "print('raw')\n"
    with pytest.raises(ValueError, match="no text"):
        extract_code("   ")


# --- end-to-end solve with mocked generation ---------------------------------


def test_solver_titanic_measured_against_sex_rule_reference(tmp_path: Path):
    solver = _solver(tmp_path, lambda prompt: _titanic_solution(), rows=TITANIC_ROWS)
    result = solver({"competition_id": "titanic", "task_type": "tabular_classification", "metric": "Accuracy"})
    assert result["status"] == "run"
    assert result["submission_valid"] is True
    # The Sex-rule solution matches the Sex-rule reference exactly.
    assert result["score"] == pytest.approx(result["baseline_score"])
    assert result["normalized_score"] == pytest.approx(1.0)
    assert result["reference_method"] == "sex_rule"
    assert result["metric_direction"] == "higher_is_better"
    assert result["target_column"] == "Survived"
    assert result["fold_scores"] == [result["score"]]


def test_solver_house_prices_lower_is_better(tmp_path: Path):
    solver = _solver(
        tmp_path,
        lambda prompt: _median_solution("SalePrice", "Id"),
        rows=HOUSE_ROWS,
    )
    result = solver({
        "competition_id": "house-prices-advanced-regression-techniques",
        "task_type": "tabular_regression", "metric": "RMSLE (log RMSE)",
    })
    assert result["status"] == "run"
    assert result["metric_direction"] == "lower_is_better"
    # Median solution == median reference: normalized score 1.0.
    assert result["normalized_score"] == pytest.approx(1.0)
    assert result["baseline_score"] is not None and result["baseline_score"] > 0


def test_solver_prompt_contains_schema_and_contract(tmp_path: Path):
    seen: dict = {}

    def _capture(prompt: str) -> str:
        seen["prompt"] = prompt
        return _titanic_solution()

    solver = _solver(tmp_path, _capture, rows=TITANIC_ROWS)
    solver({"competition_id": "titanic", "metric": "Accuracy"})
    prompt = seen["prompt"]
    assert "titanic" in prompt
    assert "Survived" in prompt and "PassengerId" in prompt
    assert "train_local.csv" in prompt and "val.csv" in prompt
    assert "predictions.csv" in prompt and "id,prediction" in prompt
    assert "Kaggle API" in prompt


def test_solver_unknown_competition_is_error_not_fake_score(tmp_path: Path):
    solver = _solver(tmp_path, lambda prompt: "print(1)", rows=TITANIC_ROWS)
    result = solver({"competition_id": "no-such-competition", "metric": "Accuracy"})
    assert result["status"] == "error"
    assert result["score"] is None and result["normalized_score"] is None


def test_solver_download_failure_is_not_run_with_null_scores(tmp_path: Path):
    solver = CompetitionSolver(
        lambda prompt: _titanic_solution(),
        data_root=tmp_path / "data",
        downloader=_failing_downloader,
    )
    result = solver({"competition_id": "titanic", "metric": "Accuracy"})
    assert result["status"] == "not_run"
    assert result["score"] is None
    assert result["normalized_score"] is None
    assert result["submission_valid"] is None


def test_solver_timeout_records_error_never_fake_score(tmp_path: Path):
    solver = _solver(
        tmp_path,
        lambda prompt: "import time\ntime.sleep(30)\n",
        rows=TITANIC_ROWS,
        timeout_s=2,
    )
    result = solver({"competition_id": "titanic", "metric": "Accuracy"})
    assert result["status"] == "error"
    assert result["score"] is None and result["normalized_score"] is None
    assert "timed out" in result["error"]


def test_solver_crashing_solution_records_error(tmp_path: Path):
    solver = _solver(
        tmp_path, lambda prompt: "raise RuntimeError('boom')\n", rows=TITANIC_ROWS
    )
    result = solver({"competition_id": "titanic", "metric": "Accuracy"})
    assert result["status"] == "error"
    assert result["score"] is None
    assert "boom" in result["error"]


def test_solver_missing_predictions_is_error(tmp_path: Path):
    solver = _solver(
        tmp_path, lambda prompt: "print('no output file')\n", rows=TITANIC_ROWS
    )
    result = solver({"competition_id": "titanic", "metric": "Accuracy"})
    assert result["status"] == "error"
    assert result["submission_valid"] is False
    assert result["score"] is None


def test_solver_wrong_row_count_is_invalid_submission(tmp_path: Path):
    bad = (
        "import csv\n"
        "with open('predictions.csv', 'w', newline='') as f:\n"
        "    w = csv.writer(f); w.writerow(['id', 'prediction']); w.writerow(['1', 1])\n"
    )
    solver = _solver(tmp_path, lambda prompt: bad, rows=TITANIC_ROWS)
    result = solver({"competition_id": "titanic", "metric": "Accuracy"})
    assert result["status"] == "error"
    assert result["submission_valid"] is False
    assert result["score"] is None


def test_solver_empty_generation_is_error(tmp_path: Path):
    solver = _solver(tmp_path, lambda prompt: "   ", rows=TITANIC_ROWS)
    result = solver({"competition_id": "titanic", "metric": "Accuracy"})
    assert result["status"] == "error"
    assert result["score"] is None


def test_solver_reference_keyword_rule_nonzero_f1(tmp_path: Path):
    # 50 rows so the hash-based val split holds both classes (a
    # 10-row fixture can leave val with no positives at all).
    keywords = ["fire", "flood", "earthquake", "explosion", "hurricane"]
    rows = [["id", "text", "target"]]
    for i in range(1, 51):
        if i % 2 == 1:
            rows.append([str(i), f"there was a {keywords[i % 5]} downtown", "1"])
        else:
            rows.append([str(i), "just had coffee and watched a movie", "0"])
    solver = _solver(tmp_path, lambda prompt: "", rows=rows)
    # Drive only the reference path via a keyword-rule solution.
    code = (
        "import csv\n"
        "words = ['fire','flood','earthquake','explosion','crash','hurricane',"
        "'tornado','bomb','killed','dead','disaster','emergency']\n"
        "with open('val.csv', newline='') as f:\n"
        "    rows = list(csv.DictReader(f))\n"
        "with open('predictions.csv', 'w', newline='') as f:\n"
        "    w = csv.writer(f); w.writerow(['id', 'prediction'])\n"
        "    for r in rows:\n"
        "        w.writerow([r['id'], 1 if any(x in r['text'].lower() for x in words) else 0])\n"
    )
    solver.generate = lambda prompt: code
    result = solver({"competition_id": "nlp-getting-started", "metric": "F1"})
    assert result["status"] == "run"
    # Keyword reference has a nonzero F1, so normalization is defined.
    assert result["baseline_score"] is not None and result["baseline_score"] > 0
    assert result["normalized_score"] is not None


def test_solver_split_is_deterministic_across_instances(tmp_path: Path):
    def _run() -> dict:
        solver = _solver(tmp_path, lambda prompt: _titanic_solution(), rows=TITANIC_ROWS)
        return solver({"competition_id": "titanic", "metric": "Accuracy"})

    first = _run()
    second = _run()
    assert first["score"] == second["score"]
    assert first["baseline_score"] == second["baseline_score"]
    assert first["val_rows"] == second["val_rows"]


def test_all_manifest_competitions_have_specs():
    manifest = json.loads((ROOT / "evaluation" / "benchmark_manifest.json").read_text())
    ids = [c["id"] for c in manifest["held_out_competitions"]]
    assert len(ids) == 12
    for competition_id in ids:
        assert competition_id in COMPETITION_SPECS, competition_id
        spec = COMPETITION_SPECS[competition_id]
        assert spec["metric"] in ("accuracy", "rmsle", "rmse", "f1", "roc_auc")
        assert spec["reference"] in (
            "majority", "median", "sex_rule", "cryosleep_rule",
            "keyword_rule", "constant_half",
        )


# --- leakage guard ------------------------------------------------------------


def test_leakage_guard_refuses_held_out_id_in_train_file(tmp_path: Path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(
        json.dumps({"competition_id": "titanic", "task_description": "x"}) + "\n"
        + json.dumps({"competition_id": "adult-census-income"}) + "\n"
    )
    with pytest.raises(ValueError, match="titanic"):
        check_training_leakage(train_file, ["titanic", "digit-recognizer"])


def test_leakage_guard_clean_file_passes(tmp_path: Path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(
        json.dumps({"competition_id": "adult-census-income"}) + "\n"
    )
    report = check_training_leakage(train_file, ["titanic"])
    assert report["clean"] is True


def test_build_solver_refuses_before_loading_model(tmp_path: Path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"competition_id": "titanic"}) + "\n")
    with pytest.raises(ValueError, match="titanic"):
        build_solver(
            "base",
            config={},
            data_root=tmp_path / "data",
            train_file=train_file,
            held_out_ids=["titanic"],
        )


# --- run_evaluation wiring ------------------------------------------------------


def test_run_evaluation_with_mocked_solver_aggregates(tmp_path: Path):
    from src.evaluation.benchmark_runner import run_evaluation

    def _mock_solver(competition: dict) -> dict:
        return {
            "status": "run",
            "score": 0.8,
            "normalized_score": 1.1,
            "submission_valid": True,
            "baseline_score": 0.7,
            "fold_scores": [0.8],
        }

    summary = run_evaluation(
        ROOT / "configs" / "evaluation.yaml",
        "base",
        solver=_mock_solver,
        generate=lambda prompt: PROBE_GENERATE_TEXT,
    )
    assert summary["normalized_score"] == pytest.approx(1.1)
    assert summary["submission_validity"] == pytest.approx(1.0)
    assert len(summary["results"]) == 12
    assert all(e["status"] == "run" for e in summary["results"])
    # Regression probes ran with the same generation callable and
    # every probe's expected substring is in the mock output.
    assert summary["regression"]["total"] == 10
    assert summary["regression"]["passed"] == 10
    assert summary["regression_probes"]["reasoning"] == pytest.approx(1.0)
    assert summary["regression_probes"]["coding"] == pytest.approx(1.0)
    assert summary["regression_probes"]["instruction_following"] == pytest.approx(1.0)


def test_run_evaluation_with_mocked_generate_uses_real_solver(tmp_path: Path):
    from src.evaluation.benchmark_runner import run_evaluation

    # Only titanic has downloadable (mocked) data here; the other
    # 11 competitions fail download and stay unmeasured.
    solver = _solver(tmp_path, lambda prompt: _titanic_solution(), rows=TITANIC_ROWS)
    # Patch the default downloader path: the solver's own downloader
    # only knows titanic rows, so other ids raise DataUnavailable.
    original = solver.downloader

    def _only_titanic(competition_id: str, dest_dir: Path) -> Path:
        if competition_id != "titanic":
            raise DataUnavailableError(f"no data in test: {competition_id}")
        return original(competition_id, dest_dir)

    solver.downloader = _only_titanic
    summary = run_evaluation(
        ROOT / "configs" / "evaluation.yaml",
        "base",
        solver=solver,
        generate=lambda prompt: PROBE_GENERATE_TEXT,
    )
    titanic = next(
        e for e in summary["results"] if e["competition_id"] == "titanic"
    )
    assert titanic["status"] == "run"
    assert titanic["normalized_score"] == pytest.approx(1.0)
    others = [e for e in summary["results"] if e["competition_id"] != "titanic"]
    assert all(e["status"] == "not_run" for e in others)
    assert all(e["normalized_score"] is None for e in others)
    # The aggregate covers only the measured competition.
    assert summary["normalized_score"] == pytest.approx(1.0)
    assert summary["submission_validity"] == pytest.approx(1.0)


def test_run_evaluation_refuses_leaked_train_file(tmp_path: Path):
    from src.evaluation.benchmark_runner import run_evaluation

    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"competition_id": "titanic"}) + "\n")
    with pytest.raises(ValueError, match="titanic"):
        run_evaluation(
            ROOT / "configs" / "evaluation.yaml",
            "base",
            solver=lambda c: {"status": "run", "score": 1.0,
                              "normalized_score": 1.0, "submission_valid": True},
            train_file=train_file,
        )


# --- eval kernel packaging ------------------------------------------------------


def test_package_eval_kernel_includes_solver_both_modes(tmp_path: Path):
    from package_eval_kernel import build_kernel

    for mode in ("candidate", "base"):
        out = tmp_path / f"eval-{mode}"
        summary = build_kernel(ROOT, out, mode)
        assert summary["mode"] == mode
        assert (out / "src" / "evaluation" / "solver.py").is_file()
        assert (out / "scripts" / "kaggle_auth.py").is_file()
        assert (out / "src" / "agent" / "executor.py").is_file()
        assert (out / "configs" / "evaluation.yaml").is_file()
        assert (out / "07_evaluation.ipynb").is_file()
