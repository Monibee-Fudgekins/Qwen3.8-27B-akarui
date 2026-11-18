"""Tests for the automated evaluate-and-promote lifecycle.

Covers the promotion gates (including the first-promotion rule:
the candidate must beat the MEASURED baseline, not merely have one
recorded), baseline recording, gate-decision records, champion
resolution, provenance building, the evaluate-and-promote
orchestrator (dry-run and stubbed publish), and eval-kernel
packaging. No test touches the network or Kaggle.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from src.evaluation.baseline import (  # noqa: E402
    canonical_baseline_record,
    is_measured,
    load_measured,
    measured_sample_size,
)
from src.publishing.gate_decision import (  # noqa: E402
    build_gate_decision,
    record_gate_decision,
    resolve_reference,
    update_champion,
)
from verify_release import check_gates, verify_release  # noqa: E402

GATES = {
    "min_normalized_score_improvement": 0.01,
    "max_allowed_regression": 0.02,
    "require_submission_validity": True,
    "min_submission_validity": 0.9,
}


def _results(n: int = 12, measured: int = 12, score: float = 0.6) -> list[dict]:
    out = []
    for i in range(n):
        out.append(
            {
                "competition_id": f"comp-{i}",
                "status": "run" if i < measured else "not_run",
                "normalized_score": score if i < measured else None,
                "submission_valid": True if i < measured else None,
            }
        )
    return out


def _evaluation(score: float = 0.62, validity: float = 1.0, **overrides) -> dict:
    payload = {
        "model": "adapter",
        "benchmark_version": "1",
        "normalized_score": score,
        "submission_validity": validity,
        "results": _results(),
    }
    payload.update(overrides)
    return payload


def _reference(score: float = 0.55, **overrides) -> dict:
    payload = {
        "model": "base",
        "benchmark_version": "1",
        "normalized_score": score,
        "submission_validity": 1.0,
        "sample_size": 12,
        "status": "measured",
        "results": _results(score=score),
    }
    payload.update(overrides)
    return payload


# --- promotion gates -------------------------------------------------------


def test_gate_passes_against_champion():
    assert check_gates(_evaluation(0.62), _reference(0.55), GATES) == []


def test_gate_fails_when_improvement_below_threshold():
    reasons = check_gates(_evaluation(0.555), _reference(0.55), GATES)
    assert reasons and "below the required 0.01" in reasons[0]


def test_first_promotion_must_beat_measured_baseline():
    # The fixed hole: previously a no-champion candidate passed as
    # long as a baseline score field was present, without any
    # comparison. Now the candidate must beat the baseline by the
    # same minimum improvement.
    candidate = _evaluation(0.54, baseline_normalized_score=0.55)
    reasons = check_gates(candidate, None, GATES)
    assert reasons and "baseline" in reasons[0]
    assert check_gates(_evaluation(0.57, baseline_normalized_score=0.55), None, GATES) == []


def test_gate_blocked_with_no_champion_and_no_baseline():
    reasons = check_gates(_evaluation(0.9), None, GATES)
    assert reasons and "must be measured" in reasons[0]


def test_gate_fails_on_low_submission_validity():
    reasons = check_gates(_evaluation(0.62, validity=0.5), _reference(0.55), GATES)
    assert reasons and "submission validity" in reasons[0]


def test_gate_fails_on_low_validity_against_baseline_too():
    candidate = _evaluation(0.62, validity=0.5, baseline_normalized_score=0.55)
    reasons = check_gates(candidate, None, GATES)
    assert reasons and "submission validity" in reasons[0]


def test_gate_fails_when_benchmark_incomplete():
    candidate = _evaluation(0.62)
    candidate["results"] = _results(measured=11)
    reasons = check_gates(candidate, _reference(0.55), GATES)
    assert reasons and "incomplete" in reasons[0]


def test_gate_fails_on_regression_probe_drop():
    candidate = _evaluation(0.62, regression_probes={"reasoning": 0.70})
    champion = _reference(0.55, regression_probes={"reasoning": 0.80})
    reasons = check_gates(candidate, champion, GATES)
    assert reasons and "reasoning" in reasons[0]


def test_gate_fails_when_candidate_has_no_score():
    assert check_gates({"results": []}, _reference(0.55), GATES)


def test_verify_release_end_to_end(tmp_path: Path):
    from tests.test_provenance import complete_provenance

    provenance = complete_provenance()
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(_evaluation(0.62)))
    champion_path = tmp_path / "champion.json"
    champion_path.write_text(json.dumps(_reference(0.55)))
    config_path = ROOT / "configs" / "evaluation.yaml"
    assert (
        verify_release(provenance_path, evaluation_path, champion_path, config_path)
        == []
    )
    # Failing candidate: same files, score below the gate.
    evaluation_path.write_text(json.dumps(_evaluation(0.55)))
    assert verify_release(
        provenance_path, evaluation_path, champion_path, config_path
    )


# --- baseline records ------------------------------------------------------


def test_not_measured_template_is_not_measured():
    template = json.loads((ROOT / "evaluation" / "baseline_results.json").read_text())
    assert template["status"] == "not_measured"
    assert not is_measured(template)
    assert load_measured(ROOT / "evaluation" / "baseline_results.json") is None


def test_canonical_baseline_record():
    evaluation = {
        "model": "base",
        "benchmark_version": "1",
        "normalized_score": 0.5,
        "submission_validity": 0.92,
        "results": _results(score=0.5),
    }
    record = canonical_baseline_record(evaluation)
    assert record["status"] == "measured"
    assert record["normalized_score"] == 0.5
    assert record["sample_size"] == 12
    assert record["base_revision"] == "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
    assert is_measured(record)
    assert measured_sample_size(record) == 12


def test_canonical_baseline_record_refuses_unmeasured_and_candidate():
    with pytest.raises(ValueError, match="base model"):
        canonical_baseline_record(_evaluation(0.62))
    with pytest.raises(ValueError, match="measured"):
        canonical_baseline_record(
            {"model": "base", "normalized_score": None, "results": []}
        )


def test_record_baseline_cli(tmp_path: Path):
    from record_baseline import main as record_main

    evaluation_path = tmp_path / "eval-base.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "model": "base",
                "benchmark_version": "1",
                "normalized_score": 0.5,
                "submission_validity": 1.0,
                "results": _results(score=0.5),
            }
        )
    )
    output_path = tmp_path / "baseline_results.json"
    assert record_main(["--evaluation", str(evaluation_path), "--output", str(output_path)]) == 0
    record = json.loads(output_path.read_text())
    assert record["status"] == "measured" and record["normalized_score"] == 0.5
    # Refuses to overwrite a measured baseline without --force.
    assert record_main(["--evaluation", str(evaluation_path), "--output", str(output_path)]) == 1
    assert record_main(
        ["--evaluation", str(evaluation_path), "--output", str(output_path), "--force"]
    ) == 0


def test_record_baseline_cli_refuses_unmeasured(tmp_path: Path):
    from record_baseline import main as record_main

    evaluation_path = tmp_path / "eval.json"
    evaluation_path.write_text(json.dumps({"model": "base", "normalized_score": None, "results": []}))
    assert (
        record_main(["--evaluation", str(evaluation_path), "--output", str(tmp_path / "out.json")])
        == 1
    )
    assert not (tmp_path / "out.json").exists()


# --- gate decisions and champion -------------------------------------------


def test_resolve_reference_prefers_champion_then_baseline(tmp_path: Path):
    champion_path = tmp_path / "champion.json"
    champion_path.write_text(json.dumps(_reference(0.60, champion_version="qlora-sft-v1")))
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_reference(0.50)))
    reference, kind, version = resolve_reference(champion_path, baseline_path)
    assert kind == "champion" and version == "qlora-sft-v1"
    assert reference["normalized_score"] == 0.60

    champion_path.write_text(json.dumps({"status": "not_measured", "normalized_score": None}))
    reference, kind, version = resolve_reference(champion_path, baseline_path)
    assert kind == "baseline" and version == "base"
    assert reference["normalized_score"] == 0.50

    baseline_path.write_text(json.dumps({"status": "not_measured", "normalized_score": None}))
    assert resolve_reference(champion_path, baseline_path) == (None, None, None)


def test_gate_decision_record_roundtrip(tmp_path: Path):
    decision = build_gate_decision(
        decision="stay_private",
        reasons=["improvement below threshold"],
        candidate=_evaluation(0.555),
        reference=_reference(0.55),
        reference_kind="champion",
        reference_version="qlora-sft-v1",
        baseline_score=0.50,
        candidate_ref="qlora-sft-v2",
        gates=GATES,
    )
    path = record_gate_decision(decision, decisions_dir=tmp_path)
    loaded = json.loads(path.read_text())
    assert loaded["decision"] == "stay_private"
    assert loaded["candidate_score"] == 0.555
    assert loaded["reference_score"] == 0.55
    assert loaded["candidate_sample_size"] == 12
    assert loaded["reference_version"] == "qlora-sft-v1"
    assert loaded["decided_at"]


def test_update_champion_refuses_unmeasured(tmp_path: Path):
    with pytest.raises(ValueError, match="unmeasured"):
        update_champion(
            {"normalized_score": None},
            champion_path=tmp_path / "champion.json",
            champion_version="qlora-sft-v1",
        )


def test_update_champion_writes_versioned_record(tmp_path: Path):
    path = update_champion(
        _evaluation(0.62),
        champion_path=tmp_path / "champion.json",
        champion_version="qlora-sft-v1",
        kaggle_model_url="https://www.kaggle.com/models/x/y",
    )
    record = json.loads(path.read_text())
    assert record["champion_version"] == "qlora-sft-v1"
    assert record["normalized_score"] == 0.62
    assert record["status"] == "measured"
    assert record["promoted_at"]


# --- provenance building ---------------------------------------------------


def _fake_adapter(tmp_path: Path, gpu_info: dict) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fake-adapter-bytes")
    (adapter / "adapter_config.json").write_text('{"r": 32}')
    training_manifest = {
        "base_model": "Qwen/Qwen3.8-27B",
        "base_revision": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
        "training_method": "QLoRA-SFT",
        "training_date": "2026-11-23T08:00:00Z",
        "training_dataset_version": "v0",
        "training_dataset_hash": "a" * 64,
        "training_config_hash": "b" * 64,
        "training_environment": "Kaggle",
        "gpu_configuration": gpu_info,
    }
    (adapter / "training_manifest.json").write_text(json.dumps(training_manifest))
    return adapter


def test_build_provenance_from_training_manifest(tmp_path: Path):
    from build_provenance import build_provenance_manifest

    adapter = _fake_adapter(
        tmp_path, {"gpus": [{"name": "L4", "total_memory_gb": 24.0}]}
    )
    manifest = build_provenance_manifest(
        adapter_dir=adapter,
        evaluation=_evaluation(0.62),
        source_commit="c" * 40,
        release_version="1",
    )
    assert manifest["gpu_configuration"] == [{"name": "L4", "total_memory_gb": 24.0}]
    assert manifest["source_commit"] == "c" * 40
    assert manifest["artifact_hashes"]["adapter_model.safetensors"]
    assert manifest["evaluation_results"]["normalized_score"] == 0.62


def test_build_provenance_refuses_cpu_only_run(tmp_path: Path):
    from build_provenance import build_provenance_manifest

    adapter = _fake_adapter(tmp_path, {"cuda_available": False, "device": "cpu"})
    with pytest.raises(ValueError, match="no GPU"):
        build_provenance_manifest(
            adapter_dir=adapter,
            evaluation=_evaluation(0.62),
            source_commit="c" * 40,
            release_version="1",
        )


# --- evaluate_and_promote orchestrator --------------------------------------


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _orchestrator_files(tmp_path: Path, *, candidate_score=0.62, baseline_score=0.55):
    from tests.test_provenance import complete_provenance

    evaluation = _write_json(tmp_path / "evaluation.json", _evaluation(candidate_score))
    provenance = complete_provenance()
    provenance_path = _write_json(tmp_path / "provenance.json", provenance)
    baseline = _write_json(tmp_path / "baseline.json", _reference(baseline_score))
    adapter = _fake_adapter(
        tmp_path, {"gpus": [{"name": "L4", "total_memory_gb": 24.0}]}
    )
    return evaluation, provenance_path, baseline, adapter


def test_orchestrator_blocked_without_measured_baseline(tmp_path: Path):
    from evaluate_and_promote import main as promote_main

    evaluation, provenance_path, _, adapter = _orchestrator_files(tmp_path)
    decisions = tmp_path / "decisions"
    rc = promote_main(
        [
            "--evaluation", str(evaluation),
            "--provenance", str(provenance_path),
            "--adapter", str(adapter),
            "--champion", str(tmp_path / "no-champion.json"),
            "--baseline", str(tmp_path / "no-baseline.json"),
            "--decisions-dir", str(decisions),
        ]
    )
    assert rc == 0
    (record,) = list(decisions.glob("*.json"))
    assert json.loads(record.read_text())["decision"] == "blocked"


def test_orchestrator_stays_private_on_gate_fail(tmp_path: Path):
    from evaluate_and_promote import main as promote_main

    evaluation, provenance_path, baseline, adapter = _orchestrator_files(
        tmp_path, candidate_score=0.555
    )
    decisions = tmp_path / "decisions"
    rc = promote_main(
        [
            "--evaluation", str(evaluation),
            "--provenance", str(provenance_path),
            "--adapter", str(adapter),
            "--champion", str(tmp_path / "no-champion.json"),
            "--baseline", str(baseline),
            "--decisions-dir", str(decisions),
        ]
    )
    assert rc == 0
    (record,) = list(decisions.glob("*.json"))
    assert json.loads(record.read_text())["decision"] == "stay_private"
    assert not (tmp_path / "champion.json").exists()


def test_orchestrator_skipped_quota(tmp_path: Path):
    from evaluate_and_promote import main as promote_main

    run_record = _write_json(
        tmp_path / "run.json", {"status": "skipped_quota", "reason": "quota"}
    )
    decisions = tmp_path / "decisions"
    rc = promote_main(
        [
            "--evaluation", str(tmp_path / "missing.json"),
            "--provenance", str(tmp_path / "missing.json"),
            "--adapter", str(tmp_path / "missing"),
            "--run-record", str(run_record),
            "--decisions-dir", str(decisions),
        ]
    )
    assert rc == 0
    (record,) = list(decisions.glob("*.json"))
    assert json.loads(record.read_text())["decision"] == "skipped"


def test_orchestrator_dry_run_promote_records_but_does_not_publish(tmp_path: Path):
    from evaluate_and_promote import main as promote_main

    evaluation, provenance_path, baseline, adapter = _orchestrator_files(tmp_path)
    decisions = tmp_path / "decisions"
    champion_path = tmp_path / "champion.json"
    rc = promote_main(
        [
            "--evaluation", str(evaluation),
            "--provenance", str(provenance_path),
            "--adapter", str(adapter),
            "--champion", str(champion_path),
            "--baseline", str(baseline),
            "--decisions-dir", str(decisions),
            "--dry-run",
        ]
    )
    assert rc == 0
    (record,) = list(decisions.glob("*.json"))
    loaded = json.loads(record.read_text())
    assert loaded["decision"] == "promote" and loaded["dry_run"] is True
    assert loaded["candidate_score"] == 0.62
    assert loaded["reference_score"] == 0.55
    assert not champion_path.exists()


def test_orchestrator_promote_with_stubbed_publish(tmp_path: Path, monkeypatch):
    from evaluate_and_promote import main as promote_main

    calls: list[list[str]] = []
    stub = types.ModuleType("publish_kaggle")

    def fake_publish(argv):
        calls.append(argv)
        return 0

    stub.main = fake_publish
    monkeypatch.setitem(sys.modules, "publish_kaggle", stub)

    evaluation, provenance_path, baseline, adapter = _orchestrator_files(tmp_path)
    decisions = tmp_path / "decisions"
    champion_path = tmp_path / "champion.json"
    releases = tmp_path / "releases"
    rc = promote_main(
        [
            "--evaluation", str(evaluation),
            "--provenance", str(provenance_path),
            "--adapter", str(adapter),
            "--champion", str(champion_path),
            "--baseline", str(baseline),
            "--decisions-dir", str(decisions),
            "--releases-dir", str(releases),
        ]
    )
    assert rc == 0
    assert len(calls) == 1
    assert "--release-version" in calls[0]
    assert calls[0][calls[0].index("--model-version") + 1] == "qlora-sft-v1"
    champion = json.loads(champion_path.read_text())
    assert champion["champion_version"] == "qlora-sft-v1"
    assert champion["normalized_score"] == 0.62
    assert (releases / "1.json").exists()
    (record,) = list(decisions.glob("*.json"))
    loaded = json.loads(record.read_text())
    assert loaded["decision"] == "promote"
    assert loaded["reference_kind"] == "baseline"


def test_orchestrator_version_increments_from_champion(tmp_path: Path, monkeypatch):
    from evaluate_and_promote import main as promote_main

    stub = types.ModuleType("publish_kaggle")
    calls: list[list[str]] = []
    stub.main = lambda argv: (calls.append(argv), 0)[1]
    monkeypatch.setitem(sys.modules, "publish_kaggle", stub)

    evaluation, provenance_path, baseline, adapter = _orchestrator_files(
        tmp_path, candidate_score=0.70, baseline_score=0.55
    )
    champion_path = _write_json(
        tmp_path / "champion.json", _reference(0.60, champion_version="qlora-sft-v1")
    )
    rc = promote_main(
        [
            "--evaluation", str(evaluation),
            "--provenance", str(provenance_path),
            "--adapter", str(adapter),
            "--champion", str(champion_path),
            "--baseline", str(baseline),
            "--decisions-dir", str(tmp_path / "decisions"),
            "--releases-dir", str(tmp_path / "releases"),
        ]
    )
    assert rc == 0
    assert calls[0][calls[0].index("--model-version") + 1] == "qlora-sft-v2"
    assert calls[0][calls[0].index("--release-version") + 1] == "2"


# --- eval kernel packaging ---------------------------------------------------


def test_package_eval_kernel_candidate_and_base(tmp_path: Path):
    from package_eval_kernel import build_kernel

    for mode, kernel_id, sources in [
        ("candidate", "monibeefudgekins/akarui-eval-candidate",
         ["monibeefudgekins/akarui-sft-training"]),
        ("base", "monibeefudgekins/akarui-eval-baseline", []),
    ]:
        out = tmp_path / f"eval-{mode}"
        summary = build_kernel(ROOT, out, mode)
        assert summary["kernel_id"] == kernel_id
        metadata = json.loads((out / "kernel-metadata.json").read_text())
        assert metadata["id"] == kernel_id
        assert metadata["code_file"] == "07_evaluation.ipynb"
        assert metadata["enable_gpu"] is True
        assert metadata["kernel_sources"] == sources
        assert json.loads((out / "eval-mode.json").read_text())["mode"] == mode
        assert (out / "07_evaluation.ipynb").exists()
        assert (out / "evaluation" / "benchmark_manifest.json").exists()
        assert (out / "data" / "train.jsonl").exists()
        assert (out / "scripts" / "evaluate.py").exists()


def test_package_eval_kernel_refuses_unknown_mode(tmp_path: Path):
    from package_eval_kernel import build_kernel

    with pytest.raises(ValueError, match="unknown eval mode"):
        build_kernel(ROOT, tmp_path / "eval", "nope")
