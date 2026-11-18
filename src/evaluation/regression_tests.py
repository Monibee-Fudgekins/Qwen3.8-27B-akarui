"""Base-capability regression tests.

Runs a configurable set of probes (reasoning, coding, math,
instruction following) against a model and compares the outcome with
a previously recorded baseline results file.

The model may be:

* ``None`` -- nothing is run; every probe is recorded as
  ``"not_run"`` with a null score. Probe results are never invented.
* a callable ``model(prompt) -> str`` -- the callable generates the
  response for each probe prompt.

A probe passes when its ``expected_substring`` (when given) appears in
the response. Probes without an expected substring are recorded as
unscored (``passed: None``). A regression is a probe that passed in
the baseline and fails now.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_probes(probe_set_path: str | Path) -> list[dict]:
    payload = _load_json(probe_set_path)
    if isinstance(payload, dict):
        probes = payload.get("probes", [])
    else:
        probes = payload
    if not isinstance(probes, list):
        raise ValueError(f"probe set {probe_set_path} must be a list or contain a 'probes' list")
    return probes


def run_regression(
    model,
    probe_set_path: str | Path,
    baseline_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Run regression probes and compare against an optional baseline.

    Returns a summary dict::

        {
            "total": int,
            "run": int, "passed": int, "failed": int, "not_run": int,
            "regressions": [{"id": ..., "category": ...}, ...],
            "results": [
                {"id": ..., "category": ..., "status": "pass"|"fail"|"not_run"|"unscored",
                 "score": float | None, "response": str | None},
                ...
            ],
        }

    The summary is also written to *output_path* as JSON when given.
    """
    probes = _load_probes(probe_set_path)
    baseline: dict[str, dict] = {}
    if baseline_path is not None and Path(baseline_path).is_file():
        baseline_payload = _load_json(baseline_path)
        baseline_results = (
            baseline_payload.get("results", [])
            if isinstance(baseline_payload, dict)
            else baseline_payload
        )
        for entry in baseline_results:
            if isinstance(entry, dict) and entry.get("id") is not None:
                baseline[str(entry["id"])] = entry

    results: list[dict] = []
    for probe in probes:
        probe_id = probe.get("id")
        category = probe.get("category")
        prompt = probe.get("prompt", "")
        expected = probe.get("expected_substring")
        if model is None:
            results.append(
                {
                    "id": probe_id,
                    "category": category,
                    "status": "not_run",
                    "score": None,
                    "response": None,
                }
            )
            continue
        try:
            response = model(prompt)
        except Exception as exc:  # noqa: BLE001 - one bad probe must not kill the run
            results.append(
                {
                    "id": probe_id,
                    "category": category,
                    "status": "error",
                    "score": None,
                    "response": None,
                    "error": str(exc),
                }
            )
            continue
        response_text = response if isinstance(response, str) else str(response)
        if expected is None:
            status, passed = "unscored", None
        elif expected in response_text:
            status, passed = "pass", True
        else:
            status, passed = "fail", False
        results.append(
            {
                "id": probe_id,
                "category": category,
                "status": status,
                "score": 1.0 if passed is True else (0.0 if passed is False else None),
                "response": response_text,
            }
        )

    regressions: list[dict] = []
    for entry in results:
        prior = baseline.get(str(entry.get("id")))
        if (
            prior is not None
            and prior.get("status") == "pass"
            and entry.get("status") == "fail"
        ):
            regressions.append({"id": entry.get("id"), "category": entry.get("category")})

    summary = {
        "total": len(results),
        "run": sum(1 for r in results if r["status"] in ("pass", "fail", "unscored")),
        "passed": sum(1 for r in results if r["status"] == "pass"),
        "failed": sum(1 for r in results if r["status"] == "fail"),
        "not_run": sum(1 for r in results if r["status"] == "not_run"),
        "regressions": regressions,
        "results": results,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return summary
