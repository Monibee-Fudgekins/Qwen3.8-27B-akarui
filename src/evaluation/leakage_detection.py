"""Leakage detection for Akarui training / evaluation data.

Two checks:

* :func:`detect_leakage` -- set overlap between training and
  evaluation competition ids. Any overlap means the benchmark is
  contaminated and must not be used.
* :func:`scan_record_for_leakage` -- heuristic scan of one record for
  hidden-test-label / private-data markers. The patterns are
  heuristics, documented as such; a finding means "review before
  training", never a proof of leakage on its own.
"""

from __future__ import annotations

import re

LEAKAGE_PATTERNS = (
    ("hidden_test_labels", re.compile(r"hidden\s+test\s+label", re.IGNORECASE)),
    ("private_test_data", re.compile(r"private\s+test\s+(data|set|labels?)", re.IGNORECASE)),
    ("test_labels", re.compile(r"test\s+labels?\s*(?:\.csv|file)", re.IGNORECASE)),
    ("leaked_solution", re.compile(r"leak(ed|age)\s+(solution|test|answer)", re.IGNORECASE)),
    ("ground_truth_test", re.compile(r"ground[\s_-]?truth\s+(test|labels?)", re.IGNORECASE)),
    ("private_leaderboard", re.compile(r"private\s+leaderboard", re.IGNORECASE)),
    ("hidden_evaluation", re.compile(r"hidden\s+evaluation\s+(label|set|data)", re.IGNORECASE)),
    ("answer_key", re.compile(r"answer\s+key", re.IGNORECASE)),
)

#: Record fields scanned by :func:`scan_record_for_leakage`.
_SCANNED_FIELDS = (
    "task_description",
    "reasoning_summary",
    "generated_code",
    "execution_output",
    "final_outcome",
    "dataset_provenance",
    "input_schema",
)


def _normalize_id(value: object) -> str:
    return str(value).strip().lower() if value is not None else ""


def detect_leakage(train_ids, eval_ids: object) -> dict:
    """Report overlap between training and evaluation competition ids.

    Returns::

        {
            "train_count": int,
            "eval_count": int,
            "overlap_count": int,
            "overlap_ids": [str, ...],   # sorted
            "clean": bool,               # True only when overlap is empty
        }
    """
    train = {_normalize_id(item) for item in (train_ids or [])}
    eval_ = {_normalize_id(item) for item in (eval_ids or [])}
    train.discard("")
    eval_.discard("")
    overlap = sorted(train & eval_)
    return {
        "train_count": len(train),
        "eval_count": len(eval_),
        "overlap_count": len(overlap),
        "overlap_ids": overlap,
        "clean": len(overlap) == 0,
    }


def scan_record_for_leakage(record: dict) -> list[dict]:
    """Scan one record for hidden-test-label / private-data markers.

    Returns a list of findings ``{"field": str, "pattern": str,
    "match": str}``. An empty list means no heuristic matched -- not a
    guarantee that the record is clean.
    """
    findings: list[dict] = []
    if not isinstance(record, dict):
        return findings
    for field in _SCANNED_FIELDS:
        value = record.get(field)
        if not isinstance(value, str):
            continue
        for name, pattern in LEAKAGE_PATTERNS:
            match = pattern.search(value)
            if match:
                findings.append(
                    {"field": field, "pattern": name, "match": match.group(0)}
                )
    return findings
