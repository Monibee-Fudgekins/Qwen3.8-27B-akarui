"""Deduplication of Akarui trajectory records.

Two records are duplicates when they describe the same solving task or
reuse the same solution code:

* **Task duplicate**: same normalized ``competition_id`` plus the same
  normalized ``task_description`` (hashed).
* **Code duplicate**: same normalized ``generated_code`` (hashed).

Normalization lower-cases and collapses whitespace so that trivial
formatting differences do not hide a duplicate. The first occurrence is
kept; later duplicates are dropped with a recorded reason.
"""

from __future__ import annotations

import hashlib


def _normalize_text(value: object) -> str:
    """Lower-case and collapse whitespace; non-strings become ``""``."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).lower()


def _sha256_hexdigest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_key(record: dict) -> str:
    """Stable key for the (competition, task) identity of a record."""
    competition = _normalize_text(record.get("competition_id"))
    task = _normalize_text(record.get("task_description"))
    return _sha256_hexdigest(f"{competition}\n{task}")


def code_key(record: dict) -> str | None:
    """Stable key for the solution code, or None when the record has none."""
    code = _normalize_text(record.get("generated_code"))
    if not code:
        return None
    return _sha256_hexdigest(code)


def dedup_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deduplicate *records*.

    Returns ``(kept, dropped_with_reasons)`` where ``kept`` is the list of
    records retained (first occurrence wins) and ``dropped_with_reasons``
    is a list of ``{"record": record, "reason": str}`` entries with one of
    the reasons ``"duplicate_task"`` or ``"duplicate_code"``.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    seen_tasks: set[str] = set()
    seen_code: set[str] = set()
    for record in records:
        key = task_key(record)
        if key in seen_tasks:
            dropped.append(
                {"record": record, "reason": "duplicate_task"}
            )
            continue
        code = code_key(record)
        if code is not None and code in seen_code:
            dropped.append(
                {"record": record, "reason": "duplicate_code"}
            )
            continue
        seen_tasks.add(key)
        if code is not None:
            seen_code.add(code)
        kept.append(record)
    return kept, dropped
