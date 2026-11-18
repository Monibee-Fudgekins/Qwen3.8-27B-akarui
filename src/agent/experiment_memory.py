"""Experiment memory for the Akarui agent.

Experiments are recorded as one JSON object per line in a JSONL
store, so the record survives process restarts and can be inspected
with ordinary line-based tools. Each record carries the experiment
id, the competition it belongs to, a hash of its configuration, the
validation score (``None`` when the experiment did not produce one),
and its outcome. The memory never invents a score: a record without
a measured score stores ``None``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ExperimentMemory:
    """Append-only JSONL store of competition experiments.

    Parameters
    ----------
    store_path:
        Path to the JSONL file. The parent directory is created on
        first write.
    """

    def __init__(self, store_path: str | Path) -> None:
        self.store_path = Path(store_path)

    @staticmethod
    def config_hash(config: dict) -> str:
        """Hash a config dict deterministically (sorted keys)."""
        canonical = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append(
        self,
        experiment_id: str,
        competition_id: str,
        config: dict,
        validation_score: float | None,
        outcome: str,
    ) -> dict:
        """Append one experiment record and return it."""
        record = {
            "experiment_id": experiment_id,
            "competition_id": competition_id,
            "config_hash": self.config_hash(config),
            "config": config,
            "validation_score": validation_score,
            "outcome": outcome,
        }
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def list(self, competition_id: str | None = None) -> list[dict]:
        """Return all records, optionally filtered to one competition."""
        if not self.store_path.exists():
            return []
        records = []
        with open(self.store_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if competition_id is None or record.get("competition_id") == competition_id:
                    records.append(record)
        return records

    def query(self, experiment_id: str) -> dict | None:
        """Return the most recent record with ``experiment_id``, or None."""
        matches = [r for r in self.list() if r.get("experiment_id") == experiment_id]
        return matches[-1] if matches else None

    def best_experiment(self, competition_id: str) -> dict | None:
        """Return the highest-scoring experiment for a competition.

        Only records with a measured (non-``None``) validation score
        are considered. Returns ``None`` when no scored experiment
        exists for the competition. Higher is assumed better; callers
        whose metric is lower-is-better must invert the score before
        storing it, and should record that in the config.
        """
        scored = [
            r
            for r in self.list(competition_id)
            if r.get("validation_score") is not None
        ]
        if not scored:
            return None
        return max(scored, key=lambda r: r["validation_score"])
