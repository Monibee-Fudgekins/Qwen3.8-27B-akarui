"""Discover public Kaggle competitions and Hugging Face datasets.

Only sources whose license permits training use are kept. Every item
records ``source_url``, ``license`` and ``collected_at``; items whose
license is unknown are marked ``license_verified: False`` and are
dropped by :func:`collect` when ``strict_license=True`` (the default).

No network call happens at import time: all API access is inside the
functions below, and a missing optional client library (``kaggle``,
``huggingface_hub``) is handled gracefully by returning an empty list
with a warning instead of crashing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

#: Licenses whose terms permit training use of the material. Compared
#: against a normalized (lower-cased, punctuation-stripped) license id.
#: "unknown" is never in this set: unverified material is not training
#: material until a human confirms its license.
PERMISSIVE_LICENSES = frozenset(
    {
        "apache-2.0",
        "apache 2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "bsd",
        "cc-by-4.0",
        "cc-by",
        "cc0-1.0",
        "cc0",
        "openrail",
        "bigscience-openrail-m",
        "odc-by",
        "pddl",
    }
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_license(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    text = value.strip().lower()
    return text or "unknown"


def license_allows_training(license_value: object) -> bool:
    """Return True only for licenses known to permit training use.

    Unknown / empty / unverified licenses return False. This is a
    conservative allow-list, not legal advice.
    """
    return _normalize_license(license_value) in PERMISSIVE_LICENSES


def discover_kaggle_competitions(
    query: str | None = None, limit: int = 20
) -> list[dict]:
    """List public Kaggle competitions.

    Uses the ``kaggle`` package when installed. Kaggle competition
    listings do not expose a per-competition data license through the
    API, so every item is returned with ``license: "unknown"`` and
    ``license_verified: False`` -- the competition's data-usage terms
    must be checked on its page before the material is used for
    training. Returns ``[]`` (with a warning) when the client is
    unavailable or the API call fails.
    """
    try:
        from kaggle import api as kaggle_api
    except ImportError:
        logger.warning("kaggle package not installed; skipping competition discovery")
        return []
    try:
        competitions = kaggle_api.competitions_list(search=query, page=1)
    except Exception as exc:  # noqa: BLE001 - API/auth failures are expected
        logger.warning("Kaggle competition listing failed: %s", exc)
        return []
    items: list[dict] = []
    for competition in list(competitions or [])[:limit]:
        ref = getattr(competition, "ref", None) or str(competition)
        items.append(
            {
                "source": "kaggle",
                "competition_id": ref,
                "source_url": f"https://www.kaggle.com/competitions/{ref}",
                "license": "unknown",
                "license_verified": False,
                "title": getattr(competition, "title", None),
                "collected_at": _now_iso(),
            }
        )
    return items


def discover_hf_datasets(query: str | None = None, limit: int = 20) -> list[dict]:
    """List public Hugging Face datasets matching *query*.

    Uses ``huggingface_hub`` when installed (no auth needed for public
    listings). Each item carries the license reported by the Hub
    (``license`` may be ``"unknown"``) and ``license_verified`` is True
    only for licenses in :data:`PERMISSIVE_LICENSES`. Returns ``[]``
    (with a warning) when the client is unavailable or the call fails.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed; skipping dataset discovery")
        return []
    try:
        api = HfApi()
        datasets = api.list_datasets(search=query, limit=limit)
    except Exception as exc:  # noqa: BLE001 - network/API failures are expected
        logger.warning("Hugging Face dataset listing failed: %s", exc)
        return []
    items: list[dict] = []
    for dataset in datasets or []:
        dataset_id = getattr(dataset, "id", None) or str(dataset)
        raw_license = None
        tags = getattr(dataset, "tags", None) or []
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("license:"):
                raw_license = tag.split(":", 1)[1]
                break
        license_value = _normalize_license(raw_license)
        items.append(
            {
                "source": "huggingface",
                "dataset_id": dataset_id,
                "source_url": f"https://huggingface.co/datasets/{dataset_id}",
                "license": license_value,
                "license_verified": license_allows_training(license_value),
                "collected_at": _now_iso(),
            }
        )
    return items


def collect(
    output_path: str | Path,
    kaggle_query: str | None = None,
    hf_query: str | None = None,
    limit: int = 20,
    strict_license: bool = True,
) -> dict:
    """Discover public sources and write a raw manifest JSONL.

    Each line is one discovered item (see the discover functions). When
    ``strict_license`` is True (default), items whose license is not
    verified to permit training use are dropped and counted under
    ``dropped_unverified_license``. Nothing here is training data yet:
    this manifest is the input to human license review and to trajectory
    generation.

    Returns a summary dict with counts and the output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = discover_kaggle_competitions(query=kaggle_query, limit=limit)
    items.extend(discover_hf_datasets(query=hf_query, limit=limit))

    kept: list[dict] = []
    dropped_unverified = 0
    for item in items:
        if strict_license and not item.get("license_verified"):
            dropped_unverified += 1
            continue
        kept.append(item)

    with output_path.open("w", encoding="utf-8") as handle:
        for item in kept:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "output_path": str(output_path),
        "collected_at": _now_iso(),
        "discovered": len(items),
        "kept": len(kept),
        "dropped_unverified_license": dropped_unverified,
        "strict_license": strict_license,
    }
