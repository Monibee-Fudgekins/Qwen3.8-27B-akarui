"""Release records for Qwen3.8-27B-akarui.

A release is a versioned JSON record that binds a provenance manifest
to the evaluation results that justified (or blocked) promotion, plus
human-readable release notes. Releases are append-only: a new version
never overwrites or deletes a previous one, and the previous stable
release is always preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

from .provenance import validate_provenance

DEFAULT_RELEASES_DIR = Path(__file__).resolve().parents[2] / "releases"


def _releases_dir(releases_dir: str | Path | None) -> Path:
    return Path(releases_dir) if releases_dir is not None else DEFAULT_RELEASES_DIR


def write_release(
    version: str,
    provenance: dict,
    evaluation_results: dict,
    notes: str,
    *,
    releases_dir: str | Path | None = None,
) -> Path:
    """Write a release record for ``version`` and return its path.

    The release JSON contains the version, the full provenance
    manifest, the evaluation results, and the release notes. The
    provenance manifest must validate; an incomplete manifest raises
    :class:`ValueError` and nothing is written. Overwriting an
    existing release version raises :class:`FileExistsError`.
    """
    errors = validate_provenance(provenance)
    if errors:
        raise ValueError("cannot write release: " + "; ".join(errors))
    if not isinstance(evaluation_results, dict):
        raise ValueError("evaluation_results must be a dict")

    directory = _releases_dir(releases_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{version}.json"
    if path.exists():
        raise FileExistsError(f"release {version} already exists at {path}; releases are append-only")

    record = {
        "version": version,
        "provenance": provenance,
        "evaluation_results": evaluation_results,
        "notes": notes,
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_release(
    version: str, *, releases_dir: str | Path | None = None
) -> dict:
    """Load the release record for ``version``.

    Raises
    ------
    FileNotFoundError
        If no release record exists for the version.
    """
    path = _releases_dir(releases_dir) / f"{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"no release record for version {version}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_releases(*, releases_dir: str | Path | None = None) -> list[str]:
    """Return the sorted list of recorded release versions."""
    directory = _releases_dir(releases_dir)
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))
