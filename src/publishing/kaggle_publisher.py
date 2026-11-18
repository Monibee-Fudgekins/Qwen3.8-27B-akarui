"""Kaggle model publication for Qwen3.8-27B-akarui.

Builds Kaggle model and variation metadata using only the fields the
Kaggle model metadata format supports (title, subtitle, description,
licenses, tags, and per-variation framework / variation type /
overview / usage), then uploads a variation version either through
the Kaggle CLI or, as a fallback, through kagglehub.

Rules enforced here:

* A variation is never uploaded when its provenance manifest is
  incomplete — :func:`upload_variation` raises before any network or
  subprocess call.
* Credentials are never printed, logged, or written into metadata,
  provenance, or upload payloads. Authentication comes from the
  environment (``KAGGLE_TOKEN`` is adapted to whatever the installed
  Kaggle client expects; see :func:`configure_kaggle_auth`).
* Adapter-only releases are labelled as adapters, never as complete
  standalone models.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .provenance import validate_provenance

MODEL_SLUG = "qwen3-8-27b-akarui"
MODEL_TITLE = "Qwen3.8-27B-akarui"
KAGGLE_OWNER = "monibeefudgekins"
FRAMEWORK = "pytorch"
LICENSE_NAME = "apache-2.0"

DEFAULT_TAGS = [
    "qwen",
    "qwen3",
    "language-model",
    "vision-language-model",
    "fine-tuned",
    "pytorch",
    "machine-learning",
    "data-science",
    "kaggle-competition",
    "coding",
    "reasoning",
    "qlora",
    "autonomous-agent",
]

# Variations this project publishes, in the order they are expected
# to appear. ``merged-bf16`` is only published when the hardware and
# storage budget support a merged model.
VARIATIONS = ("qlora-sft", "qlora-dpo", "qlora-grpo", "merged-bf16")


def configure_kaggle_auth() -> dict[str, str]:
    """Adapt the supplied ``KAGGLE_TOKEN`` secret for the Kaggle client.

    Kaggle's client tooling has historically expected either
    ``KAGGLE_USERNAME``/``KAGGLE_KEY`` or ``KAGGLE_API_TOKEN``; the
    repository secret is named ``KAGGLE_TOKEN`` and is not
    automatically recognised by every Kaggle library. This adapter
    reads ``KAGGLE_TOKEN`` and, when it holds a ``username:key``
    pair, exposes it as ``KAGGLE_USERNAME``/``KAGGLE_KEY``; otherwise
    it exposes the raw token as ``KAGGLE_API_TOKEN``. The token value
    is returned only in the mapping the caller passes to a subprocess
    environment — it is never printed or logged.

    Raises
    ------
    RuntimeError
        If ``KAGGLE_TOKEN`` is not set in the environment.
    """
    token = os.environ.get("KAGGLE_TOKEN")
    if not token:
        raise RuntimeError(
            "KAGGLE_TOKEN is not set. Add it under GitHub repository "
            "Settings > Secrets and variables > Actions; it is never "
            "supplied in chat or committed to the repository."
        )
    env = dict(os.environ)
    if ":" in token:
        username, key = token.split(":", 1)
        env["KAGGLE_USERNAME"] = username
        env["KAGGLE_KEY"] = key
    else:
        env["KAGGLE_API_TOKEN"] = token
    return env


def build_model_metadata(
    model_name: str,
    variation: str,
    description: str,
    tags: list[str] | None = None,
) -> dict:
    """Build Kaggle model metadata for ``model_name``.

    Returns a dict using only fields supported by Kaggle's model
    metadata format: ``title``, ``subtitle``, ``description``,
    ``licenses``, ``tags``, and ``instances`` (one entry per
    variation, with ``framework``, ``fineTunable``,
    ``variationType``, ``overview`` and ``usage``). No invented JSON
    properties are added. Authorship, provenance, and base-model
    information are conveyed through the description and the
    ``provenance.json`` uploaded with the variation, and are also
    completed through the Kaggle model UI — they are not metadata
    JSON keys. Tags must be verified against the Kaggle model UI
    before publication; the defaults here are the project's
    proposed tags.
    """
    if variation not in VARIATIONS:
        raise ValueError(f"unknown variation {variation!r}; expected one of {VARIATIONS}")
    return {
        "title": model_name,
        "subtitle": f"Competition-solving fine-tune of Qwen3.8-27B ({variation})",
        "description": description,
        "licenses": [{"name": LICENSE_NAME}],
        "tags": list(tags) if tags is not None else list(DEFAULT_TAGS),
        "instances": [
            {
                "framework": FRAMEWORK,
                "fineTunable": True,
                "variationType": variation,
                "overview": description,
                "usage": (
                    "Adapter weights for the Qwen/Qwen3.8-27B base model. "
                    "Load the base model and apply this adapter with PEFT; "
                    "the adapter is not a standalone model."
                    if variation.startswith("qlora")
                    else "Merged inference-ready weights based on Qwen/Qwen3.8-27B."
                ),
            }
        ],
    }


def build_variation_metadata(
    variation: str,
    *,
    framework: str = FRAMEWORK,
    overview: str = "",
    usage: str = "",
) -> dict:
    """Build the per-variation metadata block for one variation."""
    if variation not in VARIATIONS:
        raise ValueError(f"unknown variation {variation!r}; expected one of {VARIATIONS}")
    return {
        "framework": framework,
        "fineTunable": True,
        "variationType": variation,
        "overview": overview,
        "usage": usage,
    }


def _require_complete_provenance(provenance: dict | None) -> None:
    if provenance is None:
        raise ValueError("refusing to upload: no provenance manifest supplied")
    errors = validate_provenance(provenance)
    if errors:
        raise ValueError("refusing to upload: incomplete provenance: " + "; ".join(errors))


def upload_variation(
    variation: str,
    model_dir: str | Path,
    provenance: dict,
    *,
    model_slug: str = MODEL_SLUG,
    owner: str = KAGGLE_OWNER,
) -> dict:
    """Upload a new version of ``variation`` to Kaggle.

    Primary path: the Kaggle CLI
    (``kaggle models version create``). Fallback path: kagglehub's
    model upload, used only if the CLI is not installed. Both paths
    refuse to run when ``provenance`` is incomplete.

    The upload directory must contain the variation's files plus a
    ``provenance.json`` written from the validated manifest. The
    return value is a small dict describing which path was used and
    what was uploaded; the caller records it in the release record.
    Credentials are never included in the returned dict or in any
    log output.
    """
    _require_complete_provenance(provenance)
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")

    provenance_path = model_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    handle = f"{owner}/{model_slug}"
    env = configure_kaggle_auth()

    cli_result = _upload_via_cli(handle, variation, model_dir, env)
    if cli_result is not None:
        return cli_result
    return _upload_via_kagglehub(handle, variation, model_dir)


def _upload_via_cli(
    handle: str, variation: str, model_dir: Path, env: dict[str, str]
) -> dict | None:
    """Upload with the Kaggle CLI; returns None if the CLI is absent."""
    try:
        completed = subprocess.run(
            [
                "kaggle",
                "models",
                "version",
                "create",
                "-m",
                handle,
                "-v",
                variation,
                "-d",
                str(model_dir),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        raise RuntimeError(
            "kaggle CLI upload failed "
            f"(exit {completed.returncode}): {completed.stderr.strip()[:2000]}"
        )
    return {
        "method": "kaggle-cli",
        "model": handle,
        "variation": variation,
        "directory": str(model_dir),
        "cli_output": completed.stdout.strip()[:2000],
    }


def _upload_via_kagglehub(handle: str, variation: str, model_dir: Path) -> dict:
    """Fallback upload with kagglehub; raises if kagglehub is absent."""
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "neither the kaggle CLI nor kagglehub is available; "
            "install one of them to publish"
        ) from exc
    upload = getattr(kagglehub, "model_upload", None)
    if upload is None:
        raise RuntimeError(
            "installed kagglehub has no model_upload API; "
            "use the kaggle CLI to publish this variation"
        )
    upload(handle, str(model_dir))
    return {
        "method": "kagglehub",
        "model": handle,
        "variation": variation,
        "directory": str(model_dir),
    }
