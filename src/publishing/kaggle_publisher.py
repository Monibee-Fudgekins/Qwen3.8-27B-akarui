"""Kaggle model publication for Qwen3.8-27B-akarui.

Builds Kaggle model and variation metadata using only the fields the
Kaggle model metadata format supports (title, subtitle, description,
licenses, tags, and per-variation framework / variation type /
overview / usage), then uploads a variation version either through
the Kaggle CLI or, as a fallback, through kagglehub.

Visibility: the model is PUBLIC by design — its entire purpose is to
be used in Kaggle competitions. Kaggle defaults a new model to
private: the model-metadata documentation
(https://github.com/Kaggle/kaggle-cli/blob/main/docs/models_metadata.md)
says of ``isPrivate`` "If not specified, will be ``true``", the
``kaggle models init`` template writes ``"isPrivate": true``, and
kagglehub's ``model_upload`` has no visibility parameter and creates
a missing model with ``isPrivate: True`` hard-coded
(``kagglehub/models_helpers.py``: ``_create_model``). So this module
stamps ``"isPrivate": false`` into every model-metadata file it
builds, and every upload finishes with :func:`ensure_model_public`,
which reads the model's visibility back from Kaggle and repairs a
private model with ``kaggle models update`` (or the Kaggle API
library when the CLI is absent). A model that cannot be verified
public fails loudly instead of publishing silently private.

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
import shutil
import subprocess
import tempfile
from pathlib import Path

from .provenance import validate_provenance

MODEL_SLUG = "qwen3-8-27b-akarui"
MODEL_TITLE = "Qwen3.8-27B-akarui"
KAGGLE_OWNER = "monibeefudgekins"
FRAMEWORK = "pytorch"
LICENSE_NAME = "apache-2.0"
# The Kaggle model-metadata schema names the license "Apache 2.0"
# (see docs/models_metadata.md); LICENSE_NAME is the slug form kept
# in the legacy ``licenses`` list below.
LICENSE_NAME_DOCS = "Apache 2.0"
# Kaggle defaults a new model to private; this project publishes
# public models only, so the metadata always carries isPrivate False.
MODEL_IS_PRIVATE = False
KAGGLE_MODEL_URL = f"https://www.kaggle.com/models/{KAGGLE_OWNER}/{MODEL_SLUG}"
# Last resort when no programmatic visibility control is available
# (no CLI, no kaggle library): the exact manual step. The Kaggle UI
# labels for this step have not been verified against the live site.
MANUAL_MAKE_PUBLIC_STEP = (
    f"Open {KAGGLE_MODEL_URL}, open the model settings, set the "
    "model visibility to Public and save, then verify with "
    f"`kaggle models get {KAGGLE_OWNER}/{MODEL_SLUG} -p <dir>` and "
    'confirm the downloaded model-metadata.json contains '
    '"isPrivate": false.'
)

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
    *,
    owner: str = KAGGLE_OWNER,
    model_slug: str = MODEL_SLUG,
) -> dict:
    """Build Kaggle model metadata for ``model_name``.

    Returns a dict using only fields supported by Kaggle's model
    metadata format: ``ownerSlug``, ``slug``, ``title``,
    ``subtitle``, ``isPrivate``, ``description``, ``licenseName``,
    plus the project's ``licenses`` / ``tags`` / ``instances``
    records (ignored by the CLI's model creation, kept so the
    project's tag and licence intent travels with the metadata).
    The per-variation ``instances`` entry carries ``framework``,
    ``fineTunable``, ``variationType``, ``overview`` and ``usage``.
    No invented JSON properties are added. Authorship, provenance,
    and base-model information are conveyed through the description
    and the ``provenance.json`` uploaded with the variation, and are
    also completed through the Kaggle model UI — they are not
    metadata JSON keys. Tags must be verified against the Kaggle
    model UI before publication; the defaults here are the
    project's proposed tags.

    ``isPrivate`` is always ``False``: Kaggle defaults a new model
    to private (the metadata docs say "If not specified, will be
    ``true``"), and this model must be public so it can be used in
    Kaggle competitions. ``ownerSlug`` and ``slug`` are required by
    ``kaggle models create`` — without them creation fails — so they
    are part of the metadata, not left to the caller.
    """
    if variation not in VARIATIONS:
        raise ValueError(f"unknown variation {variation!r}; expected one of {VARIATIONS}")
    return {
        "ownerSlug": owner,
        "slug": model_slug,
        "title": model_name,
        "subtitle": f"Competition-solving fine-tune of Qwen3.8-27B ({variation})",
        "isPrivate": MODEL_IS_PRIVATE,
        "description": description,
        "licenseName": LICENSE_NAME_DOCS,
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


def _cli_env() -> dict[str, str] | None:
    """Adapted environment for Kaggle CLI calls, or None.

    Returns None when ``KAGGLE_TOKEN`` is not set, so callers fall
    back to the ambient environment instead of failing before the
    CLI is even tried.
    """
    try:
        return configure_kaggle_auth()
    except RuntimeError:
        return None


def _kaggle_api():
    """Return an authenticated ``KaggleApi`` instance.

    Used only when the ``kaggle`` CLI is absent, to read back or
    repair a model's visibility through the Kaggle Python library.
    Raises RuntimeError when the library is unavailable or the
    token is not set.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "cannot verify model visibility: neither the kaggle CLI "
            "nor the kaggle Python library is available; "
            f"manual step: {MANUAL_MAKE_PUBLIC_STEP}"
        ) from exc
    env = configure_kaggle_auth()
    os.environ.update(env)
    api = KaggleApi()
    api.authenticate()
    return api


def get_model_is_private(handle: str) -> bool:
    """Read back whether the Kaggle model ``handle`` is private.

    Primary path: ``kaggle models get <handle> -p <tmpdir>``, which
    writes a ``model-metadata.json`` carrying the server's
    ``isPrivate`` value (kaggle 1.7.4
    ``KaggleApi.model_get_cli``). Fallback path, when the CLI is
    absent: the Kaggle Python library's ``model_get``, whose
    returned object exposes ``is_private``.

    Raises RuntimeError when visibility cannot be determined —
    callers treat that as a failed verification, never as public.
    """
    if shutil.which("kaggle") is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["kaggle", "models", "get", handle, "-p", tmpdir],
                env=_cli_env(),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"cannot verify model visibility: "
                    f"`kaggle models get {handle}` failed: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            meta_file = Path(tmpdir) / "model-metadata.json"
            if not meta_file.exists():
                raise RuntimeError(
                    "cannot verify model visibility: `kaggle models get "
                    f"{handle}` wrote no model-metadata.json"
                )
            try:
                metadata = json.loads(meta_file.read_text())
            except ValueError as exc:
                raise RuntimeError(
                    "cannot verify model visibility: downloaded "
                    "model-metadata.json is not valid JSON"
                ) from exc
            if "isPrivate" not in metadata:
                raise RuntimeError(
                    "cannot verify model visibility: downloaded "
                    "model-metadata.json has no isPrivate field"
                )
            return bool(metadata["isPrivate"])
    api = _kaggle_api()
    return bool(api.model_get(handle).is_private)


def ensure_model_public(
    handle: str | None = None,
) -> dict:
    """Verify the model is public; repair it if it is private.

    Reads the model's visibility back from Kaggle
    (:func:`get_model_is_private`). When the model is already
    public, nothing is changed. When it is private — which is what
    kagglehub's ``model_upload`` produces, since it creates a
    missing model with ``isPrivate: True`` and offers no visibility
    parameter — the model is flipped to public with
    ``kaggle models update -p <dir>`` from a minimal metadata file
    carrying ``isPrivate: false`` (the CLI's ``model_update`` puts
    ``isPrivate`` in the update mask, so only visibility changes),
    or with the Kaggle library's ``model_update`` when the CLI is
    absent. The visibility is read back again afterwards.

    Returns ``{"handle": ..., "visibility": "public", "repaired":
    bool}``. Raises RuntimeError when the model is still private
    after the repair, or when visibility cannot be determined —
    the publish must then be treated as failed, and the error
    message carries the manual step.
    """
    handle = handle or f"{KAGGLE_OWNER}/{MODEL_SLUG}"
    if not get_model_is_private(handle):
        return {"handle": handle, "visibility": "public", "repaired": False}
    owner, slug = handle.split("/", 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "model-metadata.json").write_text(
            json.dumps(
                {"ownerSlug": owner, "slug": slug, "isPrivate": False},
                indent=2,
            )
        )
        if shutil.which("kaggle") is not None:
            result = subprocess.run(
                ["kaggle", "models", "update", "-p", tmpdir],
                env=_cli_env(),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"could not make model {handle} public: "
                    f"`kaggle models update` failed: "
                    f"{result.stderr.strip() or result.stdout.strip()}; "
                    f"manual step: {MANUAL_MAKE_PUBLIC_STEP}"
                )
        else:
            _kaggle_api().model_update(tmpdir)
    if get_model_is_private(handle):
        raise RuntimeError(
            f"model {handle} is still private after the visibility "
            f"update; manual step: {MANUAL_MAKE_PUBLIC_STEP}"
        )
    return {"handle": handle, "visibility": "public", "repaired": True}


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
    version_notes: str | None = None,
) -> dict:
    """Upload a new version of ``variation`` to Kaggle.

    Primary path: the Kaggle CLI
    (``kaggle models instances versions create
    <owner>/<model>/<framework>/<variation>`` — kaggle 1.7.4
    ``kaggle/cli.py`` ``parse_model_instance_versions``; there is no
    ``kaggle models version create`` subcommand). Fallback path: kagglehub's
    model upload, used only if the CLI is not installed. Both paths
    refuse to run when ``provenance`` is incomplete.

    Whichever path uploads, the model is then verified public:
    kagglehub creates a missing model private and the CLI path
    relies on the staged metadata, so :func:`ensure_model_public`
    reads the visibility back from Kaggle, repairs a private model,
    and raises when the model cannot be verified public. The upload
    directory must contain the variation's files plus a
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

    cli_result = _upload_via_cli(handle, variation, model_dir, env, version_notes=version_notes)
    if cli_result is not None:
        cli_result["visibility"] = ensure_model_public(handle)
        return cli_result
    hub_result = _upload_via_kagglehub(handle, variation, model_dir)
    hub_result["visibility"] = ensure_model_public(handle)
    return hub_result


def _upload_via_cli(
    handle: str,
    variation: str,
    model_dir: Path,
    env: dict[str, str],
    *,
    version_notes: str | None = None,
) -> dict | None:
    """Upload with the Kaggle CLI; returns None if the CLI is absent.

    The version-create command is ``kaggle models instances versions
    create <owner>/<model>/<framework>/<variation> -p <dir>
    [-n <notes>]`` (kaggle 1.7.4 ``kaggle/cli.py``
    ``parse_model_instance_versions``; the instance handle is
    ``[owner]/[model-name]/[framework]/[instance-slug]`` per
    ``KaggleApi.model_instance_version_create``). There is no
    ``kaggle models version create`` subcommand — ``models`` only
    offers get/list/init/create/delete/update.
    """
    instance_handle = f"{handle}/{FRAMEWORK}/{variation}"
    command = [
        "kaggle",
        "models",
        "instances",
        "versions",
        "create",
        instance_handle,
        "-p",
        str(model_dir),
    ]
    if version_notes:
        command += ["-n", version_notes]
    try:
        completed = subprocess.run(
            command,
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
