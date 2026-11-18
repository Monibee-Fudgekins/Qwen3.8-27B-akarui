"""Publish a verified release to Kaggle Models.

Flow:
1. Run the release gate (:mod:`verify_release`). If it fails, publication
   is refused and nothing is uploaded.
2. Build Kaggle model + variation metadata from the provenance manifest
   and the model card. Metadata uses only fields from the Kaggle CLI
   model-metadata documentation
   (https://github.com/Kaggle/kaggle-cli/blob/main/docs/models_metadata.md);
   re-verify the field names against the installed CLI version before the
   first real publication, because Kaggle's metadata schema evolves.
3. Stage the adapter files plus metadata and upload with the Kaggle CLI
   (``kaggle models create`` for a new model, ``kaggle models instances
   versions create <owner>/<model>/<framework>/<variation>`` for a new
   version — there is no ``kaggle models version create`` subcommand),
   falling back to kagglehub's ``model_upload`` when the CLI is unavailable.
   The version command's variation is the BARE variation (``qlora-sft``),
   never the release label (``qlora-sft-v1``), and the instance is created
   first when it does not exist yet (``kaggle models instances get`` /
   ``kaggle models instances create``), so the first publish works.
4. Verify the published model is PUBLIC and repair it if it is not:
   the staged metadata carries ``"isPrivate": false`` (Kaggle
   defaults a new model to private), and after the upload the
   model's visibility is read back from Kaggle; a private model is
   flipped public with ``kaggle models update``. If visibility
   cannot be verified public, publication fails loudly instead of
   leaving a silently private model.
5. Never overwrite or delete previous stable versions.

``--dry-run`` writes the staged metadata and files without uploading, so
the release can be inspected first. No credentials are ever written into
metadata, provenance, or upload logs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The publishing modules live in src/ at the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kaggle_auth import configure_kaggle_auth  # noqa: E402
from verify_release import verify_release  # noqa: E402

from src.evaluation.baseline import load_measured  # noqa: E402
from src.publishing.kaggle_publisher import (  # noqa: E402
    DEFAULT_TAGS,
    FRAMEWORK,
    MODEL_SLUG,
    MODEL_TITLE,
    KAGGLE_OWNER,
    VARIATIONS,
    build_model_metadata as build_kaggle_model_metadata,
    build_variation_metadata as build_kaggle_variation_metadata,
    ensure_instance_exists,
    ensure_model_public,
)

KAGGLE_MODEL_URL = f"https://www.kaggle.com/models/{KAGGLE_OWNER}/{MODEL_SLUG}"
BASE_MODEL_URL = "https://huggingface.co/Qwen/Qwen3.8-27B"


def _describe_release(manifest: dict) -> str:
    """Provenance-rich description shared by model and variation metadata."""
    return (
        "Qwen3.8-27B-akarui is a competition-solving fine-tune of "
        "Qwen/Qwen3.8-27B, developed to improve autonomous data science "
        "workflows, machine-learning experimentation, feature engineering, "
        "statistical validation, programming, debugging and competition "
        "optimization.\n\n"
        f"Base model: {manifest.get('base_model')} "
        f"(revision {manifest.get('base_revision')}).\n"
        f"Training method: {manifest.get('training_method')}.\n"
        f"Training dataset version: {manifest.get('training_dataset_version')}.\n"
        f"Source repository: {manifest.get('source_repository')} "
        f"(commit {manifest.get('source_commit')}).\n\n"
        "See the model card for full provenance, limitations and usage."
    )


def _variation_for_model_version(model_version: str) -> str:
    """Derive the variation slug from a version label like ``qlora-sft-v1``."""
    for variation in VARIATIONS:
        if model_version == variation or model_version.startswith(variation + "-"):
            return variation
    raise ValueError(
        f"model version {model_version!r} does not start with a known "
        f"variation: {VARIATIONS}"
    )


def build_model_metadata(manifest: dict, variation: str) -> dict:
    """Build Kaggle model metadata.

    Delegates to :mod:`src.publishing.kaggle_publisher`, the single
    source of truth for the metadata shape, so the script and the
    library can never disagree about the format.
    """
    return build_kaggle_model_metadata(
        MODEL_TITLE, variation, _describe_release(manifest), list(DEFAULT_TAGS)
    )


def build_variation_metadata(manifest: dict, model_version: str) -> dict:
    """Build Kaggle variation metadata via ``src.publishing.kaggle_publisher``."""
    variation = _variation_for_model_version(model_version)
    usage = (
        "Adapter weights for the Qwen/Qwen3.8-27B base model. "
        "Load the base model and apply this adapter with PEFT; "
        "the adapter is not a standalone model."
        if variation.startswith("qlora")
        else "Merged inference-ready weights based on Qwen/Qwen3.8-27B."
    )
    return build_kaggle_variation_metadata(
        variation, overview=_describe_release(manifest), usage=usage
    )


def stage_release(
    adapter_dir: Path,
    manifest: dict,
    model_card_text: str,
    model_version: str,
    staging_dir: Path,
) -> Path:
    """Copy adapter files and write metadata into the staging directory.

    The model card is staged as a ``MODEL_CARD.md`` file, not embedded
    in the metadata JSON (Kaggle's metadata format has no model-card
    field).
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for item in adapter_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, staging_dir / item.name)
    variation = _variation_for_model_version(model_version)
    (staging_dir / "model-metadata.json").write_text(
        json.dumps(build_model_metadata(manifest, variation), indent=2)
    )
    (staging_dir / "model-variation-metadata.json").write_text(
        json.dumps(build_variation_metadata(manifest, model_version), indent=2)
    )
    (staging_dir / "provenance.json").write_text(
        json.dumps(manifest, indent=2)
    )
    if model_card_text:
        (staging_dir / "MODEL_CARD.md").write_text(model_card_text)
    return staging_dir


def _measured_reference_path(champion_path: Path, baseline_path: Path) -> Path | None:
    """Return the promotion reference file, when one is measured.

    The measured champion (``evaluation/champion_results.json``)
    wins when one exists; otherwise the measured baseline
    (``evaluation/baseline_results.json``) is the reference for the
    first promotion. The shipped baseline file is an honest
    not-measured template (``normalized_score`` is null), so it
    counts as no reference at all -- verification then fails on the
    missing baseline instead of publishing against nothing. A
    missing or unparseable file also counts as no reference.
    """
    if load_measured(champion_path) is not None:
        return champion_path
    if load_measured(baseline_path) is not None:
        return baseline_path
    return None


def _model_exists() -> bool:
    """Return True if the Kaggle model already exists (CLI check)."""
    result = subprocess.run(
        ["kaggle", "models", "get", f"{KAGGLE_OWNER}/{MODEL_SLUG}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode == 0


def _staged_variation_text(staging_dir: Path) -> tuple[str, str]:
    """Read overview/usage from the staged variation metadata, if present."""
    meta_file = staging_dir / "model-variation-metadata.json"
    if not meta_file.exists():
        return "", ""
    try:
        data = json.loads(meta_file.read_text())
    except (ValueError, OSError):
        return "", ""
    return str(data.get("overview") or ""), str(data.get("usage") or "")


def upload_with_cli(
    staging_dir: Path, release_version: str, model_version: str
) -> None:
    """Upload via the Kaggle CLI. Raises RuntimeError on failure.

    New model: ``kaggle models create -p <dir>`` (creates only the
    model shell — no instance, no files), so the first publish
    continues with an instance ensure and a version create below.
    Existing model: straight to the instance ensure + version
    create. The version command is ``kaggle models instances
    versions create <owner>/<model>/<framework>/<variation> -p
    <dir> -n <notes>`` (kaggle 1.7.4 ``kaggle/cli.py``
    ``parse_model_instance_versions``; there is no ``kaggle models
    version create`` subcommand). The 4th handle component is the
    BARE variation (``qlora-sft``), derived from the release label
    (``qlora-sft-v1``) — Kaggle treats it as the instance slug, so
    the label must never appear there. ``model_version`` is the
    release variation label, e.g. ``qlora-sft-v1``.

    The version create requires the instance to exist, so
    :func:`ensure_instance_exists` checks with ``kaggle models
    instances get`` and creates the instance (``kaggle models
    instances create -p <tmpdir>``) only when it is missing.
    """
    if shutil.which("kaggle") is None:
        raise RuntimeError("kaggle CLI not found on PATH")
    variation = _variation_for_model_version(model_version)
    instance_handle = f"{KAGGLE_OWNER}/{MODEL_SLUG}/{FRAMEWORK}/{variation}"
    overview, usage = _staged_variation_text(staging_dir)
    if not _model_exists():
        result = subprocess.run(
            ["kaggle", "models", "create", "-p", str(staging_dir)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "kaggle upload failed: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
    ensure_instance_exists(instance_handle, variation, overview=overview, usage=usage)
    command = [
        "kaggle",
        "models",
        "instances",
        "versions",
        "create",
        instance_handle,
        "-p",
        str(staging_dir),
        "-n",
        f"Release {release_version}",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            "kaggle upload failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def upload_with_kagglehub(
    staging_dir: Path, release_version: str, model_version: str
) -> None:
    """Upload via kagglehub. Raises RuntimeError on failure.

    kagglehub 0.3.12 ``model_upload`` parses the handle with
    ``parse_model_handle``, which requires 4 parts
    (``owner/model/framework/variation``) and raises ValueError
    otherwise — so the handle is the 4-part form with the BARE
    variation, never the 2-part ``owner/model`` form. kagglehub
    creates a missing model and a missing instance itself
    (``models_helpers.create_model_if_missing`` /
    ``create_model_instance_or_version``), so this path needs no
    separate instance-create step.
    """
    try:
        import kagglehub  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(f"kagglehub is not installed: {exc}") from exc
    variation = _variation_for_model_version(model_version)
    handle = f"{KAGGLE_OWNER}/{MODEL_SLUG}/{FRAMEWORK}/{variation}"
    kagglehub.model_upload(
        handle, str(staging_dir), version_notes=f"Release {release_version}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success (or dry-run), else 1."""
    parser = argparse.ArgumentParser(
        description="Publish a verified release to Kaggle Models."
    )
    parser.add_argument("--provenance", required=True, help="Provenance manifest JSON")
    parser.add_argument("--evaluation", default="evaluation/results.json")
    parser.add_argument("--champion", default="evaluation/champion_results.json")
    parser.add_argument("--baseline", default="evaluation/baseline_results.json")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument(
        "--adapter", required=True, help="Adapter directory to publish"
    )
    parser.add_argument(
        "--model-card",
        default="model_cards/MODEL_CARD.md",
        help="Model card markdown",
    )
    parser.add_argument("--release-version", required=True, help="Release version, e.g. 1")
    parser.add_argument("--model-version", required=True, help="Variation label, e.g. qlora-sft-v1")
    parser.add_argument(
        "--staging-dir",
        default="output/kaggle-publish",
        help="Staging directory for the upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write metadata and stage files without uploading",
    )
    args = parser.parse_args(argv)

    provenance_path = Path(args.provenance)
    if not provenance_path.exists():
        print(f"error: provenance manifest not found: {provenance_path}", file=sys.stderr)
        return 1
    manifest = json.loads(provenance_path.read_text())

    # Gate 1: the automated release verification must pass.
    # The reference is the measured champion when one exists, else
    # the measured baseline (first promotion). An unmeasured
    # baseline template counts as no reference, so verification
    # fails instead of publishing against nothing.
    reference_path = _measured_reference_path(
        Path(args.champion), Path(args.baseline)
    )
    reasons = verify_release(
        provenance_path,
        Path(args.evaluation),
        reference_path,
        Path(args.config),
    )
    if reasons:
        print("Publication REFUSED — release verification failed:")
        for reason in reasons:
            print(f"  - {reason}")
        return 1

    adapter_dir = Path(args.adapter)
    if not adapter_dir.exists():
        print(f"error: adapter directory not found: {adapter_dir}", file=sys.stderr)
        return 1
    model_card_path = Path(args.model_card)
    model_card_text = model_card_path.read_text() if model_card_path.exists() else ""

    staging_dir = stage_release(
        adapter_dir,
        manifest,
        model_card_text,
        args.model_version,
        Path(args.staging_dir) / args.model_version,
    )
    print(f"Release staged at {staging_dir}")
    if args.dry_run:
        print("Dry run: metadata and files staged, nothing uploaded.")
        return 0

    try:
        auth = configure_kaggle_auth()
        print(f"Kaggle auth configured via {auth['method']}")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        upload_with_cli(staging_dir, args.release_version, args.model_version)
    except RuntimeError as cli_error:
        print(f"kaggle CLI upload failed ({cli_error}); trying kagglehub...")
        try:
            upload_with_kagglehub(staging_dir, args.release_version, args.model_version)
        except RuntimeError as hub_error:
            print(f"error: kagglehub upload also failed: {hub_error}", file=sys.stderr)
            return 1

    # The model must be public: kagglehub creates a missing model
    # private and the CLI path relies on the staged metadata, so
    # read the visibility back and repair it if needed. A model
    # that cannot be verified public is a failed publication.
    try:
        visibility = ensure_model_public(f"{KAGGLE_OWNER}/{MODEL_SLUG}")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if visibility["repaired"]:
        print("Model was private after upload; visibility repaired to public.")
    print(f"Published: {KAGGLE_MODEL_URL} (release {args.release_version})")
    print(f"Visibility verified: public ({KAGGLE_MODEL_URL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
