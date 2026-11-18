"""Publishing utilities for Qwen3.8-27B-akarui.

Covers Kaggle model metadata and uploads, model-card rendering,
provenance manifests, and release records. Nothing in this package
ever prints or stores credentials.
"""

from .kaggle_publisher import build_model_metadata, build_variation_metadata, upload_variation
from .model_card import render_model_card
from .provenance import (
    build_provenance,
    compute_config_hash,
    compute_file_hash,
    validate_provenance,
)
from .release_manifest import list_releases, load_release, write_release

__all__ = [
    "build_model_metadata",
    "build_variation_metadata",
    "upload_variation",
    "render_model_card",
    "build_provenance",
    "validate_provenance",
    "compute_file_hash",
    "compute_config_hash",
    "write_release",
    "load_release",
    "list_releases",
]
