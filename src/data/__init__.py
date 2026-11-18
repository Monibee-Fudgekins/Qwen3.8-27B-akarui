"""Data collection, dataset building and validation for Akarui."""

from .competition_collector import collect, discover_hf_datasets, discover_kaggle_competitions
from .dataset_builder import build_dataset
from .dataset_validation import CATEGORIES, REQUIRED_FIELDS, validate_file, validate_record
from .deduplication import dedup_records
from .trajectory_generator import generate_trajectory

__all__ = [
    "CATEGORIES",
    "REQUIRED_FIELDS",
    "build_dataset",
    "collect",
    "dedup_records",
    "discover_hf_datasets",
    "discover_kaggle_competitions",
    "generate_trajectory",
    "validate_file",
    "validate_record",
]
