# Provenance

How every Qwen3.8-27B-akarui release is traced to the code, data, and
configuration that produced it.

## Provenance manifest

Every published version ships a machine-readable provenance manifest
(`provenance.json` in the uploaded Kaggle variation) with exactly
these fields:

| Field | Meaning |
| --- | --- |
| `model` | `Qwen3.8-27B-akarui` |
| `base_model` | `Qwen/Qwen3.8-27B` |
| `base_revision` | Pinned base-model revision |
| `training_method` | e.g. `QLoRA`, `QLoRA+DPO` |
| `training_date` | ISO date of the training run |
| `training_dataset_version` | Dataset version, e.g. `v1` |
| `training_dataset_hash` | SHA-256 of the dataset file |
| `training_config_hash` | SHA-256 of the training config |
| `source_repository` | GitHub repository URL |
| `source_commit` | Git commit that produced the release |
| `training_environment` | e.g. `Kaggle` |
| `gpu_configuration` | Detected GPUs (model, count, VRAM) |
| `evaluation_results` | Measured benchmark results |
| `artifact_hashes` | SHA-256 of each published artifact |
| `release_version` | Kaggle model version number |

No provenance field is ever filled with a placeholder in a published
manifest. `src/publishing/provenance.py` builds the manifest and
`validate_provenance()` rejects any manifest with a missing or empty
required field; `src/publishing/kaggle_publisher.py` refuses to
upload an incomplete manifest.

## Pinned base model

- **Base model:** Qwen/Qwen3.8-27B
- **Pinned revision:** `37561816e66b5b9153b1b64c10968c18c1e6f4a0`
- **Base model page:** https://huggingface.co/Qwen/Qwen3.8-27B
- **Official repository:** https://github.com/QwenLM/Qwen3.8
- **License:** Apache 2.0

Every training cycle starts from this pinned revision, so releases
are reproducible and comparable.

## Releases

No model version has been published yet. When releases exist, they
are recorded in `model_cards/RELEASE_HISTORY.md` and as JSON records
under `releases/` (see `src/publishing/release_manifest.py`).
