"""Stage 2: Direct Preference Optimization (DPO).

Preference pairs must be grounded in verified execution: a pair is
only used when its chosen response carries execution evidence
(``execution_verified: true``). Pairs whose chosen response lacks that
evidence are logged and skipped -- Akarui never trains on
opinion-only preference labels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .checkpointing import write_training_manifest
from .model_loader import get_lora_config, load_base_model, load_tokenizer
from .trainer_utils import get_device_info, load_config, set_seed

logger = logging.getLogger(__name__)


def _as_messages(value, default_role: str) -> list[dict]:
    """Normalize a prompt / response to a list of chat messages.

    A single message dict keeps only ``role``/``content`` -- metadata
    keys such as ``execution_verified`` are stripped so nothing
    unexpected reaches the trainer.
    """
    if isinstance(value, list):
        return [
            {k: v for k, v in item.items() if k in ("role", "content")}
            if isinstance(item, dict)
            else {"role": default_role, "content": str(item)}
            for item in value
        ]
    if isinstance(value, dict):
        return [
            {k: v for k, v in value.items() if k in ("role", "content")}
        ]
    return [{"role": default_role, "content": str(value)}]


def _chosen_is_verified(pair: dict) -> bool:
    """True when the pair documents verified execution for the chosen
    response -- either at the pair level or on the chosen entry."""
    if pair.get("execution_verified") is True:
        return True
    chosen = pair.get("chosen")
    if isinstance(chosen, dict) and chosen.get("execution_verified") is True:
        return True
    return False


def _load_preference_pairs(preference_file: str | Path) -> tuple[list[dict], int]:
    """Load and filter preference pairs.

    Returns ``(kept_pairs, skipped_count)``. A pair is kept only when
    it has a prompt, a chosen and a rejected response, and the chosen
    response has verified execution evidence.
    """
    kept: list[dict] = []
    skipped = 0
    with Path(preference_file).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            if not isinstance(pair, dict):
                logger.warning("skipping line %d: not a JSON object", line_number)
                skipped += 1
                continue
            if not pair.get("prompt") or not pair.get("chosen") or not pair.get("rejected"):
                logger.warning(
                    "skipping line %d: missing prompt/chosen/rejected", line_number
                )
                skipped += 1
                continue
            if not _chosen_is_verified(pair):
                logger.warning(
                    "skipping line %d: chosen response lacks verified "
                    "execution evidence; opinion-only labels are never trained on",
                    line_number,
                )
                skipped += 1
                continue
            kept.append(
                {
                    "prompt": _as_messages(pair["prompt"], "user"),
                    "chosen": _as_messages(pair["chosen"], "assistant"),
                    "rejected": _as_messages(pair["rejected"], "assistant"),
                }
            )
    return kept, skipped


def run_dpo(
    config: dict | str | Path,
    preference_file: str | Path,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
) -> dict:
    """Run DPO on verified preference pairs.

    ``config`` is a config dict or a path to a YAML config file; the
    DPO beta defaults to 0.1 (``config["dpo"]["beta"]`` overrides it).
    When ``resume_from`` names a checkpoint directory, training resumes
    from it. Returns a summary dict with kept/skipped pair counts.
    """
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    if isinstance(config, dict):
        config_dict, config_path = config, None
    else:
        config_path = Path(config)
        config_dict = load_config(config_path)
    dpo_cfg = config_dict.get("dpo", {})
    training = config_dict.get("training", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(training.get("seed", 42))
    set_seed(seed)
    gpu_info = get_device_info()
    logger.info("device info: %s", gpu_info)

    pairs, skipped = _load_preference_pairs(preference_file)
    if not pairs:
        raise ValueError(
            f"no usable preference pairs in {preference_file} "
            f"({skipped} skipped for missing fields or unverified execution)"
        )
    logger.info("loaded %d preference pairs (%d skipped)", len(pairs), skipped)
    dataset = Dataset.from_list(pairs)

    tokenizer = load_tokenizer(config_dict)
    model = load_base_model(config_dict)
    model.config.use_cache = False
    lora_config = get_lora_config(config_dict)

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        beta=float(dpo_cfg.get("beta", 0.1)),
        per_device_train_batch_size=int(
            dpo_cfg.get("per_device_train_batch_size", 1)
        ),
        gradient_accumulation_steps=int(
            dpo_cfg.get("gradient_accumulation_steps", 4)
        ),
        learning_rate=float(dpo_cfg.get("learning_rate", 5e-5)),
        lr_scheduler_type=str(dpo_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(dpo_cfg.get("warmup_ratio", 0.03)),
        num_train_epochs=float(dpo_cfg.get("num_train_epochs", 1)),
        bf16=bool(training.get("bf16", True)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        optim=str(training.get("optim", "adamw_bnb_8bit")),
        logging_steps=int(training.get("logging_steps", 10)),
        save_steps=int(training.get("save_steps", 100)),
        seed=seed,
        max_length=int(dpo_cfg.get("max_length", 4096)),
        max_prompt_length=int(dpo_cfg.get("max_prompt_length", 2048)),
        report_to="none",
        save_total_limit=5,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    _, manifest = write_training_manifest(
        output_dir,
        config=config_dict,
        config_path=config_path,
        dataset_path=preference_file,
        seed=seed,
        gpu_info=gpu_info,
        extra={
            "training_method": "QLoRA-DPO",
            "preference_pairs": len(pairs),
            "preference_pairs_skipped": skipped,
            "beta": float(dpo_cfg.get("beta", 0.1)),
        },
    )
    logger.info("DPO complete; adapter saved to %s", output_dir)
    return {
        "output_dir": str(output_dir),
        "preference_pairs": len(pairs),
        "preference_pairs_skipped": skipped,
        "manifest": manifest,
    }
