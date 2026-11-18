"""Stage 1: supervised fine-tuning (QLoRA) of Qwen3.8-27B.

Runs TRL's :class:`SFTTrainer` with the LoRA config, 4-bit NF4 base
weights, BF16 compute, cosine schedule and gradient checkpointing.
Training is resumable: pass ``resume_from`` with a checkpoint directory
and the trainer continues from it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .checkpointing import write_training_manifest
from .model_loader import get_lora_config, load_base_model, load_tokenizer
from .trainer_utils import (
    format_messages_for_training,
    get_device_info,
    load_config,
    set_seed,
)

logger = logging.getLogger(__name__)


def _resolve_config(config: dict | str | Path) -> tuple[dict, Path | None]:
    if isinstance(config, dict):
        return config, None
    path = Path(config)
    return load_config(path), path


def _record_to_text(record: dict, tokenizer) -> str | None:
    """Render one training record as a single chat-formatted string.

    Records may carry ``messages`` (used directly), ``text`` (used as
    is), or trajectory fields (assembled into a user/assistant turn
    whose assistant message keeps the reasoning summary and the
    generated code together). Returns None when the record has no
    usable content.
    """
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        return format_messages_for_training(tokenizer, messages)
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return text
    if record.get("task_description") and record.get("generated_code"):
        messages = [
            {
                "role": "user",
                "content": str(record["task_description"]),
            },
            {
                "role": "assistant",
                "content": (
                    str(record.get("reasoning_summary", "")).strip()
                    + "\n\n```python\n"
                    + str(record["generated_code"])
                    + "\n```"
                ).strip(),
            },
        ]
        return format_messages_for_training(tokenizer, messages)
    return None


def _load_train_texts(train_file: str | Path, tokenizer) -> list[str]:
    texts: list[str] = []
    with Path(train_file).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = _record_to_text(record, tokenizer)
            if text is None:
                logger.warning("skipping line %d: no usable text content", line_number)
                continue
            texts.append(text)
    if not texts:
        raise ValueError(f"no usable training records found in {train_file}")
    return texts


def run_sft(
    config: dict | str | Path,
    train_file: str | Path,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
) -> dict:
    """Run QLoRA SFT.

    ``config`` is a config dict or a path to a YAML config file.
    ``train_file`` is a JSONL of trajectory / chat records.
    ``output_dir`` receives the adapter, tokenizer and training
    manifest. When ``resume_from`` names a checkpoint directory,
    training resumes from it. Returns a summary dict.
    """
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    config, config_path = _resolve_config(config)
    training = config.get("training", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(training.get("seed", 42))
    set_seed(seed)
    gpu_info = get_device_info()
    logger.info("device info: %s", gpu_info)

    tokenizer = load_tokenizer(config)
    model = load_base_model(config)
    model.config.use_cache = False
    lora_config = get_lora_config(config)

    texts = _load_train_texts(train_file, tokenizer)
    dataset = Dataset.from_dict({"text": texts})
    logger.info("loaded %d training examples from %s", len(texts), train_file)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(
            training.get("per_device_train_batch_size", 1)
        ),
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 4)
        ),
        learning_rate=float(training.get("learning_rate", 1e-4)),
        lr_scheduler_type=str(training.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        num_train_epochs=float(training.get("num_train_epochs", 1)),
        max_length=int(training.get("max_seq_length", 4096)),
        bf16=bool(training.get("bf16", True)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        optim=str(training.get("optim", "adamw_bnb_8bit")),
        logging_steps=int(training.get("logging_steps", 10)),
        save_steps=int(training.get("save_steps", 100)),
        seed=seed,
        dataset_text_field="text",
        report_to="none",
        save_total_limit=5,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    _, manifest = write_training_manifest(
        output_dir,
        config=config,
        config_path=config_path,
        dataset_path=train_file,
        seed=seed,
        gpu_info=gpu_info,
        extra={
            "training_method": "QLoRA-SFT",
            "train_examples": len(texts),
            "resumed_from": str(resume_from) if resume_from else None,
        },
    )
    logger.info("SFT complete; adapter saved to %s", output_dir)
    return {
        "output_dir": str(output_dir),
        "train_examples": len(texts),
        "manifest": manifest,
    }
