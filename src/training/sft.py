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
from .runtime_guard import RuntimeGuard, resolve_runtime_guard, runtime_guard_callback
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

    11h guard: the run is bounded by ``runtime.max_runtime_hours``
    (default 10h) and ``training.max_steps`` (default 600 for SFT).
    When the wall clock runs out the trainer is stopped gracefully,
    a checkpoint is saved, a ``partial_complete.json`` record is
    written, and the summary reports ``status="partial_complete"``
    -- the run still exits 0, because a partial adapter is a usable
    outcome and a Kaggle 12h kill is what is being avoided. A run
    that finishes its epoch or step cap reports
    ``status="complete"``.
    """
    import time

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    config, config_path = _resolve_config(config)
    training = config.get("training", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve (and validate) the runtime guard BEFORE loading the
    # model, so an inconsistent cap fails fast instead of mid-run.
    guard_cfg = resolve_runtime_guard(config)
    guard = RuntimeGuard(guard_cfg["max_runtime_hours"])
    max_steps = guard_cfg["max_steps"]
    logger.info(
        "runtime guard: max %.1fh, step cap %s, worst-case estimate %s",
        guard_cfg["max_runtime_hours"],
        max_steps,
        (
            f"{guard_cfg['estimated_total_hours']:.2f}h total"
            if guard_cfg["estimated_total_hours"] is not None
            else "uncapped"
        ),
    )

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
        # max_steps overrides num_train_epochs in TrainingArguments;
        # -1 means "no cap" (full epoch). The LR schedule is computed
        # over max_steps when it is set, so a capped run is a proper
        # short run, not a chopped epoch.
        max_steps=int(max_steps) if max_steps is not None else -1,
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

    # Wall-clock callback: checked after every optimizer step. The
    # import is lazy so this module stays importable without torch.
    guard_callback = runtime_guard_callback(guard, logger)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[guard_callback],
    )
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    steps_completed = trainer.state.global_step
    elapsed_hours = guard.elapsed_seconds() / 3600
    stopped_early = bool(guard_callback.triggered)
    status = "partial_complete" if stopped_early else "complete"
    checkpoint_dir: Path | None = None
    if stopped_early:
        # Save a resumable checkpoint at the stop point so the next
        # run can --resume from it. _save_checkpoint is a private
        # Trainer API, so fall back to an adapter-only save if its
        # signature ever changes.
        checkpoint_dir = output_dir / f"checkpoint-{steps_completed}"
        try:
            trainer._save_checkpoint(trainer.model, trial=None)
        except Exception as exc:  # noqa: BLE001 - best effort, fallback below
            logger.warning(
                "trainer._save_checkpoint failed (%s); saving adapter only", exc
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(checkpoint_dir))
            tokenizer.save_pretrained(str(checkpoint_dir))
        partial_record = {
            "status": "partial_complete",
            "reason": (
                f"runtime guard: max_runtime_hours="
                f"{guard_cfg['max_runtime_hours']} reached"
            ),
            "steps_completed": steps_completed,
            "max_steps": max_steps,
            "elapsed_hours": round(elapsed_hours, 3),
            "checkpoint": str(checkpoint_dir),
            "resume_hint": (
                "resume with: python scripts/train.py --config "
                "configs/sft.yaml --stage sft --resume"
            ),
        }
        (output_dir / "partial_complete.json").write_text(
            json.dumps(partial_record, indent=2)
        )
        logger.info("partial-complete record written; checkpoint: %s", checkpoint_dir)
    else:
        logger.info("SFT complete; adapter saved to %s", output_dir)

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
            "status": status,
            "steps_completed": steps_completed,
            "elapsed_hours": round(elapsed_hours, 3),
            "runtime_guard": guard.as_dict(),
        },
    )
    return {
        "status": status,
        "output_dir": str(output_dir),
        "train_examples": len(texts),
        "steps_completed": steps_completed,
        "max_steps": max_steps,
        "elapsed_hours": round(elapsed_hours, 3),
        "stopped_early": stopped_early,
        "checkpoint": str(checkpoint_dir) if checkpoint_dir else None,
        "manifest": manifest,
    }
