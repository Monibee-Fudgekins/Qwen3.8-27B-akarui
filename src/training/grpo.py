"""Stage 3 (EXPERIMENTAL): Group Relative Policy Optimization (GRPO).

GRPO is disabled unless the config explicitly enables it
(``grpo.enabled: true``). It is experimental: the reward must come from
measured execution of generated solutions, never from an LLM judge.

The reward hook shipped here, :func:`execution_reward`, is a
placeholder adapter: it extracts a ``validation_score`` value that the
execution harness wrote into the completion text and returns 0.0 when
no measured score is present. Before any real GRPO run, replace it
with a reward function wired to the actual sandboxed execution
harness, where completions are executed and scored by measurement.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .checkpointing import write_training_manifest
from .model_loader import get_lora_config, load_base_model, load_tokenizer
from .runtime_guard import RuntimeGuard, resolve_runtime_guard, runtime_guard_callback
from .trainer_utils import get_device_info, load_config, set_seed

logger = logging.getLogger(__name__)

_SCORE_PATTERN = re.compile(
    r"validation_score\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE
)


def execution_reward(prompts, completions, **kwargs) -> list[float]:
    """Placeholder execution-based reward.

    Returns the ``validation_score`` recorded in each completion (a
    number the execution harness wrote after running the generated
    solution), or 0.0 when no measured score is present. A completion
    with no measured execution result earns no reward -- it is never
    rewarded on the basis of how plausible it reads.

    This is a placeholder: wire it to the real sandboxed execution
    harness before running GRPO for real.
    """
    rewards: list[float] = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        match = _SCORE_PATTERN.search(text)
        if match:
            try:
                rewards.append(float(match.group(1)))
                continue
            except ValueError:
                pass
        rewards.append(0.0)
    return rewards


def run_grpo(
    config: dict | str | Path,
    tasks_file: str | Path,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
) -> dict:
    """Run GRPO on competition tasks.

    ``config`` is a config dict or a path to a YAML config file.
    ``tasks_file`` is a JSONL of tasks, each with a ``prompt`` (string
    or chat messages). ``num_generations`` comes from
    ``config["grpo"]["num_generations"]`` (default 4). When
    ``resume_from`` names a checkpoint directory, training resumes
    from it.

    Raises :class:`RuntimeError` when GRPO is not explicitly enabled in
    the config -- this stage is experimental and opt-in only. The
    enabled check runs before any heavy import, so a disabled call
    fails fast with this message even without trl installed.

    11h guard: the run is bounded by ``runtime.max_runtime_hours``
    (default 10h) and ``training.max_steps`` (default 600 for GRPO),
    exactly as for SFT. The guard is resolved (and validated) before
    any heavy import or model load, so an inconsistent cap fails fast
    instead of mid-run. When the wall clock runs out the trainer is
    stopped gracefully, a checkpoint is saved, a
    ``partial_complete.json`` record is written, and the summary
    reports ``status="partial_complete"`` -- the run still exits 0,
    because a partial adapter is a usable outcome and a Kaggle 12h
    kill is what is being avoided. A run that finishes its epoch or
    step cap reports ``status="complete"``.
    """
    if isinstance(config, dict):
        config_dict, config_path = config, None
    else:
        config_path = Path(config)
        config_dict = load_config(config_path)
    grpo_cfg = config_dict.get("grpo", {})
    if not grpo_cfg.get("enabled", False):
        raise RuntimeError(
            "GRPO is EXPERIMENTAL and disabled: set grpo.enabled: true in "
            "the config to run it, and only once the execution-based "
            "reward harness is wired up."
        )

    # Resolve (and validate) the runtime guard BEFORE the heavy
    # imports and model load, so an inconsistent cap fails fast
    # instead of mid-run.
    guard_cfg = resolve_runtime_guard(config_dict)
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

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    training = config_dict.get("training", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(training.get("seed", 42))
    set_seed(seed)
    gpu_info = get_device_info()
    logger.info("device info: %s", gpu_info)

    tasks: list[dict] = []
    with Path(tasks_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("prompt"):
                tasks.append({"prompt": record["prompt"]})
    if not tasks:
        raise ValueError(f"no tasks with a 'prompt' field found in {tasks_file}")
    logger.info("loaded %d GRPO tasks from %s", len(tasks), tasks_file)
    dataset = Dataset.from_list(tasks)

    num_generations = int(grpo_cfg.get("num_generations", 4))
    per_device = int(grpo_cfg.get("per_device_train_batch_size", 4))
    if per_device % num_generations != 0:
        raise ValueError(
            "GRPO requires per_device_train_batch_size to be divisible by "
            f"num_generations ({per_device} % {num_generations} != 0)"
        )

    tokenizer = load_tokenizer(config_dict)
    model = load_base_model(config_dict)
    model.config.use_cache = False
    lora_config = get_lora_config(config_dict)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_generations=num_generations,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=int(
            grpo_cfg.get("gradient_accumulation_steps", 1)
        ),
        learning_rate=float(grpo_cfg.get("learning_rate", 1e-5)),
        lr_scheduler_type=str(grpo_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(grpo_cfg.get("warmup_ratio", 0.03)),
        num_train_epochs=float(grpo_cfg.get("num_train_epochs", 1)),
        # max_steps overrides num_train_epochs in TrainingArguments;
        # -1 means "no cap" (full epoch). The LR schedule is computed
        # over max_steps when it is set, so a capped run is a proper
        # short run, not a chopped epoch.
        max_steps=int(max_steps) if max_steps is not None else -1,
        max_completion_length=int(grpo_cfg.get("max_completion_length", 1024)),
        max_prompt_length=int(grpo_cfg.get("max_prompt_length", 2048)),
        bf16=bool(training.get("bf16", True)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        optim=str(training.get("optim", "adamw_bnb_8bit")),
        logging_steps=int(training.get("logging_steps", 10)),
        save_steps=int(training.get("save_steps", 100)),
        seed=seed,
        report_to="none",
        save_total_limit=5,
    )
    # Wall-clock callback: checked after every optimizer step (shared
    # implementation, see src/training/runtime_guard.py).
    guard_callback = runtime_guard_callback(guard, logger)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[execution_reward],
        args=grpo_config,
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
                "configs/grpo.yaml --stage grpo --force --resume"
            ),
        }
        (output_dir / "partial_complete.json").write_text(
            json.dumps(partial_record, indent=2)
        )
        logger.info("partial-complete record written; checkpoint: %s", checkpoint_dir)
    else:
        logger.info("GRPO complete; adapter saved to %s", output_dir)

    _, manifest = write_training_manifest(
        output_dir,
        config=config_dict,
        config_path=config_path,
        dataset_path=tasks_file,
        seed=seed,
        gpu_info=gpu_info,
        extra={
            "training_method": "QLoRA-GRPO-experimental",
            "grpo_tasks": len(tasks),
            "num_generations": num_generations,
            "status": status,
            "steps_completed": steps_completed,
            "elapsed_hours": round(elapsed_hours, 3),
            "runtime_guard": guard.as_dict(),
        },
    )
    return {
        "status": status,
        "output_dir": str(output_dir),
        "grpo_tasks": len(tasks),
        "num_generations": num_generations,
        "steps_completed": steps_completed,
        "max_steps": max_steps,
        "elapsed_hours": round(elapsed_hours, 3),
        "stopped_early": stopped_early,
        "checkpoint": str(checkpoint_dir) if checkpoint_dir else None,
        "manifest": manifest,
    }
