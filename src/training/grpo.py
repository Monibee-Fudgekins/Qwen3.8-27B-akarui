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
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[execution_reward],
        args=grpo_config,
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
        dataset_path=tasks_file,
        seed=seed,
        gpu_info=gpu_info,
        extra={
            "training_method": "QLoRA-GRPO-experimental",
            "grpo_tasks": len(tasks),
            "num_generations": num_generations,
        },
    )
    logger.info("GRPO complete; adapter saved to %s", output_dir)
    return {
        "output_dir": str(output_dir),
        "grpo_tasks": len(tasks),
        "num_generations": num_generations,
        "manifest": manifest,
    }
