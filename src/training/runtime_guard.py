"""Wall-clock guard for long Kaggle training runs.

Kaggle hard-kills notebooks at 12 hours (exit 137) with no chance to
save. This module is the single place that turns that platform limit
into a training setting:

* ``EFFECTIVE_CAP_HOURS`` (11h) is the cap the project plans against
  -- one hour of margin under Kaggle's 12h kill.
* ``DEFAULT_MAX_RUNTIME_HOURS`` (10h) is the default training
  wall-clock budget, leaving a further hour of margin under the
  effective cap.
* ``DEFAULT_SETUP_ALLOWANCE_HOURS`` (1h) is the time budgeted for
  setup before training starts: base-model download, tokenizer and
  dataset load. The step cap is sized from what is left.
* ``ASSUMED_WORST_STEP_SECONDS`` (50s) is the per-step rate the cap
  is sized from. It is the worst rate measured on the sibling Glimmer
  kernel (v9: ~50 s/step on T4-class hardware for a ~30B model);
  Akarui is a 27B model on the same class of Kaggle GPU, so 50 s/step
  is the conservative planning rate, not a measured Akarui rate. The
  first real Akarui run must measure its own rate and this constant
  must be revised from that measurement.

Arithmetic for the SFT defaults (``max_steps=600`` in
``configs/sft.yaml``)::

    training budget = 10.0h - 1.0h setup = 9.0h = 32,400s
    steps at worst rate = 32,400 / 50 = 648 -> capped at 600
    600 steps x 50 s/step = 30,000s = 8.33h training
    total = 1.0h setup + 8.33h training = 9.33h
    margin = 11h cap - 9.33h = 1.67h (2.67h under Kaggle's 12h kill)

DPO and GRPO share the same 10h / 1h budget and the same 600-step
cap, sized from the same (borrowed, unmeasured) 50 s/step
assumption. Their steps are expected to be slower than SFT steps --
DPO runs a chosen and a rejected pass per step, GRPO generates
``num_generations`` completions and executes them per step -- so for
them the wall-clock guard is the binding limit and the step cap is a
backstop, not the primary bound. A run that hits the wall clock stops
gracefully with a checkpoint and a partial-complete record instead of
being killed.

At a faster 25 s/step the same 600 steps take 4.17h. The step cap is
an upper bound, not a target: a run whose epoch ends first simply
finishes early, and a run that hits the wall-clock guard stops
gracefully with a checkpoint and a partial-complete record instead of
being killed.

This module uses only the standard library so it can be imported by
both the training code and the planning script without a GPU stack.
"""

from __future__ import annotations

import math
import time

#: Kaggle's hard kill for a notebook session, in hours.
KAGGLE_HARD_LIMIT_HOURS = 12.0

#: Effective cap the project plans against: 1h of margin under the kill.
EFFECTIVE_CAP_HOURS = 11.0

#: Default wall-clock budget for a training run, in hours.
DEFAULT_MAX_RUNTIME_HOURS = 10.0

#: Time budgeted for setup (base download, tokenizer/dataset load).
DEFAULT_SETUP_ALLOWANCE_HOURS = 1.0

#: Conservative per-step planning rate (see module docstring).
ASSUMED_WORST_STEP_SECONDS = 50.0


def resolve_runtime_guard(config: dict) -> dict:
    """Resolve the runtime guard settings from a merged config dict.

    Reads ``runtime.max_runtime_hours`` (default 10.0),
    ``runtime.setup_allowance_hours`` (default 1.0) and
    ``training.max_steps`` (optional step cap). Validates that the
    hours are within (0, 11] and that, when a step cap is set, the
    worst-case estimate (setup + steps x 50s) fits inside the hour
    budget. Raises :class:`ValueError` on any inconsistency so a bad
    config fails before a model is loaded, never mid-run.

    Returns a dict with the resolved hours, step cap, worst-case
    estimates and the margin under the 11h effective cap.
    """
    runtime = config.get("runtime", {}) or {}
    training = config.get("training", {}) or {}

    try:
        max_hours = float(runtime.get("max_runtime_hours", DEFAULT_MAX_RUNTIME_HOURS))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime.max_runtime_hours must be a number of hours, "
            f"got {runtime.get('max_runtime_hours')!r}"
        ) from exc
    if not 0 < max_hours <= EFFECTIVE_CAP_HOURS:
        raise ValueError(
            f"runtime.max_runtime_hours must be in (0, {EFFECTIVE_CAP_HOURS}] "
            f"hours (margin under Kaggle's {KAGGLE_HARD_LIMIT_HOURS}h kill); "
            f"got {max_hours}"
        )

    try:
        setup_hours = float(
            runtime.get("setup_allowance_hours", DEFAULT_SETUP_ALLOWANCE_HOURS)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime.setup_allowance_hours must be a number of hours, "
            f"got {runtime.get('setup_allowance_hours')!r}"
        ) from exc
    if not 0 <= setup_hours < max_hours:
        raise ValueError(
            "runtime.setup_allowance_hours must be in [0, "
            f"max_runtime_hours); got {setup_hours}"
        )

    max_steps = training.get("max_steps")
    if max_steps is not None:
        try:
            max_steps = int(max_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"training.max_steps must be a positive integer, got {max_steps!r}"
            ) from exc
        if max_steps < 1:
            raise ValueError(
                f"training.max_steps must be a positive integer, got {max_steps}"
            )

    training_budget_seconds = (max_hours - setup_hours) * 3600
    steps_at_worst_rate = int(training_budget_seconds // ASSUMED_WORST_STEP_SECONDS)
    if max_steps is not None:
        estimated_training_seconds = max_steps * ASSUMED_WORST_STEP_SECONDS
        estimated_total_hours = setup_hours + estimated_training_seconds / 3600
        if estimated_total_hours > max_hours:
            raise ValueError(
                f"training.max_steps={max_steps} at the worst-case "
                f"{ASSUMED_WORST_STEP_SECONDS:.0f} s/step needs "
                f"{estimated_total_hours:.2f}h (setup {setup_hours}h + "
                f"training {estimated_training_seconds / 3600:.2f}h), "
                f"over the max_runtime_hours budget of {max_hours}h. "
                f"Lower max_steps to at most {steps_at_worst_rate} or "
                "raise max_runtime_hours (max 11h)."
            )
    else:
        estimated_training_seconds = None
        estimated_total_hours = None

    return {
        "max_runtime_hours": max_hours,
        "setup_allowance_hours": setup_hours,
        "max_steps": max_steps,
        "kaggle_hard_limit_hours": KAGGLE_HARD_LIMIT_HOURS,
        "effective_cap_hours": EFFECTIVE_CAP_HOURS,
        "assumed_worst_step_seconds": ASSUMED_WORST_STEP_SECONDS,
        "training_budget_seconds": training_budget_seconds,
        "steps_at_worst_rate": steps_at_worst_rate,
        "estimated_training_seconds": estimated_training_seconds,
        "estimated_total_hours": estimated_total_hours,
        "margin_hours_under_cap": (
            EFFECTIVE_CAP_HOURS - estimated_total_hours
            if estimated_total_hours is not None
            else None
        ),
    }


class RuntimeGuard:
    """Wall-clock guard: stop training gracefully before Kaggle kills it.

    Constructed with the hour budget and a monotonic start time; the
    training callback asks :meth:`exceeded` after each step. Pure
    standard library, so it is unit-testable without torch.
    """

    def __init__(
        self, max_runtime_hours: float, start: float | None = None
    ) -> None:
        if not 0 < max_runtime_hours <= EFFECTIVE_CAP_HOURS:
            raise ValueError(
                f"max_runtime_hours must be in (0, {EFFECTIVE_CAP_HOURS}]; "
                f"got {max_runtime_hours}"
            )
        self.max_runtime_hours = float(max_runtime_hours)
        self.start = time.monotonic() if start is None else float(start)
        self.deadline = self.start + self.max_runtime_hours * 3600

    def elapsed_seconds(self, now: float | None = None) -> float:
        """Seconds elapsed since the guard started."""
        return (time.monotonic() if now is None else now) - self.start

    def remaining_seconds(self, now: float | None = None) -> float:
        """Seconds left before the deadline (negative once past it)."""
        return self.deadline - (time.monotonic() if now is None else now)

    def exceeded(self, now: float | None = None) -> bool:
        """True once the wall-clock budget is used up."""
        return (time.monotonic() if now is None else now) >= self.deadline

    def as_dict(self) -> dict:
        """Summary of the guard settings for run records."""
        return {
            "max_runtime_hours": self.max_runtime_hours,
            "effective_cap_hours": EFFECTIVE_CAP_HOURS,
            "kaggle_hard_limit_hours": KAGGLE_HARD_LIMIT_HOURS,
        }


def steps_for_budget(max_runtime_hours: float, setup_allowance_hours: float) -> int:
    """Largest step cap that fits the hour budget at the worst rate."""
    budget_seconds = (max_runtime_hours - setup_allowance_hours) * 3600
    return max(0, math.floor(budget_seconds / ASSUMED_WORST_STEP_SECONDS))


def runtime_guard_callback(guard: RuntimeGuard, logger=None):
    """Build the wall-clock stop callback for a TRL trainer.

    Checked after every optimizer step: once the guard's deadline is
    reached, the run is stopped gracefully (``should_training_stop``)
    so the trainer saves a checkpoint instead of Kaggle killing the
    notebook at 12h. The returned callback instance carries a
    ``triggered`` attribute, True once it has stopped a run -- the
    stage runner uses it to write the partial-complete record.

    The ``transformers`` import is lazy so this module stays
    importable without the training stack; callers pass the callback
    instance to the trainer's ``callbacks=`` argument. All three
    stages (SFT, DPO, GRPO) share this one implementation so the stop
    logic cannot drift between them.
    """
    from transformers import TrainerCallback

    class _RuntimeGuardCallback(TrainerCallback):
        def __init__(self) -> None:
            self.triggered = False

        def on_step_end(self, args, state, control, **kwargs):
            if guard.exceeded():
                self.triggered = True
                if logger is not None:
                    logger.warning(
                        "runtime guard: %.2fh wall clock reached at step %d; "
                        "stopping gracefully with a checkpoint",
                        guard.elapsed_seconds() / 3600,
                        state.global_step,
                    )
                control.should_training_stop = True

    return _RuntimeGuardCallback()
