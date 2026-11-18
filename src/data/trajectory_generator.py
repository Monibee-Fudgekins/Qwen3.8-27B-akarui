"""Generate schema-valid trajectory records from competition-solving runs.

A trajectory record captures one solving attempt: the task, the model's
reasoning, the code it wrote, what happened when the code ran, and the
measured validation score.

Security note: this module NEVER executes code. It only structures
results that were produced elsewhere. Execution must happen in an
isolated sandbox with no access to Kaggle credentials, GitHub secrets,
or unrelated filesystem contents; the caller supplies the execution
results (output, errors, validation score) as plain data. Any
``execution_environment`` recorded here must describe that sandbox.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .dataset_validation import validate_record


def make_execution_environment(
    *,
    sandbox: bool = True,
    python_version: str | None = None,
    packages: list[str] | None = None,
    cpu_limit: str | None = None,
    memory_limit: str | None = None,
    time_limit_seconds: int | None = None,
    network: str = "none",
) -> dict:
    """Describe the isolated environment a run executed in.

    The default is a sandbox with no network access. Callers must pass
    the real values; do not record an environment that was not used.
    """
    environment: dict = {
        "sandbox": sandbox,
        "network": network,
    }
    if python_version is not None:
        environment["python_version"] = python_version
    if packages is not None:
        environment["packages"] = list(packages)
    if cpu_limit is not None:
        environment["cpu_limit"] = cpu_limit
    if memory_limit is not None:
        environment["memory_limit"] = memory_limit
    if time_limit_seconds is not None:
        environment["time_limit_seconds"] = time_limit_seconds
    return environment


def generate_trajectory(
    *,
    competition_id: str,
    source_url: str,
    dataset_provenance: str,
    license: str,
    task_description: str,
    input_schema,
    target_definition: str,
    evaluation_metric: str,
    training_validation_strategy: str,
    reasoning_summary: str,
    generated_code: str,
    execution_environment,
    execution_output: str,
    errors_and_debugging: list | None = None,
    validation_score: float | None = None,
    final_outcome: str,
    category: str,
    dataset_version: str,
    collected_at: str | None = None,
) -> dict:
    """Build one trajectory record from a completed solving run.

    All arguments map directly onto the trajectory schema fields. The
    record is validated before it is returned; a :class:`ValueError` is
    raised listing every validation error, so an invalid trajectory can
    never be silently written into a dataset.

    ``execution_output`` and ``validation_score`` must come from a real,
    measured execution run -- never from an LLM's guess at what the code
    would do. Pass ``validation_score=None`` when no score was measured.
    """
    record = {
        "competition_id": competition_id,
        "source_url": source_url,
        "dataset_provenance": dataset_provenance,
        "license": license,
        "task_description": task_description,
        "input_schema": input_schema,
        "target_definition": target_definition,
        "evaluation_metric": evaluation_metric,
        "training_validation_strategy": training_validation_strategy,
        "reasoning_summary": reasoning_summary,
        "generated_code": generated_code,
        "execution_environment": execution_environment,
        "execution_output": execution_output,
        "errors_and_debugging": list(errors_and_debugging or []),
        "validation_score": validation_score,
        "final_outcome": final_outcome,
        "category": category,
        "dataset_version": dataset_version,
        "collected_at": collected_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    errors = validate_record(record)
    if errors:
        raise ValueError(
            "trajectory record failed validation: " + "; ".join(errors)
        )
    return record


def trajectory_from_run(run: dict, *, dataset_version: str) -> dict:
    """Build a trajectory record from a raw run dict.

    ``run`` uses the same field names as :func:`generate_trajectory`
    (plus an optional ``collected_at``). ``dataset_version`` is supplied
    by the caller so the dataset version is always set explicitly.
    """
    return generate_trajectory(
        competition_id=run["competition_id"],
        source_url=run["source_url"],
        dataset_provenance=run["dataset_provenance"],
        license=run["license"],
        task_description=run["task_description"],
        input_schema=run["input_schema"],
        target_definition=run["target_definition"],
        evaluation_metric=run["evaluation_metric"],
        training_validation_strategy=run["training_validation_strategy"],
        reasoning_summary=run["reasoning_summary"],
        generated_code=run["generated_code"],
        execution_environment=run["execution_environment"],
        execution_output=run["execution_output"],
        errors_and_debugging=run.get("errors_and_debugging") or [],
        validation_score=run.get("validation_score"),
        final_outcome=run["final_outcome"],
        category=run["category"],
        dataset_version=dataset_version,
        collected_at=run.get("collected_at"),
    )
