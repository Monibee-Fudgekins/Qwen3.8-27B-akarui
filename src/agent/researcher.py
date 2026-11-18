"""Researcher component for the Akarui competition-solving agent.

The Researcher reads a competition description, identifies the task
type, inspects the data schema, develops hypotheses, selects a
validation strategy, and plans experiments. It does not run code
itself — that is the Executor's job — and it never fabricates
results: every plan it returns is a plan, not a measurement. Model
output comes through a pluggable ``generate`` callable supplied by
the caller, so this module never hardcodes a model call or a
provider.
"""

from __future__ import annotations

from typing import Callable

# Task types the planner recognises from a competition description.
# Anything else is reported as "unknown" rather than guessed.
_TASK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tabular_classification": ("classification", "classify", "predict the class", "label"),
    "tabular_regression": ("regression", "predict the value", "predict the price", "predict the score"),
    "computer_vision": ("image", "images", "vision", "object detection", "segmentation"),
    "nlp": ("text", "nlp", "language", "sentiment", "summarisation", "summarization"),
    "time_series": ("time series", "forecast", "forecasting", "temporal"),
}


def _infer_task_type(task_description: str) -> str:
    lowered = task_description.lower()
    for task_type, keywords in _TASK_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return task_type
    return "unknown"


class Researcher:
    """Plans competition-solving work for one competition.

    Parameters
    ----------
    generate:
        Optional callable ``generate(prompt) -> str`` that produces
        model text. When supplied, the Researcher may ask the model
        for hypotheses or experiment ideas and records them as
        model-suggested (never as measured). When omitted, plans are
        built purely from the structured inputs.
    """

    def __init__(self, generate: Callable[[str], str] | None = None) -> None:
        self._generate = generate

    def plan(
        self,
        task_description: str,
        schema: dict,
        metric: str,
    ) -> dict:
        """Return a structured plan for the competition.

        The plan contains the inferred task type, the evaluation
        metric, hypotheses, a validation strategy, and an ordered
        experiment list. Scores and outcomes are absent by design:
        a plan predicts nothing about performance.
        """
        task_type = _infer_task_type(task_description)
        target = schema.get("target", "not specified")
        features = schema.get("features", [])

        validation = self._validation_strategy(task_type, schema)
        hypotheses = self._hypotheses(task_type, metric, features)
        experiments = self._experiment_list(task_type, validation, metric)

        model_notes = None
        if self._generate is not None:
            prompt = (
                "Competition task: " + task_description + "\n"
                f"Task type: {task_type}\nMetric: {metric}\n"
                "Suggest up to three experiment hypotheses, one per line."
            )
            model_notes = self._generate(prompt)

        return {
            "task_type": task_type,
            "task_description": task_description,
            "target": target,
            "features": features,
            "metric": metric,
            "hypotheses": hypotheses,
            "validation_strategy": validation,
            "experiments": experiments,
            "model_suggested_notes": model_notes,
            "status": "planned",
        }

    def _validation_strategy(self, task_type: str, schema: dict) -> dict:
        """Choose a validation strategy appropriate to the task type.

        Time-series data gets a time-based split; everything else
        defaults to stratified K-fold for classification-like targets
        and plain K-fold otherwise. The strategy is a recommendation
        recorded in the plan, not a claim that it was run.
        """
        if task_type == "time_series" or schema.get("has_time_column"):
            return {
                "method": "time_based_split",
                "n_splits": 5,
                "note": "Folds respect chronological order; no future data in training folds.",
            }
        if task_type == "tabular_classification":
            return {
                "method": "stratified_kfold",
                "n_splits": 5,
                "note": "Stratification preserves the target distribution in each fold.",
            }
        return {
            "method": "kfold",
            "n_splits": 5,
            "note": "Standard K-fold cross-validation.",
        }

    def _hypotheses(self, task_type: str, metric: str, features: list) -> list[str]:
        hypotheses = [
            f"Gradient-boosted trees are a strong first baseline for this {task_type} task.",
            f"A simple, well-validated baseline should be established before any ensembling, scored by {metric}.",
        ]
        if len(features) > 20:
            hypotheses.append(
                "Feature selection or target encoding may help given the number of features."
            )
        if task_type == "time_series":
            hypotheses.append(
                "Lag and rolling-window features are likely to carry most of the signal."
            )
        return hypotheses

    def _experiment_list(self, task_type: str, validation: dict, metric: str) -> list[dict]:
        return [
            {
                "experiment_id": "baseline_lgbm",
                "description": "LightGBM baseline with default-ish hyperparameters.",
                "validation": validation["method"],
                "metric": metric,
                "status": "planned",
            },
            {
                "experiment_id": "baseline_catboost",
                "description": "CatBoost baseline for comparison, especially if categorical features dominate.",
                "validation": validation["method"],
                "metric": metric,
                "status": "planned",
            },
            {
                "experiment_id": "feature_engineering_v1",
                "description": "First feature-engineering pass on top of the best baseline.",
                "validation": validation["method"],
                "metric": metric,
                "status": "planned",
            },
            {
                "experiment_id": "tuned_best",
                "description": "Hyperparameter tuning (Optuna) of the best baseline.",
                "validation": validation["method"],
                "metric": metric,
                "status": "planned",
            },
        ]
