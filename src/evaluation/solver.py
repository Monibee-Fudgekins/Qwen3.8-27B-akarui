"""Competition solver for the akarui held-out benchmark.

This module turns the benchmark from a ``not_run`` template into
measured scores. It is eval-time only: it never trains on the
held-out competitions and never writes to training data.

Pipeline for one competition
----------------------------
1. **Data.** The competition's train file is downloaded with the
   Kaggle CLI (auth via ``scripts/kaggle_auth.py``; ``KAGGLE_TOKEN``
   lives in GitHub secrets only and is never printed). The download
   runs in the parent process; generated code never sees it.
2. **Local split.** Kaggle test labels are hidden, so a local score
   cannot come from the leaderboard -- and the eval kernel must NOT
   submit to the leaderboard (rate limits, account side effects).
   Instead the competition's *train* file is split into a fixed
   local train/val split (80/20, seed 42): the model-generated
   solution trains/fits on the local train part and predicts the
   local val part, whose labels are known locally. Time-series
   competitions use a chronological split (last 20% is val); the
   rest use a deterministic hash-based split, stratified by target
   for classification. The split is a property of (seed, row) only,
   so it is reproducible without storing rows.
3. **Generation.** The loaded model (pinned base, or base + LoRA
   adapter) is prompted once per competition with a fixed prompt
   (task description, file listing, column schema, target/id
   columns, metric, expected output format) and generates one
   ``solution.py`` with greedy decoding (``do_sample=False``),
   fixed seed 42, ``max_new_tokens=4096``. The prompt template is
   :func:`build_prompt`; decoding choices are fixed module
   constants so base and candidate runs are comparable.
4. **Execution.** The generated code runs ONLY through
   :class:`src.agent.executor.Executor` (wall-clock timeout,
   address-space limit, restricted working directory, environment
   scrubbed of ``KAGGLE_*`` / ``HF_TOKEN`` / ``AWS_*``). A timeout
   or crash records the competition as unmeasured (``status:
   "error"``, null scores) -- never a fake score.
5. **Scoring.** The solution writes ``predictions.csv`` with
   ``id,prediction`` for the val rows. Predictions are validated
   (row count and id set must match val exactly; all values finite)
   and scored with the competition's metric, implemented here in
   pure stdlib (no sklearn dependency at import time, so the module
   is testable without the GPU stack).
6. **Normalization / baselines.** The manifest carries no baseline
   scores and none are invented. Instead the solver computes, on
   the SAME local val split, a fixed deterministic reference
   solution per competition (see :data:`COMPETITION_SPECS` --
   majority/median predictors, a Sex rule, a CryoSleep rule, a
   disaster-keyword rule, a constant 0.5 AUC score). The reference
   never uses the model under test. ``normalized_score`` follows
   :func:`src.evaluation.competition_evaluator._normalized_score`
   semantics: higher-is-better ``score / baseline``,
   lower-is-better ``baseline / score``.

Sandbox limitation (honest note)
--------------------------------
:class:`src.agent.executor.Executor` scrubs credentials from the
child environment and confines the child's working directory, but
it does NOT block network access: in the eval kernel the internet
is on (the base weights are downloaded from Hugging Face), so
generated code could in principle reach the network. The prompt
forbids network/Kaggle-API use, the credentials are scrubbed so
generated code cannot authenticate to Kaggle or Hugging Face, and
generated code is never given the parent's environment -- but
network isolation is a documented limitation, not a guarantee.
Do not rely on the Executor alone for network containment.

Per-competition scoring table
-----------------------------
See :data:`COMPETITION_SPECS`. For each competition: metric
implementation, direction, target/id columns, split kind, and
reference predictor. Two target columns are NOT known from the
repo and are auto-detected at runtime as the last column of
``train.csv`` (recorded in the result as
``target_detected: True``): ``playground-series-s3e11`` and
``tabular-playground-series-jan-2024``. If the auto-detected
column is wrong the first live run will show it (the reference
and the generated solution both use the recorded column, so the
run stays honest but the score is meaningless for that
competition) -- verify the recorded ``target_column`` in the
first live results before citing them.

Competitions that cannot be scored as designed: none
categorically. Three are at elevated risk on the first live run
(see evaluation/evaluation_protocol.md): ``store-sales-...``
(3M+ train rows; the generated solution may time out fitting on
the full local train part), ``digit-recognizer`` (42k x 785
pixels; a slow model fit may hit the execution timeout), and
``petfinder-pawpularity-score`` (image files in the download;
the reference and the prompt use the tabular metadata only, so
image signal is unscored). A timeout or download failure is
recorded unmeasured, never fabricated, and the gate fails closed.

Leakage guard
------------
:func:`build_solver` (and :func:`run_evaluation` in
``benchmark_runner``) refuse to run when any held-out
competition id appears in the training inputs
(``data/train.jsonl``), using
:func:`src.evaluation.leakage_detection.detect_leakage`. The
solver never writes to training data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import zipfile
from pathlib import Path

from src.agent.executor import Executor
from src.evaluation.competition_evaluator import _normalized_score
from src.evaluation.leakage_detection import detect_leakage

# --- fixed determinism / budget constants ---------------------------------

#: Fixed seed for the local split, generation, and any seeded code
#: the solver itself runs. Generated solution code is instructed
#: to use the same seed.
SEED = 42
#: Fraction of the competition train file held out as local val.
VAL_FRACTION = 0.2
#: Max new tokens for one solution generation (greedy decoding).
MAX_NEW_TOKENS = 4096
#: Sandboxed execution budget per competition (wall clock + memory).
EXECUTION_TIMEOUT_S = 600
EXECUTION_MEMORY_MB = 4096
#: Rows sampled to infer column types for the prompt schema.
SCHEMA_SAMPLE_ROWS = 2000
#: Max characters of sample rows embedded in the prompt.
PROMPT_SAMPLE_CHARS = 2000

#: Metric string (as written in the manifest) -> canonical metric key.
METRIC_ALIASES = {
    "accuracy": "accuracy",
    "rmsle": "rmsle",
    "rmsle (log rmse)": "rmsle",
    "rmse": "rmse",
    "f1": "f1",
    "roc auc": "roc_auc",
    "roc_auc": "roc_auc",
    "auc": "roc_auc",
}

#: Canonical metric -> direction.
METRIC_DIRECTIONS = {
    "accuracy": "higher_is_better",
    "f1": "higher_is_better",
    "roc_auc": "higher_is_better",
    "rmsle": "lower_is_better",
    "rmse": "lower_is_better",
}

#: Words whose presence predicts the positive class for the
#: nlp-getting-started keyword reference. Fixed, deterministic,
#: deliberately small -- it is a floor, not a model.
DISASTER_KEYWORDS = (
    "fire", "flood", "earthquake", "explosion", "crash", "hurricane",
    "tornado", "bomb", "killed", "dead", "disaster", "emergency",
    "evacuat", "wildfire", "tsunami", "collapse",
)


# --- per-competition table -------------------------------------------------

#: Per-competition spec. ``target`` / ``id_column`` are the Kaggle
#: train-file columns; ``None`` means "auto-detect at runtime"
#: (target = last column of train.csv; recorded in the result).
#: ``reference`` names the deterministic reference predictor
#: (see _reference_predict); ``time_column`` forces a
#: chronological split for time series.
COMPETITION_SPECS = {
    "titanic": {
        "metric": "accuracy", "target": "Survived",
        "id_column": "PassengerId", "time_column": None,
        "split": "stratified", "reference": "sex_rule",
        "reference_note": "Female -> 1, male -> 0 (Sex column); "
        "falls back to majority class when Sex is absent.",
    },
    "house-prices-advanced-regression-techniques": {
        "metric": "rmsle", "target": "SalePrice", "id_column": "Id",
        "time_column": None, "split": "random",
        "reference": "median",
        "reference_note": "Median SalePrice of the local train part.",
    },
    "digit-recognizer": {
        "metric": "accuracy", "target": "label",
        "id_column": None, "time_column": None,
        "split": "stratified", "reference": "majority",
        "reference_note": "Majority digit class (train.csv has no id "
        "column; val rows are identified by row position).",
    },
    "spaceship-titanic": {
        "metric": "accuracy", "target": "Transported",
        "id_column": "PassengerId", "time_column": None,
        "split": "stratified", "reference": "cryosleep_rule",
        "reference_note": "CryoSleep true -> 1, else 0; falls back to "
        "majority class when CryoSleep is absent.",
    },
    "playground-series-s3e11": {
        "metric": "rmsle", "target": None, "id_column": "id",
        "time_column": None, "split": "random",
        "reference": "median",
        "reference_note": "Median target of the local train part; "
        "target column auto-detected as the last column of train.csv.",
    },
    "store-sales-time-series-forecasting": {
        "metric": "rmsle", "target": "sales", "id_column": "id",
        "time_column": "date", "split": "chronological",
        "reference": "median",
        "reference_note": "Median sales of the local train part. "
        "Chronological split: last 20% of dates is val.",
    },
    "nlp-getting-started": {
        "metric": "f1", "target": "target", "id_column": "id",
        "time_column": None, "split": "stratified",
        "reference": "keyword_rule",
        "reference_note": "Predict 1 when the tweet text contains any "
        "of a fixed disaster-word list, else 0.",
    },
    "petfinder-pawpularity-score": {
        "metric": "rmse", "target": "Pawpularity",
        "id_column": "Id", "time_column": None, "split": "random",
        "reference": "median",
        "reference_note": "Median Pawpularity of the local train part. "
        "Tabular metadata only; image files are present in the data "
        "dir but the reference ignores them.",
    },
    "tabular-playground-series-jan-2024": {
        "metric": "roc_auc", "target": None, "id_column": "id",
        "time_column": None, "split": "stratified",
        "reference": "constant_half",
        "reference_note": "Constant predicted probability 0.5, i.e. "
        "ROC AUC 0.5 by construction. Target auto-detected as the "
        "last column of train.csv; predictions must be probabilities.",
    },
    "forest-cover-type-prediction": {
        "metric": "accuracy", "target": "Cover_Type",
        "id_column": "Id", "time_column": None, "split": "stratified",
        "reference": "majority",
        "reference_note": "Majority cover-type class.",
    },
    "bike-sharing-demand": {
        "metric": "rmsle", "target": "count",
        "id_column": "datetime", "time_column": "datetime",
        "split": "chronological", "reference": "median",
        "reference_note": "Median count of the local train part. "
        "Chronological split on datetime; the id is the datetime.",
    },
    "commonlitreadabilityprize": {
        "metric": "rmse", "target": "target", "id_column": "id",
        "time_column": None, "split": "random",
        "reference": "median",
        "reference_note": "Median target of the local train part.",
    },
}


class DataUnavailableError(RuntimeError):
    """A competition's data could not be downloaded or found."""


class LeakageError(ValueError):
    """A held-out competition id appears in the training inputs."""


# --- metric implementations (pure stdlib) ---------------------------------


def _as_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def compute_metric(metric: str, y_true: list[float], y_pred: list[float]) -> float:
    """Score predictions with a canonical metric.

    ``y_true`` / ``y_pred`` are parallel lists. Accuracy and F1
    threshold predictions at 0.5 (predictions are class labels or
    probabilities); RMSLE clips both sides at 0 before log1p.
    """
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("metric needs non-empty, equal-length inputs")
    if metric == "accuracy":
        correct = sum(
            1 for y, p in zip(y_true, y_pred) if int(round(p)) == int(round(y))
        )
        return correct / len(y_true)
    if metric == "rmse":
        return math.sqrt(
            sum((y - p) ** 2 for y, p in zip(y_true, y_pred)) / len(y_true)
        )
    if metric == "rmsle":
        return math.sqrt(
            sum(
                (math.log1p(max(y, 0.0)) - math.log1p(max(p, 0.0))) ** 2
                for y, p in zip(y_true, y_pred)
            )
            / len(y_true)
        )
    if metric == "f1":
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p >= 0.5)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != 1 and p >= 0.5)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p < 0.5)
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)
    if metric == "roc_auc":
        # Mann-Whitney rank AUC with average ranks for ties.
        order = sorted(range(len(y_true)), key=lambda i: y_pred[i])
        ranks = [0.0] * len(y_true)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and y_pred[order[j + 1]] == y_pred[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1  # 1-based average rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        n_pos = sum(1 for y in y_true if y == 1)
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            raise ValueError("ROC AUC needs both classes in val")
        return (
            sum(r for y, r in zip(y_true, ranks) if y == 1)
            - n_pos * (n_pos + 1) / 2
        ) / (n_pos * n_neg)
    raise ValueError(f"unknown metric: {metric!r}")


def canonical_metric(metric: str) -> str:
    """Normalize a manifest metric string to a canonical metric key."""
    key = METRIC_ALIASES.get(str(metric).strip().lower())
    if key is None:
        raise ValueError(f"unknown metric: {metric!r}")
    return key


# --- code extraction --------------------------------------------------------


_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)\n?```", re.DOTALL | re.IGNORECASE
)


def extract_code(generated_text: str) -> str:
    """Extract Python code from generated text.

    Prefers the first fenced `````python`` block; falls back to the
    whole text when there is no fence. Raises ValueError on empty
    output -- an empty generation is never executed as a no-op.
    """
    if not generated_text or not generated_text.strip():
        raise ValueError("model generated no text")
    match = _CODE_FENCE_RE.search(generated_text)
    code = match.group(1) if match else generated_text
    if not code.strip():
        raise ValueError("model generated no code")
    return code.strip() + "\n"


# --- prompt -----------------------------------------------------------------


def build_prompt(
    competition_id: str,
    spec: dict,
    *,
    target_column: str,
    id_column: str | None,
    columns: list[str],
    column_types: dict[str, str],
    n_train_rows: int,
    n_val_rows: int,
    sample_rows: str,
    files: list[str],
) -> str:
    """Build the fixed deterministic prompt for one competition.

    The prompt states the task, the files in the working directory,
    the schema, the target/id columns, the metric, and the exact
    output contract (read ``train_local.csv`` / ``val.csv`` from
    the working directory, write ``predictions.csv`` with columns
    ``id,prediction``). It forbids network and Kaggle-API use and
    fixes seed 42. Nothing in the prompt depends on the model
    under test, so base and candidate runs get identical prompts
    for the same competition.
    """
    prediction_kind = (
        "the predicted probability of class 1 (a number between 0 and 1)"
        if spec["metric"] == "roc_auc"
        else "the predicted value of the target column"
    )
    id_line = (
        f"ID column: {id_column}. The predictions file must use these IDs."
        if id_column
        else "There is no ID column: val rows are identified by their "
        "0-based row position in val.csv (0, 1, 2, ...); use that as id."
    )
    schema_lines = "\n".join(
        f"- {name}: {column_types.get(name, 'string')}" for name in columns
    )
    return (
        f"You are solving the Kaggle competition '{competition_id}' "
        f"({spec.get('task_type', 'tabular')} task, metric: "
        f"{spec['metric']}).\n\n"
        f"Files in your working directory:\n"
        + "".join(f"- {name}\n" for name in files)
        + f"\n- train_local.csv: {n_train_rows} training rows "
        f"(the local train split; fit only on this file).\n"
        f"- val.csv: {n_val_rows} validation rows WITH the target "
        f"column present. Predict the target for these rows.\n\n"
        f"Schema of train_local.csv / val.csv:\n{schema_lines}\n\n"
        f"Target column: {target_column}. {id_line}\n\n"
        f"Sample rows (truncated):\n{sample_rows}\n\n"
        f"Write a complete, runnable Python solution. Rules:\n"
        f"- Read train_local.csv and val.csv from the working "
        f"directory (relative paths only).\n"
        f"- Write predictions.csv in the working directory with "
        f"exactly two columns: id,prediction, one row per val row, "
        f"in val.csv order. prediction is {prediction_kind}.\n"
        f"- Use only the Python standard library, pandas, numpy, "
        f"and scikit-learn. Set random_state=42 / seed 42 everywhere.\n"
        f"- Do NOT use the internet, the Kaggle API, or any "
        f"credentials. Do NOT read any file outside the working "
        f"directory. Fit only on train_local.csv, never on val.csv "
        f"labels beyond reading them to write predictions.\n"
        f"- The code must run with `python solution.py` and finish "
        f"quickly on the data given.\n\n"
        f"Reply with only the Python code in a single ```python "
        f"fenced block, nothing else."
    )


# --- data access ------------------------------------------------------------


def default_downloader(competition_id: str, dest_dir: Path) -> Path:
    """Download a competition with the Kaggle CLI into *dest_dir*.

    Configures auth via ``scripts/kaggle_auth.py`` (KAGGLE_TOKEN
    from the environment; never printed). Raises
    :class:`DataUnavailableError` on any failure -- including a
    rules-acceptance failure, which can only be fixed by accepting
    the competition rules on kaggle.com, not from the kernel.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
        from kaggle_auth import configure_kaggle_auth  # noqa: PLC0415

        configure_kaggle_auth()
    except Exception as exc:  # noqa: BLE001 - auth failure is a data failure
        raise DataUnavailableError(
            f"could not configure Kaggle auth for {competition_id}: {exc}"
        ) from exc
    try:
        completed = subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition_id,
             "-p", str(dest_dir)],
            capture_output=True, text=True, timeout=1800, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DataUnavailableError(
            f"kaggle download failed for {competition_id}: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise DataUnavailableError(
            f"kaggle download failed for {competition_id} "
            f"(exit {completed.returncode}): {detail[:500]}"
        )
    for archive in dest_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest_dir)
        except zipfile.BadZipFile as exc:
            raise DataUnavailableError(
                f"downloaded archive for {competition_id} is not a zip: {exc}"
            ) from exc
    return dest_dir


def find_train_file(data_dir: Path) -> Path:
    """Locate the competition train file inside a downloaded data dir."""
    data_dir = Path(data_dir)
    direct = data_dir / "train.csv"
    if direct.is_file():
        return direct
    candidates = sorted(data_dir.rglob("train*.csv"))
    if candidates:
        return candidates[0]
    csvs = sorted(data_dir.rglob("*.csv"))
    if csvs:
        return csvs[0]
    raise DataUnavailableError(f"no CSV file found in {data_dir}")


def _is_val_row(seed: int, strat_key: str, row_index: int, val_fraction: float) -> bool:
    """Deterministic hash-based val assignment (seed, key, index)."""
    digest = hashlib.sha256(
        f"{seed}:{strat_key}:{row_index}".encode("utf-8")
    ).hexdigest()
    return (int(digest[:8], 16) % 10000) < int(val_fraction * 10000)


def _infer_type(values: list[str]) -> str:
    """Infer a column type from sample string values."""
    non_empty = [v for v in values if v not in ("", None)]
    if not non_empty:
        return "string"
    try:
        for v in non_empty:
            int(v)
        return "int"
    except ValueError:
        pass
    try:
        for v in non_empty:
            float(v)
        return "float"
    except ValueError:
        pass
    return "string"


# --- reference predictors ---------------------------------------------------


def _reference_predict(
    reference: str,
    *,
    train_targets: list[float],
    val_rows: list[list[str]],
    header: list[str],
    target_idx: int,
) -> list[float]:
    """Deterministic reference predictions for the val rows.

    ``train_targets`` are the local-train targets (for
    median/majority fitting); ``val_rows`` are the val rows as
    value lists aligned with ``header``. Never uses the model.
    """
    if reference == "median":
        if not train_targets:
            raise ValueError("median reference needs train targets")
        value = statistics.median(train_targets)
        return [value] * len(val_rows)
    if reference == "majority":
        if not train_targets:
            raise ValueError("majority reference needs train targets")
        counts: dict[float, int] = {}
        for target in train_targets:
            counts[target] = counts.get(target, 0) + 1
        value = max(sorted(counts), key=lambda k: counts[k])
        return [value] * len(val_rows)
    if reference == "constant_half":
        return [0.5] * len(val_rows)
    if reference == "sex_rule":
        idx = header.index("Sex") if "Sex" in header else None
        if idx is None:
            return _reference_predict(
                "majority", train_targets=train_targets,
                val_rows=val_rows, header=header, target_idx=target_idx,
            )
        return [
            1.0 if str(row[idx]).strip().lower().startswith("f") else 0.0
            for row in val_rows
        ]
    if reference == "cryosleep_rule":
        idx = header.index("CryoSleep") if "CryoSleep" in header else None
        if idx is None:
            return _reference_predict(
                "majority", train_targets=train_targets,
                val_rows=val_rows, header=header, target_idx=target_idx,
            )
        return [
            1.0 if str(row[idx]).strip().lower() in ("true", "1", "t") else 0.0
            for row in val_rows
        ]
    if reference == "keyword_rule":
        text_idx = None
        for name in ("text", "tweet", "excerpt"):
            if name in header:
                text_idx = header.index(name)
                break
        if text_idx is None:
            # Fall back to the longest string column of the first row.
            text_idx = 0
        predictions = []
        for row in val_rows:
            text = str(row[text_idx]).lower() if text_idx < len(row) else ""
            predictions.append(
                1.0 if any(word in text for word in DISASTER_KEYWORDS) else 0.0
            )
        return predictions
    raise ValueError(f"unknown reference: {reference!r}")


# --- generation interface ---------------------------------------------------


class ModelGenerator:
    """Greedy text generation wrapping a loaded model + tokenizer.

    Deterministic: ``do_sample=False``, fixed seed, fixed
    ``max_new_tokens``. Heavy imports (torch) happen only when
    generating, so importing this module never needs the GPU stack.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
        seed: int = SEED,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.seed = seed

    def __call__(self, prompt: str) -> str:
        import torch  # noqa: PLC0415 - lazy: GPU stack only in-kernel

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def load_model_for_eval(model_ref: str, config: dict):
    """Load the pinned base model, plus a LoRA adapter when asked.

    ``model_ref`` is ``"base"`` or an adapter directory. Adapters
    are applied with PEFT on top of the pinned base (adapters
    first; a merged model is used only when one is supplied as a
    full model directory -- this loader does not merge).
    Returns ``(model, tokenizer)``.
    """
    from src.training.model_loader import (  # noqa: PLC0415 - lazy
        load_base_model,
        load_tokenizer,
    )

    model = load_base_model(config)
    if model_ref != "base":
        from peft import PeftModel  # noqa: PLC0415 - lazy

        model = PeftModel.from_pretrained(model, str(model_ref))
    tokenizer = load_tokenizer(config)
    return model, tokenizer


# --- leakage guard ------------------------------------------------------------


def check_training_leakage(
    train_file: str | Path | None,
    held_out_ids: list[str],
) -> dict:
    """Refuse when any held-out id appears in the training inputs.

    Returns the leakage report. Raises :class:`LeakageError` on
    overlap. A missing training file means there is nothing to
    check in-kernel (packaging copies it in); it is not treated
    as leakage, but the report records ``train_file_missing``.
    """
    if train_file is None or not Path(train_file).is_file():
        return {
            "train_count": 0,
            "eval_count": len(held_out_ids),
            "overlap_count": 0,
            "overlap_ids": [],
            "clean": True,
            "train_file_missing": True,
        }
    train_ids = []
    with Path(train_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("competition_id"):
                train_ids.append(str(record["competition_id"]))
    report = detect_leakage(train_ids, held_out_ids)
    if not report["clean"]:
        raise LeakageError(
            "held-out competition(s) in training inputs: "
            + ", ".join(report["overlap_ids"])
            + "; refusing to evaluate a contaminated benchmark"
        )
    return report


# --- the solver -----------------------------------------------------------------


class CompetitionSolver:
    """Solve one held-out competition per call; see module docstring.

    ``generate`` is a callable ``generate(prompt) -> str`` (a
    :class:`ModelGenerator` in-kernel, a mock in tests).
    ``data_root`` holds one downloaded competition per
    sub-directory plus a ``work`` area per competition.
    """

    def __init__(
        self,
        generate,
        *,
        data_root: str | Path,
        seed: int = SEED,
        val_fraction: float = VAL_FRACTION,
        max_new_tokens: int = MAX_NEW_TOKENS,
        execution_timeout_s: int = EXECUTION_TIMEOUT_S,
        execution_memory_mb: int = EXECUTION_MEMORY_MB,
        downloader=None,
        executor_factory=None,
    ) -> None:
        self.generate = generate
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.val_fraction = val_fraction
        self.max_new_tokens = max_new_tokens
        self.execution_timeout_s = execution_timeout_s
        self.execution_memory_mb = execution_memory_mb
        self.downloader = downloader or default_downloader
        self.executor_factory = executor_factory or (
            lambda sandbox_dir: Executor(
                sandbox_dir,
                default_timeout_s=execution_timeout_s,
                default_memory_mb=execution_memory_mb,
            )
        )

    # -- data preparation -------------------------------------------------

    def _prepare_split(self, competition_id: str, spec: dict) -> dict:
        """Download (if needed) and split the train file.

        Returns a dict with the work dir, header, target/id
        columns, row counts, val rows (lists aligned with the
        header), and schema info. The split files are written
        once per competition and reused on repeat calls.
        """
        comp_dir = self.data_root / competition_id
        train_file = comp_dir / "train.csv"
        if not train_file.is_file():
            self.downloader(competition_id, comp_dir)
            train_file = find_train_file(comp_dir)
            if train_file.name != "train.csv":
                # Normalize the layout the rest of the solver expects.
                train_file.rename(comp_dir / "train.csv")
                train_file = comp_dir / "train.csv"
        work_dir = comp_dir / "work"
        train_local = work_dir / "train_local.csv"
        val_file = work_dir / "val.csv"
        meta_file = work_dir / "split_meta.json"
        if train_local.is_file() and val_file.is_file() and meta_file.is_file():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            val_rows = self._read_rows(val_file)
            return {**meta, "work_dir": work_dir, "val_rows": val_rows}

        work_dir.mkdir(parents=True, exist_ok=True)
        with train_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if spec.get("target") and spec["target"] in header:
                target_column, target_detected = spec["target"], False
            else:
                target_column, target_detected = header[-1], True
            if spec.get("id_column") and spec["id_column"] in header:
                id_column = spec["id_column"]
            elif "id" in [c.lower() for c in header]:
                id_column = header[[c.lower() for c in header].index("id")]
            else:
                id_column = None
            target_idx = header.index(target_column)
            id_idx = header.index(id_column) if id_column else None
            time_idx = None
            if spec.get("time_column") and spec["time_column"] in header:
                time_idx = header.index(spec["time_column"])

            # First pass (chronological only): count rows.
            total_rows = 0
            if spec["split"] == "chronological":
                for _ in reader:
                    total_rows += 1
                train_cutoff = int(total_rows * (1 - self.val_fraction))
                handle.seek(0)
                next(reader)  # skip header again
            sample: dict[str, list[str]] = {name: [] for name in header}
            train_targets: list[float] = []
            val_rows: list[list[str]] = []
            with train_local.open("w", encoding="utf-8", newline="") as train_out, \
                    val_file.open("w", encoding="utf-8", newline="") as val_out:
                train_writer = csv.writer(train_out)
                val_writer = csv.writer(val_out)
                train_writer.writerow(header)
                val_writer.writerow(header)
                n_train = 0
                for row_index, row in enumerate(reader):
                    if len(row) < len(header):
                        row = row + [""] * (len(header) - len(row))
                    row = row[: len(header)]
                    if spec["split"] == "chronological":
                        is_val = row_index >= train_cutoff
                    elif spec["split"] == "stratified":
                        is_val = _is_val_row(
                            self.seed, row[target_idx], row_index,
                            self.val_fraction,
                        )
                    else:
                        is_val = _is_val_row(
                            self.seed, "", row_index, self.val_fraction,
                        )
                    for name, value in zip(header, row):
                        if len(sample[name]) < 50:
                            sample[name].append(value)
                    if is_val:
                        val_writer.writerow(row)
                        val_rows.append(row)
                    else:
                        train_writer.writerow(row)
                        target = _as_float(row[target_idx])
                        if target is not None:
                            train_targets.append(target)
                        n_train += 1
        column_types = {
            name: _infer_type(values) for name, values in sample.items()
        }
        sample_text = ""
        with val_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for i, row in enumerate(reader):
                if i >= 3:
                    break
                sample_text += ",".join(row) + "\n"
        sample_text = sample_text[:PROMPT_SAMPLE_CHARS]
        meta = {
            "competition_id": competition_id,
            "header": header,
            "target_column": target_column,
            "target_detected": target_detected,
            "id_column": id_column,
            "target_idx": target_idx,
            "id_idx": id_idx,
            "time_idx": time_idx,
            "column_types": column_types,
            "n_train_rows": n_train,
            "n_val_rows": len(val_rows),
            "sample_rows": sample_text,
            "train_targets": train_targets,
        }
        # train_targets can be large (store-sales); keep them out of
        # the cached meta file and recompute from train_local when
        # reusing a cached split.
        meta_to_store = {k: v for k, v in meta.items() if k != "train_targets"}
        meta_file.write_text(json.dumps(meta_to_store), encoding="utf-8")
        return {**meta, "work_dir": work_dir, "val_rows": val_rows}

    @staticmethod
    def _read_rows(path: Path) -> list[list[str]]:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            return [row for row in reader if row]

    def _train_targets(self, work_dir: Path, target_idx: int) -> list[float]:
        targets: list[float] = []
        with (work_dir / "train_local.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.reader(handle)
            next(reader)
            for row in reader:
                value = _as_float(row[target_idx]) if target_idx < len(row) else None
                if value is not None:
                    targets.append(value)
        return targets

    # -- scoring ------------------------------------------------------------

    def _score_predictions(
        self,
        spec: dict,
        meta: dict,
        predictions_path: Path,
    ) -> tuple[float, bool, bool]:
        """Validate predictions.csv and score it.

        Returns ``(score, submission_valid, clipped)``. Raises
        ValueError when the submission is invalid (wrong rows,
        unknown ids, non-numeric values).
        """
        header = meta["header"]
        target_idx = meta["target_idx"]
        id_idx = meta["id_idx"]
        val_rows = meta["val_rows"]
        expected_ids = [
            (row[id_idx] if id_idx is not None else str(i))
            for i, row in enumerate(val_rows)
        ]
        predictions: dict[str, float] = {}
        with predictions_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "id" not in [
                (name or "").strip().lower() for name in reader.fieldnames
            ] or "prediction" not in [
                (name or "").strip().lower() for name in reader.fieldnames
            ]:
                raise ValueError(
                    "predictions.csv must have id,prediction columns"
                )
            id_key = next(
                name for name in reader.fieldnames
                if (name or "").strip().lower() == "id"
            )
            pred_key = next(
                name for name in reader.fieldnames
                if (name or "").strip().lower() == "prediction"
            )
            for row in reader:
                raw_id = str(row[id_key]).strip()
                value = _as_float(row[pred_key])
                if value is None:
                    raise ValueError(
                        f"non-numeric prediction for id {raw_id!r}"
                    )
                if raw_id in predictions:
                    raise ValueError(f"duplicate prediction for id {raw_id!r}")
                predictions[raw_id] = value
        if len(predictions) != len(expected_ids):
            raise ValueError(
                f"predictions.csv has {len(predictions)} rows, "
                f"expected {len(expected_ids)}"
            )
        missing = [i for i in expected_ids if i not in predictions]
        if missing:
            raise ValueError(
                f"predictions.csv missing {len(missing)} val ids "
                f"(e.g. {missing[0]!r})"
            )
        y_true = [
            _as_float(row[target_idx]) for row in val_rows
        ]
        if any(v is None for v in y_true):
            raise ValueError("val targets contain non-numeric values")
        y_pred = [predictions[i] for i in expected_ids]
        clipped = False
        if spec["metric"] == "rmsle" and any(p < 0 for p in y_pred):
            y_pred = [max(p, 0.0) for p in y_pred]
            clipped = True
        if spec["metric"] == "roc_auc":
            if any(p < 0 or p > 1 for p in y_pred):
                y_pred = [min(max(p, 0.0), 1.0) for p in y_pred]
                clipped = True
        score = compute_metric(spec["metric"], y_true, y_pred)
        return score, True, clipped

    # -- main entry point ---------------------------------------------------

    def __call__(self, competition: dict) -> dict:
        """Solve one competition; returns the benchmark result dict."""
        competition_id = competition.get("competition_id") or competition.get("id")
        spec = COMPETITION_SPECS.get(str(competition_id)) if competition_id else None
        if spec is None:
            return {
                "status": "error",
                "score": None,
                "normalized_score": None,
                "submission_valid": None,
                "error": f"no solver spec for competition {competition_id!r}",
            }
        try:
            metric = canonical_metric(
                competition.get("metric") or spec["metric"]
            )
        except ValueError as exc:
            return {
                "status": "error", "score": None, "normalized_score": None,
                "submission_valid": None, "error": str(exc),
            }
        spec = {**spec, "metric": metric, "task_type": competition.get("task_type")}
        try:
            meta = self._prepare_split(str(competition_id), spec)
        except DataUnavailableError as exc:
            return {
                "status": "not_run",
                "score": None,
                "normalized_score": None,
                "submission_valid": None,
                "baseline_score": None,
                "metric_direction": METRIC_DIRECTIONS[metric],
                "error": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 - one bad task must not kill the run
            return {
                "status": "error",
                "score": None,
                "normalized_score": None,
                "submission_valid": None,
                "error": f"split failed for {competition_id}: {exc}",
            }

        # Reference baseline on the same val split (no model involved).
        try:
            train_targets = meta.get("train_targets")
            if train_targets is None:
                train_targets = self._train_targets(
                    meta["work_dir"], meta["target_idx"]
                )
            reference_preds = _reference_predict(
                spec["reference"],
                train_targets=train_targets,
                val_rows=meta["val_rows"],
                header=meta["header"],
                target_idx=meta["target_idx"],
            )
            y_true = [
                _as_float(row[meta["target_idx"]]) for row in meta["val_rows"]
            ]
            reference_score = compute_metric(metric, y_true, reference_preds)
        except Exception as exc:  # noqa: BLE001 - reference failure is unmeasured
            return {
                "status": "error",
                "score": None,
                "normalized_score": None,
                "submission_valid": None,
                "error": f"reference baseline failed for {competition_id}: {exc}",
            }

        direction = METRIC_DIRECTIONS[metric]
        work_dir: Path = meta["work_dir"]
        files = ["train_local.csv", "val.csv"]
        # Copy the downloaded data files the solution may need
        # (e.g. petfinder images, digit-recognizer extras) into the
        # listing only -- generated code runs with cwd=work_dir, so
        # list what is actually there plus the split files.
        try:
            data_files = sorted(
                p.name for p in (self.data_root / str(competition_id)).iterdir()
                if p.is_file() and p.name not in ("train.csv",)
            )[:20]
        except OSError:
            data_files = []
        prompt = build_prompt(
            str(competition_id),
            spec,
            target_column=meta["target_column"],
            id_column=meta["id_column"],
            columns=meta["header"],
            column_types=meta["column_types"],
            n_train_rows=meta["n_train_rows"],
            n_val_rows=meta["n_val_rows"],
            sample_rows=meta["sample_rows"],
            files=files + data_files,
        )
        try:
            generated_text = self.generate(prompt)
            code = extract_code(generated_text)
        except Exception as exc:  # noqa: BLE001 - generation failure is unmeasured
            return {
                "status": "error",
                "score": None,
                "normalized_score": None,
                "submission_valid": None,
                "baseline_score": reference_score,
                "metric_direction": direction,
                "error": f"generation failed for {competition_id}: {exc}",
            }
        executor = self.executor_factory(work_dir)
        result = executor.run_code(code)
        if result.timed_out:
            return {
                "status": "error",
                "score": None,
                "normalized_score": None,
                "submission_valid": None,
                "baseline_score": reference_score,
                "metric_direction": direction,
                "error": (
                    f"solution for {competition_id} timed out after "
                    f"{self.execution_timeout_s}s"
                ),
            }
        if not result.succeeded:
            return {
                "status": "error",
                "score": None,
                "normalized_score": None,
                "submission_valid": None,
                "baseline_score": reference_score,
                "metric_direction": direction,
                "error": (
                    f"solution for {competition_id} exited with code "
                    f"{result.returncode}: "
                    f"{(result.stderr or '').strip()[:500]}"
                ),
            }
        predictions_path = work_dir / "predictions.csv"
        if not predictions_path.is_file():
            return {
                "status": "error",
                "score": None,
                "normalized_score": None,
                "submission_valid": False,
                "baseline_score": reference_score,
                "metric_direction": direction,
                "error": f"solution for {competition_id} wrote no predictions.csv",
            }
        try:
            score, valid, clipped = self._score_predictions(
                spec, meta, predictions_path
            )
        except Exception as exc:  # noqa: BLE001 - invalid submission is unmeasured
            return {
                "status": "error",
                "score": None,
                "normalized_score": None,
                "submission_valid": False,
                "baseline_score": reference_score,
                "metric_direction": direction,
                "error": f"invalid submission for {competition_id}: {exc}",
            }
        return {
            "status": "run",
            "score": score,
            "normalized_score": _normalized_score(score, reference_score, direction),
            "submission_valid": valid,
            "baseline_score": reference_score,
            "metric_direction": direction,
            "reference_method": spec["reference"],
            "reference_note": spec.get("reference_note"),
            "target_column": meta["target_column"],
            "target_detected": meta["target_detected"],
            "id_column": meta["id_column"],
            "val_rows": meta["n_val_rows"],
            "train_rows": meta["n_train_rows"],
            "fold_scores": [score],
            "predictions_clipped": clipped,
        }


def build_solver(
    model_ref: str,
    *,
    config: dict,
    data_root: str | Path,
    train_file: str | Path | None = None,
    held_out_ids: list[str] | None = None,
    seed: int = SEED,
    val_fraction: float = VAL_FRACTION,
    max_new_tokens: int = MAX_NEW_TOKENS,
    execution_timeout_s: int = EXECUTION_TIMEOUT_S,
    execution_memory_mb: int = EXECUTION_MEMORY_MB,
) -> CompetitionSolver:
    """Build a solver around the loaded model.

    Loads the pinned base model (plus the LoRA adapter when
    ``model_ref`` names one), wraps it in a :class:`ModelGenerator`,
    and runs the leakage guard against ``train_file`` first --
    a contaminated training file refuses before any model is
    loaded or any competition is attempted.
    """
    held_out = list(held_out_ids or [])
    if train_file is not None:
        check_training_leakage(train_file, held_out)
    model, tokenizer = load_model_for_eval(model_ref, config)
    generate = ModelGenerator(
        model, tokenizer, max_new_tokens=max_new_tokens, seed=seed
    )
    return CompetitionSolver(
        generate,
        data_root=data_root,
        seed=seed,
        val_fraction=val_fraction,
        max_new_tokens=max_new_tokens,
        execution_timeout_s=execution_timeout_s,
        execution_memory_mb=execution_memory_mb,
    )
