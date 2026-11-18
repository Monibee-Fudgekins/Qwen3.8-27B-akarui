"""Tool registry for the Akarui competition-solving agent.

The registry lists the libraries the agent is allowed to use, with a
short description of each. :func:`get_available_tools` checks which
of them are actually importable in the current environment, so the
agent only ever attempts to use tools that are present — an absent
library is reported as unavailable, never assumed.
"""

from __future__ import annotations

import importlib.util

# import name -> (display name, description)
_TOOLS: dict[str, tuple[str, str]] = {
    "pandas": ("pandas", "Tabular data manipulation and analysis."),
    "polars": ("polars", "Fast columnar DataFrames for large tabular data."),
    "numpy": ("numpy", "Numerical arrays and linear algebra."),
    "duckdb": ("duckdb", "In-process SQL analytics over files and DataFrames."),
    "sklearn": ("scikit-learn", "Classical ML models, preprocessing, and metrics."),
    "catboost": ("catboost", "Gradient boosting, strong on categorical features."),
    "lightgbm": ("lightgbm", "Fast gradient-boosted trees for tabular data."),
    "xgboost": ("xgboost", "Gradient-boosted trees for tabular data."),
    "torch": ("torch", "PyTorch: deep learning and model training."),
    "transformers": ("transformers", "Hugging Face model loading and inference."),
    "optuna": ("optuna", "Hyperparameter optimisation."),
    "matplotlib": ("matplotlib", "Plotting and visualisation."),
}


def get_available_tools() -> list[dict]:
    """Return the registry entries for tools importable here.

    Each entry is ``{"name", "import_name", "description",
    "available"}``. The list is sorted by name and covers every
    registered tool, available or not, so callers can see what is
    missing as well as what is present.
    """
    entries = []
    for import_name, (display_name, description) in sorted(
        _TOOLS.items(), key=lambda item: item[1][0]
    ):
        available = importlib.util.find_spec(import_name) is not None
        entries.append(
            {
                "name": display_name,
                "import_name": import_name,
                "description": description,
                "available": available,
            }
        )
    return entries
