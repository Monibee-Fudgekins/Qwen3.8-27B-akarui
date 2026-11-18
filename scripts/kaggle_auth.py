"""Kaggle authentication adapter for Qwen3.8-27B-akarui.

The repository secret is ``KAGGLE_TOKEN``. Kaggle's client tooling does not
all recognise that name, so this module translates it into the form the
supported clients expect:

* If ``KAGGLE_TOKEN`` parses as JSON with ``username``/``key`` fields, it is
  written to ``~/.kaggle/kaggle.json`` (mode 0600) and ``KAGGLE_USERNAME`` /
  ``KAGGLE_KEY`` are set in the process environment (legacy kaggle CLI).
* Otherwise a non-empty ``KAGGLE_TOKEN`` is treated as an API token and
  exposed as ``KAGGLE_API_TOKEN`` (kagglehub / current kaggle client).
* If ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` are already set in the
  environment, they are passed through unchanged.

The credential value is never printed, logged, or returned. Functions
return only a description of which mechanism was configured.

No network calls are made at import time.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

KAGGLE_CONFIG_DIR = Path.home() / ".kaggle"
KAGGLE_JSON_PATH = KAGGLE_CONFIG_DIR / "kaggle.json"


def _write_kaggle_json(username: str, key: str) -> Path:
    """Write ``~/.kaggle/kaggle.json`` with mode 0600.

    Args:
        username: Kaggle username.
        key: Kaggle API key.

    Returns:
        The path written.
    """
    KAGGLE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    KAGGLE_JSON_PATH.write_text(json.dumps({"username": username, "key": key}))
    os.chmod(KAGGLE_JSON_PATH, stat.S_IRUSR | stat.S_IWUSR)
    return KAGGLE_JSON_PATH


def configure_kaggle_auth() -> dict:
    """Configure Kaggle authentication from the environment.

    Reads ``KAGGLE_TOKEN`` (the repository secret) and translates it into
    the form Kaggle clients expect. Falls back to legacy
    ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` when those are already set.

    Returns:
        A dict describing the mechanism configured, e.g.
        ``{"method": "kaggle_json", "path": "~/.kaggle/kaggle.json"}``.
        The credential value is never included.

    Raises:
        RuntimeError: If no usable credential is available.
    """
    token = os.environ.get("KAGGLE_TOKEN", "").strip()

    if token:
        parsed = None
        if token.startswith("{"):
            try:
                parsed = json.loads(token)
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, dict) and parsed.get("username") and parsed.get("key"):
            _write_kaggle_json(str(parsed["username"]), str(parsed["key"]))
            os.environ["KAGGLE_USERNAME"] = str(parsed["username"])
            os.environ["KAGGLE_KEY"] = str(parsed["key"])
            return {"method": "kaggle_json", "path": str(KAGGLE_JSON_PATH)}
        os.environ["KAGGLE_API_TOKEN"] = token
        return {"method": "api_token", "env_var": "KAGGLE_API_TOKEN"}

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return {"method": "legacy_env", "env_vars": ["KAGGLE_USERNAME", "KAGGLE_KEY"]}

    raise RuntimeError(
        "No Kaggle credential found. Set KAGGLE_TOKEN (repository secret), "
        "or KAGGLE_USERNAME and KAGGLE_KEY."
    )


def auth_status() -> dict:
    """Report which Kaggle auth mechanism is currently configured.

    Returns:
        A dict with the detected method, or ``{"method": "none"}``.
        No credential values are included.
    """
    if KAGGLE_JSON_PATH.exists():
        return {"method": "kaggle_json", "path": str(KAGGLE_JSON_PATH)}
    if os.environ.get("KAGGLE_API_TOKEN"):
        return {"method": "api_token", "env_var": "KAGGLE_API_TOKEN"}
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return {"method": "legacy_env", "env_vars": ["KAGGLE_USERNAME", "KAGGLE_KEY"]}
    return {"method": "none"}


if __name__ == "__main__":
    try:
        result = configure_kaggle_auth()
    except RuntimeError as exc:
        print(f"error: {exc}")
        raise SystemExit(1)
    print(f"Kaggle auth configured via {result['method']}")
