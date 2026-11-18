"""Diagnose a Kaggle kernel push failure by printing the API's response body.

The `kaggle` CLI reports a failed push as a bare HTTP status (e.g.
"400 Client Error: Bad Request for url: .../SaveKernel") and discards
the response body, which is where the API explains *why* it rejected
the request. This script re-attempts the push in-process through the
Kaggle API client and prints the full error response (status + body),
so the actual cause can be read from the workflow log instead of
guessed at.

Usage (in the workflow, after a failed push step):
    python scripts/diagnose_kernel_push.py --notebook-dir kernels/eval

It exits 1 after printing the diagnosis. If the re-attempted push
succeeds, it prints the success response instead -- that means the
earlier failure was transient and the kernel is now running on Kaggle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_auth import configure_kaggle_auth  # noqa: E402


def _print_response_body(exc: BaseException) -> None:
    """Print the HTTP response body attached to an API exception, if any."""
    response = getattr(exc, "response", None)
    if response is None:
        print("no HTTP response attached to the exception")
        return
    print(f"HTTP status: {getattr(response, 'status_code', '?')}")
    text = getattr(response, "text", "") or ""
    print(f"response body ({len(text)} chars):")
    print(text if text else "(empty body)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-attempt a Kaggle kernel push and print the API error body."
    )
    parser.add_argument(
        "--notebook-dir",
        default="kernels/eval",
        help="Packaged kernel directory (default: kernels/eval)",
    )
    args = parser.parse_args(argv)

    notebook_dir = Path(args.notebook_dir)
    meta_path = notebook_dir / "kernel-metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        print(f"kernel id: {meta.get('id')}")
        print(f"kernel title: {meta.get('title')!r}")
        print(f"kernel type: {meta.get('kernel_type')}")
        print(f"code file: {meta.get('code_file')}")
    else:
        print(f"error: {meta_path} not found", file=sys.stderr)
        return 1

    auth = configure_kaggle_auth()
    print(f"Kaggle auth configured via {auth['method']}")

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    try:
        result = api.kernels_push(str(notebook_dir))
    except Exception as exc:  # noqa: BLE001 - diagnostic: print everything
        print(f"push raised {type(exc).__name__}: {exc}")
        _print_response_body(exc)
        # Some client versions wrap the HTTP error; unwrap one level.
        cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
        if cause is not None and cause is not exc:
            print(f"wrapped cause: {type(cause).__name__}: {cause}")
            _print_response_body(cause)
        return 1

    print("push SUCCEEDED on the re-attempt (earlier failure was transient):")
    print(f"  error: {getattr(result, 'error', None)}")
    print(f"  url: {getattr(result, 'url', None)}")
    print(f"  version: {getattr(result, 'versionNumber', None)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
