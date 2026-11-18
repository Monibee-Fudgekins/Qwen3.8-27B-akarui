"""Sandboxed code execution for the Akarui agent.

The Executor runs generated Python in a child subprocess with:

* a hard wall-clock timeout,
* CPU and memory limits enforced with the ``resource`` module,
* a working directory restricted to a caller-supplied sandbox
  directory,
* an environment scrubbed of credentials: any variable starting
  with ``KAGGLE_``, ``HF_TOKEN`` or ``AWS_`` is removed, so generated
  code can never read the parent process's secrets.

Generated code is never executed in the parent process and never
receives the parent's environment. This module does not decide
whether code is safe to run in general — callers must also apply
their own review and allow-listing policy — but it guarantees the
mechanical isolation listed above.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Environment variable prefixes/names that must never reach the child.
_BLOCKED_ENV_PREFIXES = ("KAGGLE_", "AWS_")
_BLOCKED_ENV_NAMES = {"HF_TOKEN", "HF_HOME", "KAGGLE_TOKEN", "KAGGLE_API_TOKEN"}


@dataclass
class ExecutionResult:
    """The outcome of one sandboxed code execution."""

    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool

    @property
    def succeeded(self) -> bool:
        """True when the code ran to completion with exit code 0."""
        return not self.timed_out and self.returncode == 0


class Executor:
    """Execute generated Python code in a sandboxed subprocess.

    Parameters
    ----------
    sandbox_dir:
        Directory the child process runs in. It is created if it
        does not exist. Generated code cannot write outside it
        through the working directory (absolute paths are still the
        caller's responsibility to police via review).
    default_timeout_s:
        Default wall-clock timeout in seconds.
    default_memory_mb:
        Default address-space limit in megabytes.
    """

    def __init__(
        self,
        sandbox_dir: str | Path,
        default_timeout_s: int = 600,
        default_memory_mb: int = 4096,
    ) -> None:
        self.sandbox_dir = Path(sandbox_dir)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.default_timeout_s = default_timeout_s
        self.default_memory_mb = default_memory_mb

    @staticmethod
    def scrub_env(env: dict[str, str] | None = None) -> dict[str, str]:
        """Return a copy of ``env`` (or the process environment) with credentials removed."""
        source = dict(env) if env is not None else dict(os.environ)
        return {
            key: value
            for key, value in source.items()
            if key not in _BLOCKED_ENV_NAMES
            and not key.startswith(_BLOCKED_ENV_PREFIXES)
        }

    def run_code(
        self,
        code: str,
        timeout_s: int | None = None,
        memory_mb: int | None = None,
    ) -> ExecutionResult:
        """Run ``code`` as a Python script in the sandbox and capture the result.

        The code is written to ``sandbox_dir/generated.py`` and run
        with the current interpreter. CPU time is limited to the
        timeout (plus a small margin) and address space to
        ``memory_mb``. A timeout kills the child and is reported as
        ``timed_out=True`` with ``returncode=None``.
        """
        timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        memory = memory_mb if memory_mb is not None else self.default_memory_mb

        script_path = self.sandbox_dir / "generated.py"
        script_path.write_text(code, encoding="utf-8")

        def _limits() -> None:
            # CPU seconds: soft limit at the timeout, hard limit shortly after.
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 30))
            # Address space in bytes.
            resource.setrlimit(resource.RLIMIT_AS, (memory * 1024 * 1024, memory * 1024 * 1024))

        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.sandbox_dir,
                env=self.scrub_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limits,
                check=False,
            )
            return ExecutionResult(
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                returncode=None,
                timed_out=True,
            )
