"""GPU detection for Qwen3.8-27B-akarui training.

Detects hardware with ``nvidia-smi`` and, when torch is installed,
``torch.cuda``. Works with torch absent: it falls back to ``nvidia-smi``
parsing only.

Output is a single JSON object on stdout describing GPU count, model,
compute capability, VRAM, CUDA version and torch version.

Exit codes:
    0  At least one GPU was found.
    1  Detection failed (e.g. ``nvidia-smi`` missing and torch absent).
    2  No GPU found.

No network calls are made. This module is safe to import; detection only
runs under ``main()`` or via :func:`detect`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys


def _run(command: list[str]) -> str | None:
    """Run a command and return its stdout, or None on any failure."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _parse_nvidia_smi() -> list[dict]:
    """Parse ``nvidia-smi --query-gpu`` output into GPU records."""
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict] = []
    if not output:
        return gpus
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            memory_mib = float(parts[2])
        except ValueError:
            continue
        gpus.append(
            {
                "index": len(gpus),
                "model": parts[0],
                "compute_capability": parts[1],
                "memory_total_gb": round(memory_mib / 1024, 2),
                "driver_version": parts[3],
            }
        )
    return gpus


def _cuda_version_from_smi() -> str | None:
    """Extract the CUDA version reported by ``nvidia-smi``."""
    output = _run(["nvidia-smi"])
    if not output:
        return None
    match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", output)
    return match.group(1) if match else None


def _torch_info() -> dict:
    """Collect torch/CUDA info when torch is importable, else empty info."""
    info: dict = {
        "torch_version": None,
        "torch_cuda_available": False,
        "torch_cuda_version": None,
    }
    try:
        import torch  # noqa: PLC0415  (optional dependency, lazy import)
    except ImportError:
        return info
    info["torch_version"] = torch.__version__
    info["torch_cuda_available"] = bool(torch.cuda.is_available())
    info["torch_cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info.setdefault("torch_gpus", []).append(
                {
                    "index": index,
                    "model": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "memory_total_gb": round(props.total_memory / (1024**3), 2),
                }
            )
    return info


def detect() -> dict:
    """Detect GPUs and return a JSON-serialisable hardware report.

    Returns:
        Dict with keys ``gpus`` (list), ``gpu_count``, ``cuda_version``,
        ``torch_version``, ``torch_cuda_available`` and
        ``torch_cuda_version``.
    """
    gpus = _parse_nvidia_smi()
    torch_info = _torch_info()
    # Prefer nvidia-smi records; fall back to torch records when nvidia-smi
    # is unavailable (e.g. inside some containers).
    if not gpus and torch_info.get("torch_gpus"):
        gpus = torch_info["torch_gpus"]
    cuda_version = _cuda_version_from_smi() or torch_info.get("torch_cuda_version")
    return {
        "gpus": gpus,
        "gpu_count": len(gpus),
        "cuda_version": cuda_version,
        "torch_version": torch_info["torch_version"],
        "torch_cuda_available": torch_info["torch_cuda_available"],
        "torch_cuda_version": torch_info["torch_cuda_version"],
    }


def main() -> int:
    """Print the hardware report as JSON and return the exit code."""
    report = detect()
    if report["gpu_count"] == 0 and report["torch_version"] is None:
        # Neither nvidia-smi nor torch could report anything: detection
        # itself failed, as opposed to a machine with no GPU.
        if _run(["nvidia-smi"]) is None:
            print(
                json.dumps(
                    {"error": "no GPU detected and nvidia-smi is unavailable"}
                )
            )
            return 1
    print(json.dumps(report, indent=2))
    return 0 if report["gpu_count"] > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
