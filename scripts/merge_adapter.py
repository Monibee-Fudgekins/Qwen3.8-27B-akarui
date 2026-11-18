"""Merge a LoRA adapter into the Qwen3.8-27B base model (bf16).

Usage:
    python scripts/merge_adapter.py --adapter output/akarui-sft-v0 --output output/akarui-merged-v0

The merge needs the full 27B model in memory: about 54 GB for bf16
weights plus overhead. It runs on GPU when the detected hardware has
enough VRAM, or on CPU when the machine has at least 110 GB of RAM.
Otherwise it refuses with a clear message and exits 2 — it never writes
a partial model and never claims a merged model was produced when it
was not.

The merged output is a full inference-ready model directory (weights,
config and tokenizer). It is large; publish it only when the storage
budget supports it. Adapter-only releases are the default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 27B parameters in bf16: 27e9 * 2 bytes ~= 54 GB, plus adapter and
# activation overhead during the merge.
MERGE_VRAM_GB = 60.0
MERGE_RAM_GB = 110.0


def _total_ram_gb() -> float:
    """Return total system RAM in GB, or 0.0 if it cannot be read."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kib = float(line.split()[1])
                    return kib / (1024**2)
    except OSError:
        pass
    return 0.0


def hardware_allows_merge() -> tuple[bool, str]:
    """Check whether this machine can hold the merged model in memory.

    Returns:
        ``(allowed, method)`` where method is ``"gpu"``, ``"cpu"`` or
        ``""`` when the merge is refused.
    """
    from detect_gpus import detect  # noqa: PLC0415

    hardware = detect()
    total_vram = sum(
        gpu.get("memory_total_gb", 0) for gpu in hardware.get("gpus", [])
    )
    if total_vram >= MERGE_VRAM_GB:
        return True, "gpu"
    total_ram = _total_ram_gb()
    if total_ram >= MERGE_RAM_GB:
        return True, "cpu"
    return False, ""


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on error, 2 on refusal."""
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the base model (bf16)."
    )
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Merged model output directory")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3.8-27B",
        help="Base model id or local path",
    )
    parser.add_argument(
        "--base-revision",
        default="37561816e66b5b9153b1b64c10968c18c1e6f4a0",
        help="Pinned base-model revision",
    )
    args = parser.parse_args(argv)

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"error: adapter not found: {adapter_path}", file=sys.stderr)
        return 1

    allowed, method = hardware_allows_merge()
    if not allowed:
        print(
            "error: cannot merge on this machine. A 27B bf16 merge needs "
            f"at least {MERGE_VRAM_GB:.0f} GB of GPU VRAM or "
            f"{MERGE_RAM_GB:.0f} GB of system RAM. No merged model was "
            "produced.",
            file=sys.stderr,
        )
        return 2

    print(f"Merging adapter {adapter_path} into {args.base_model} ({method})...")
    try:
        import torch  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        print(f"error: missing merge dependency: {exc}", file=sys.stderr)
        return 1

    device = "cuda" if method == "gpu" else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = model.merge_and_unload()

    output_path = Path(args.output)
    merged.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.base_revision, trust_remote_code=False
    )
    tokenizer.save_pretrained(str(output_path))
    print(f"Merged model written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
