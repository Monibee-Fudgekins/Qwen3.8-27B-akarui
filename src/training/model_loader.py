"""Load the pinned Qwen/Qwen3.8-27B base model and its tokenizer.

Base model facts (verified against the Hugging Face model card and
config):

* Model: ``Qwen/Qwen3.8-27B``, pinned revision
  ``37561816e66b5b9153b1b64c10968c18c1e6f4a0``.
* Architecture: ``Qwen3_5ForConditionalGeneration`` (``model_type``
  ``qwen3_5``), a dense vision-language causal LM.
* Text: hidden 5120, 64 layers, 24 attention heads, 4 KV heads,
  intermediate 17408, vocab 248320, max positions 262144.
* Vision: hidden 1152, patch size 16.
* License: Apache 2.0.

All heavy imports (torch, transformers, peft, bitsandbytes) happen
inside the functions, so importing this module works on a machine
without the GPU stack. ``trust_remote_code`` is always False: the
stock transformers implementation is used, never remote code.
"""

from __future__ import annotations

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
EXPECTED_MODEL_TYPE = "qwen3_5"

#: Verified architecture numbers, used for memory / LoRA estimates.
ARCH = {
    "params_billion": 27,
    "text_hidden_size": 5120,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "max_position_embeddings": 262144,
    "vision_hidden_size": 1152,
    "vision_patch_size": 16,
}

#: Head dimension assumed for the LoRA parameter estimate. Qwen3-family
#: models use head_dim 128; the estimate is labelled as such and is not
#: a substitute for counting the real model's parameters.
_ASSUMED_HEAD_DIM = 128

_DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _quantization_config(config: dict):
    """Build a BitsAndBytesConfig from the ``quantization`` config block."""
    import torch
    from transformers import BitsAndBytesConfig

    quant = config.get("quantization", {})
    compute_dtype_name = str(
        quant.get("bnb_4bit_compute_dtype", "bfloat16")
    ).lower()
    compute_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(compute_dtype_name)
    if compute_dtype is None:
        raise ValueError(
            f"unsupported bnb_4bit_compute_dtype: {compute_dtype_name!r}"
        )
    return BitsAndBytesConfig(
        load_in_4bit=bool(quant.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(quant.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(
            quant.get("bnb_4bit_use_double_quant", True)
        ),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_base_model(config: dict, device_map: str = "auto"):
    """Load Qwen/Qwen3.8-27B at the pinned revision with 4-bit NF4
    double quantization and BF16 compute.

    Verifies that the installed transformers understands ``model_type``
    ``qwen3_5`` and raises a clear error otherwise (an old transformers
    cannot load this model -- upgrade it instead of working around it).
    The tokenizer and chat template are never modified.
    """
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    base_model = config.get("base_model", BASE_MODEL)
    base_revision = config.get("base_revision", BASE_REVISION)
    try:
        model_config = AutoConfig.from_pretrained(
            base_model, revision=base_revision, trust_remote_code=False
        )
    except Exception as exc:
        raise RuntimeError(
            f"transformers could not load the config of {base_model} "
            f"revision {base_revision}: {exc}. This usually means the "
            "installed transformers predates qwen3_5 support -- upgrade "
            "transformers instead of patching the model."
        ) from exc
    if getattr(model_config, "model_type", None) != EXPECTED_MODEL_TYPE:
        raise RuntimeError(
            f"unexpected model_type {getattr(model_config, 'model_type', None)!r} "
            f"for {base_model}; expected {EXPECTED_MODEL_TYPE!r}. Refusing "
            "to load a model that is not the pinned Qwen3.8-27B."
        )
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        revision=base_revision,
        quantization_config=_quantization_config(config),
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    return model


def load_tokenizer(config: dict):
    """Load the original Qwen tokenizer, preserving it exactly.

    The tokenizer and its chat template are returned as shipped: this
    function never modifies the tokenizer, the vocabulary, or the chat
    template. Native reasoning / thinking behaviour is therefore kept.
    """
    from transformers import AutoTokenizer

    base_model = config.get("base_model", BASE_MODEL)
    base_revision = config.get("base_revision", BASE_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, revision=base_revision, trust_remote_code=False
    )
    return tokenizer


def get_lora_config(config: dict):
    """Build a PEFT :class:`LoraConfig` from the ``lora`` config block."""
    from peft import LoraConfig, TaskType

    lora = config.get("lora", {})
    target_modules = list(
        lora.get("target_modules", _DEFAULT_LORA_TARGET_MODULES)
    )
    return LoraConfig(
        r=int(lora.get("r", 32)),
        lora_alpha=int(lora.get("alpha", 64)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def estimate_memory(config: dict) -> dict:
    """Estimate memory for a 4-bit QLoRA run of the pinned base model.

    Returns a dict with ``weights_gb`` (4-bit weights), an estimated
    trainable-parameter count for the configured LoRA, a rough total
    (weights + adapter + gradients + 8-bit AdamW states + activations
    with gradient checkpointing + framework overhead), and
    ``fits_single_l4_24gb`` with a ``reasoning`` string. All numbers
    are estimates, clearly labelled as such -- they are for planning,
    not a guarantee.
    """
    lora = config.get("lora", {})
    r = int(lora.get("r", 32))
    training = config.get("training", {})
    seq_len = int(training.get("max_seq_length", 4096))

    hidden = ARCH["text_hidden_size"]
    intermediate = ARCH["intermediate_size"]
    layers = ARCH["num_hidden_layers"]
    q_out = ARCH["num_attention_heads"] * _ASSUMED_HEAD_DIM
    kv_out = ARCH["num_key_value_heads"] * _ASSUMED_HEAD_DIM

    # LoRA params per layer: r * (in + out) summed over target modules.
    per_layer = r * (
        (hidden + q_out)  # q_proj
        + (hidden + kv_out)  # k_proj
        + (hidden + kv_out)  # v_proj
        + (q_out + hidden)  # o_proj
        + (hidden + intermediate)  # gate_proj
        + (hidden + intermediate)  # up_proj
        + (intermediate + hidden)  # down_proj
    )
    trainable_params = per_layer * layers

    weights_gb = ARCH["params_billion"] * 0.5  # 4-bit: 0.5 bytes/param
    adapter_gb = trainable_params * 2 / 1e9  # BF16 adapter weights
    grads_gb = trainable_params * 2 / 1e9  # BF16 gradients
    optimizer_gb = trainable_params * 2 / 1e9  # 8-bit AdamW: 2 x 1 byte/param
    # Activations with gradient checkpointing scale ~linearly with seq len;
    # ~1.5 GB at 4k tokens for this shape, plus ~1.5 GB CUDA/framework.
    activations_gb = round(1.5 * seq_len / 4096, 2)
    overhead_gb = 1.5
    total_gb = round(
        weights_gb + adapter_gb + grads_gb + optimizer_gb + activations_gb + overhead_gb,
        2,
    )
    fits = total_gb < 24.0
    reasoning = (
        f"4-bit weights ~{weights_gb:.1f} GB + LoRA adapter/grads/8-bit AdamW "
        f"~{adapter_gb + grads_gb + optimizer_gb:.2f} GB (trainable params "
        f"~{trainable_params / 1e6:.0f}M, estimate assuming head_dim "
        f"{_ASSUMED_HEAD_DIM}) + activations ~{activations_gb:.2f} GB at "
        f"seq {seq_len} with gradient checkpointing + ~{overhead_gb:.1f} GB "
        f"framework overhead = ~{total_gb:.1f} GB estimated total vs 24 GB "
        f"on one L4. This is a planning estimate, not a measured value; "
        "confirm with a real run before full training."
    )
    return {
        "weights_gb": round(weights_gb, 2),
        "trainable_params_estimate": trainable_params,
        "adapter_gb": round(adapter_gb, 2),
        "gradients_gb": round(grads_gb, 2),
        "optimizer_gb": round(optimizer_gb, 2),
        "activations_gb": activations_gb,
        "estimated_total_gb": total_gb,
        "fits_single_l4_24gb": fits,
        "reasoning": reasoning,
    }
