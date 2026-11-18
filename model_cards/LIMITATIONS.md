# Limitations

Known limitations of Qwen3.8-27B-akarui. This list is honest about
what has not been done yet; it is updated as measurements arrive.

## No model has been published yet

- No training run has completed. No adapter, merged model, or
  checkpoint has been published to Kaggle.
- No benchmark has run. No performance claim of any kind can be
  made about the fine-tune.
- No baseline has been recorded for the original Qwen3.8-27B on
  the held-out benchmark (`evaluation/baseline_results.json` is an
  honest null template).

## Capability retention is unproven

- The fine-tune is intended to preserve the base model's reasoning,
  coding, instruction-following, and vision-language capabilities,
  but no regression test has run. Do not claim the fine-tune
  retains every base capability until regression testing confirms it.
- QLoRA training touches a subset of the model's weights; the
  effect on general capabilities is a measured question, not an
  assumption.

## Hardware and training constraints

- The target is 4 × NVIDIA L4 (24GB each), but Kaggle notebooks
  typically provide a single GPU and a 30-hour weekly GPU quota.
  Four GPUs are a target, not a guarantee: the training code
  detects actual hardware and degrades gracefully.
- A 27B model in 4-bit QLoRA is memory-heavy; sequence length,
  batch size, and gradient accumulation must be validated against
  the hardware actually obtained before a full run.
- GRPO is experimental and is not required for the initial public
  release; it is enabled only when compute, evaluation
  infrastructure, and reward reliability are sufficient.

## Data and evaluation limits

- The held-out benchmark (12 competitions) is an initial proposal;
  competition IDs must be verified against Kaggle before the first
  run.
- Preference (DPO) labels come only from verified execution
  results; preference data does not exist yet.
- The agent executes generated code in a sandbox with resource
  limits and scrubbed credentials, but sandboxing is a mitigation,
  not a proof of safety — generated code must still be reviewed
  before it runs outside the sandbox.

## Scope

- The model is specialised for data-science competitions. It is
  not a general-purpose chatbot fine-tune, and competition-solving
  ability does not imply general capability.
- Outputs must be validated by execution. A plan or a code sample
  from the model is a proposal until it has run and produced a
  measured result.
