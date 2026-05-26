# Kaggle Discussion Findings

Snapshot source: Kaggle competition discussion tab, saved locally under
`public_artifacts/discussion_snapshot/`.

## What Strong Public Runs Do

- They are not plain "train all rows once" SFT runs.
- The strongest public writeup targets roughly `0.877` with:
  - deterministic chain-of-thought traces,
  - about `27.85M` training tokens for the key run,
  - difficulty/min-logprob-driven data selection,
  - external/Tinker-style compute,
  - rank-32 LoRA and long `8192` context training.
- Bit manipulation is the biggest published unlock. The public strategy focuses on compact
  per-bit reasoning traces instead of asking the model to do parallel bit-string operations.
- Correct solver output is not enough. Several participants report that "better" synthetic
  datasets can score worse when traces are too long, too hard for the model to learn, duplicated,
  or oversampled enough to cause forgetting.

## Evaluator Gotchas

- Kaggle evaluates by loading the base Nemotron model plus our LoRA adapter with vLLM.
- The overview settings matter:
  - `max_lora_rank=32`
  - `max_model_len=8192`
  - `max_tokens=7680`
  - `temperature=0.0`
- The answer extractor prefers `\boxed{...}`. Answers containing `}` had metric issues and were
  discussed multiple times.
- Binary answers are string-sensitive after the metric update. Numeric tolerance does not rescue
  wrong binary strings.
- Submissions can vary slightly across reruns, but a collapse from `0.86` to `0.06` is not normal
  variance; it points to a bad/incompatible adapter or artifact.

## Why The Bobber Replica Scored 0.06

The submitted artifact was:

`/kaggle/input/notebooks/bobber/vex506-reinst-kaggle-replica/submission.zip`

It was structurally valid, but discussion/notebook evidence is weaker than the title implies:

- The notebook says "Predicted: 0.85 -> 0.86 LB".
- The author says Tinker was used for the real SFT/GRPO work and Kaggle for evaluation.
- The downloaded adapter contains PEFT/Unsloth-specific fields and many MoE tensors.
- It includes a large `base_layer` tensor, which is risky for a pure LoRA/vLLM evaluator path.

Conclusion: do not treat public notebook output as a scored adapter unless the exact artifact,
version, logs, and vLLM load behavior are verified.

## Missing Pieces In This Repo

- Deterministic, validated CoT generation per category.
- A compact bit-manipulation trace generator.
- Tokenized prompt/response masks so the loss is only on the solution/answer region.
- Min-logprob tracking to decide which examples to train or repeat.
- A rank-32 long-context training path on RTX Pro 6000/Tinker/Modal-class hardware.
- vLLM-compatible adapter validation before every submission.

## Immediate Rule

No more submissions until `scripts/validate_adapter_submission.py` passes, and risky warnings are
understood. A valid zip only proves the evaluator can read the file; it does not prove the adapter
is actually useful.
