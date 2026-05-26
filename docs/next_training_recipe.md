# Next Training Recipe

This is the practical recipe we should implement next, based on the public discussion tab.

## Objective

Train a vLLM-compatible rank-32 LoRA adapter for Nemotron that learns deterministic reasoning
traces and always ends with a clean `\boxed{answer}`.

## Data Plan

1. Start from public Huikang/Tong-style traces rather than raw `train.csv` only.
2. Preserve strong existing trace formats for easy categories:
   - cipher
   - gravity
   - numeral
   - unit conversion
3. Add a compact bit-manipulation trace generator.
4. Avoid long or generic duplicated traces.
5. Do not blindly train all 9500 rows. Track difficulty with min logprob and repeat only hard,
   learnable examples.

## Training Plan

Use a custom masked SFT loop instead of generic `SFTTrainer`:

- tokenize prompts and completions ahead of time,
- mask prompt tokens with loss weight `0`,
- train answer/reasoning tokens with loss weight `1`,
- use cut-cross-entropy to avoid materializing full logits,
- train rank `32`,
- keep sequence length at or below `8192`,
- prefer RTX Pro 6000/Tinker/Modal-class GPU for the real run.

## Adapter Targets

Start with vLLM-safe modules:

```text
q_proj
k_proj
v_proj
o_proj
up_proj
down_proj
lm_head
```

Treat `in_proj`, `out_proj`, and `target_parameters` as risky until we can prove Kaggle's vLLM
loader applies them correctly.

## Validation Before Submit

Every candidate adapter must pass:

```powershell
python scripts/validate_adapter_submission.py submissions/submission.zip --strict
```

Then we need a behavioral check:

- load base + adapter in the same style as evaluation when possible,
- run a small fixed validation set,
- confirm outputs change versus base,
- confirm final answer format is `\boxed{...}`,
- record zip SHA256, source notebook/run id, trainable parameter count, final loss, and key warnings.

## Stop Conditions

Do not submit if:

- `r > 32`,
- root files are missing,
- LoRA A/B tensor counts mismatch,
- adapter includes unexplained full `base_layer` weights,
- configured target modules do not appear in tensor keys,
- the artifact source cannot be traced to logs or a known run.
