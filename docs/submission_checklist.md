# Submission Checklist

Use this checklist before every Kaggle upload.

## Needed From User

- Kaggle competition rules must be accepted.
- Kaggle API access must be available locally.
- GPU/runtime choice must be approved before training.
- Do not paste or commit private tokens.

## Required Files

`submission.zip` must contain root-level:

```text
adapter_config.json
adapter_model.safetensors
```

Do not put these inside a subfolder.

## Structural Check

Run:

```powershell
python scripts/validate_adapter_submission.py submissions/submission.zip --strict
```

The check must report:

- rank `<= 32`,
- non-empty tensor file,
- matching LoRA A/B tensor counts,
- no missing root files.

Warnings must be understood before submitting.

## Behavioral Check

Before upload, record:

- adapter source/run id,
- zip SHA256,
- training data source,
- final loss or min-logprob report,
- small validation outputs,
- whether answers use `\boxed{...}`.

## Guardrails

- Do not include raw private credentials, full model weights, or generated caches.
- Do not blindly submit public notebook outputs unless the exact artifact and logs are verified.
- Treat `target_parameters`, `in_proj`, `out_proj`, and full `base_layer` tensors as risky until vLLM compatibility is proven.
- Keep submitted artifacts under `submissions/` locally, but do not commit large zips.

## Upload

After validation:

```powershell
python scripts/submit_to_kaggle.py submissions/submission.zip --message "short run description"
```

Use `--skip-validation` only for emergency debugging. Do not use it for real leaderboard attempts.
