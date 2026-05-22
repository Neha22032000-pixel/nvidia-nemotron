# Kaggle Submission Checklist

## Needed From User

- Accept the Kaggle competition rules on the competition page.
- Provide Kaggle API access locally, preferably by placing `kaggle.json` in the standard Kaggle config folder rather than pasting secrets into chat.
- Confirm Hugging Face access to `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` and any required license terms.
- Provide or approve a GPU runtime for training; local Windows CPU is not realistic for a 30B LoRA workflow.

## Before First Submission

- Download official competition files into `data/raw/`.
- Inspect schema with `python scripts/inspect_data.py`.
- Confirm the exact adapter directory and archive format Kaggle expects.
- Confirm LoRA constraints, especially rank, target modules, tokenizer handling, and model revision.
- Run a tiny end-to-end smoke package before training the real adapter.

## Submission Guardrails

- Do not include raw private credentials, full model weights, or generated caches.
- Include only the adapter and required metadata files.
- Record the training data mix, seed, LoRA config, and commit SHA for every submission.
- Keep each submitted artifact under `submissions/` locally, but do not commit it unless it is small and allowed.
