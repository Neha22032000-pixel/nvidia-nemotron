# NVIDIA Nemotron Model Reasoning Challenge

Working repository for the Kaggle NVIDIA Nemotron Model Reasoning Challenge.

Competition URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

## Goal

Improve the reasoning accuracy of `NVIDIA-Nemotron-3-Nano-30B` on NVIDIA's hidden reasoning benchmark and submit a compatible LoRA adapter.

Publicly described task families include:

- bit manipulation
- cipher/encryption transformations
- numeral and base conversions
- unit conversions
- modified gravity/physics rules
- symbolic equations
- numeric equations

## Repository Layout

- `docs/competition_notes.md` - distilled competition notes and assumptions
- `docs/project_plan.md` - working plan and milestones
- `docs/submission_checklist.md` - what must be verified before submitting
- `data/raw/` - official Kaggle files and external raw references
- `data/processed/` - normalized train/eval data
- `notebooks/` - exploratory analysis
- `src/nemotron_challenge/` - reusable code
- `scripts/` - command-line entry points
- `models/` - local adapter checkpoints; ignored by git
- `submissions/` - packaged Kaggle submissions; ignored by git

## First Commands

Once Kaggle credentials and a Python environment are ready:

```powershell
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data/raw
```

Inspect the downloaded files:

```powershell
python scripts/inspect_data.py
```

Generate starter synthetic reasoning data:

```powershell
python scripts/generate_synthetic.py --count 1000 --output data/processed/synthetic_train.jsonl
```

Run tests:

```powershell
pytest
```

## Submission Status

No Kaggle submission has been made yet. The exact adapter packaging format must be confirmed from the official competition files before submitting.
