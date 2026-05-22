# Kaggle T4 Training

Use a Kaggle Notebook with GPU set to T4.

## Secrets

Add a Kaggle Secret:

- `HF_TOKEN`: Hugging Face token with access to the NVIDIA model.

Do not commit this token to GitHub or write it into notebook cells.

## Notebook Setup

```bash
git clone https://github.com/Neha22032000-pixel/nvidia-nemotron.git
cd nvidia-nemotron
pip install -r requirements-kaggle.txt
```

In a Kaggle notebook cell:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Download the competition files:

```bash
python - <<'PY'
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.competition_download_files(
    "nvidia-nemotron-model-reasoning-challenge",
    path="data/raw",
    quiet=False,
)
PY
unzip -o data/raw/nvidia-nemotron-model-reasoning-challenge.zip -d data/raw
```

## Smoke Submission

This verifies credentials and packaging:

```bash
PYTHONPATH=src python scripts/create_lookup_submission.py
PYTHONPATH=src python scripts/package_submission.py
PYTHONPATH=src python scripts/submit_to_kaggle.py --message "lookup smoke submission"
```

## First LoRA Run

Start small on T4:

```bash
PYTHONPATH=src python scripts/train_lora.py --max-rows 256 --epochs 0.1
```

Then scale if memory and time are healthy:

```bash
PYTHONPATH=src python scripts/train_lora.py --epochs 1 --gradient-accumulation-steps 16
```

## Notes

- T4 does not support BF16 efficiently, so the script uses FP16 plus 4-bit NF4 loading.
- The LoRA rank defaults to 32 to stay within the publicly discussed competition constraint.
- Confirm the official adapter packaging format before submitting a trained adapter. The current smoke path submits a CSV inside `submission.zip`.
