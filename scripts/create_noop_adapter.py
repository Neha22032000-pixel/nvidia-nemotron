from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from safetensors.numpy import save_file

from nemotron_challenge.paths import MODELS_DIR


app = typer.Typer(add_completion=False)
console = Console()


BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
ATTENTION_LAYERS = [5, 12, 19, 26, 33, 42]
HIDDEN_SIZE = 2688
ATTENTION_QO_SIZE = 4096
ATTENTION_KV_SIZE = 256
LORA_RANK = 32


@app.command()
def main(output_dir: Path = MODELS_DIR / "noop-lora-r32") -> None:
    """Create a zero-weight PEFT adapter package for Kaggle format validation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": BASE_MODEL,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": LORA_RANK,
        "rank_pattern": {},
        "revision": None,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }

    (output_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tensors: dict[str, np.ndarray] = {}
    for layer in ATTENTION_LAYERS:
        for module_name, in_features, out_features in [
            ("q_proj", HIDDEN_SIZE, ATTENTION_QO_SIZE),
            ("k_proj", HIDDEN_SIZE, ATTENTION_KV_SIZE),
            ("v_proj", HIDDEN_SIZE, ATTENTION_KV_SIZE),
            ("o_proj", ATTENTION_QO_SIZE, HIDDEN_SIZE),
        ]:
            prefix = f"base_model.model.backbone.layers.{layer}.mixer.{module_name}"
            tensors[f"{prefix}.lora_A.weight"] = np.zeros(
                (LORA_RANK, in_features), dtype=np.float16
            )
            tensors[f"{prefix}.lora_B.weight"] = np.zeros(
                (out_features, LORA_RANK), dtype=np.float16
            )

    # Zero tensors preserve base-model behavior while giving PEFT real adapter
    # weights to load. A trained adapter will replace these tensors.
    save_file(tensors, output_dir / "adapter_model.safetensors")
    (output_dir / "README.md").write_text(
        "# Zero-Weight LoRA Adapter\n\n"
        "Format-validation adapter for the NVIDIA Nemotron Kaggle competition. "
        "Replace with trained adapter weights for a competitive submission.\n",
        encoding="utf-8",
    )
    console.print(f"Wrote no-op adapter files to {output_dir}")


if __name__ == "__main__":
    app()
