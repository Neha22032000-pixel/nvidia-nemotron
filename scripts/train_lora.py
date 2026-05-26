from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
import typer
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from nemotron_challenge.paths import MODELS_DIR, RAW_DATA_DIR


app = typer.Typer(add_completion=False)
console = Console()


TARGET_PRESETS = {
    "safe": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    "safe_lm_head": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "lm_head"],
    "huikang_style": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "in_proj",
        "out_proj",
        "lm_head",
    ],
}


def format_prompt(prompt: str) -> str:
    return (
        f"{prompt.strip()}\n\n"
        "Solve the problem carefully. Put only the final answer inside a LaTeX boxed command."
        "\n\nResponse:\n"
    )


def format_completion(answer: str) -> str:
    return f"We need produce the requested result. The final answer is \\boxed{{{str(answer).strip()}}}"


class CompletionOnlyCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder

        input_ids = []
        attention_mask = []
        labels = []
        pad_id = self.tokenizer.pad_token_id
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@app.command()
def main(
    train_path: Path = RAW_DATA_DIR / "train.csv",
    output_dir: Path = MODELS_DIR / "nemotron-lora-r32",
    model_name: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    max_rows: int | None = None,
    max_length: int = 1024,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    target_preset: str = "safe_lm_head",
    learning_rate: float = 2e-4,
    epochs: float = 1.0,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    gpu_memory_gib: int = 13,
    cpu_memory_gib: int = 28,
    offload_dir: Path = MODELS_DIR / "offload",
) -> None:
    hf_token = os.environ.get("HF_TOKEN")

    train = pd.read_csv(train_path, dtype={"id": str})
    if max_rows:
        train = train.sample(n=min(max_rows, len(train)), random_state=13).reset_index(drop=True)
    dataset = Dataset.from_pandas(train[["prompt", "answer"]])

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        input_ids = []
        labels = []
        for prompt, answer in zip(batch["prompt"], batch["answer"], strict=True):
            prompt_ids = tokenizer(format_prompt(prompt), add_special_tokens=True)["input_ids"]
            completion_ids = tokenizer(
                format_completion(answer) + tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]
            ids = (prompt_ids + completion_ids)[:max_length]
            prompt_len = min(len(prompt_ids), len(ids))
            input_ids.append(ids)
            labels.append([-100] * prompt_len + ids[prompt_len:])
        return {"input_ids": input_ids, "labels": labels}

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    gpu_count = torch.cuda.device_count()
    max_memory = {idx: f"{gpu_memory_gib}GiB" for idx in range(gpu_count)}
    max_memory["cpu"] = f"{cpu_memory_gib}GiB"
    offload_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Detected {gpu_count} CUDA devices; max_memory={max_memory}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        quantization_config=quant_config,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload_dir),
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    console.print(f"Device map: {getattr(model, 'hf_device_map', None)}")
    if target_preset not in TARGET_PRESETS:
        raise typer.BadParameter(f"target_preset must be one of {sorted(TARGET_PRESETS)}")
    target_modules = TARGET_PRESETS[target_preset]
    console.print(f"Using LoRA target preset {target_preset}: {target_modules}")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Import lazily so local data utilities remain usable without PEFT training dependencies.
    from peft import get_peft_model

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=CompletionOnlyCollator(tokenizer),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    console.print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    app()
