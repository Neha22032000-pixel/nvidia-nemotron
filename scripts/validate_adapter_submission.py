from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import typer
from rich.console import Console
from rich.table import Table

from nemotron_challenge.paths import SUBMISSIONS_DIR


app = typer.Typer(add_completion=False)
console = Console()

REQUIRED_ROOT_FILES = {"adapter_config.json", "adapter_model.safetensors"}
ALLOWED_ROOT_PREFIXES = ("adapter_", "README", "README.")
VLLM_SUPPORTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
}
KNOWN_NEMOTRON_TARGET_MODULES = VLLM_SUPPORTED_TARGET_MODULES | {"in_proj", "out_proj"}
WARN_ONLY_TARGET_MODULES = {"in_proj", "out_proj"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_safetensor_keys(path: Path) -> tuple[list[str], dict[str, list[int]]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "Install safetensors to inspect adapter_model.safetensors "
            "(for example: pip install safetensors)."
        ) from exc

    keys: list[str] = []
    shapes: dict[str, list[int]] = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            keys.append(key)
            shapes[key] = list(f.get_slice(key).get_shape())
    return keys, shapes


def _normalize_target_modules(raw: object) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {str(item) for item in raw}
    return {str(raw)}


@app.command()
def main(
    submission_zip: Path = typer.Argument(SUBMISSIONS_DIR / "submission.zip"),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail on vLLM-risky but not definitively invalid adapter fields.",
    ),
) -> None:
    errors: list[str] = []
    warnings: list[str] = []

    if not submission_zip.exists():
        raise FileNotFoundError(submission_zip)
    if submission_zip.name != "submission.zip":
        warnings.append("Kaggle upload requires the file to be named submission.zip")

    with ZipFile(submission_zip) as archive:
        names = archive.namelist()
        root_files = {name for name in names if "/" not in name.rstrip("/")}
        nested_files = [name for name in names if "/" in name.rstrip("/")]

        missing = REQUIRED_ROOT_FILES - root_files
        if missing:
            errors.append(f"Missing root files: {sorted(missing)}")
        if nested_files:
            warnings.append(
                "Zip contains nested paths. Kaggle expects adapter files at the root: "
                f"{nested_files[:5]}"
            )

        unexpected = [
            name
            for name in root_files
            if name not in REQUIRED_ROOT_FILES and not name.startswith(ALLOWED_ROOT_PREFIXES)
        ]
        if unexpected:
            warnings.append(f"Unexpected root files: {unexpected}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive.extractall(tmp_path)
            config_path = tmp_path / "adapter_config.json"
            weights_path = tmp_path / "adapter_model.safetensors"

            config: dict[str, object] = {}
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))

            if config.get("peft_type") != "LORA":
                errors.append(f"Expected peft_type=LORA, found {config.get('peft_type')!r}")

            rank = config.get("r")
            if not isinstance(rank, int):
                errors.append(f"adapter_config.json must contain integer r, found {rank!r}")
            elif rank > 32:
                errors.append(f"LoRA rank r={rank} exceeds competition max_lora_rank=32")

            target_modules = _normalize_target_modules(config.get("target_modules"))
            unknown_targets = target_modules - KNOWN_NEMOTRON_TARGET_MODULES
            warn_targets = target_modules & WARN_ONLY_TARGET_MODULES
            if unknown_targets:
                warnings.append(f"Unknown target_modules for Nemotron/vLLM: {sorted(unknown_targets)}")
            if warn_targets:
                warnings.append(
                    "These targets appear in Unsloth public recipes but may be vLLM-risky: "
                    f"{sorted(warn_targets)}"
                )

            if config.get("target_parameters"):
                warnings.append(
                    "adapter_config.json uses target_parameters. This is PEFT/Unsloth-specific "
                    "and must be validated against Kaggle's vLLM loader before trusting score."
                )

            if weights_path.exists():
                keys, shapes = _load_safetensor_keys(weights_path)
                if not keys:
                    errors.append("adapter_model.safetensors has no tensors")

                key_counts = Counter(
                    "lora_A"
                    if ".lora_A." in key
                    else "lora_B"
                    if ".lora_B." in key
                    else "base_layer"
                    if ".base_layer." in key
                    else "other"
                    for key in keys
                )
                if key_counts["lora_A"] != key_counts["lora_B"]:
                    errors.append(
                        "LoRA A/B tensor count mismatch: "
                        f"A={key_counts['lora_A']} B={key_counts['lora_B']}"
                    )
                if key_counts["base_layer"]:
                    warnings.append(
                        f"Found {key_counts['base_layer']} base_layer tensors. "
                        "A pure LoRA adapter usually should not include full base weights."
                    )

                module_hits = Counter()
                for key in keys:
                    for module in KNOWN_NEMOTRON_TARGET_MODULES:
                        if f".{module}." in key or f".{module}_" in key:
                            module_hits[module] += 1
                missing_targets_in_weights = sorted(target_modules - set(module_hits))
                if missing_targets_in_weights:
                    warnings.append(
                        "Configured target_modules with no obvious tensors: "
                        f"{missing_targets_in_weights}"
                    )

                table = Table(title="Adapter Tensor Summary")
                table.add_column("Metric")
                table.add_column("Value")
                table.add_row("zip", str(submission_zip))
                table.add_row("zip_size_bytes", str(submission_zip.stat().st_size))
                table.add_row("sha256", _sha256(submission_zip))
                table.add_row("rank", str(rank))
                table.add_row("tensor_count", str(len(keys)))
                table.add_row("lora_A", str(key_counts["lora_A"]))
                table.add_row("lora_B", str(key_counts["lora_B"]))
                table.add_row("base_layer", str(key_counts["base_layer"]))
                table.add_row("target_modules", ", ".join(sorted(target_modules)))
                table.add_row("module_hits", json.dumps(dict(sorted(module_hits.items()))))
                first_shape = next(iter(shapes.items())) if shapes else None
                if first_shape:
                    table.add_row("first_tensor", f"{first_shape[0]} {first_shape[1]}")
                console.print(table)

    if warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"- {warning}")
    if errors:
        console.print("[red]Errors:[/red]")
        for error in errors:
            console.print(f"- {error}")
        raise typer.Exit(code=1)
    if strict and warnings:
        console.print("[red]Strict mode failed because warnings were present.[/red]")
        raise typer.Exit(code=1)
    console.print("[green]Adapter submission passed structural validation.[/green]")


if __name__ == "__main__":
    app()
