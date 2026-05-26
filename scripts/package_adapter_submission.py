from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import typer
from rich.console import Console

from nemotron_challenge.paths import MODELS_DIR, SUBMISSIONS_DIR
from validate_adapter_submission import main as validate_submission


app = typer.Typer(add_completion=False)
console = Console()


REQUIRED_FILES = {
    "adapter_config.json",
}
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


@app.command()
def main(
    adapter_dir: Path = typer.Argument(MODELS_DIR / "nemotron-lora-r32"),
    output: Path = typer.Argument(SUBMISSIONS_DIR / "submission.zip"),
) -> None:
    if output.name != "submission.zip":
        raise ValueError("Kaggle requires the uploaded file to be named submission.zip")
    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)

    existing = {path.name for path in adapter_dir.iterdir() if path.is_file()}
    missing = REQUIRED_FILES - existing
    if missing:
        raise ValueError(f"Adapter directory is missing required files: {sorted(missing)}")
    weight_file = next((name for name in WEIGHT_FILES if name in existing), None)
    if weight_file is None:
        raise ValueError("Adapter directory must contain adapter_model.safetensors or adapter_model.bin")

    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name in ("adapter_config.json", weight_file):
            archive.write(adapter_dir / name, arcname=name)

    console.print(f"Wrote {output} from adapter files in {adapter_dir}")
    validate_submission(output)


if __name__ == "__main__":
    app()
