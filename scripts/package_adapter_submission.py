from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import typer
from rich.console import Console

from nemotron_challenge.paths import MODELS_DIR, SUBMISSIONS_DIR


app = typer.Typer(add_completion=False)
console = Console()


REQUIRED_FILES = {
    "adapter_config.json",
}


@app.command()
def main(
    adapter_dir: Path = MODELS_DIR / "nemotron-lora-r32",
    output: Path = SUBMISSIONS_DIR / "submission.zip",
) -> None:
    if output.name != "submission.zip":
        raise ValueError("Kaggle requires the uploaded file to be named submission.zip")
    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)

    existing = {path.name for path in adapter_dir.iterdir() if path.is_file()}
    missing = REQUIRED_FILES - existing
    if missing:
        raise ValueError(f"Adapter directory is missing required files: {sorted(missing)}")
    if not ({"adapter_model.safetensors", "adapter_model.bin"} & existing):
        raise ValueError("Adapter directory must contain adapter_model.safetensors or adapter_model.bin")

    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(adapter_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(adapter_dir))

    console.print(f"Wrote {output} from adapter files in {adapter_dir}")


if __name__ == "__main__":
    app()
