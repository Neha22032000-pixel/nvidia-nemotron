from __future__ import annotations

from pathlib import Path

import typer
from kaggle.api.kaggle_api_extended import KaggleApi
from rich.console import Console

from nemotron_challenge.paths import SUBMISSIONS_DIR


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    file_path: Path = SUBMISSIONS_DIR / "submission.zip",
    message: str = "submission from repo script",
) -> None:
    if file_path.name != "submission.zip":
        raise ValueError("Kaggle requires the uploaded file to be named submission.zip")
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    api = KaggleApi()
    api.authenticate()
    api.competition_submit(str(file_path), message, COMPETITION)
    console.print(f"Submitted {file_path} to {COMPETITION}")


if __name__ == "__main__":
    app()
