from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import typer
from rich.console import Console

from nemotron_challenge.paths import SUBMISSIONS_DIR


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    source: Path = SUBMISSIONS_DIR / "lookup_submission.csv",
    output: Path = SUBMISSIONS_DIR / "submission.zip",
) -> None:
    if output.name != "submission.zip":
        raise ValueError("Kaggle requires this competition's uploaded file to be named submission.zip")
    if not source.exists():
        raise FileNotFoundError(source)

    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(source, arcname=source.name)
    console.print(f"Wrote {output} containing {source.name}")


if __name__ == "__main__":
    app()
