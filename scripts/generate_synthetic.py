from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from nemotron_challenge.paths import PROCESSED_DATA_DIR
from nemotron_challenge.synthetic import GENERATORS, generate_examples


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    output: Path = PROCESSED_DATA_DIR / "synthetic_train.jsonl",
    count: int = 1000,
    seed: int = 13,
    chat_format: bool = True,
    categories: list[str] | None = typer.Option(None, help=f"Choices: {', '.join(GENERATORS)}"),
) -> None:
    examples = generate_examples(count=count, seed=seed, categories=categories)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            record = example.to_chat_record() if chat_format else example.to_record()
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    console.print(f"Wrote {len(examples)} examples to {output}")


if __name__ == "__main__":
    app()
