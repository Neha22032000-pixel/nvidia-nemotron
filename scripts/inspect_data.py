from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from nemotron_challenge.paths import RAW_DATA_DIR


console = Console()


def discover_tables(raw_dir: Path) -> list[Path]:
    patterns = ("*.csv", "*.parquet", "*.jsonl", "*.json")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(raw_dir.glob(pattern))
    return sorted(files)


def load_preview(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, nrows=5)
    if path.suffix == ".parquet":
        return pd.read_parquet(path).head(5)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True).head(5)
    if path.suffix == ".json":
        return pd.read_json(path).head(5)
    raise ValueError(f"Unsupported file type: {path}")


def main() -> None:
    files = discover_tables(RAW_DATA_DIR)
    if not files:
        console.print(
            f"No data files found in {RAW_DATA_DIR}. "
            "Download competition files there before running this script."
        )
        return

    table = Table(title="Raw Data Files")
    table.add_column("file")
    table.add_column("columns")
    table.add_column("preview rows")

    for path in files:
        try:
            preview = load_preview(path)
        except Exception as exc:
            table.add_row(path.name, f"error: {exc}", "0")
            continue
        table.add_row(path.name, ", ".join(map(str, preview.columns)), str(len(preview)))

    console.print(table)


if __name__ == "__main__":
    main()
