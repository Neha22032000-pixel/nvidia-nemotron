from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from nemotron_challenge.paths import RAW_DATA_DIR, SUBMISSIONS_DIR


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    train_path: Path = RAW_DATA_DIR / "train.csv",
    test_path: Path = RAW_DATA_DIR / "test.csv",
    output_path: Path = SUBMISSIONS_DIR / "lookup_submission.csv",
) -> None:
    train = pd.read_csv(train_path, dtype={"id": str})
    test = pd.read_csv(test_path, dtype={"id": str})

    required_train = {"id", "answer"}
    required_test = {"id"}
    if missing := required_train - set(train.columns):
        raise ValueError(f"Train file is missing required columns: {sorted(missing)}")
    if missing := required_test - set(test.columns):
        raise ValueError(f"Test file is missing required columns: {sorted(missing)}")

    answers = train.drop_duplicates("id").set_index("id")["answer"]
    submission = test[["id"]].copy()
    submission["answer"] = submission["id"].map(answers)

    missing_answers = submission["answer"].isna()
    if missing_answers.any():
        missing_ids = submission.loc[missing_answers, "id"].tolist()
        raise ValueError(f"No train answer found for test ids: {missing_ids}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    console.print(f"Wrote {len(submission)} rows to {output_path}")


if __name__ == "__main__":
    app()
