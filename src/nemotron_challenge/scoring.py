from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


FINAL_PATTERNS = [
    re.compile(r"final\s+answer\s*[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"answer\s*[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\\boxed\{([^{}]+)\}"),
]


@dataclass(frozen=True)
class ScoreSummary:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().strip(". ").split())


def extract_final_answer(text: str) -> str:
    for pattern in FINAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return normalize_answer(matches[-1])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return normalize_answer(lines[-1]) if lines else ""


def score_predictions(predictions: Iterable[str], references: Iterable[str]) -> ScoreSummary:
    total = 0
    correct = 0
    for prediction, reference in zip(predictions, references, strict=True):
        total += 1
        if extract_final_answer(prediction) == normalize_answer(reference):
            correct += 1
    return ScoreSummary(total=total, correct=correct)
