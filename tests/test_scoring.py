from nemotron_challenge.scoring import extract_final_answer, score_predictions


def test_extract_final_answer_label() -> None:
    assert extract_final_answer("Reasoning...\nFinal answer: 42") == "42"


def test_extract_boxed_answer() -> None:
    assert extract_final_answer("Therefore \\boxed{ABC}") == "ABC"


def test_score_predictions() -> None:
    summary = score_predictions(["Final answer: 9", "Final answer: 10"], ["9", "11"])
    assert summary.total == 2
    assert summary.correct == 1
    assert summary.accuracy == 0.5
