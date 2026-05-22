from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class ReasoningExample:
    category: str
    prompt: str
    reasoning: str
    answer: str

    def to_chat_record(self) -> dict[str, str]:
        return {
            "category": self.category,
            "prompt": self.prompt,
            "response": f"{self.reasoning}\n\nFinal answer: {self.answer}",
            "answer": self.answer,
        }

    def to_record(self) -> dict[str, str]:
        return asdict(self)


def _base_digits(value: int, base: int) -> str:
    if value == 0:
        return "0"
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out: list[str] = []
    n = value
    while n:
        n, rem = divmod(n, base)
        out.append(digits[rem])
    return "".join(reversed(out))


def make_base_conversion(rng: random.Random) -> ReasoningExample:
    value = rng.randint(16, 4095)
    src_base = rng.choice([2, 3, 4, 5, 7, 8, 12, 16])
    dst_base = rng.choice([2, 3, 4, 5, 7, 8, 10, 12, 16])
    while dst_base == src_base:
        dst_base = rng.choice([2, 3, 4, 5, 7, 8, 10, 12, 16])
    source = _base_digits(value, src_base)
    answer = _base_digits(value, dst_base)
    return ReasoningExample(
        category="base_conversion",
        prompt=f"Convert {source} from base {src_base} to base {dst_base}.",
        reasoning=(
            f"First interpret {source} in base {src_base}, which equals {value} in decimal. "
            f"Then write {value} in base {dst_base}."
        ),
        answer=answer,
    )


def make_bitwise(rng: random.Random) -> ReasoningExample:
    a = rng.randint(0, 255)
    b = rng.randint(0, 255)
    op_name, symbol, func = rng.choice(
        [
            ("AND", "&", lambda x, y: x & y),
            ("OR", "|", lambda x, y: x | y),
            ("XOR", "^", lambda x, y: x ^ y),
        ]
    )
    answer = str(func(a, b))
    return ReasoningExample(
        category="bitwise",
        prompt=f"Compute the bitwise {op_name} of {a} and {b}. Give the decimal result.",
        reasoning=f"Apply {a} {symbol} {b} bit by bit and convert the result back to decimal.",
        answer=answer,
    )


def make_caesar_cipher(rng: random.Random) -> ReasoningExample:
    words = ["REASON", "NVIDIA", "MODEL", "VECTOR", "LOGIC", "TOKEN", "MATRIX"]
    text = rng.choice(words)
    shift = rng.randint(1, 25)

    def enc(ch: str) -> str:
        return chr((ord(ch) - ord("A") + shift) % 26 + ord("A"))

    cipher = "".join(enc(ch) for ch in text)
    return ReasoningExample(
        category="cipher",
        prompt=f"A Caesar cipher shifts letters forward by {shift}. Decode {cipher}.",
        reasoning=f"Shift each letter in {cipher} backward by {shift} positions.",
        answer=text,
    )


def make_unit_conversion(rng: random.Random) -> ReasoningExample:
    choices: list[tuple[str, str, int]] = [
        ("kilometers", "meters", 1000),
        ("hours", "minutes", 60),
        ("minutes", "seconds", 60),
        ("kilograms", "grams", 1000),
    ]
    src, dst, factor = rng.choice(choices)
    value = rng.randint(2, 250)
    answer = str(value * factor)
    return ReasoningExample(
        category="unit_conversion",
        prompt=f"Convert {value} {src} to {dst}.",
        reasoning=f"Each {src[:-1] if src.endswith('s') else src} is {factor} {dst}, so multiply {value} by {factor}.",
        answer=answer,
    )


def make_linear_equation(rng: random.Random) -> ReasoningExample:
    x = rng.randint(-20, 20)
    a = rng.choice([n for n in range(-9, 10) if n not in (0, 1)])
    b = rng.randint(-30, 30)
    c = a * x + b
    return ReasoningExample(
        category="linear_equation",
        prompt=f"Solve for x: {a}x + {b} = {c}.",
        reasoning=f"Subtract {b} from both sides to get {a}x = {c - b}. Divide by {a}.",
        answer=str(x),
    )


GENERATORS: dict[str, Callable[[random.Random], ReasoningExample]] = {
    "base_conversion": make_base_conversion,
    "bitwise": make_bitwise,
    "cipher": make_caesar_cipher,
    "unit_conversion": make_unit_conversion,
    "linear_equation": make_linear_equation,
}


def generate_examples(count: int, seed: int = 13, categories: Iterable[str] | None = None) -> list[ReasoningExample]:
    rng = random.Random(seed)
    selected = list(categories or GENERATORS)
    unknown = sorted(set(selected) - set(GENERATORS))
    if unknown:
        raise ValueError(f"Unknown categories: {', '.join(unknown)}")

    examples: list[ReasoningExample] = []
    for idx in range(count):
        category = selected[idx % len(selected)]
        examples.append(GENERATORS[category](rng))
    rng.shuffle(examples)
    return examples
