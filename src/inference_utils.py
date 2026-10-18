from __future__ import annotations

import re
from typing import Any


def parse_answer_letter(text: str) -> str:
    """Parse a single A-D choice from model output.

    Prefer an explicit 'Answer:' label. Otherwise only accept a leading choice
    token, preventing accidental matches on unrelated words such as 'apple'.
    """
    stripped = text.strip()
    explicit = re.search(r"(?i)\banswer\s*[:\-]\s*\(?([A-D])\)?(?=\s|[.)]|$)", stripped)
    if explicit:
        return explicit.group(1).upper()

    leading = re.match(r"(?i)^\s*\(?([A-D])\)?(?=\s|[.)]|$)", stripped)
    return leading.group(1).upper() if leading else "?"


def format_question(row: dict[str, Any]) -> str:
    """Build a choice prompt without exposing the dataset's gold answer."""
    options = row.get("options", {})
    if isinstance(options, dict):
        option_lines = [f"{key}. {value}" for key, value in options.items()]
    elif isinstance(options, list):
        option_lines = [
            f"{chr(ord('A') + index)}. {value}"
            for index, value in enumerate(options)
        ]
    else:
        raise ValueError("Unsupported multiple-choice options format.")
    if len(option_lines) != 4:
        raise ValueError(f"Expected four answer options; found {len(option_lines)}.")
    return (
        "Answer using only visual evidence in the provided video frames. "
        "Choose exactly one option letter A, B, C, or D. Give a short reason, "
        "but put the answer letter first. Do not mention dataset labels.\n\n"
        f"Question: {row.get('question', '')}\n"
        + "\n".join(option_lines)
        + "\n\nRespond as: Answer: X. Reason: ..."
    )
