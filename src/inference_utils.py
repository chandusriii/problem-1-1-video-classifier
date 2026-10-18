from __future__ import annotations

import re
from typing import Any

CHOICE_LABELS = ("A", "B", "C", "D")


def parse_answer_letter(text: str) -> str:
    """Parse one A-D answer conservatively; return '?' when ambiguous."""
    stripped = text.strip()
    explicit = re.search(
        r"(?i)\banswer\s*(?::|-|\bis\b)\s*\(?([A-D])\)?(?=\s|[.)]|$)",
        stripped,
    )
    if explicit:
        return explicit.group(1).upper()
    leading = re.match(r"(?i)^\s*\(?([A-D])\)?(?=\s|[.)]|$)", stripped)
    return leading.group(1).upper() if leading else "?"


def format_question(row: dict[str, Any]) -> str:
    """Build the multiple-choice prompt without including the gold answer."""
    options = row.get("options")
    if isinstance(options, dict):
        normalized = {str(key).strip().upper(): value for key, value in options.items()}
        if len(normalized) != 4 or set(normalized) != set(CHOICE_LABELS):
            raise ValueError("Options dictionary must have exactly the A, B, C, and D labels.")
        option_lines = [f"{label}. {normalized[label]}" for label in CHOICE_LABELS]
    elif isinstance(options, list):
        if len(options) != 4:
            raise ValueError(f"Expected four answer options; found {len(options)}.")
        option_lines = [f"{label}. {value}" for label, value in zip(CHOICE_LABELS, options)]
    else:
        raise ValueError("Unsupported multiple-choice options format.")

    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError("Question text is empty.")

    return (
        "Use only visual evidence in the provided video frames. Choose exactly one answer "
        "letter A, B, C, or D. Put the letter first and provide a brief reason.\n\n"
        f"Question: {question}\n"
        + "\n".join(option_lines)
        + "\n\nRespond as: Answer: X. Reason: ..."
    )
