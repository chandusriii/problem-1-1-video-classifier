"""Lightweight query-intent classification and routing for adaptive video RAG."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

INTENTS = (
    "FRAME_RECOGNITION", "OCR_FRAME_QUERY", "TRANSCRIPT_LOOKUP",
    "VIDEO_SUMMARY", "EVENT_COUNTING", "TEMPORAL_LOCALIZATION",
    "COMPARISON", "CAUSAL_REASONING", "CROSS_SECTION_REASONING",
    "GENERAL_VIDEO_QUESTION",
)
SCOPES = ("FRAME", "CLIP", "WHOLE_VIDEO", "MULTI_SECTION")
COMPLEXITIES = ("L1_DIRECT", "L2_RETRIEVAL", "L3_GRAPH")
TIMESTAMP_RE = re.compile(r"(?<!\\d)(?:\\d{1,2}:)?\\d{1,2}:\\d{2}(?:\\.\\d+)?(?!\\d)")

def normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    cleaned = re.sub(r"\\s+", " ", query).strip()
    if not cleaned:
        raise ValueError("query must not be empty")
    return cleaned

def engineer_query_features(query: str) -> dict[str, Any]:
    """Create interpretable cheap features; no video decoding or LLM required."""
    text = normalize_query(query)
    lower = text.lower()
    timestamps = TIMESTAMP_RE.findall(text)
    tokens = re.findall(r"\\b[\\w']+\\b", lower)
    clauses = len(re.findall(r"[,;]|\\b(?:and|but|while|whereas|although)\\b", lower)) + 1
    cues = {
        "asks_why": bool(re.search(r"\\bwhy\\b|\\bwhat caused\\b|\\breason for\\b", lower)),
        "asks_when": bool(re.search(r"\\bwhen\\b|\\bwhat time\\b", lower)),
        "asks_count": bool(re.search(r"\\bhow many\\b|\\bhow often\\b|\\bcount\\b|\\bnumber of times\\b", lower)),
        "asks_summary": bool(re.search(r"\\b(summarize|summary|overview|recap|main points)\\b", lower)),
        "asks_compare": bool(re.search(r"\\b(compare|difference|similar|versus|vs\\.?|change from .* to)\\b", lower)),
        "asks_ocr": bool(re.search(r"\\b(read|text|written|sign|caption|subtitle|on-screen text)\\b", lower)),
        "asks_audio": bool(re.search(r"\\b(say|said|says|speak|speaker|transcript|mention|mentions|quote)\\b", lower)),
        "asks_global": bool(re.search(r"\\b(entire|whole|overall|throughout|in the video|video as a whole|beginning to end|from the beginning to the end)\\b", lower)),
        "asks_multiple_events": bool(re.search(r"\\b(all|every|each|throughout|repeatedly|again)\\b", lower)),
        "asks_temporal_relation": bool(re.search(r"\\b(before|after|then|earlier|later|following|until|while)\\b", lower)),
        "has_timestamp": bool(timestamps),
        "asks_visual_attribute": bool(re.search(r"\\b(color|colour|wearing|holding|look like|appear|visible|object|person|car|animal)\\b", lower)),
        "asks_entity_relation": bool(re.search(r"\\b(relationship|relation|interact|influence|connect|between)\\b", lower)),
    }
    return {
        "normalized_length": len(text), "token_count": len(tokens),
        "clause_count": clauses, "timestamp_count": len(timestamps),
        "question_word": next((word for word in ("why", "when", "where", "who", "how", "what") if re.search(rf"\\b{word}\\b", lower)), "none"),
        **cues,
    }

def _heuristic_predictions(query: str) -> dict[str, str | float]:
    f = engineer_query_features(query)
    text = query.lower()

    # Narrow operations take precedence over broad question forms.
    if f["asks_ocr"]:
        intent = "OCR_FRAME_QUERY"
    elif f["asks_count"]:
        intent = "EVENT_COUNTING"
    elif f["asks_summary"]:
        intent = "VIDEO_SUMMARY"
    elif f["asks_compare"]:
        intent = "CROSS_SECTION_REASONING" if f["asks_global"] or f["asks_temporal_relation"] else "COMPARISON"
    elif f["asks_why"]:
        intent = "CAUSAL_REASONING"
    elif f["asks_audio"]:
        intent = "TRANSCRIPT_LOOKUP"
    elif f["has_timestamp"] and f["asks_visual_attribute"]:
        intent = "FRAME_RECOGNITION"
    elif f["asks_temporal_relation"] or f["asks_when"]:
        intent = "TEMPORAL_LOCALIZATION"
    elif f["asks_global"] and (f["asks_entity_relation"] or f["asks_multiple_events"]):
        intent = "CROSS_SECTION_REASONING"
    elif f["asks_visual_attribute"] and re.search(r"\\b(frame|image|screen)\\b", text):
        intent = "FRAME_RECOGNITION"
    else:
        intent = "GENERAL_VIDEO_QUESTION"

    # Scope is separate from complexity.
    if intent == "VIDEO_SUMMARY":
        scope = "WHOLE_VIDEO" if re.search(r"\\b(entire|whole|overall|in the video|video as a whole)\\b", text) else "CLIP"
    elif f["has_timestamp"]:
        scope = "FRAME" if intent in {"FRAME_RECOGNITION", "OCR_FRAME_QUERY"} else "CLIP"
    elif intent in {"CROSS_SECTION_REASONING"}:
        scope = "MULTI_SECTION"
    elif intent == "EVENT_COUNTING":
        scope = "MULTI_SECTION" if f["asks_multiple_events"] or re.search(r"\\bhow many times\\b", text) else "CLIP"
    elif intent == "CAUSAL_REASONING":
        scope = "MULTI_SECTION" if f["asks_temporal_relation"] or f["asks_entity_relation"] else "CLIP"
    elif intent == "COMPARISON":
        scope = "MULTI_SECTION" if re.search(r"\\b(first and last|beginning|end|throughout)\\b", text) else "CLIP"
    elif f["asks_global"] and f["asks_multiple_events"]:
        scope = "MULTI_SECTION"
    elif intent in {"TRANSCRIPT_LOOKUP", "TEMPORAL_LOCALIZATION"}:
        scope = "CLIP"
    else:
        scope = "CLIP"

    if intent in {"CAUSAL_REASONING", "CROSS_SECTION_REASONING"}:
        complexity = "L3_GRAPH"
    elif scope == "MULTI_SECTION":
        complexity = "L2_RETRIEVAL" if intent == "EVENT_COUNTING" else "L3_GRAPH"
    elif intent in {"VIDEO_SUMMARY", "TRANSCRIPT_LOOKUP", "TEMPORAL_LOCALIZATION", "EVENT_COUNTING", "COMPARISON"} or scope == "CLIP":
        complexity = "L2_RETRIEVAL"
    else:
        complexity = "L1_DIRECT"

    return {"intent": intent, "scope": scope, "complexity": complexity,
            "confidence": 0.55, "method": "rules_baseline"}

def predict_intent(query: str, model_path: str | Path | None = None) -> dict[str, Any]:
    cleaned = normalize_query(query)
    if model_path:
        import joblib
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Classifier artifact not found: {path}")
        artifact = joblib.load(path)
        features = artifact["vectorizer"].transform([cleaned])
        predictions: dict[str, Any] = {}
        scores = []
        for axis in ("intent", "scope", "complexity"):
            classifier = artifact["classifiers"][axis]
            probabilities = classifier.predict_proba(features)[0]
            index = int(probabilities.argmax())
            predictions[axis] = str(classifier.classes_[index])
            scores.append(float(probabilities[index]))
        predictions["confidence"] = round(sum(scores) / len(scores), 4)
        predictions["method"] = "tfidf_logistic_regression"
    else:
        predictions = _heuristic_predictions(cleaned)

    route = {
        "L1_DIRECT": "DIRECT_OR_TARGETED_LOOKUP",
        "L2_RETRIEVAL": "MULTIMODAL_RETRIEVAL",
        "L3_GRAPH": "GRAPH_RETRIEVAL",
    }[predictions["complexity"]]
    predictions.update({"query": cleaned, "retrieval_strategy": route})
    predictions["fallback"] = "SAFE_HYBRID_RETRIEVAL" if float(predictions["confidence"]) < 0.45 else None
    return predictions

def load_jsonl_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate labelled JSONL examples."""
    import json
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("query", "intent", "scope", "complexity"):
                if field not in row:
                    raise ValueError(f"line {line_number}: missing field {field!r}")
            if row["intent"] not in INTENTS:
                raise ValueError(f"line {line_number}: unsupported intent {row['intent']!r}")
            if row["scope"] not in SCOPES:
                raise ValueError(f"line {line_number}: unsupported scope {row['scope']!r}")
            if row["complexity"] not in COMPLEXITIES:
                raise ValueError(f"line {line_number}: unsupported complexity {row['complexity']!r}")
            rows.append(row)
    if not rows:
        raise ValueError("dataset has no labelled examples")
    return rows
