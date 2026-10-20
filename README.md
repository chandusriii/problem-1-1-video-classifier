# Adaptive Video RAG — User Intent Classifier

This branch implements the **User Intent Classifier** task: classify a user's video-related query so a router can choose a retrieval scope and reasoning path.

## Design

Predictions are separate across three axes:

- **Intent:** frame recognition, OCR, transcript lookup, summarization, counting, temporal localization, comparison, causal reasoning, cross-section reasoning, or general video question.
- **Scope:** `FRAME`, `CLIP`, `WHOLE_VIDEO`, or `MULTI_SECTION`.
- **Complexity:** `L1_DIRECT`, `L2_RETRIEVAL`, or `L3_GRAPH`, inspired by AdaVideoRAG.

The axes are separate: a whole-video question does not always require graph retrieval, and a timestamp-specific question can still need complex reasoning.

## Quick start

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements-intent.txt
python -m unittest discover -s tests -v
python scripts/predict_intent.py --query "What color is the car at 00:15?"
python scripts/predict_intent.py --query "How does the speaker's opinion change from the beginning to the end?"
python scripts/train_intent_classifier.py --data data/intent_examples.jsonl --output models/intent_classifier.joblib
python scripts/predict_intent.py --query "How many times does the player score?" --model models/intent_classifier.joblib
```

The deterministic rules baseline runs without model weights. The optional trained model uses word/character TF-IDF with Logistic Regression.

## Key files

- `src/intent_classifier.py` — feature engineering, baseline routing, optional model loading.
- `scripts/train_intent_classifier.py` — lightweight model training.
- `scripts/predict_intent.py` — CLI inference.
- `data/intent_examples.jsonl` — seed queries and labels.
- `tests/test_intent_classifier.py` — unit tests.
- `docs/INTENT_CLASSIFIER_DESIGN.md` — feature engineering, routing, evaluation, and deployment design.

## Honest evaluation note

The seed examples validate the pipeline; they are not a representative benchmark of a diverse video corpus. Classification confidence from the rules baseline is a fixed heuristic score. The trained model's probability scores also require calibration before production claims. Use a held-out, representative labelled query set before reporting accuracy or reliability.
