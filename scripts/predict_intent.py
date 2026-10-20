"""Run intent/scope/complexity classification for a video query."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Allow direct execution as "python scripts/predict_intent.py" from repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intent_classifier import predict_intent

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Video-related user query")
    parser.add_argument("--model", help="Optional .joblib artifact; rules used otherwise")
    args = parser.parse_args()
    print(json.dumps(predict_intent(args.query, args.model), indent=2))

if __name__ == "__main__":
    main()
