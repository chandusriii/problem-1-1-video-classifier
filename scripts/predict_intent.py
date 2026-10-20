"""Run intent/scope/complexity classification for a video query."""
from __future__ import annotations
import argparse
import json
from src.intent_classifier import predict_intent

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Video-related user query")
    parser.add_argument("--model", help="Optional .joblib artifact; rules used otherwise")
    args = parser.parse_args()
    print(json.dumps(predict_intent(args.query, args.model), indent=2))

if __name__ == "__main__":
    main()
