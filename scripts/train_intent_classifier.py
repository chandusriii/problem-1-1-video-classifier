"""Train lightweight independent intent, scope, and complexity classifiers."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from src.intent_classifier import load_jsonl_dataset

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Labelled JSONL dataset")
    parser.add_argument("--output", required=True, help="Output .joblib artifact path")
    args = parser.parse_args()
    rows = load_jsonl_dataset(args.data)
    texts = [row["query"] for row in rows]
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=30000, sublinear_tf=True)),
    ])
    matrix = vectorizer.fit_transform(texts)
    classifiers = {}
    for axis in ("intent", "scope", "complexity"):
        labels = [row[axis] for row in rows]
        if len(set(labels)) < 2:
            raise ValueError(f"need at least two distinct {axis} labels for training")
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(matrix, labels)
        classifiers[axis] = model
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifiers": classifiers, "label_schema_version": 1}, output)
    print(f"Saved artifact: {output}")
    print(f"Training examples: {len(rows)}")
    print("This command trains the model but does not evaluate it on a held-out set.")

if __name__ == "__main__":
    main()
