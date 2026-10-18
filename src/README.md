from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

DATASET_ID = "aintropy-ai/video-mme-sample10"
QA_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/qa/test.jsonl"


def load_questions(local_path: str | None = None) -> list[dict]:
    if local_path:
        text = Path(local_path).read_text(encoding="utf-8-sig")
    else:
        request = urllib.request.Request(
            QA_URL, headers={"User-Agent": "LiteVideo-assessment/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Video-MME question metadata and validate local video availability."
    )
    parser.add_argument("--qa", help="Optional path to a local qa/test.jsonl file.")
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--limit", type=int, default=5, help="Rows to inspect; 0 means all.")
    parser.add_argument("--help-task", action="store_true")
    args = parser.parse_args()

    if args.help_task:
        print("Stage 1: load question metadata and check local MP4 files.")
        print("Next stage: frame sampling and model inference.")
        return 0

    rows = load_questions(args.qa)
    if args.limit:
        rows = rows[:args.limit]
    video_dir = Path(args.video_dir)
    unique_ids = sorted({row.get("video_id", "") for row in rows if row.get("video_id")})
    available = [
        video_id for video_id in unique_ids
        if (video_dir / f"{video_id}.mp4").is_file()
    ]

    print(f"Dataset: {DATASET_ID}")
    print(f"Questions inspected: {len(rows)}")
    print(f"Unique videos referenced: {len(unique_ids)}")
    print(f"Matching local videos: {len(available)}")
    print(f"Video directory: {video_dir.resolve()}")
    for row in rows[:3]:
        print(f"- {row.get('question_id')}: {row.get('question', '')}")
        print(f"  video_id={row.get('video_id')} answer={row.get('answer')}")
    if unique_ids and not available:
        print("No matching MP4s found. Place videos at data/videos/<video_id>.mp4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
