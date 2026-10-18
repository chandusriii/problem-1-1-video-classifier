from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

DATASET_ID = "aintropy-ai/video-mme-sample10"
QA_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/qa/test.jsonl"


def load_questions(local_path: str | None = None) -> list[dict[str, Any]]:
    """Load question rows from local JSONL or the public Hugging Face dataset."""
    if local_path:
        text = Path(local_path).read_text(encoding="utf-8-sig")
    else:
        request = urllib.request.Request(
            QA_URL, headers={"User-Agent": "LiteVideo-assessment/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sample_indices(frame_count: int, sample_count: int) -> list[int]:
    """Choose uniformly spaced frame indices including both endpoints."""
    if frame_count <= 0:
        return []
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    count = min(frame_count, sample_count)
    if count == 1:
        return [frame_count // 2]
    return [round(i * (frame_count - 1) / (count - 1)) for i in range(count)]


def inspect_videos(rows: list[dict[str, Any]], video_dir: Path) -> tuple[list[str], list[str]]:
    video_ids = sorted({str(row.get("video_id", "")) for row in rows if row.get("video_id")})
    available = [video_id for video_id in video_ids if (video_dir / f"{video_id}.mp4").is_file()]
    return video_ids, available


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Video-MME question metadata and validate local video availability."
    )
    parser.add_argument("--qa", help="Optional local path to qa/test.jsonl.")
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--limit", type=int, default=5, help="Rows to inspect; 0 means all.")
    parser.add_argument("--check-frames", action="store_true", help="Decode a few uniformly sampled frames.")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.frames < 1:
        parser.error("--frames must be >= 1")

    try:
        rows = load_questions(args.qa)
    except Exception as exc:
        print(f"Could not load question metadata: {exc}")
        return 2

    selected = rows if args.limit == 0 else rows[:args.limit]
    video_dir = Path(args.video_dir)
    video_ids, available = inspect_videos(selected, video_dir)
    print(f"Dataset: {DATASET_ID}")
    print(f"Question rows loaded: {len(rows)}")
    print(f"Question rows selected: {len(selected)}")
    print(f"Unique videos referenced: {len(video_ids)}")
    print(f"Matching local videos: {len(available)}")
    print(f"Video directory: {video_dir.resolve()}")

    for row in selected[:3]:
        print(f"- {row.get('question_id')}: {row.get('question', '')}")
        print(f"  video_id={row.get('video_id')}")

    if args.check_frames:
        if not available:
            print("Frame check skipped: no matching local videos were found.")
        else:
            import cv2
            import time

            video_path = video_dir / f"{available[0]}.mp4"
            capture = cv2.VideoCapture(str(video_path))
            started = time.perf_counter()
            try:
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = sample_indices(total_frames, args.frames)
                decoded = 0
                for frame_index in indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, _frame = capture.read()
                    decoded += int(ok)
            finally:
                capture.release()
            print(f"Frame check video: {video_path.name}")
            print(f"Target frames: {len(indices)}; decoded frames: {decoded}")
            print(f"Frame sampling/decoding time: {time.perf_counter() - started:.3f} sec")

    print("Use src/infer.py for model inference; this command is metadata inspection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
