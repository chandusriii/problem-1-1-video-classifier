from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

DATASET_ROOT = "https://huggingface.co/datasets/aintropy-ai/video-mme-sample10/resolve/main"
DEFAULT_QA_PATH = Path("data/qa/test.jsonl")
DEFAULT_VIDEO_DIR = Path("data/videos")


def download_file(url: str, destination: Path, timeout: int = 120) -> None:
    """Download a public dataset file to a destination using a temporary file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "LiteVideo-assessment/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_qa(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    required = {"video_id", "question_id", "question", "options", "answer"}
    if not rows:
        raise ValueError(f"QA manifest is empty: {path}")
    for index, row in enumerate(rows, start=1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Row {index} is missing fields: {sorted(missing)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the Video-MME QA manifest and selected question videos."
    )
    parser.add_argument("--qa-path", default=str(DEFAULT_QA_PATH))
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--limit", type=int, default=1, help="Number of first questions whose videos are required; 0 means all unique videos.")
    parser.add_argument("--force", action="store_true", help="Redownload existing files.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")

    qa_path = Path(args.qa_path)
    if args.force or not qa_path.is_file():
        print(f"Downloading QA manifest to {qa_path}...")
        download_file(f"{DATASET_ROOT}/qa/test.jsonl", qa_path)
    rows = load_qa(qa_path)
    selected = rows if args.limit == 0 else rows[:args.limit]
    video_ids = list(dict.fromkeys(str(row["video_id"]) for row in selected))
    video_dir = Path(args.video_dir)
    for index, video_id in enumerate(video_ids, start=1):
        destination = video_dir / f"{video_id}.mp4"
        if destination.is_file() and not args.force:
            print(f"[{index}/{len(video_ids)}] Exists: {destination.name}")
            continue
        url = f"{DATASET_ROOT}/videos/{video_id}.mp4"
        print(f"[{index}/{len(video_ids)}] Downloading {video_id}...")
        download_file(url, destination)
        print(f"  Saved {destination.stat().st_size:,} bytes")
    print(f"QA rows: {len(rows)}; videos prepared: {len(video_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
