from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from inference_utils import format_question, parse_answer_letter

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DATASET_ID = "aintropy-ai/video-mme-sample10"
QA_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/qa/test.jsonl"


def load_questions(path: str | None) -> list[dict[str, Any]]:
    """Read JSONL question metadata from disk or the public dataset URL."""
    if path:
        text = Path(path).read_text(encoding="utf-8-sig")
    else:
        request = urllib.request.Request(QA_URL, headers={"User-Agent": "LiteVideo-assessment/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8-sig")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    required = {"video_id", "question_id", "question", "options", "answer"}
    for row_number, row in enumerate(rows, 1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Question row {row_number} is missing fields: {sorted(missing)}")
    return rows


def sample_video_frames(video_path: Path, count: int) -> list[Image.Image]:
    """Decode uniformly spaced RGB frames while keeping memory use bounded."""
    if count < 1:
        raise ValueError("Frame count must be positive.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")
        indices = (
            [total_frames // 2]
            if count == 1
            else [round(i * (total_frames - 1) / (count - 1)) for i in range(count)]
        )
        frames: list[Image.Image] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if not frames:
            raise RuntimeError(f"Could not decode any frames from: {video_path}")
        return frames
    finally:
        capture.release()


def predict(
    model: Any,
    processor: Any,
    video_path: Path,
    row: dict[str, Any],
    frame_count: int,
    max_new_tokens: int = 48,
) -> dict[str, Any]:
    frames = sample_video_frames(video_path, frame_count)
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": frame} for frame in frames],
            {"type": "text", "text": format_question(row)},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    latency = time.perf_counter() - started
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return {
        "question_id": row["question_id"],
        "video_id": row["video_id"],
        "prediction": parse_answer_letter(output_text),
        "raw_output": output_text,
        "gold_answer": str(row["answer"]).strip().upper(),
        "latency_seconds": round(latency, 4),
        "frames_decoded": len(frames),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SmolVLM2 multiple-choice inference on the Video-MME sample."
    )
    parser.add_argument("--qa", help="Local path to qa/test.jsonl; downloads if omitted.")
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--limit", type=int, default=1, help="Questions to run; 0 means all.")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument("--continue-on-error", action="store_true", help="Skip failed questions and continue.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.frames < 1:
        parser.error("--frames must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")

    questions = load_questions(args.qa)
    selected = questions if args.limit == 0 else questions[:args.limit]
    if not selected:
        raise SystemExit("No question rows found.")

    missing = [
        str(row["video_id"]) for row in selected
        if not (Path(args.video_dir) / f"{row['video_id']}.mp4").is_file()
    ]
    if missing and not args.continue_on_error:
        raise SystemExit(
            f"{len(missing)} required video(s) are missing (first: {missing[0]}). "
            f"Download required MP4 files to {Path(args.video_dir).resolve()} or pass --continue-on-error."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Questions selected: {len(selected)}; frames/question: {args.frames}")
    print("Loading model and processor; weights download from Hugging Face on first use.")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.float32 if device == "cpu" else torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    skipped = 0
    run_started = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(selected, start=1):
            video_path = Path(args.video_dir) / f"{row['video_id']}.mp4"
            if not video_path.is_file():
                print(f"[{index}/{len(selected)}] SKIP missing video: {video_path}")
                skipped += 1
                continue
            try:
                result = predict(model, processor, video_path, row, args.frames, args.max_new_tokens)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(f"[{index}/{len(selected)}] ERROR {row['question_id']}: {exc}")
                skipped += 1
                continue
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            predictions.append(result)
            print(
                f"[{index}/{len(selected)}] {result['question_id']}: "
                f"predicted={result['prediction']}, gold={result['gold_answer']}, "
                f"latency={result['latency_seconds']:.2f}s"
            )

    latencies = [float(result["latency_seconds"]) for result in predictions]
    valid = [r for r in predictions if r["prediction"] in {"A", "B", "C", "D"}]
    correct = sum(r["prediction"] == r["gold_answer"] for r in valid)
    elapsed = time.perf_counter() - run_started
    sorted_latencies = sorted(latencies)
    p95_index = max(0, int(0.95 * len(sorted_latencies) + 0.999999) - 1) if sorted_latencies else None
    summary = {
        "model": args.model,
        "device": device,
        "questions_requested": len(selected),
        "questions_completed": len(predictions),
        "valid_predictions": len(valid),
        "invalid_predictions": len(predictions) - len(valid),
        "correct": correct,
        "accuracy_over_valid_predictions": round(correct / len(valid), 4) if valid else None,
        "mean_generation_latency_seconds": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "median_generation_latency_seconds": round(sorted_latencies[len(sorted_latencies)//2], 4) if sorted_latencies else None,
        "p95_generation_latency_seconds": round(sorted_latencies[p95_index], 4) if p95_index is not None else None,
        "elapsed_seconds_including_model_load": round(elapsed, 4),
        "skipped": skipped,
        "frames_per_question": args.frames,
        "max_new_tokens": args.max_new_tokens,
    }
    summary_path = output_path.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Predictions saved to: {output_path.resolve()}")
    print(f"Summary saved to: {summary_path.resolve()}")
    return 0 if predictions else 2


if __name__ == "__main__":
    raise SystemExit(main())
