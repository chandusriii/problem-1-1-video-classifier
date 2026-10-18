from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DATASET_ID = "aintropy-ai/video-mme-sample10"
QA_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/qa/test.jsonl"


def load_questions(path: str | None) -> list[dict[str, Any]]:
    if path:
        text = Path(path).read_text(encoding="utf-8-sig")
    else:
        request = urllib.request.Request(QA_URL, headers={"User-Agent": "LiteVideo-assessment/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            text = response.read().decode("utf-8-sig")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sample_video_frames(video_path: Path, count: int) -> list[Image.Image]:
    """Decode uniformly spaced video frames and return RGB PIL images."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")
        indices = ([total_frames // 2] if count == 1 else
                   [round(i * (total_frames - 1) / (count - 1)) for i in range(count)])
        frames = []
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


def parse_answer_letter(text: str) -> str:
    """Parse a leading or explicitly labeled A-D answer; otherwise return '?'."""
    patterns = (
        r"(?i)^\s*(?:answer\s*[:\-]\s*)?\(?([A-D])\)?(?:[\s.)]|$)",
        r"(?i)\banswer\s*[:\-]\s*\(?([A-D])\)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text.strip())
        if match:
            return match.group(1).upper()
    return "?"


def format_question(row: dict[str, Any]) -> str:
    options = row.get("options", {})
    if isinstance(options, dict):
        option_lines = [f"{key}. {value}" for key, value in options.items()]
    elif isinstance(options, list):
        option_lines = [f"{chr(ord('A') + i)}. {value}" for i, value in enumerate(options)]
    else:
        raise ValueError("Unsupported multiple-choice options format.")
    return (
        "Answer using visual evidence in the video frames. Choose exactly one letter A, B, C, or D "
        "and give a brief reason.\n\n"
        f"Question: {row.get('question', '')}\n"
        + "\n".join(option_lines)
        + "\n\nRespond as: Answer: X. Reason: ..."
    )


def predict(model: Any, processor: Any, video_path: Path, row: dict[str, Any], frame_count: int) -> dict[str, Any]:
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
        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    latency = time.perf_counter() - started
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return {
        "question_id": row.get("question_id"),
        "video_id": row.get("video_id"),
        "prediction": parse_answer_letter(output_text),
        "raw_output": output_text,
        "gold_answer": str(row.get("answer", "")).strip().upper(),
        "latency_seconds": round(latency, 4),
        "frames_decoded": len(frames),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SmolVLM2 multiple-choice inference over sampled video frames.")
    parser.add_argument("--qa", help="Path to qa/test.jsonl; downloads metadata if omitted.")
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--limit", type=int, default=1, help="Number of questions; 0 means all.")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    args = parser.parse_args()
    if args.limit < 0 or args.frames < 1:
        parser.error("--limit must be >= 0 and --frames must be >= 1")
    questions = load_questions(args.qa)
    selected = questions if args.limit == 0 else questions[:args.limit]
    if not selected:
        raise SystemExit("No question rows found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}\nDevice: {device}")
    print("Loading model and processor; model files download on first run.")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.float32 if device == "cpu" else torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correct = valid = skipped = 0
    latencies = []
    run_started = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(selected, 1):
            video_path = Path(args.video_dir) / f"{row.get('video_id', '')}.mp4"
            if not video_path.is_file():
                print(f"[{index}/{len(selected)}] SKIP missing video: {video_path}")
                skipped += 1
                continue
            try:
                result = predict(model, processor, video_path, row, args.frames)
            except Exception as exc:
                print(f"[{index}/{len(selected)}] ERROR {row.get('question_id')}: {exc}")
                skipped += 1
                continue
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            prediction, gold = result["prediction"], result["gold_answer"]
            if prediction in {"A", "B", "C", "D"}:
                valid += 1
                correct += int(prediction == gold)
            latencies.append(result["latency_seconds"])
            print(f"[{index}/{len(selected)}] {result['question_id']}: predicted={prediction}, gold={gold}, latency={result['latency_seconds']:.2f}s")
            print(f"  Model output: {result['raw_output']}")
    elapsed = time.perf_counter() - run_started
    summary = {
        "model": args.model,
        "device": device,
        "questions_requested": len(selected),
        "valid_predictions": valid,
        "correct": correct,
        "accuracy": round(correct / valid, 4) if valid else None,
        "mean_generation_latency_seconds": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "elapsed_seconds_including_model_load": round(elapsed, 4),
        "skipped": skipped,
        "frames_per_question": args.frames,
    }
    summary_path = output_path.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Predictions saved to: {output_path.resolve()}")
    print(f"Summary saved to: {summary_path.resolve()}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
