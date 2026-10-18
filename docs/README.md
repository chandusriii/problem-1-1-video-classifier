# LiteVideo — lightweight Video-MME baseline

## Objective
Build a reproducible video-question-answering baseline for AIntropy's 10% Video-MME sample, using a small vision-language model and uniform temporal sampling.

## Implementation
The repository includes a metadata-inspection CLI and a SmolVLM2-500M-Video-Instruct inference entrypoint. The inference entrypoint samples frames uniformly with OpenCV, supplies frames and a multiple-choice prompt to the vision-language model, parses a predicted answer letter, and records raw output and timing.

## Smoke-test observation
A single local test question (ID 020-1, four frames) completed on CPU. The model predicted A, matching the provided answer A. Generation latency was 124.49 seconds; elapsed time including model loading was 126.10 seconds. The generated rationale said “the copper is rusty,” which is questionable and should not be treated as a validated explanation. This one-question result is a smoke test only and is not a meaningful estimate of benchmark accuracy.

## Evaluation protocol
- Exact-match accuracy = correct A–D predictions / valid A–D predictions.
- Generation latency is measured after model loading; total elapsed time is reported separately.
- Compare 4, 8, and 16 frames on the same question subset and model.
- Invalid responses and missing-video questions must be disclosed rather than counted as correct.
- Record CPU/GPU, RAM, software versions, model ID, and question count.

## Dataset and constraints
Dataset: https://huggingface.co/datasets/aintropy-ai/video-mme-sample10
The manifest contains 90 videos and 270 questions; videos total about 11.2 GB. Do not commit dataset videos or model weights.

## Limitations
Uniformly sampled frames can miss brief events. CPU inference is slow (about 124 seconds for the one smoke-test question). Results from one question are not a stable estimate of overall benchmark performance.
