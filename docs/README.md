# LiteVideo — lightweight Video-MME baseline

## Objective
Build a reproducible video-question-answering baseline for AIntropy's 10% Video-MME sample using a small vision-language model and uniform temporal sampling.

## Implementation
The repository contains a metadata inspection CLI and an inference entrypoint based on SmolVLM2-500M-Video-Instruct. The inference script extracts uniformly spaced frames using OpenCV, supplies the frames plus a multiple-choice question to the model, parses an A–D prediction, and saves raw model output and timing metrics.

## Initial smoke-test result
One question (ID `020-1`) was run locally using four sampled frames and CPU inference. Prediction: A. Dataset answer: A. Generation latency: 124.49 seconds. Elapsed time including model loading: 126.10 seconds. This is a single-question smoke test, not a reliable estimate of benchmark accuracy. The model's generated explanation (“the copper is rusty”) was questionable, so only the matched answer letter is counted as correct for this test.

## Evaluation protocol
- Exact-match accuracy = correct A–D predictions / valid A–D predictions.
- Generation latency excludes model loading; total elapsed time is reported separately.
- Compare 4, 8, and 16 frames on the same question subset and model.
- Report skipped questions and invalid answers; do not count them as correct.
- Record hardware, software versions, model ID, frame count, and number of questions.

## Dataset
Dataset: https://huggingface.co/datasets/aintropy-ai/video-mme-sample10
The sample manifest contains 90 videos and 270 multiple-choice questions. The videos total about 11.2 GB. Dataset media is not committed to this repository. The evaluator needs the QA manifest and relevant video MP4 files available locally.

## Limitations
Uniform temporal sampling can miss brief actions or events. CPU inference is slow; the first smoke-test question took about two minutes including model loading. Results from one question do not represent overall benchmark performance.
