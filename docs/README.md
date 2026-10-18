# LiteVideo: CPU-first Video-MME baseline

## Summary
This submission implements multiple-choice video question answering with SmolVLM2-500M-Video-Instruct. OpenCV samples a fixed number of frames uniformly across each clip; a vision-language model receives the frames and question; outputs include a parsed answer, raw response, and per-question latency.

## Reproducibility
The `run.yaml` smoke-test sequence installs dependencies, runs the unit tests, downloads the Video-MME sample question manifest and only the first referenced MP4, then executes one inference with four frames. Model weights are downloaded from Hugging Face the first time the inference command runs. This sequence is intended to verify the pipeline without downloading the dataset's full ~11.2 GB of videos.

## Initial smoke test
On a Windows CPU-only development machine, one question (ID `020-1`) was run with four sampled frames. Prediction: `A`; provided answer: `A`; generation latency: `124.49 s`; elapsed time including model loading: `126.10 s`. The rationale ("the copper is rusty") was questionable. This result proves that the model invocation can complete locally; it is not a meaningful accuracy estimate.

## Metrics
The inference script records exact-match answer-letter accuracy over valid predictions, invalid outputs, mean/median/p95 generation latency, elapsed time, and skipped questions. Generation latency excludes model loading, while total elapsed time includes it. Comparisons should hold the model and question subset constant while varying the number of sampled frames (e.g. 4, 8, 16).

## Limitations
- Uniform sampling can miss short-lived actions or events.
- CPU inference is slow; the first-question smoke test took about two minutes.
- A single question cannot support a benchmark-level accuracy claim.
- The grader must have network access to the dataset and model host for the configured smoke test.

## Dataset
[aintropy-ai/video-mme-sample10](https://huggingface.co/datasets/aintropy-ai/video-mme-sample10)
