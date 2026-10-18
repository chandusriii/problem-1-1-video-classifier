# LiteVideo: CPU-first Video-MME baseline

## Summary
This submission implements multiple-choice video question answering with SmolVLM2-500M-Video-Instruct. OpenCV samples a fixed number of frames uniformly across each clip. The model receives the sampled frames and question; predictions are written alongside raw output and per-stage timings.

## Reproducibility
The `run.yaml` sequence installs dependencies, runs unit tests, downloads the QA manifest and only the first referenced MP4, then performs one inference with four frames. It does not download the complete ~11.2 GB video collection. Model weights are downloaded from Hugging Face on first use.

## Initial smoke-test result
On a Windows CPU-only development machine, one question (ID `020-1`) was run with four sampled frames. Prediction: `A`; provided answer: `A`; generation latency: `124.49 s`; elapsed time including model loading: `126.10 s`. The rationale ("the copper is rusty") was questionable. This result demonstrates that the model call completed; it is not a meaningful accuracy estimate.

## Metrics
The inference script reports valid-answer accuracy, exact-match accuracy over all requested questions, valid-prediction coverage, invalid/skipped questions, frame decoding time, processor time, model generation latency (mean/median/p95), model loading time, and total elapsed time. Comparisons should hold the model and question subset constant while varying frame counts (e.g. 4, 8, and 16).

## Limitations
- Uniform sampling can miss short-lived actions or events.
- CPU inference is slow; the initial smoke test took about two minutes including loading.
- One question cannot support a benchmark-level accuracy claim.
- The evaluator needs network access to the dataset and model host.

## Dataset
[aintropy-ai/video-mme-sample10](https://huggingface.co/datasets/aintropy-ai/video-mme-sample10)
