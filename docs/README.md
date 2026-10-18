# LiteVideo — Video-MME lightweight baseline

## Goal
Build a fast, reproducible video-understanding pipeline for AIntropy's Video-MME 10% sample, measuring accuracy, inference latency and model size.

## Current status
The repository entrypoint currently inspects question metadata and checks matching local MP4 files. A local preprocessing experiment successfully extracted 8 uniformly sampled frames in 0.257 s and 16 frames in 0.554 s from one 104.1-second 1280x720 video on Windows. These are preprocessing-only measurements, not model inference results.

## Evaluation metrics
- Exact-match multiple-choice accuracy: correctly predicted answer letters / evaluated questions.
- Median and p95 model-generation latency, excluding model initialization.
- Mean inference latency.
- Model weight size on disk and parameter-weight memory estimate, reported separately.
- Peak process memory and frames processed per question.
- Frame preprocessing time measured separately from inference.

## Planned experiments
Compare 4, 8, and 16 uniformly sampled frames using the same question/video subset and same model. Report overall and per-duration performance where sample size permits. Include model ID, software versions, and hardware in the report.

## Dataset
https://huggingface.co/datasets/aintropy-ai/video-mme-sample10

The dataset manifest reports 90 videos and 270 multiple-choice questions. Videos total about 11.2 GB, so download only necessary clips for development and do not commit video assets.

## Limitations
No model accuracy claims are made until inference has actually run. Uniform sampling can miss short events and may not capture temporal relationships adequately. This repository is an initial scaffold; final submission must include a working, tested inference pipeline and accurate `run.yaml` commands.
