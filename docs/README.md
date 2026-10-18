# Lightweight Video-Type Classification for Adaptive Chunking

## Problem framing
The goal is not to answer questions about video content. It is to predict a video-level class that lets a downstream indexing system choose an appropriate chunking policy. For a diverse corpus (gameplay, egocentric footage, podcasts, wildlife, sports, and more), the output should include both a class and a calibrated confidence, with an abstain/fallback option for ambiguous or out-of-distribution material.

## Proposed label set and chunking policy
Start with operationally useful coarse classes rather than dozens of narrow topics:

| Class | Signals | Suggested downstream strategy |
|---|---|---|
| `talking_head_podcast` | One/few dominant faces, speech-dominant audio, low visual motion | Speech/VAD boundaries, speaker turns, sentence-level transcript chunks; retain occasional visual keyframes |
| `gameplay_screen` | Screen/UI-like imagery, HUD/text overlays, rapid scene changes and game audio | Scene-change and event-aware chunks; shorter windows around UI/state changes; preserve temporally adjacent context |
| `egocentric_activity` | First-person viewpoint, hand/object interaction, camera motion | Action/event boundaries and short overlapping temporal windows |
| `wildlife_nature` | Animals, outdoor habitats, environmental audio | Shot/scene boundaries plus activity/event segmentation; longer stable-scene chunks where appropriate |
| `sports_action` | Field/court, players, ball/equipment, crowd/commentary | Motion/event boundaries; short chunks around plays and longer context windows for buildup/replay |
| `general_video` | Movies, animation, travel, tutorials, mixed or uncertain content | Hybrid scene-change + transcript segmentation |
| `unknown_mixed` | Low confidence, conflicting cues, unseen domain, missing modalities | Conservative fixed-duration chunks with overlap until further analysis |

These are **proposed operational labels**, not validated classes from the sample Video-MME QA metadata. The sample manifest contains question/task labels, not a labeled video-type classification dataset.

## Lightweight feature engineering

The deployed classifier should derive a compact feature vector rather than run a large vision-language model on every frame:

1. **Temporal visual sampling:** decode at low resolution (e.g. 160–224 pixels on the short side) and sample a small number of frames per fixed time interval. Extract a compact image embedding with a small pretrained image encoder; optionally reduce embeddings to a compact vector offline.
2. **Motion features:** calculate frame-to-frame embedding distance and low-cost optical-flow or image-difference summaries. Aggregate mean, variance, and high-percentile motion; keep both clip-level statistics and coarse temporal bins.
3. **Audio features (when audio is available):** use VAD/speech ratio, RMS energy, coarse spectral statistics, and optionally speaker-turn counts. Do not assume every video has usable audio.
4. **Text features (optional):** use ASR transcript length, speech density, and a small set of keywords/entities. Treat ASR as an optional feature source; transcript content alone should not decide the class.
5. **Aggregation:** concatenate mean, standard deviation, and temporal-bin summaries of visual embeddings and motion/audio features. Normalize numeric features; add explicit missing-modality flags. Avoid retaining raw audio/video beyond the inference window unless the indexing system needs it.

A small Logistic Regression or LightGBM classifier over these features is the preferred first model: low latency, low memory, easy to calibrate, and easy to retrain. The classifier should be benchmarked against a rules-only baseline and a small linear classifier before considering larger models.

## Training and evaluation plan
There is no verified labeled train/validation set for these proposed video-type classes in the provided metadata. Therefore, no classification accuracy is claimed. To train this classifier, create a stratified, manually checked labeled set covering source types, duration buckets, creators/channels, and recording conditions. Split by source video/channel—not by individual frames—to avoid leakage. Keep a held-out out-of-domain set (e.g. unseen channels or categories). Record label definitions and disagreements.

Report macro-F1, per-class precision/recall, balanced accuracy, confusion matrix, calibration (e.g. Brier score or expected calibration error), abstention coverage, feature-extraction latency, classifier latency, and peak memory. Measure on CPU as well as any available accelerator. For chunking, additionally evaluate whether class-conditioned chunks preserve events and reduce boundary errors relative to fixed-duration chunking.

## Deployment design

```text
Incoming video
    │
    ├─ metadata probe (duration, FPS, dimensions, audio presence)
    ├─ low-resolution interval sampler
    ├─ parallel lightweight features
    │    ├─ visual embeddings + temporal changes
    │    ├─ optional motion summary
    │    └─ optional VAD / audio summary / transcript statistics
    ├─ normalized feature vector + missing-modality flags
    ├─ small classifier + confidence calibration
    ├─ confidence / OOD gate
    │    ├─ high confidence → predicted class + chunking policy
    │    └─ low confidence → unknown_mixed + safe fallback chunking
    └─ persist class, confidence, model version, and policy decision in index metadata
```

Run feature extraction as a streaming worker so only a few frames/audio windows reside in memory at once. Cache extracted features by video checksum and model version. Batch independent frame embeddings where possible; keep feature extraction and classification separately timed. Keep the classifier stateless, versioned, and behind a small API or worker queue. Log confidence and downstream chunk-quality signals; retrain only after reviewing class drift and errors. The class predicts a chunking policy, not the final chunk boundaries themselves.

## Current repository prototype
The current code is a **video-question-answering smoke-test baseline**, built with SmolVLM2-500M-Video-Instruct and uniform frame sampling. It is not yet the requested lightweight video-type classifier. One local question (ID `020-1`) matched its answer letter, with 124.49 s generation latency and 126.10 s elapsed time including model loading on the development CPU. The generated explanation was questionable; this single result is not an accuracy estimate for either video QA or video-type classification.

## Limitations and next step
The present sample provides 270 multiple-choice questions across 90 videos, but not ground-truth labels for the proposed chunking classes. The next necessary step is to label a representative subset by video type and implement a compact feature baseline. Until that data exists, a rigorous supervised classifier and credible classification score cannot be reported.
