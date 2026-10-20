# User Intent Classifier for Adaptive Video RAG

## Objective
Classify a user's video-related query by requested operation, retrieval scope, and reasoning complexity, then recommend a retrieval strategy. The design follows AdaVideoRAG's general principle of adaptive routing; the taxonomy below is a task-specific extension, not a label set copied verbatim from the paper.

## Output schema
- `intent`: requested operation.
- `scope`: `FRAME`, `CLIP`, `WHOLE_VIDEO`, or `MULTI_SECTION`.
- `complexity`: `L1_DIRECT`, `L2_RETRIEVAL`, or `L3_GRAPH`.
- `confidence`: estimated score, to be calibrated on held-out labelled data.
- `retrieval_strategy`: router destination derived from complexity.
- `fallback`: safe strategy for low confidence.

Scope and complexity stay separate: a whole-video summary can use precomputed global summaries, while a timestamped question may still require temporal reasoning.

## Initial intent taxonomy

| Intent | Meaning | Common scope | Typical strategy |
|---|---|---|---|
| `FRAME_RECOGNITION` | Objects, colors, people, attributes in a frame | FRAME | Direct visual lookup |
| `OCR_FRAME_QUERY` | Text on a sign, screen, caption, or slide | FRAME / CLIP | OCR plus visual evidence |
| `TRANSCRIPT_LOOKUP` | Spoken words or mentions | CLIP | ASR/transcript retrieval |
| `VIDEO_SUMMARY` | Video-wide synopsis or major topics | WHOLE_VIDEO | Global index/summary |
| `EVENT_COUNTING` | Count occurrences in a video | MULTI_SECTION | Retrieve events and aggregate |
| `TEMPORAL_LOCALIZATION` | Find when an event/topic occurs or order events | CLIP / MULTI_SECTION | Temporal index and local context |
| `COMPARISON` | Compare entities, events, or scenes | CLIP / MULTI_SECTION | Retrieve comparison evidence |
| `CAUSAL_REASONING` | Explain why/how an event led to another | MULTI_SECTION | Temporal evidence, optionally graph |
| `CROSS_SECTION_REASONING` | Relate items across distant segments | MULTI_SECTION | Graph retrieval |
| `GENERAL_VIDEO_QUESTION` | Other video-specific requests | CLIP by default | Hybrid search and fallback |

Labels are intentionally coarse and require validation on the actual query distribution. Compound or ambiguous queries should permit more than one retrieval route.

## Feature engineering

### Lightweight text features
The baseline operates on the query alone: it does not decode video, run ASR, or invoke a large language model at classification time.

- Normalize whitespace while preserving timestamp punctuation.
- Word TF-IDF unigrams/bigrams capture phrases such as “how many times” and “summarize the video”.
- Character TF-IDF n-grams capture spelling variation, compounds, and timestamp-like strings.
- Interpretable cues include timestamp presence, question words, temporal markers (before/after/throughout), counting terms, OCR terms, speech/transcript terms, comparison and causal language, global-scope cues, query length, and clause count.
- Video-side modality availability belongs in index metadata, not query features, unless supplied by the client.

The repository baseline uses transparent rules. An optional trainable baseline uses word+character TF-IDF and independent Logistic Regression models for intent, scope, and complexity. It is CPU-friendly and produces a small model artifact.

Keyword cues are hints, not ground truth. For example, “What happened before the crash?” requires temporal retrieval, while “summarize this ten-second clip” is clip-scoped rather than video-scoped. Training data should include paraphrases, hard negatives, ambiguous cases, and compound queries.

## Routing policy

1. Normalize the query; predict intent, scope, and complexity.
2. Apply confidence/ambiguity checks.
3. **L1:** use the indexed target frame/metadata or a direct model call when enough context is already available.
4. **L2:** hybrid retrieval over transcript/ASR, OCR, captions, and visual embeddings, scoped to relevant timestamps or segments.
5. **L3:** retrieve across sections and use entity/relationship and temporal graphs for global or multi-hop evidence.
6. Deduplicate and rerank evidence, preserve timestamps/chronology, then generate the answer.
7. If confidence is low or predictions conflict, use conservative hybrid retrieval; allow multi-route execution for compound questions.

The classifier recommends a route; it does not guarantee the final answer. Retrieval and generation must be evaluated end to end.

## Training and evaluation
The repository seed examples demonstrate schema and training flow only. They are not representative enough to support accuracy claims across a diverse video corpus.

For production, collect a larger adjudicated set of real and synthetic queries across topics, durations, timestamp patterns, modalities, paraphrases, and difficulty. Use a labelling guide and review a subset. Split by source video and query-template family to reduce leakage. Report per-axis macro-F1, per-class precision/recall, confusion matrices, calibration/Brier score, routing regret, end-to-end answer support, CPU latency, throughput, memory, and artifact size. Compare against always-L1, always-L2, always-L3, and rules-only baselines. No accuracy should be claimed until tested on a representative held-out set.

## Deployment
Serve the serialized vectorizer and small classifiers in a stateless API or worker. A request performs normalization, sparse vectorization, three linear predictions, confidence handling, and returns routing JSON. Version the artifact and schema; log model version and outcomes; sample low-confidence cases for annotation; avoid retaining sensitive query text without approval. Retrain offline, evaluate before rollout, monitor latency/error rates, and keep a rollback path to the rules baseline or previous model. The classifier should be much cheaper than the retrieval and generation stages.

## Limitations
The paper motivates adaptive routing but does not validate these newly proposed labels. Rules may fail on paraphrases; seed examples do not represent the full corpus; confidence is not calibrated until evaluated. The downstream system must still verify retrieved evidence and timestamps.
