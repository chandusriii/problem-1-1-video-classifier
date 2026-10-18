# Problem 1.1: Video Classifier

This fork contains a CPU-first video-question-answering baseline using SmolVLM2-500M-Video-Instruct. The assessment instructions are preserved below.

## Solution quick links
- Implementation: [`src/infer.py`](src/infer.py)
- Lightweight answer parsing and prompt validation: [`src/inference_utils.py`](src/inference_utils.py)
- Dataset preparation: [`src/prepare_dataset.py`](src/prepare_dataset.py)
- Tests: [`tests/`](tests/)
- Method, measured smoke test, and limitations: [`docs/README.md`](docs/README.md)
- Exact evaluator commands: [`run.yaml`](run.yaml)

## Submission instructions

1. Install the **AIntropy Grader Bot** GitHub App (https://github.com/apps/aintropy-grader-bot) and grant it access to your fork, if you haven't already — we can't evaluate your submission without this.
2. Put your code under `src/`.
3. Put your write-up under `docs/` — markdown, slides, a notebook, or another format of your choosing are all fine, as long as it's committed in this repo.
4. Fill in `run.yaml` with the **exact** commands needed to set up and run your code. We run these commands exactly as written to evaluate your submission — this is the one part of the submission we read literally, not loosely: if `run.yaml` doesn't work, we can't see your code run.
5. Push to your own fork's default branch any time during your test's time limit. No pull request is needed or expected.

Once you start the test you have until your test's time limit to complete the assignment — see your invitation email for the exact duration. We ignore any commits pushed to your repo after that.

Questions? careers@aintropy.ai
