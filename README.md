# LiteVideo — Video-Type Classifier for Adaptive Chunking

This project targets fast video-level classification so a downstream video index can select an appropriate chunking policy. The current write-up proposes classes, lightweight feature engineering, a deployment architecture, and an evaluation plan.

> **Implementation status:** The inference code currently contains a SmolVLM2 video-question-answering smoke test. It is not yet the requested trained video-type classifier because the supplied sample metadata does not provide labels for the proposed video-type categories. The write-up distinguishes the design proposal from the implemented, tested code.

## Project files
- [Design, feature engineering, chunking policy, deployment, evaluation plan](docs/README.md)
- [Question metadata inspection CLI](src/main.py)
- [Video QA inference smoke test](src/infer.py)
- [Answer parsing and prompt validation](src/inference_utils.py)
- [Selective dataset preparation](src/prepare_dataset.py)
- [Run commands](run.yaml)
- [Unit tests](tests/)
- [GitHub Actions test workflow](.github/workflows/tests.yml)

## Quick local checks
```bash
python -m unittest discover -s tests -v
python src/main.py --help
```

The full video sample totals approximately 11.2 GB. Dataset preparation downloads only the video files needed for selected questions. The QA smoke test downloads model weights from Hugging Face on first use.

