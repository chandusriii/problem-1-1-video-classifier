import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_dataset import load_qa, download_file


class DatasetPreparationTests(unittest.TestCase):
    def test_load_qa_validates_required_fields(self):
        row = {
            "video_id": "abc",
            "question_id": "q1",
            "question": "What is shown?",
            "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
            "answer": "A",
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "qa.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(load_qa(path)[0]["question_id"], "q1")

    def test_load_qa_rejects_missing_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "qa.jsonl"
            path.write_text(json.dumps({"video_id": "abc"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing fields"):
                load_qa(path)

    def test_load_qa_rejects_empty_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "qa.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_qa(path)

    def test_download_file_cleans_partial_file_on_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "file.bin"
            def broken_download(*args, **kwargs):
                raise OSError("network down")
            with patch("prepare_dataset.urllib.request.urlopen", side_effect=broken_download):
                with self.assertRaises(OSError):
                    download_file("https://example.invalid/file", destination)
            self.assertFalse(destination.with_suffix(".bin.part").exists())


if __name__ == "__main__":
    unittest.main()
