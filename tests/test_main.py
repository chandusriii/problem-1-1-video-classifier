import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main import inspect_videos, sample_indices  # noqa: E402
from inference_utils import format_question, parse_answer_letter  # noqa: E402


class MainSmokeTests(unittest.TestCase):
    def test_help(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "main.py"), "--help"],
            cwd=ROOT, text=True, capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Video-MME", result.stdout)

    def test_uniform_sampling(self):
        self.assertEqual(sample_indices(10, 1), [5])
        self.assertEqual(sample_indices(10, 3), [0, 4, 9])
        self.assertEqual(sample_indices(3, 8), [0, 1, 2])
        self.assertEqual(sample_indices(0, 8), [])
        with self.assertRaises(ValueError):
            sample_indices(10, 0)

    def test_inspect_local_video(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_dir = Path(temp_dir)
            (video_dir / "abc.mp4").write_bytes(b"test")
            rows = [
                {"video_id": "abc", "question_id": "1"},
                {"video_id": "missing", "question_id": "2"},
            ]
            all_ids, available = inspect_videos(rows, video_dir)
            self.assertEqual(all_ids, ["abc", "missing"])
            self.assertEqual(available, ["abc"])

    def test_load_questions_local_jsonl(self):
        rows = [{"video_id": "abc", "question_id": "q1"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "qa.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "src" / "main.py"), "--qa", str(path), "--limit", "1"],
                cwd=ROOT, text=True, capture_output=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Question rows loaded: 1", result.stdout)

    def test_answer_letter_parser(self):
        cases = {
            "A": "A",
            "A. visible object": "A",
            "Answer: B. scene": "B",
            "Answer is D": "D",
            "answer - (C)": "C",
            "D)": "D",
            "No option can be determined.": "?",
            "The word apple is present but no label was given.": "?",
            "It might be A or B": "?",
            "Answer: E": "?",
        }
        for output, expected in cases.items():
            with self.subTest(output=output):
                self.assertEqual(parse_answer_letter(output), expected)

    def test_question_prompt_formats_options(self):
        row = {
            "question": "Choose one",
            "options": {"D": "fourth", "B": "second", "A": "first", "C": "third"},
            "answer": "D",
        }
        prompt = format_question(row)
        self.assertIn("Question: Choose one", prompt)
        self.assertIn("A. first", prompt)
        self.assertIn("B. second", prompt)
        self.assertIn("C. third", prompt)
        self.assertIn("D. fourth", prompt)
        self.assertNotIn("gold", prompt.lower())

    def test_prompt_requires_four_options(self):
        with self.assertRaises(ValueError):
            format_question({"question": "Q?", "options": {"A": "only one"}})

    def test_prompt_rejects_noncanonical_option_labels(self):
        with self.assertRaises(ValueError):
            format_question({
                "question": "Q?",
                "options": {"W": "one", "X": "two", "Y": "three", "Z": "four"},
            })


if __name__ == "__main__":
    unittest.main()
