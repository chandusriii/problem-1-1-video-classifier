import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MainSmokeTests(unittest.TestCase):
    def test_help(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "main.py"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Video-MME", result.stdout)


if __name__ == "__main__":
    unittest.main()
