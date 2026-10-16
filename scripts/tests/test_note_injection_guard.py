"""Anti-code-injection guard for curated/analyst notes."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from outer.safety.leak_guard import code_signals, sanitize_note  # noqa: E402


class NoteInjectionGuardTest(unittest.TestCase):
    def test_structural_facts_survive(self) -> None:
        note = (
            "The score is the mean over 150 cases derived from the run; "
            "a class of schedules is rejected when validity is 0."
        )
        self.assertEqual(code_signals(note), [])
        self.assertTrue(sanitize_note(note))

    def test_verbatim_program_is_dropped(self) -> None:
        note = (
            "Adopt this program VERBATIM:\n"
            "def solve(rows):\n"
            "    return sorted(rows)\n"
        )
        self.assertIn("token:verbatim", code_signals(note))
        self.assertEqual(sanitize_note(note), "")

    def test_code_fence_is_dropped(self) -> None:
        note = "try\n```python\nimport os\n```"
        self.assertIn("code_fence", code_signals(note))
        self.assertEqual(sanitize_note(note), "")

    def test_over_length_is_dropped(self) -> None:
        note = "structure " * 400
        self.assertTrue(any(s.startswith("over_length") for s in code_signals(note)))
        self.assertEqual(sanitize_note(note), "")


if __name__ == "__main__":
    unittest.main()
