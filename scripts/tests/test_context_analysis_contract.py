from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from outer.rounds.analysis import run_analysis  # noqa: E402


class ContextAnalysisContractTest(unittest.TestCase):
    def test_post_cold_context_uses_exactly_two_read_only_specialists(self) -> None:
        calls = []

        def chat(system: str, user: str) -> str:
            calls.append((system, user))
            if "performance analyzer" in system:
                return (
                    '{"findings":[{"finding":"candidate 1 had positive delta",'
                    '"confidence":"high"}],"regressions_or_noops":[],"uncertainties":[]}'
                )
            return (
                '{"tested_axes":["system prompt"],"invalid_patterns":[],'
                '"unexplored_directions":["sampling"]}'
            )

        feedback = {
            "round": 1,
            "base_score": 0.9,
            "best_score": 0.91,
            "candidates": [{"k": 0, "score": 0.9}, {"k": 1, "score": 0.91}],
        }
        brief = run_analysis(feedback, chat=chat, want_perf=True, want_design=True)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("tool" in system.lower() for system, _ in calls))
        self.assertIn("## Analysis brief", brief)
        self.assertIn("MEASURED", brief)
        self.assertIn("UNEXPLORED DIRECTIONS", brief)

    def test_cold_round_without_feedback_makes_no_analyzer_call(self) -> None:
        calls = []
        brief = run_analysis({}, chat=lambda *args: calls.append(args) or "{}")
        self.assertEqual(brief, "")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
