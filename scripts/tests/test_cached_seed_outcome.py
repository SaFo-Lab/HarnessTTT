"""A hash-bound outer-loop score can initialize an inner NexAU session."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inner.evaluation.eval_runner import EvalOutcome  # noqa: E402
from inner.runtime.session import BudgetLedger, InnerSession  # noqa: E402


class CachedSeedOutcomeTest(unittest.TestCase):
    def test_reuse_initializes_best_without_spending_budget(self) -> None:
        session = InnerSession(
            task=SimpleNamespace(initial_program="x = 1"),
            ledger=BudgetLedger(max_evaluator_calls=1),
        )
        outcome = EvalOutcome(
            combined_score=3.5,
            validity=1.0,
            error=None,
            wall_s=9.0,
            metrics={"quality": 7.0},
        )

        observed = session.seed_baseline_from_outcome(outcome)

        self.assertIs(observed, outcome)
        self.assertEqual(session.best_score, 3.5)
        self.assertEqual(session.best_program, "x = 1")
        self.assertEqual(session.best_metrics, {"quality": 7.0})
        self.assertEqual(session.ledger.evaluator_calls, 0)
        self.assertEqual(session.ledger.sandbox_seconds, 0.0)
        self.assertEqual(len(session.history), 1)
        self.assertEqual(session.history[0].kind, "seed")
        self.assertTrue(session.history[0].is_new_best)

    def test_invalid_cached_seed_is_not_promoted(self) -> None:
        session = InnerSession(
            task=SimpleNamespace(initial_program="x = 1"),
            ledger=BudgetLedger(max_evaluator_calls=1),
        )
        outcome = EvalOutcome(99.0, 0.0, "invalid", 1.0, metrics={})

        session.seed_baseline_from_outcome(outcome)

        self.assertEqual(session.best_score, float("-inf"))
        self.assertEqual(session.best_program, "")
        self.assertFalse(session.history[0].is_new_best)
        self.assertEqual(session.ledger.evaluator_calls, 0)


if __name__ == "__main__":
    unittest.main()
