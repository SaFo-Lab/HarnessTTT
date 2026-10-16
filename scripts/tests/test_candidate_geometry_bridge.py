"""Trusted current-candidate geometry bridge and budget accounting."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inner.runtime.harness_sdk import MockContext, ToolContext
from inner.runtime.session import BudgetLedger, InnerSession
from inner.evaluation.eval_runner import EvalOutcome


def _program(rows: int = 26) -> str:
    return (
        "def run_packing():\n"
        f"    centers = [[0.5, 0.5] for _ in range({rows})]\n"
        f"    radii = [0.0 for _ in range({rows})]\n"
        "    return centers, radii, sum(radii)\n"
    )


def _task(root: Path, program: str):
    return SimpleNamespace(
        task_id="eft__math__circle_packing",
        initial_program=program,
        task_dir=root,
        shim_path=root,
        language="python",
    )


class CandidateGeometryBridgeTest(unittest.TestCase):
    def test_current_candidate_is_executed_and_metered(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = InnerSession(
                task=_task(root, _program()),
                ledger=BudgetLedger(max_evaluator_calls=2, max_probe_calls=2),
                eval_timeout_s=5.0,
            )
            result = session.candidate_geometry()
            self.assertTrue(result["ok"], result)
            self.assertEqual(len(result["centers"]), 26)
            self.assertEqual(len(result["radii"]), 26)
            self.assertEqual(result["source"], "current_program_sandbox")
            self.assertEqual(session.ledger.probe_calls, 1)
            self.assertEqual(session.ledger.candidate_geometry_calls, 1)
            self.assertEqual(session.ledger.candidate_geometry_failures, 0)
            self.assertEqual(session.ledger.max_executor_model_calls, 36)
            self.assertEqual(session.ledger.max_executor_context_tokens, 131072)
            self.assertEqual(
                session.ledger.max_executor_input_tokens, 36 * 131072
            )
            self.assertEqual(
                session.ledger.max_executor_output_tokens, 36 * 8192
            )
            self.assertGreater(session.ledger.sandbox_seconds, 0.0)
            self.assertEqual(session.history[-1].kind, "candidate_geometry")

    def test_runtime_context_clamp_resolves_the_token_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict(
            "os.environ", {"SAH_MAX_CONTEXT_TOKENS": "49152"}
        ):
            session = InnerSession(
                task=_task(Path(td), _program()), ledger=BudgetLedger()
            )
            self.assertEqual(session.ledger.max_executor_context_tokens, 49152)
            self.assertEqual(
                session.ledger.max_executor_input_tokens, 36 * 49152
            )

    def test_bad_candidate_shape_is_audited_without_evaluator_debit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = InnerSession(
                task=_task(root, _program(rows=25)),
                ledger=BudgetLedger(max_evaluator_calls=2, max_probe_calls=2),
                eval_timeout_s=5.0,
            )
            result = session.candidate_geometry()
            self.assertFalse(result["ok"])
            self.assertIn("shape must be (26, 2)", result["error"])
            self.assertEqual(session.ledger.evaluator_calls, 0)
            self.assertEqual(session.ledger.probe_calls, 1)
            self.assertEqual(session.ledger.candidate_geometry_failures, 1)

    def test_exhausted_probe_budget_prevents_execution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ledger = BudgetLedger(max_evaluator_calls=2, max_probe_calls=0)
            session = InnerSession(task=_task(root, _program()), ledger=ledger)
            result = session.candidate_geometry()
            self.assertFalse(result["ok"])
            self.assertIn("probe budget is exhausted", result["error"])
            self.assertEqual(ledger.candidate_geometry_calls, 0)
            self.assertEqual(ledger.sandbox_seconds, 0.0)

    def test_regular_probe_wall_time_is_in_the_same_sandbox_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ledger = BudgetLedger(max_evaluator_calls=2, max_probe_calls=2)
            session = InnerSession(task=_task(root, _program()), ledger=ledger)
            outcome = EvalOutcome(
                combined_score=0.1, validity=1.0, error=None,
                wall_s=2.75, metrics={},
            )
            with patch("inner.runtime.session.evaluate_program", return_value=outcome):
                session.probe(subsample=26)
            self.assertEqual(ledger.probe_calls, 1)
            self.assertAlmostEqual(ledger.sandbox_seconds, 2.75)

    def test_tool_context_and_mock_expose_only_normalized_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session = InnerSession(
                task=_task(root, _program()),
                ledger=BudgetLedger(max_probe_calls=2),
            )
            context = ToolContext(session, root / "scratch")
            result = context.get_candidate_geometry()
            self.assertTrue(result["ok"])
            mock = MockContext(root / "mock")
            mock_result = mock.get_candidate_geometry()
            self.assertTrue(mock_result["ok"])
            self.assertIn("get_candidate_geometry", mock.calls)


if __name__ == "__main__":
    unittest.main()
