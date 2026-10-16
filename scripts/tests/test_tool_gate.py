"""Mechanical middleware tool gate: enforcement, exemption, auto-lift."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inner.harness.tools import discovery  # noqa: E402
from inner.harness.tools.runtime import session_scope  # noqa: E402
from inner.runtime.session import (  # noqa: E402
    BudgetLedger,
    InnerSession,
    classify_generated_tool_result,
)


def _session() -> InnerSession:
    session = InnerSession(
        task=SimpleNamespace(initial_program="x = 1"),
        ledger=BudgetLedger(max_evaluator_calls=5, max_probe_calls=5),
    )
    session.best_score = 1.0
    return session


class ToolGateTest(unittest.TestCase):
    def test_generated_tool_result_audit_separates_transport_from_semantics(self):
        session = _session()
        session.register_tool("checker", "custom_tools/checker.py")
        session.record_tool_event("checker", "invoked")
        session.record_tool_event("checker", "completed")
        rejected = session.record_tool_result("checker", {
            "ok": False, "reject_reason": "bridge_stale",
            "geometry": {"centers": [[0.1, 0.1]]},
        })
        self.assertFalse(rejected["usable"])
        self.assertTrue(rejected["semantic_rejection"])
        audit = session.tool_audit["checker"]
        self.assertEqual(audit["invocations"], 1)
        self.assertEqual(audit["completed"], 1)
        self.assertEqual(audit["semantic_successful_invocations"], 0)
        self.assertEqual(audit["semantic_rejections"], 1)
        self.assertEqual(audit["reject_reasons"], {"bridge_stale": 1})

        session.record_tool_event("checker", "invoked")
        session.record_tool_event("checker", "completed")
        accepted = session.record_tool_result("checker", {
            "ok": True, "feasible": False, "violations": ["overlap"],
        })
        self.assertTrue(accepted["usable"])
        self.assertTrue(accepted["semantic_success"])
        self.assertEqual(audit["usable_outputs"], 1)
        self.assertEqual(audit["semantic_successful_invocations"], 1)

    def test_only_literal_top_level_ok_true_is_semantic_success(self):
        self.assertFalse(classify_generated_tool_result({"feasible": True})[
            "semantic_success"
        ])
        self.assertFalse(classify_generated_tool_result('{"ok": true}')[
            "semantic_success"
        ])
        self.assertTrue(classify_generated_tool_result({"ok": True})[
            "semantic_success"
        ])
        contradictory = classify_generated_tool_result({
            "ok": True, "error": "stale bridge",
        })
        self.assertFalse(contradictory["semantic_success"])
        self.assertTrue(contradictory["semantic_rejection"])
        disguised_rejection = classify_generated_tool_result({
            "ok": True, "reject_reason": "bridge_stale",
        })
        self.assertFalse(disguised_rejection["semantic_success"])
        self.assertTrue(disguised_rejection["semantic_rejection"])
        self.assertEqual(
            disguised_rejection["reject_reason"], "bridge_stale"
        )
        timed_out = classify_generated_tool_result({
            "ok": True, "status": "timeout",
        })
        self.assertFalse(timed_out["semantic_success"])
        self.assertTrue(timed_out["semantic_rejection"])

    def test_refusal_consumes_no_budget_and_probe_satisfies(self) -> None:
        session = _session()
        session.request_tool_gate("g1", ["probe_solution"])
        with session_scope(session):
            out = discovery.evaluate_solution()
        self.assertIn("GATED by middleware 'g1'", out)
        self.assertEqual(session.ledger.evaluator_calls, 0)
        self.assertIsNotNone(session.tool_gate)
        # satisfying tool clears the gate
        self.assertIsNone(session.check_tool_gate("probe_solution"))
        self.assertIsNone(session.tool_gate)
        gate = session.middleware_audit["g1"]["gate"]
        self.assertEqual(gate["enforced"], 1)
        self.assertEqual(gate["refused"], 1)
        self.assertEqual(gate["satisfied"], 1)
        self.assertEqual(gate["auto_lifted"], 0)

    def test_finish_is_never_gated(self) -> None:
        session = _session()
        session.request_tool_gate("g1", ["probe_solution"])
        self.assertIsNone(session.check_tool_gate("finish"))
        self.assertIsNotNone(session.tool_gate)

    def test_auto_lift_after_repeated_refusals(self) -> None:
        session = _session()
        session.request_tool_gate("g1", ["probe_solution"])
        self.assertIsNotNone(session.check_tool_gate("evaluate_solution"))
        self.assertIsNotNone(session.check_tool_gate("edit_solution"))
        # two refusals -> lifted
        self.assertIsNone(session.tool_gate)
        self.assertIsNone(session.check_tool_gate("evaluate_solution"))
        gate = session.middleware_audit["g1"]["gate"]
        self.assertEqual(gate["auto_lifted"], 1)

    def test_same_gate_reassertion_preserves_refusal_count(self) -> None:
        session = _session()
        session.request_tool_gate("g1", ["probe_solution"])
        self.assertIsNotNone(session.check_tool_gate("evaluate_solution"))
        self.assertEqual(session.tool_gate["refusals"], 1)

        # Simulate the generated before_model hook firing again before the
        # executor's next attempt.  This must not reset the auto-lift clock.
        session.request_tool_gate("g1", ["probe_solution"])
        self.assertEqual(session.tool_gate["refusals"], 1)
        self.assertIsNotNone(session.check_tool_gate("edit_solution"))
        self.assertIsNone(session.tool_gate)
        gate = session.middleware_audit["g1"]["gate"]
        self.assertEqual(gate["enforced"], 1)
        self.assertEqual(gate["refused"], 2)
        self.assertEqual(gate["auto_lifted"], 1)

    def test_competing_middlewares_cannot_overwrite_or_reset_active_gate(self):
        session = _session()
        self.assertTrue(session.request_tool_gate("g1", ["probe_solution"]))
        self.assertIsNotNone(session.check_tool_gate("evaluate_solution"))
        self.assertEqual(session.tool_gate["refusals"], 1)

        # A different hook and a changed requirement from the same hook both
        # lose first-wins arbitration while g1 is active.
        self.assertFalse(session.request_tool_gate("g2", ["edit_solution"]))
        self.assertFalse(session.request_tool_gate("g1", ["edit_solution"]))
        self.assertEqual(session.tool_gate["set_by"], "g1")
        self.assertEqual(session.tool_gate["require"], ("probe_solution",))
        self.assertEqual(session.tool_gate["refusals"], 1)

        self.assertIsNotNone(session.check_tool_gate("edit_solution"))
        self.assertIsNone(session.tool_gate)
        self.assertEqual(session.middleware_audit["g1"]["gate"]["auto_lifted"], 1)
        self.assertEqual(session.middleware_audit["g1"]["gate"]["suppressed"], 1)
        self.assertEqual(session.middleware_audit["g2"]["gate"]["suppressed"], 1)

    def test_invalid_gate_requests_are_rejected(self) -> None:
        session = _session()
        with self.assertRaises(ValueError):
            session.request_tool_gate("g1", [])
        with self.assertRaises(ValueError):
            session.request_tool_gate("g1", ["finish"])
        with self.assertRaises(ValueError):
            session.request_tool_gate("g1", ["not_a_tool"])

    def test_structure_gate_requires_new_ast_family(self) -> None:
        from inner.runtime.program_structure import normalized_ast_digest

        session = _session()
        self.assertEqual(session.ledger.distinct_program_families, 1)
        baseline = normalized_ast_digest(session.current_program)
        session.request_tool_gate(
            "family_switch", ["edit_solution"],
            require_structure_change_from=baseline,
        )
        self.assertIsNone(session.check_tool_gate("edit_solution"))
        same_family = session.apply_edit("x = 2")
        self.assertIn("STRUCTURE GATE NOT SATISFIED", same_family)
        self.assertEqual(session.ledger.distinct_program_families, 1)
        self.assertIsNotNone(session.tool_gate)
        self.assertEqual(session.tool_gate["require_structure_change_from"], baseline)

        self.assertIsNone(session.check_tool_gate("edit_solution"))
        new_family = session.apply_edit("x = [2]")
        self.assertIn("STRUCTURE GATE SATISFIED", new_family)
        self.assertEqual(session.ledger.distinct_program_families, 2)
        self.assertIsNone(session.tool_gate)
        gate = session.middleware_audit["family_switch"]["gate"]
        self.assertEqual(gate["postcondition_failed"], 1)
        self.assertEqual(gate["satisfied"], 1)

    def test_structure_gate_rejects_stale_or_non_edit_baseline(self) -> None:
        from inner.runtime.program_structure import normalized_ast_digest

        session = _session()
        with self.assertRaises(ValueError):
            session.request_tool_gate(
                "g1", ["probe_solution"],
                require_structure_change_from=normalized_ast_digest(
                    session.current_program
                ),
            )
        with self.assertRaises(ValueError):
            session.request_tool_gate(
                "g1", ["edit_solution"],
                require_structure_change_from="stale-family",
            )

    def test_snapshot_exposes_active_gate(self) -> None:
        from inner.harness.middleware.generated_context import GeneratedHookTracker
        session = _session()
        tracker = GeneratedHookTracker()
        session.request_tool_gate("g1", ["probe_solution"])
        state = tracker.snapshot(SimpleNamespace(current_iteration=0), session)
        self.assertEqual(state["active_tool_gate"]["require"], ("probe_solution",))

    def test_exhausted_evaluator_budget_refuses_dead_edit(self) -> None:
        session = _session()
        session.ledger.evaluator_calls = session.ledger.max_evaluator_calls
        original = session.current_program
        with session_scope(session):
            out = discovery.edit_solution("x = 2")
        self.assertIn("evaluation budget is exhausted", out)
        self.assertIn("No edit was staged", out)
        self.assertEqual(session.current_program, original)
        self.assertEqual(session.ledger.edit_calls, 0)

    def test_exhausted_evaluator_budget_refuses_dead_probe(self) -> None:
        session = _session()
        session.ledger.evaluator_calls = session.ledger.max_evaluator_calls
        with session_scope(session):
            out = discovery.probe_solution()
        self.assertIn("evaluation budget is exhausted", out)
        self.assertIn("No probe was run", out)
        self.assertEqual(session.ledger.probe_calls, 0)


if __name__ == "__main__":
    unittest.main()
