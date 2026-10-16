"""TOOLSCHEMA-001 contract: H1 tool calls are repaired or refused mechanically.

fair16-v5 failure modes locked here: CP round 4 burned all eight proposer
slots on ``harness_shell`` calls missing the required ``command``; an LLM-SQL
context round lost ``edit_harness_file`` calls to hallucinated ``avg_len`` /
``unique_count`` properties.  The guard drops hallucinated extras (call
proceeds), refuses missing-required calls with the exact contract plus a
literal example, escalates to the full catalog on repetition, and leaves an
attribution trail on the session.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.outer.workspace.propose_session import ProposeSession
from src.outer.proposing.proposer_io import H1_VERSION
from src.outer.safety.tool_schema_guard import (
    check_tool_args,
    full_catalog,
    load_input_schemas,
    refusal_text,
    usage_hint,
)

H1_PACKAGE = Path(__file__).resolve().parents[2] / "src" / "outer" / "harness"


class LoadSchemasTest(unittest.TestCase):
    def test_all_ten_mounted_tools_are_guarded(self):
        schemas = load_input_schemas(H1_PACKAGE)
        self.assertEqual(sorted(schemas), [
            "delete_harness_file", "edit_harness_file", "harness_shell",
            "repair_component", "submit_harness", "upsert_middleware", "upsert_skill", "upsert_tool",
            "validate_harness", "write_harness_file",
        ])

    def test_guarded_schemas_match_the_served_contracts(self):
        schemas = load_input_schemas(H1_PACKAGE)
        self.assertEqual(schemas["harness_shell"].get("required"), ["command"])
        self.assertEqual(schemas["edit_harness_file"].get("required"), ["path"])
        for name, schema in schemas.items():
            self.assertIs(
                schema.get("additionalProperties"), False,
                f"{name} must declare additionalProperties: false",
            )


class CheckToolArgsTest(unittest.TestCase):
    def setUp(self):
        self.schemas = load_input_schemas(H1_PACKAGE)

    def test_wellformed_call_passes_untouched(self):
        decision = check_tool_args(
            "harness_shell", self.schemas["harness_shell"],
            {"command": "cat agent.yaml"},
        )
        self.assertEqual(decision.action, "pass")
        self.assertEqual(decision.repaired_args, {"command": "cat agent.yaml"})

    def test_hallucinated_properties_are_dropped_and_call_proceeds(self):
        # The observed LLM-SQL failure: avg_len / unique_count extras.
        decision = check_tool_args(
            "edit_harness_file", self.schemas["edit_harness_file"],
            {"path": "prompt.md", "append_text": "## hint",
             "avg_len": 12, "unique_count": 3},
        )
        self.assertEqual(decision.action, "repair")
        self.assertEqual(decision.dropped, ["avg_len", "unique_count"])
        self.assertEqual(
            decision.repaired_args,
            {"path": "prompt.md", "append_text": "## hint"},
        )

    def test_missing_required_refuses_with_contract_and_example(self):
        # The observed CP round-4 failure: harness_shell without command.
        decision = check_tool_args(
            "harness_shell", self.schemas["harness_shell"], {},
        )
        self.assertEqual(decision.action, "refuse")
        self.assertEqual(decision.missing, ["command"])
        self.assertIn("REFUSED", decision.message)
        self.assertIn("command (string, REQUIRED)", decision.message)
        self.assertIn('harness_shell(command="cat agent.yaml")', decision.message)

    def test_missing_required_dominates_unknown_extras(self):
        decision = check_tool_args(
            "edit_harness_file", self.schemas["edit_harness_file"],
            {"avg_len": 12},
        )
        self.assertEqual(decision.action, "refuse")
        self.assertEqual(decision.missing, ["path"])
        self.assertIn("avg_len", decision.message)

    def test_structured_component_repair_needs_no_exact_text_patch(self):
        decision = check_tool_args(
            "repair_component", self.schemas["repair_component"], {
                "kind": "tool", "name": "packing_checker",
                "surface": "input_schema",
                "replacement_schema": {"type": "object", "properties": {}},
            },
        )
        self.assertEqual(decision.action, "pass")


class RefusalEscalationTest(unittest.TestCase):
    def setUp(self):
        self.schemas = load_input_schemas(H1_PACKAGE)
        self.decision = check_tool_args(
            "harness_shell", self.schemas["harness_shell"], {},
        )

    def test_first_refusal_is_compact(self):
        text = refusal_text(self.decision, self.schemas, 1, escalate_after=2)
        self.assertNotIn("Complete tool catalog", text)

    def test_repeated_refusals_escalate_to_the_full_catalog(self):
        text = refusal_text(self.decision, self.schemas, 2, escalate_after=2)
        self.assertIn("Complete tool catalog", text)
        for name in ("harness_shell", "edit_harness_file", "validate_harness",
                     "submit_harness"):
            self.assertIn(name, text)

    def test_catalog_lists_every_mounted_tool(self):
        catalog = full_catalog(self.schemas)
        for name in self.schemas:
            self.assertIn(name, catalog)

    def test_usage_hint_marks_required_arguments(self):
        hint = usage_hint("harness_shell", self.schemas["harness_shell"])
        self.assertIn("REQUIRED", hint)


class SessionAttributionTest(unittest.TestCase):
    def test_summary_counts_repairs_refusals_and_escalation(self):
        session = ProposeSession(base_spec={})
        session.record_schema_guard(
            {"action": "repaired", "tool": "edit_harness_file",
             "dropped": ["avg_len"]}
        )
        session.record_schema_guard(
            {"action": "refused", "tool": "harness_shell",
             "missing": ["command"], "consecutive": 1, "escalated": False}
        )
        session.record_schema_guard(
            {"action": "refused", "tool": "harness_shell",
             "missing": ["command"], "consecutive": 2, "escalated": True}
        )
        summary = session.schema_guard_summary()
        self.assertEqual(summary["repaired_calls"], 1)
        self.assertEqual(summary["refused_calls"], 2)
        self.assertIs(summary["escalated"], True)
        self.assertEqual(summary["refusals_by_tool"], {"harness_shell": 2})


class MountAndVersionTest(unittest.TestCase):
    def test_schema_guard_is_mounted_first_in_h1(self):
        agent = yaml.safe_load((H1_PACKAGE / "agent.yaml").read_text())
        imports = [m.get("import", "") for m in agent.get("middlewares") or []]
        self.assertTrue(
            imports and imports[0].endswith("SchemaGuardMiddleware"),
            f"schema guard must wrap the chain first, got: {imports}",
        )
        self.assertTrue(any(i.endswith("SubmitReminderMiddleware") for i in imports))

    def test_h1_version_records_the_guard(self):
        self.assertEqual(H1_VERSION, "v1.1")


if __name__ == "__main__":
    unittest.main()
