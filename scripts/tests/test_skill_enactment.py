"""Generated-skill enactment and proposer-credit attribution contract."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inner.runtime.harness_runner import (  # noqa: E402
    _initial_message,
    _is_score_eligible,
    _runtime_component_contract,
    _record_skill_loads,
    _required_generated_skills,
    _mounted_skill_playbooks,
)
from inner.runtime.session import BudgetLedger, InnerSession  # noqa: E402


def _session() -> InnerSession:
    session = InnerSession(
        task=SimpleNamespace(initial_program="x = 1"),
        ledger=BudgetLedger(max_evaluator_calls=2),
    )
    session.register_skill("c2-optimization", "./skills/c2-optimization")
    return session


def _load_request(call_id: str = "load-1") -> dict:
    return {
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "id": call_id,
            "name": "LoadSkill",
            "input": {"skill_name": "c2-optimization"},
        }],
    }


def _load_result(call_id: str = "load-1", *, is_error: bool = False) -> dict:
    return {
        "role": "tool",
        "content": [{
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": "loaded" if not is_error else "not found",
            "is_error": is_error,
        }],
    }


def _edit_request() -> dict:
    return {
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "id": "edit-1",
            "name": "edit_solution",
            "input": {"code": "x = 2"},
        }],
    }


class SkillEnactmentTest(unittest.TestCase):
    def test_only_added_or_updated_skills_are_required_for_credit(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp)
            (package / "component_manifest.json").write_text(json.dumps({
                "lineage": {"new_skills": [
                    {"name": "old-playbook", "status": "inherited"},
                    {"name": "new-playbook", "status": "added"},
                    {"name": "revised-playbook", "status": "updated"},
                ]},
            }))
            required = _required_generated_skills(package, {
                "old-playbook": "old",
                "new-playbook": "new",
                "revised-playbook": "revised",
            })
        self.assertEqual(required, ["new-playbook", "revised-playbook"])

    def test_legacy_package_without_lineage_requires_all_mounted_skills(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            required = _required_generated_skills(
                Path(temp), {"old-playbook": "old", "new-playbook": "new"}
            )
        self.assertEqual(required, ["new-playbook", "old-playbook"])

    def test_successful_load_before_edit_is_enacted(self) -> None:
        session = _session()
        _record_skill_loads(
            session,
            [_load_request(), _load_result(), _edit_request()],
        )
        audit = session.skill_audit["c2-optimization"]
        self.assertEqual(audit["loads"], 1)
        self.assertEqual(audit["tool_loads"], 1)
        self.assertEqual(audit["runtime_injections"], 0)
        self.assertEqual(audit["loads_before_first_edit"], 1)
        self.assertEqual(audit["late_loads"], 0)
        self.assertEqual(
            session.skill_participation_issues(["c2-optimization"]), []
        )

    def test_failed_load_is_not_enacted(self) -> None:
        session = _session()
        _record_skill_loads(
            session,
            [_load_request(), _load_result(is_error=True), _edit_request()],
        )
        audit = session.skill_audit["c2-optimization"]
        self.assertEqual(audit["loads"], 0)
        issues = session.skill_participation_issues(["c2-optimization"])
        self.assertEqual(
            issues, ["c2-optimization: mounted but never loaded"]
        )
        self.assertFalse(_is_score_eligible("completed", issues))

    def test_load_after_first_edit_is_too_late_for_credit(self) -> None:
        session = _session()
        _record_skill_loads(
            session,
            [_edit_request(), _load_request(), _load_result()],
        )
        audit = session.skill_audit["c2-optimization"]
        self.assertEqual(audit["loads"], 1)
        self.assertEqual(audit["loads_before_first_edit"], 0)
        self.assertEqual(audit["late_loads"], 1)
        self.assertEqual(
            session.skill_participation_issues(["c2-optimization"]),
            ["c2-optimization: loaded only after program editing began"],
        )

    def test_runtime_injection_enacts_skill_before_edit(self) -> None:
        session = _session()
        session.record_skill_load(
            "c2-optimization",
            before_first_edit=True,
            delivery_mode="runtime_injection",
        )
        audit = session.skill_audit["c2-optimization"]
        self.assertEqual(audit["runtime_injections"], 1)
        self.assertEqual(audit["tool_loads"], 0)
        self.assertEqual(audit["loads"], 0)
        self.assertEqual(
            session.skill_participation_issues(["c2-optimization"]), []
        )

    def test_required_playbook_is_read_from_mounted_skill(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp)
            skill_dir = package / "skills" / "c2-optimization"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("# Exact playbook\nUse steps.")
            playbooks = _mounted_skill_playbooks(
                package / "agent.yaml",
                {"c2-optimization": "./skills/c2-optimization"},
                ["c2-optimization"],
            )
        self.assertEqual(
            playbooks, {"c2-optimization": "# Exact playbook\nUse steps."}
        )

    def test_initial_message_includes_every_required_playbook(self) -> None:
        task = SimpleNamespace(spec="maximize", initial_program="x = 1")
        message = _initial_message(
            task,
            seed_score=0.5,
            seed_valid=1.0,
            budget=2,
            generated_skill_playbooks={
                "c2-optimization": "# C2\nUse steps.",
                "shape-search": "# Shape\nUse splines.",
            },
        )
        self.assertIn("Automatically enacted", message)
        self.assertIn("`c2-optimization`", message)
        self.assertIn("`shape-search`", message)
        self.assertIn("Use steps.", message)
        self.assertIn("Use splines.", message)

    def test_runtime_contract_exposes_exact_component_delivery_semantics(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp)
            (package / "tools").mkdir()
            (package / "middlewares").mkdir()
            (package / "tools" / "edit.tool.yaml").write_text(
                "description: Edit the candidate program.\n"
            )
            (package / "tools" / "shape_probe.tool.yaml").write_text(
                "description: Use when task inputs expose an unknown target shape.\n"
            )
            (package / "middlewares" / "stall_guard.middleware.yaml").write_text(
                "description: Forces a structural restart after repeated stalls.\n"
            )
            (package / "agent.yaml").write_text(
                "tools:\n"
                "  - name: edit_solution\n"
                "    yaml_path: ./tools/edit.tool.yaml\n"
                "    binding: inner.harness.tools.discovery:edit_solution\n"
                "  - name: shape_probe\n"
                "    yaml_path: ./tools/shape_probe.tool.yaml\n"
                "    binding: inner.harness.tools.custom_runtime:custom_tool\n"
                "skills:\n"
                "  - ./skills/discovery-optimization\n"
                "  - ./skills/c2-optimization\n"
                "middlewares:\n"
                "  - import: middlewares.stall_guard:GeneratedMiddleware\n"
                "    params:\n"
                "      stall_rounds: 2\n"
            )
            contract = _runtime_component_contract(
                package / "agent.yaml",
                evaluator_budget=2,
                probe_budget=30,
                generated_skill_playbooks={"c2-optimization": "# C2\nUse steps."},
                current_candidate_skills=["c2-optimization"],
            )

        self.assertIn("Official evaluator budget: 2 call(s).", contract)
        self.assertIn("Approximate probe budget: 30 call(s).", contract)
        self.assertIn("`edit_solution` [CORE]", contract)
        self.assertIn("`shape_probe` [GENERATED, CONDITIONAL]", contract)
        self.assertIn("unknown target shape", contract)
        self.assertIn("`discovery-optimization` [BASE, LOAD REQUIRED]", contract)
        self.assertIn(
            "`c2-optimization` [AUTO-ENACTED, CURRENT CANDIDATE]", contract
        )
        self.assertIn("`stall_guard` [GENERATED, AUTOMATIC", contract)
        self.assertIn('"stall_rounds": 2', contract)
        self.assertIn("Generated-tool rule", contract)
        self.assertIn("Component plan:", contract)

    def test_initial_message_places_contract_before_generated_playbook(self) -> None:
        task = SimpleNamespace(spec="maximize", initial_program="x = 1")
        message = _initial_message(
            task,
            seed_score=0.5,
            seed_valid=1.0,
            budget=2,
            generated_skill_playbooks={"c2-optimization": "# C2\nUse steps."},
            component_contract="<RuntimeComponentContract>exact</RuntimeComponentContract>",
        )
        self.assertLess(
            message.index("<RuntimeComponentContract>"),
            message.index("Automatically enacted proposer-generated skills"),
        )


if __name__ == "__main__":
    unittest.main()
