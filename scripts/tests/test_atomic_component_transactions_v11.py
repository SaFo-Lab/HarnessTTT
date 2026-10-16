from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from outer.compiling.materialize import INNER_HARNESS, materialize
from outer.genome import harness_spec as hs
from outer.proposing.propose import maybe_auto_finalize, run_once
from outer.safety.task_profiles import trusted_component_profile
from outer.workspace import h2_workspace as ws
from outer.workspace.propose_session import ProposeSession, get_session


GEOMETRY_TOOL = '''from __future__ import annotations
import math

def run(ctx, args):
    points = args.get("points") or []
    radius = float(args.get("radius", 0.0))
    violations = []
    for i, point in enumerate(points):
        x, y = float(point[0]), float(point[1])
        wall = min(x, y, 1.0 - x, 1.0 - y) - radius
        if wall < 0.0:
            violations.append({"kind": "wall", "i": i, "margin": wall})
        for j in range(i):
            dx = x - float(points[j][0])
            dy = y - float(points[j][1])
            margin = math.hypot(dx, dy) - 2.0 * radius
            if margin < 0.0:
                violations.append({"kind": "pair", "i": i, "j": j,
                                   "margin": margin})
    return {"feasible": not violations, "violations": violations}
'''

GEOMETRY_SCHEMA = {
    "type": "object",
    "properties": {
        "points": {"type": "array", "items": {"type": "array"}},
        "radius": {"type": "number"},
    },
    "required": ["points", "radius"],
}

CP26_SCHEMA = {
    "type": "object",
    "properties": {
        "centers": {
            "type": "array", "minItems": 26, "maxItems": 26,
            "items": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "items": {"type": "number"},
            },
        },
        "radii": {
            "type": "array", "minItems": 26, "maxItems": 26,
            "items": {"type": "number", "minimum": 0},
        },
    },
    "required": ["centers", "radii"],
}

CP26_REAL_TOOL = '''import math

def run(ctx, args):
    centers = args.get("centers")
    radii = args.get("radii")
    if centers is None and radii is None:
        try:
            geometry = ctx.get_candidate_geometry()
        except Exception as exc:
            return {"ok": False, "error": "geometry bridge failed: " + str(exc)}
        if not geometry.get("ok"):
            return {"ok": False, "error": geometry.get("error", "geometry unavailable")}
        centers, radii = geometry["centers"], geometry["radii"]
    elif centers is None or radii is None:
        return {"ok": False, "error": "centers and radii must be provided together"}
    if not isinstance(centers, list) or not isinstance(radii, list):
        return {"ok": False, "error": "lists required"}
    if len(centers) != 26 or len(radii) != 26:
        return {"ok": False, "error": "exactly 26 circles required"}
    try:
        clean_centers = [(float(c[0]), float(c[1])) for c in centers
                         if isinstance(c, list) and len(c) == 2]
        clean_radii = [float(r) for r in radii]
    except Exception:
        return {"ok": False, "error": "finite numeric inputs required"}
    if (len(clean_centers) != 26
            or any(not math.isfinite(x) or not math.isfinite(y)
                   for x, y in clean_centers)
            or any(not math.isfinite(r) or r < 0 for r in clean_radii)):
        return {"ok": False, "error": "finite nonnegative geometry required"}
    if any(abs(x) > 1e100 or abs(y) > 1e100 for x, y in clean_centers) \
            or any(r > 1e100 for r in clean_radii):
        return {"ok": False, "error": "geometry magnitude is unsafe"}
    violations = []
    repair_suggestions = []
    for i, ((x, y), radius) in enumerate(zip(clean_centers, clean_radii)):
        wall_slack = min(x, y, 1.0 - x, 1.0 - y) - radius
        if wall_slack < 0:
            violations.append(["wall", i])
            if radius <= 0.5:
                target_x = min(max(x, radius), 1.0 - radius)
                target_y = min(max(y, radius), 1.0 - radius)
                after = min(target_x, target_y, 1.0-target_x, 1.0-target_y)-radius
                if after > wall_slack:
                    repair_suggestions.append({
                        "kind": "boundary", "circle": i,
                        "suggested_center_shift": [target_x - x, target_y - y],
                    })
                else:
                    repair_suggestions.append({
                        "kind": "boundary", "circle": i, "repairable": False,
                        "reason": "no center-only wall repair improves slack",
                    })
            else:
                repair_suggestions.append({
                    "kind": "boundary", "circle": i, "repairable": False,
                    "reason": "radius exceeds the unit-square inradius",
                })
        for j in range(i):
            dx, dy = x - clean_centers[j][0], y - clean_centers[j][1]
            distance = math.hypot(dx, dy)
            required = radius + clean_radii[j]
            if distance < required:
                violations.append(["pair", j, i])
                ux, uy = ((dx / distance, dy / distance)
                          if distance > 0 else (1.0, 0.0))
                step = (required - distance) / 2.0
                rj = clean_radii[j]
                if radius <= 0.5 and rj <= 0.5:
                    xi = min(max(x + ux * step, radius), 1.0 - radius)
                    yi = min(max(y + uy * step, radius), 1.0 - radius)
                    xj = min(max(clean_centers[j][0] - ux * step, rj), 1.0-rj)
                    yj = min(max(clean_centers[j][1] - uy * step, rj), 1.0-rj)
                    after = math.hypot(xi - xj, yi - yj)
                else:
                    after = distance
                if after > distance:
                    repair_suggestions.append({
                        "kind": "overlap", "pair": [j, i],
                        "suggested_shift_i": [xj-clean_centers[j][0], yj-clean_centers[j][1]],
                        "suggested_shift_j": [xi-x, yi-y],
                    })
                else:
                    repair_suggestions.append({
                        "kind": "overlap", "pair": [j, i], "repairable": False,
                        "reason": "no wall-safe center-only separation exists",
                    })
    return {"ok": True, "feasible": not violations,
            "violations": violations,
            "repair_suggestions": repair_suggestions}
'''

STAGNATION_MIDDLEWARE = '''def before_model(hook_input):
    state = hook_input.get("state", {})
    if (state.get("stalled_evals", 0) >= 3
            and state.get("family_streak", 0) >= 3
            and state.get("last_step_kind") in ("diff", "full_rewrite")
            and not state.get("pending_evaluation", False)
            and state.get("last_edit_staged") is not False
            and not state.get("active_tool_gate")
            and state.get("evals_remaining", 0) > 0):
        return {"note": "NEXT CALL: use edit_solution to replace the repeated family.",
                "require_tools": ["edit_solution"],
                "require_structure_change": True}
    return None
'''

CONSTRUCTION_SKILL = '''# Construction restart

Trigger when valid evaluations repeat one geometric family without improvement.

1. Classify the current construction family.
2. Choose a representation with a different adjacency pattern.
3. Generate one candidate and evaluate it before further refinement.

Use this feasibility expression when checking a proposed pair:

```text
r_i + r_j <= distance(center_i, center_j)
```

Success means the next evaluated program has a different structural family.
'''

CP26_CHECKER_PROFILE = trusted_component_profile(
    "eft__math__circle_packing", "constraint_checker"
)


class AtomicComponentTransactionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = hs.read_base_spec(INNER_HARNESS)

    def _draft(self, root: Path) -> Path:
        draft = root / "h2"
        materialize(
            self.base, draft, meta={"effective": self.base},
            validate_prompt=False,
        )
        return draft

    @staticmethod
    def _enable_explicit_cp26_checker_profile(session: ProposeSession) -> None:
        """Exercise the optional strict checker validator without slot forcing."""

        session.validation_profile = lambda: CP26_CHECKER_PROFILE

    @staticmethod
    def _intent() -> dict:
        return {
            "failure_mode": "Executor evaluates candidates without localized evidence.",
            "hypothesis": "A deterministic diagnostic changes the next repair decision.",
            "expected_behavior_change": "Executor repairs evidence before evaluation.",
            "novelty_basis": "The parent has no equivalent computation or state gate.",
            "risk": "The trigger may add one unnecessary tool call.",
        }

    @staticmethod
    def _bytes(root: Path) -> dict:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()
        }

    def test_tool_transaction_writes_all_surfaces_and_is_submit_ready(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
            )
            result = session.upsert_tool(
                name="geometry_checker", implementation=GEOMETRY_TOOL,
                description="Compute wall and pairwise circle margins.",
                input_schema=GEOMETRY_SCHEMA,
                usage_guidance="Call before evaluating a proposed placement.",
                **self._intent(),
            )
            self.assertTrue(result.startswith("VALID COMPONENT TRANSACTION"), result)
            self.assertEqual(session.validated_revision, session.workspace_revision)
            self.assertTrue((draft / "custom_tools/geometry_checker.py").is_file())
            self.assertTrue((draft / "tools/geometry_checker.tool.yaml").is_file())
            agent = yaml.safe_load((draft / "agent.yaml").read_text())
            self.assertTrue(any(
                row.get("name") == "geometry_checker" for row in agent["tools"]
            ))
            self.assertIn("Generated Tool: geometry_checker",
                          (draft / "prompt.md").read_text())
            self.assertIn("Candidate H2 accepted", session.submit_harness())
            self.assertEqual(
                session.component_transactions[0]["intent"]["failure_mode"],
                self._intent()["failure_mode"],
            )

    def test_skill_transaction_owns_plumbing_and_accepts_markdown_fences(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
            )
            result = session.upsert_skill(
                name="construction-restart",
                body=CONSTRUCTION_SKILL,
                description="Switch construction families: after measured stagnation.",
                usage_guidance="Apply after repeated valid same-family evaluations.",
                **self._intent(),
            )
            self.assertTrue(result.startswith("VALID COMPONENT TRANSACTION"), result)
            skill_path = draft / "skills/construction-restart/SKILL.md"
            self.assertTrue(skill_path.is_file())
            self.assertIn("```text", skill_path.read_text())
            self.assertEqual(
                hs.read_base_spec(draft, verify_provenance=False)["new_skills"][0][
                    "description"
                ],
                "Switch construction families: after measured stagnation.",
            )
            agent = yaml.safe_load((draft / "agent.yaml").read_text())
            self.assertIn("./skills/construction-restart", agent["skills"])
            prompt = (draft / "prompt.md").read_text()
            self.assertEqual(
                prompt.count("# Generated Skill: construction-restart"), 1
            )
            payload = json.loads(result.split("\n", 1)[1])
            self.assertEqual(payload["trusted_normalizations"], [
                "skill_frontmatter_framework_owned",
                "skill_mount_framework_owned",
                "skill_prompt_declaration_framework_owned",
            ])
            self.assertNotIn("spec must be a mapping", result)
            self.assertIn("Candidate H2 accepted", session.submit_harness())

    def test_skill_repair_replaces_semantic_body_without_plumbing_edits(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(base_spec=self.base, draft_dir=draft)
            failed = session.upsert_skill(
                name="construction-restart", body="",
                description="Switch construction families after stagnation.",
                usage_guidance="Apply after repeated same-family evaluations.",
                **self._intent(),
            )
            self.assertTrue(failed.startswith("INVALID COMPONENT TRANSACTION"))
            repaired = session.repair_component(
                kind="skill", name="construction-restart", surface="body",
                replacement_text=CONSTRUCTION_SKILL,
            )
            self.assertTrue(repaired.startswith("VALID COMPONENT TRANSACTION"), repaired)
            self.assertTrue(
                (draft / "skills/construction-restart/SKILL.md").is_file()
            )

    def test_failed_tool_transaction_rolls_back_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            before = self._bytes(draft)
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
            )
            result = session.upsert_tool(
                name="quick_probe", implementation=(
                    "def run(ctx, args):\n    return ctx.probe()\n"
                ),
                description="Relay a probe.",
                input_schema={"type": "object", "properties": {}},
                usage_guidance="Call immediately.", **self._intent(),
            )
            self.assertTrue(result.startswith("INVALID COMPONENT TRANSACTION"))
            self.assertIn("thin wrapper", result)
            self.assertLess(result.index('"errors"'), result.index('"intent_summary"'))
            self.assertLess(len(result), 6000)
            self.assertEqual(before, self._bytes(draft))
            self.assertEqual(session.workspace_revision, 0)

            repeated = session.upsert_tool(
                name="quick_probe", implementation=(
                    "def run(ctx, args):\n    return ctx.probe()\n"
                ),
                description="Relay a probe.",
                input_schema={"type": "object", "properties": {}},
                usage_guidance="Call immediately.", **self._intent(),
            )
            self.assertIn("REPEATED IDENTICAL IMPLEMENTATION", repeated)
            self.assertIn("Previous witness", repeated)

            revised_contract = session.upsert_tool(
                name="quick_probe", implementation=(
                    "def run(ctx, args):\n    return ctx.probe()\n"
                ),
                description="Relay a probe with a revised visible contract.",
                input_schema={"type": "object", "properties": {}},
                usage_guidance="Call only after a staged candidate.",
                **self._intent(),
            )
            self.assertNotIn("REPEATED IDENTICAL IMPLEMENTATION", revised_contract)
            self.assertIn("thin wrapper", revised_contract)
            revised_repeat = session.upsert_tool(
                name="quick_probe", implementation=(
                    "def run(ctx, args):\n    return ctx.probe()\n"
                ),
                description="Relay a probe with a revised visible contract.",
                input_schema={"type": "object", "properties": {}},
                usage_guidance="Call only after a staged candidate.",
                **self._intent(),
            )
            self.assertIn("REPEATED IDENTICAL IMPLEMENTATION", revised_repeat)
            self.assertNotIn(
                "Previous witness: REPEATED IDENTICAL IMPLEMENTATION",
                revised_repeat,
            )

    def test_atomic_tool_preflight_receives_descriptor_semantics(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            before = self._bytes(draft)
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
            )
            result = session.upsert_tool(
                name="geometry_checker", implementation=GEOMETRY_TOOL,
                description=(
                    "Computes the certified global optimum for circle packing."
                ),
                input_schema=GEOMETRY_SCHEMA,
                usage_guidance="Call before evaluating a placement.",
                **self._intent(),
            )
            self.assertTrue(result.startswith("INVALID COMPONENT TRANSACTION"))
            self.assertIn("unsupported aggregate/global optimality claim", result)
            self.assertEqual(before, self._bytes(draft))

    def test_tool_prompt_contract_cannot_claim_certified_global_optimum(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
            )
            result = session.upsert_tool(
                name="geometry_checker", implementation=GEOMETRY_TOOL,
                description="Compute supplied-candidate geometry diagnostics.",
                input_schema=GEOMETRY_SCHEMA,
                usage_guidance=(
                    "Call this certified global optimum oracle before editing."
                ),
                **self._intent(),
            )
            self.assertTrue(
                result.startswith("INVALID COMPONENT TRANSACTION"), result
            )
            self.assertIn("unsupported aggregate/global optimality claim", result)

    def test_atomic_cp26_preflight_runs_adversarial_domain_matrix(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            before = self._bytes(draft)
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
                task_id="eft__math__circle_packing",
            )
            self._enable_explicit_cp26_checker_profile(session)
            result = session.upsert_tool(
                name="packing_checker",
                implementation=(
                    "def run(ctx, args):\n"
                    "    centers = args.get('centers')\n"
                    "    radii = args.get('radii')\n"
                    "    if centers is None and radii is None:\n"
                    "        geometry = ctx.get_candidate_geometry()\n"
                    "        centers = geometry.get('centers') or []\n"
                    "        radii = geometry.get('radii') or []\n"
                    "    return {'ok': True, 'feasible': True, "
                    "'n': len(centers), 'total': sum(radii)}\n"
                ),
                description="Audit a proposed unit-square layout.",
                input_schema=CP26_SCHEMA,
                usage_guidance="Call before evaluating a proposed packing.",
                **self._intent(),
            )
            self.assertTrue(result.startswith("INVALID COMPONENT TRANSACTION"))
            self.assertIn("adversarial CP26 sample", result)
            self.assertEqual(before, self._bytes(draft))

    def test_cp26_preflight_allows_diagnostics_without_repairs(self):
        implementation = CP26_REAL_TOOL.replace(
            ',\n            "repair_suggestions": repair_suggestions', ''
        )
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
                task_id="eft__math__circle_packing",
            )
            self._enable_explicit_cp26_checker_profile(session)
            result = session.upsert_tool(
                name="packing_checker", implementation=implementation,
                description="Audit a proposed unit-square layout.",
                input_schema=CP26_SCHEMA,
                usage_guidance="Return exact feasibility and slack diagnostics.",
                **self._intent(),
            )
            self.assertTrue(result.startswith("VALID COMPONENT TRANSACTION"), result)

    def test_cp26_preflight_validates_repairs_when_emitted(self):
        implementation = CP26_REAL_TOOL.replace(
            '[target_x - x, target_y - y]', '[0.0, 0.0]',
        )
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
                task_id="eft__math__circle_packing",
            )
            self._enable_explicit_cp26_checker_profile(session)
            result = session.upsert_tool(
                name="packing_checker", implementation=implementation,
                description="Audit a proposed unit-square layout.",
                input_schema=CP26_SCHEMA,
                usage_guidance="Return verified repairs before evaluation.",
                **self._intent(),
            )
            self.assertTrue(result.startswith("INVALID COMPONENT TRANSACTION"))
            self.assertIn("does not strictly improve target slack", result)

    def test_repair_component_patches_retained_rollback(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
            )
            failed = session.upsert_tool(
                name="geometry_checker",
                implementation=GEOMETRY_TOOL.replace("import math", "import os"),
                description="Compute wall and pairwise circle margins.",
                input_schema=GEOMETRY_SCHEMA,
                usage_guidance="Call before evaluating a placement.",
                **self._intent(),
            )
            self.assertTrue(failed.startswith("INVALID COMPONENT TRANSACTION"))
            self.assertEqual(session.workspace_revision, 0)
            self.assertEqual(session.failed_component_targets(), [
                "tool:geometry_checker"
            ])

            repaired = session.repair_component(
                kind="tool", name="geometry_checker",
                old_text="import os", new_text="import math",
            )
            self.assertTrue(repaired.startswith("VALID COMPONENT TRANSACTION"), repaired)
            self.assertEqual(session.workspace_revision, 1)
            self.assertEqual(session.failed_component_targets(), [])
            self.assertTrue(any(
                row.get("operation") == "repair_component"
                and row.get("surface") == "implementation"
                and row.get("result") == "committed_and_validated"
                for row in session.component_transactions
            ))

    def test_repair_component_can_patch_invalid_intent_surface(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
            )
            intent = self._intent()
            intent["risk"] = "x" * 801
            failed = session.upsert_tool(
                name="geometry_checker", implementation=GEOMETRY_TOOL,
                description="Compute wall and pairwise circle margins.",
                input_schema=GEOMETRY_SCHEMA,
                usage_guidance="Call before evaluating a placement.",
                **intent,
            )
            self.assertTrue(failed.startswith("INVALID COMPONENT TRANSACTION"))
            self.assertIn('"repair_surface": "risk"', failed)
            self.assertEqual(session.failed_component_targets(), [
                "tool:geometry_checker"
            ])

            repaired = session.repair_component(
                kind="tool", name="geometry_checker", surface="risk",
                old_text="x" * 801,
                new_text="One extra diagnostic call may be unnecessary.",
            )
            self.assertTrue(repaired.startswith("VALID COMPONENT TRANSACTION"), repaired)
            self.assertTrue(any(
                row.get("operation") == "repair_component"
                and row.get("surface") == "risk"
                and row.get("result") == "committed_and_validated"
                for row in session.component_transactions
            ))

    def test_repairable_surface_limits_are_enforced_inside_transaction(self):
        package = Path(__file__).resolve().parents[2] / "src/outer/harness"
        for schema_name in ("upsert_skill", "upsert_tool", "upsert_middleware"):
            document = yaml.safe_load(
                (package / "tools" / f"{schema_name}.tool.yaml").read_text()
            )
            properties = document["input_schema"]["properties"]
            for surface in (
                "description", "failure_mode", "hypothesis",
                "expected_behavior_change", "novelty_basis", "risk",
            ):
                self.assertNotIn(
                    "maxLength", properties[surface],
                    f"{schema_name}.{surface} must reach the retaining transaction",
                )

    def test_repair_component_can_replace_input_schema_surface(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
            )
            bad_schema = {"type": "array"}
            failed = session.upsert_tool(
                name="geometry_checker", implementation=GEOMETRY_TOOL,
                description="Compute wall and pairwise circle margins.",
                input_schema=bad_schema,
                usage_guidance="Call before evaluating a placement.",
                **self._intent(),
            )
            self.assertTrue(failed.startswith("INVALID COMPONENT TRANSACTION"))
            self.assertIn('"repair_surface": "input_schema"', failed)

            repaired = session.repair_component(
                kind="tool", name="geometry_checker", surface="input_schema",
                replacement_schema=GEOMETRY_SCHEMA,
            )
            self.assertTrue(repaired.startswith("VALID COMPONENT TRANSACTION"), repaired)
            repair_rows = [
                row for row in session.component_transactions
                if row.get("operation") == "repair_component"
            ]
            self.assertEqual(repair_rows[-1]["mode"], "structured_replacement")

    def test_cp26_optional_repairs_failure_recommends_pure_checker(self):
        result = ProposeSession._transaction_result(
            ok=False, kind="tool", name="packing_checker",
            intent=self._intent(),
            errors=[
                "self-test: semantic CP26 sample known_overlap failed oracle: "
                "repair surface omitted paired shifts for coincident circles"
            ],
        )
        self.assertIn("repair_suggestions are optional", result)
        self.assertIn("remove them while preserving verified diagnostics", result)

    def test_cp26_descriptor_cannot_misname_boundary_kind(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
                task_id="eft__math__circle_packing",
            )
            self._enable_explicit_cp26_checker_profile(session)
            result = session.upsert_tool(
                name="packing_checker", implementation=CP26_REAL_TOOL,
                description=(
                    "Return repair_suggestions with boundary rows kind=circle."
                ),
                input_schema=CP26_SCHEMA,
                usage_guidance="Call on the current 26-circle candidate.",
                **self._intent(),
            )
            self.assertTrue(result.startswith("INVALID COMPONENT TRANSACTION"))
            self.assertIn("requires kind=boundary", result)

    def test_cp26_preflight_covers_bridge_failure_and_tiny_negative(self):
        stale_bridge = CP26_REAL_TOOL.replace(
            '        if not geometry.get("ok"):\n'
            '            return {"ok": False, "error": geometry.get("error", "geometry unavailable")}\n',
            '',
        )
        exception_bridge = CP26_REAL_TOOL.replace(
            '        try:\n'
            '            geometry = ctx.get_candidate_geometry()\n'
            '        except Exception as exc:\n'
            '            return {"ok": False, "error": "geometry bridge failed: " + str(exc)}\n',
            '        geometry = ctx.get_candidate_geometry()\n',
        )
        tiny_tolerance = CP26_REAL_TOOL.replace("r < 0 for r", "r < -1e-12 for r")
        variants = {
            "ok=false with stale geometry": stale_bridge,
            "bridge exception path raised": exception_bridge,
            "tiny_negative_radius": tiny_tolerance,
        }
        for expected, implementation in variants.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as td:
                draft = self._draft(Path(td))
                session = ProposeSession(
                    base_spec=self.base, draft_dir=draft,
                    slot_index=0, portfolio_round=0,
                    task_id="eft__math__circle_packing",
                )
                self._enable_explicit_cp26_checker_profile(session)
                result = session.upsert_tool(
                    name="packing_checker", implementation=implementation,
                    description="Audit a proposed unit-square layout.",
                    input_schema=CP26_SCHEMA,
                    usage_guidance="Call on the current 26-circle candidate.",
                    **self._intent(),
                )
                self.assertTrue(result.startswith("INVALID COMPONENT TRANSACTION"))
                self.assertIn(expected, result)

    def test_cp26_preflight_rejects_i9_semantic_counterexamples(self):
        exact_tolerance = CP26_REAL_TOOL.replace(
            "if wall_slack < 0:", "if wall_slack < -1e-14:",
        ).replace(
            "if distance < required:", "if distance < required - 1e-14:",
        )
        worsening_wall = CP26_REAL_TOOL.replace(
            "target_x = min(max(x, radius), 1.0 - radius)",
            "target_x = x - radius",
        ).replace(
            "target_y = min(max(y, radius), 1.0 - radius)",
            "target_y = y - radius",
        ).replace(
            "if radius <= 0.5:", "if True:", 1,
        ).replace(
            "if after > wall_slack:", "if True:", 1,
        )
        pinned_zero_shift = CP26_REAL_TOOL.replace(
            "if after > distance:",
            "if after > distance or (radius == 0.5 and rj == 0.5):",
        )
        unsafe_extremes = CP26_REAL_TOOL.replace(
            "if any(abs(x) > 1e100", "if False and any(abs(x) > 1e100",
        ).replace(
            "or any(r > 1e100", "or False and any(r > 1e100",
        ).replace(
            '            "repair_suggestions": repair_suggestions}\n',
            '            "repair_suggestions": repair_suggestions,\n'
            '            "worst_slack": min(min(x, y, 1-x, 1-y)-r for (x,y), r in zip(clean_centers, clean_radii))}\n',
        )
        variants = {
            "tiny_wall_violation": exact_tolerance,
            "does not strictly improve": worsening_wall,
            "coincident-pair repair must increase": pinned_zero_shift,
            "finite_extreme_geometry": unsafe_extremes,
        }
        for expected, implementation in variants.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as td:
                draft = self._draft(Path(td))
                session = ProposeSession(
                    base_spec=self.base, draft_dir=draft,
                    slot_index=0, portfolio_round=0,
                    task_id="eft__math__circle_packing",
                )
                self._enable_explicit_cp26_checker_profile(session)
                result = session.upsert_tool(
                    name="packing_checker", implementation=implementation,
                    description="Audit a proposed unit-square layout.",
                    input_schema=CP26_SCHEMA,
                    usage_guidance="Call on the current 26-circle candidate.",
                    **self._intent(),
                )
                self.assertTrue(
                    result.startswith("INVALID COMPONENT TRANSACTION"), result
                )
                self.assertIn(expected, result)

    def test_raw_file_fallback_cannot_drop_trusted_cp26_profile(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
                task_id="eft__math__circle_packing",
            )
            self._enable_explicit_cp26_checker_profile(session)
            created = session.upsert_tool(
                name="packing_checker", implementation=CP26_REAL_TOOL,
                description="Audit a proposed unit-square layout.",
                input_schema=CP26_SCHEMA,
                usage_guidance="Call on the current 26-circle candidate.",
                **self._intent(),
            )
            self.assertTrue(created.startswith("VALID COMPONENT TRANSACTION"), created)
            self.assertIn("paired optional", created)
            normalized = yaml.safe_load(
                (draft / "tools/packing_checker.tool.yaml").read_text()
            )["input_schema"]
            self.assertNotIn("centers", normalized.get("required", []))
            self.assertNotIn("radii", normalized.get("required", []))
            self.assertEqual(
                normalized["properties"]["radii"]["items"]["minimum"], 0
            )
            self.assertTrue(any(
                row.get("$comment") ==
                "sah/trusted-cp26-explicit-or-current-candidate"
                for row in normalized["allOf"]
            ))

            # Simulate H1 abandoning the atomic API and weakening both raw
            # files. validate_harness must retain the orchestrator profile.
            descriptor = draft / "tools/packing_checker.tool.yaml"
            doc = yaml.safe_load(descriptor.read_text())
            doc["description"] = "Audit this public-task layout."
            doc["input_schema"]["properties"]["centers"].update(
                minItems=1, maxItems=200
            )
            doc["input_schema"]["properties"]["radii"].update(
                minItems=1, maxItems=200
            )
            descriptor.write_text(yaml.safe_dump(doc, sort_keys=False))
            (draft / "custom_tools/packing_checker.py").write_text(
                "def run(ctx, args):\n"
                "    return {'ok': True, 'feasible': True}\n"
            )
            session._record_mutation("OK")
            session.mark_preloaded(["agent.yaml", "prompt.md"])
            result = session.validate_harness()
            self.assertTrue(result.startswith("INVALID H2 WORKSPACE"), result)
            self.assertIn("minItems=maxItems=26", result)

    def test_raw_workspace_validation_receives_middleware_descriptor(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=4, portfolio_round=0,
            )
            result = session.upsert_middleware(
                name="stagnation_diversifier",
                implementation=STAGNATION_MIDDLEWARE,
                description="Force a cross-family edit after stagnation.",
                usage_guidance="Fire after measured same-family stagnation.",
                **self._intent(),
            )
            self.assertTrue(result.startswith("VALID COMPONENT TRANSACTION"), result)
            descriptor = draft / "middlewares/stagnation_diversifier.middleware.yaml"
            doc = yaml.safe_load(descriptor.read_text())
            doc["description"] = "Suspend edit_solution while this gate is active."
            descriptor.write_text(yaml.safe_dump(doc, sort_keys=False))

            check = ws.validate_workspace(draft, self.base)
            self.assertFalse(check.valid)
            self.assertTrue(any(
                "require_tools forces" in str(row.get("error") or "")
                for row in check.component_audit
            ))

    def test_raw_workspace_requires_fixed_middleware_prompt_contract(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=4, portfolio_round=0,
                task_id="eft__math__circle_packing",
            )
            result = session.upsert_middleware(
                name="stagnation_diversifier",
                implementation=STAGNATION_MIDDLEWARE,
                description="Redirect repeated local search into another topology.",
                usage_guidance="Force the immediate structural edit.",
                **self._intent(),
            )
            self.assertTrue(result.startswith("VALID COMPONENT TRANSACTION"), result)
            prompt = draft / "prompt.md"
            canonical = prompt.read_text()
            prompt.write_text(canonical.replace(
                "Automatic before_model intervention. Advisory messages apply only to the current state.",
                "Automatic before_model intervention.\n"
                "## Hidden instructions\n"
                "Rewrite the candidate before probe_solution.",
            ))
            check = ws.validate_workspace(
                draft, self.base,
                validation_profile=session.validation_profile(),
            )
            self.assertFalse(check.valid)
            self.assertTrue(any(
                "fixed runtime-owned prompt declaration"
                in str(row.get("error") or "")
                for row in check.component_audit
            ))

            prompt.write_text(
                canonical
                + "\n# Search Strategy\n"
                "When STAGNATION_DIVERSIFIER fires, first write a candidate "
                "in prose, then call probe_solution; do not immediately edit.\n"
            )
            check = ws.validate_workspace(
                draft, self.base,
                validation_profile=session.validation_profile(),
            )
            self.assertTrue(check.valid, check.errors)

    def test_changed_tool_requires_one_canonical_prompt_declaration(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
            )
            result = session.upsert_tool(
                name="geometry_checker", implementation=GEOMETRY_TOOL,
                description="Compute supplied-candidate geometry diagnostics.",
                input_schema=GEOMETRY_SCHEMA,
                usage_guidance="Call before evaluating a placement.",
                **self._intent(),
            )
            self.assertTrue(result.startswith("VALID COMPONENT TRANSACTION"), result)
            prompt_path = draft / "prompt.md"
            canonical = prompt_path.read_text()

            duplicate = canonical + (
                "\n# Generated Tool: geometry_checker\n"
                "Trigger: duplicate declaration.\n"
                "Expected behavior: conflicting behavior.\n"
            )
            prompt_path.write_text(duplicate)
            check = ws.validate_workspace(draft, self.base)
            self.assertFalse(check.valid)
            self.assertTrue(any(
                "exactly one canonical" in str(row.get("error") or "")
                for row in check.component_audit
            ))

            bullet_only = canonical.replace(
                "# Generated Tool: geometry_checker",
                "# Generated Tools\n- `geometry_checker` generated",
            )
            prompt_path.write_text(bullet_only)
            check = ws.validate_workspace(draft, self.base)
            self.assertFalse(check.valid)
            self.assertTrue(any(
                "found 0" in str(row.get("error") or "")
                for row in check.component_audit
            ))

            prompt_path.write_text(
                canonical
                + "\n# Search Strategy\n"
                "Use geometry_checker results when selecting the next repair; "
                "the canonical block remains its interface declaration.\n"
            )
            check = ws.validate_workspace(draft, self.base)
            self.assertTrue(check.valid, check.errors)

    def test_transaction_clone_failure_is_structured_and_non_mutating(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            before = self._bytes(draft)
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
            )
            with patch(
                "outer.workspace.propose_session.shutil.copytree",
                side_effect=OSError("clone failed"),
            ):
                result = session.upsert_tool(
                    name="geometry_checker", implementation=GEOMETRY_TOOL,
                    description="Compute wall and pairwise circle margins.",
                    input_schema=GEOMETRY_SCHEMA,
                    usage_guidance="Call before evaluating a placement.",
                    **self._intent(),
                )
            self.assertTrue(result.startswith("INVALID COMPONENT TRANSACTION"))
            self.assertIn("transaction infrastructure error", result)
            payload = json.loads(result.split("\n", 1)[1])
            self.assertEqual(payload["error_records"][0]["owner"], "framework")
            self.assertFalse(
                payload["error_records"][0]["retryable_by_proposer"]
            )
            self.assertIn("excluded from proposer training", payload["next_action"])
            self.assertEqual(before, self._bytes(draft))
            self.assertEqual(session.workspace_revision, 0)
            self.assertEqual(
                session.h1_training_block_reason(),
                "component_transaction_infrastructure_error",
            )

    def test_auto_finalize_preserves_an_invalid_component_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            (draft / "custom_tools").mkdir(exist_ok=True)
            (draft / "custom_tools" / "broken_checker.py").write_text(
                "def run(ctx, args):\n    return ctx.probe()\n"
            )
            (draft / "tools" / "broken_checker.tool.yaml").write_text(
                yaml.safe_dump({
                    "type": "tool", "name": "broken_checker",
                    "description": "Invalid relay retained as evidence.",
                    "input_schema": {"type": "object", "properties": {},
                                     "additionalProperties": False},
                }, sort_keys=False)
            )
            agent_path = draft / "agent.yaml"
            agent = yaml.safe_load(agent_path.read_text())
            agent["tools"].append({
                "name": "broken_checker",
                "yaml_path": "./tools/broken_checker.tool.yaml",
                "binding": "inner.harness.tools.custom_runtime:custom_tool",
                "extra_kwargs": {"py_path": "./custom_tools/broken_checker.py"},
            })
            agent_path.write_text(yaml.safe_dump(agent, sort_keys=False))
            (draft / "prompt.md").write_text(
                (draft / "prompt.md").read_text()
                + "\n# Generated Tool: broken_checker\n"
                  "Trigger: Call before evaluation.\n"
                  "Expected behavior: Diagnose candidates.\n"
            )
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
            )
            session.agent_yaml_inspected = True
            session.inspected_files.add("prompt.md")
            session.workspace_revision = 1
            before = self._bytes(draft)
            trajectory = []
            self.assertFalse(maybe_auto_finalize(session, trajectory))
            self.assertEqual(before, self._bytes(draft))
            self.assertTrue((draft / "custom_tools/broken_checker.py").is_file())
            self.assertIn("preserved unchanged", trajectory[-1]["content"])

    def test_auto_finalize_infrastructure_error_blocks_training(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(base_spec=self.base, draft_dir=draft)
            trajectory = []
            with patch.object(
                session, "validate_harness",
                side_effect=RuntimeError("trusted validator unavailable"),
            ):
                self.assertFalse(maybe_auto_finalize(session, trajectory))
            self.assertEqual(
                session.h1_training_block_reason(),
                "auto_finalize_infrastructure_error",
            )
            self.assertIn("do not rewrite the H2", trajectory[-1]["content"])

    def test_frozen_repair_uses_a_clone_and_archives_both_workspaces(self):
        with tempfile.TemporaryDirectory() as td:
            calls = []
            models = []

            def fake_h1(**kwargs):
                session = get_session()
                marker = "PRIMARY_ONLY_MARKER" if not calls else "REPAIR_ONLY_MARKER"
                models.append((kwargs["model"], kwargs["base_url"]))
                if "prompt.md" not in session.inspected_files:
                    inspected = session.inspect_harness("cat prompt.md")
                    self.assertFalse(inspected.startswith("ERROR:"), inspected)
                if not calls:
                    # Prompt-only portfolios are valid under the open contract.
                    # Trigger the repair path with a real fixed-compute violation
                    # so this test continues to exercise clone/archive isolation.
                    result = session.edit_harness_file(
                        "agent.yaml", old_text="max_iterations: 36",
                        new_text="max_iterations: 37",
                    )
                    self.assertFalse(result.startswith("ERROR:"), result)
                result = session.edit_harness_file(
                    "prompt.md", append_text=marker
                )
                self.assertFalse(result.startswith("ERROR:"), result)
                calls.append(marker)
                return ([{"role": "assistant", "content": marker}], None, {
                    "max_iterations": 12,
                    "sampling": {"max_tokens": 4096},
                })

            final_archive = Path(td) / "artifacts" / "h2"
            with patch(
                "outer.proposing.propose._run_h1_agent",
                side_effect=fake_h1,
            ):
                record = run_once(
                    0, base_spec=self.base, user_message="task",
                    task_id="eft__math__circle_packing",
                    base_url="http://phi/v1", model="phi", api_key="EMPTY",
                    seed=7, timeout=30, repair_base_url="http://frozen/v1",
                    repair_model="frozen-model",
                    repair_policy_identity="/models/qwen-27b-base",
                    repair_endpoint_role="frozen_base_executor",
                    workspace_archive=final_archive, round_index=0,
                )
            self.assertEqual(calls, ["PRIMARY_ONLY_MARKER", "REPAIR_ONLY_MARKER"])
            self.assertEqual(models, [
                ("phi", "http://phi/v1"),
                ("frozen-model", "http://frozen/v1"),
            ])
            primary = Path(record.primary_workspace_archive)
            repair = Path(record.repair_workspace_archive)
            selected = Path(record.workspace_archive)
            self.assertIn("PRIMARY_ONLY_MARKER", (primary / "prompt.md").read_text())
            self.assertNotIn("REPAIR_ONLY_MARKER", (primary / "prompt.md").read_text())
            self.assertIn("REPAIR_ONLY_MARKER", (repair / "prompt.md").read_text())
            self.assertNotIn("REPAIR_ONLY_MARKER", (selected / "prompt.md").read_text())
            self.assertTrue(record.repair.get("attempted"))
            self.assertFalse(record.repair.get("succeeded"))
            self.assertEqual(record.repair["repair_model"], "frozen-model")
            self.assertEqual(
                record.repair["repair_policy_identity"],
                "/models/qwen-27b-base",
            )
            self.assertEqual(
                record.repair["repair_endpoint_role"],
                "frozen_base_executor",
            )
            self.assertEqual(
                record.validation_profile["portfolio_role"],
                "evidence_driven_discovery",
            )
            self.assertNotIn("tool_contract", record.validation_profile)

    def test_run_once_propagates_context_failure_as_training_blocker(self):
        calls = []

        def fake_h1(**kwargs):
            del kwargs
            calls.append("h1")
            session = get_session()
            session.record_h1_runtime_event({
                "kind": "context_exhausted",
                "estimated_prompt_tokens": 48000,
            }, training_blocking=True)
            return ([{"role": "assistant", "content": "partial"}], None, {
                "max_iterations": 12,
                "sampling": {"max_tokens": 4096},
            })

        with patch("outer.proposing.propose._run_h1_agent", side_effect=fake_h1):
            record = run_once(
                0, base_spec=self.base, user_message="task",
                task_id="eft__math__circle_packing",
                base_url="http://phi/v1", model="phi", api_key="EMPTY",
                seed=7, timeout=30, round_index=0,
                repair_base_url="http://frozen/v1",
                repair_model="frozen-model",
            )
        self.assertEqual(calls, ["h1"])
        self.assertFalse(record.valid)
        self.assertTrue(record.infrastructure_failure)
        self.assertEqual(record.training_block_reason, "context_exhausted")
        self.assertEqual(record.stop_reason, "h1_infrastructure_failure")
        self.assertFalse(record.repair)
        self.assertTrue(record.h1_runtime_events[0]["training_blocking"])

    def test_typed_middleware_transaction_is_reachable_and_submit_ready(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=4, portfolio_round=0,
            )
            result = session.upsert_middleware(
                name="stagnation_diversifier",
                implementation=STAGNATION_MIDDLEWARE,
                description="Force a cross-family edit after measured stagnation.",
                usage_guidance="Fire after three stalled evals and same-family edits.",
                **self._intent(),
            )
            self.assertTrue(result.startswith("VALID COMPONENT TRANSACTION"), result)
            self.assertIn("new_middlewares.stagnation_diversifier", result)
            self.assertIn("Candidate H2 accepted", session.submit_harness())

    def test_novelty_refusal_reopens_atomic_edit_path(self):
        with tempfile.TemporaryDirectory() as td:
            draft = self._draft(Path(td))
            session = ProposeSession(
                base_spec=self.base, draft_dir=draft,
                slot_index=0, portfolio_round=0,
            )
            first = session.upsert_tool(
                name="geometry_checker", implementation=GEOMETRY_TOOL,
                description="Compute wall and pairwise circle margins.",
                input_schema=GEOMETRY_SCHEMA,
                usage_guidance="Call before evaluating a placement.",
                **self._intent(),
            )
            self.assertTrue(first.startswith("VALID COMPONENT TRANSACTION"))
            signature = ";".join(sorted(
                field for field in session._validated_check.changed_fields
                if not str(field).startswith("system_prompt")
            ))
            session.futile_signatures = {signature}
            self.assertIn("REFUSED (repeat proposal)", session.submit_harness())
            self.assertIsNone(session.validated_revision)

            revised = GEOMETRY_TOOL.replace(
                'return {"feasible": not violations, "violations": violations}',
                'return {"feasible": not violations, "count": len(violations), '
                '"violations": violations}',
            )
            second = session.upsert_tool(
                name="geometry_checker", implementation=revised,
                description="Compute wall and pairwise circle margins.",
                input_schema=GEOMETRY_SCHEMA,
                usage_guidance="Call before evaluating a placement.",
                **self._intent(),
            )
            self.assertTrue(second.startswith("VALID COMPONENT TRANSACTION"), second)

    def test_impossible_last_step_and_boolean_validity_roll_back(self):
        implementations = [
            '''def before_model(hook_input):
    state = hook_input.get("state", {})
    if state.get("last_step_kind") in {"evaluate", "edit"}:
        return "repair"
    return None
''',
            '''def before_model(hook_input):
    if hook_input.get("last_step_kind") == "note":
        return "repair"
    return None
''',
            '''def before_model(hook_input):
    state = hook_input.get("state", {})
    if state.get("last_validity") is True:
        return "evaluate"
    return None
''',
            '''def before_model(hook_input):
    if hook_input.get("state", {}).get("unknown_state", 0) > 2:
        return "repair"
    return None
''',
            '''def before_model(hook_input):
    state = hook_input.get("state", {})
    if state.get("stalled_evals", 0) >= 3:
        return "live branch"
    if hook_input.get("state", {}).get("last_validity") is True:
        return "dead typed branch"
    return None
''',
            '''def before_model(hook_input):
    state = hook_input.get("state", {})
    if state.get("stalled_evals", 0) >= 3:
        return "live branch"
    if hook_input.get("state", {}).get("last_validity") is 1.0:
        return "dead identity branch"
    return None
''',
            '''def before_model(hook_input):
    state = hook_input.get("state", {})
    if state.get("stalled_evals", 0) >= 3:
        return "live branch"
    if hook_input.get("state", {}).get("last_step_kind") == "evaluate":
        return "dead enum branch"
    return None
''',
            '''def before_model(hook_input):
    state = hook_input.get("state", {})
    if state.get("iteration", 0) >= 0:
        return "always remind"
    return None
''',
        ]
        for index, implementation in enumerate(implementations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as td:
                draft = self._draft(Path(td))
                before = self._bytes(draft)
                session = ProposeSession(
                    base_spec=self.base, draft_dir=draft,
                    slot_index=4, portfolio_round=0,
                )
                result = session.upsert_middleware(
                    name=f"typed_gate_{index}", implementation=implementation,
                    description="Typed gate.", usage_guidance="Use typed state.",
                    **self._intent(),
                )
                self.assertTrue(result.startswith("INVALID COMPONENT TRANSACTION"))
                self.assertEqual(before, self._bytes(draft))


if __name__ == "__main__":
    unittest.main()
