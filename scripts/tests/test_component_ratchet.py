"""COMPRATCHET-001: the component-level selection operator, verified.

Each admission clause is exercised against REAL materialized packages so
the grafted base is a runnable H2, not a bookkeeping fiction.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from outer.compiling.materialize import materialize, INNER_HARNESS  # noqa: E402
from outer.genome import harness_spec as hs  # noqa: E402
from outer.rounds.component_ratchet import (  # noqa: E402
    _used_by_executor,
    apply_component_ratchet,
)
from outer.workspace.h2_workspace import (  # noqa: E402
    generated_component_prompt_block,
)

SKILL = {
    "name": "isolated-winner",
    "description": "A construction family with a measured positive delta.",
    "body": "# Isolated winner\nUse a different construction family first.\n",
}
TOOL = {
    "name": "structure_probe",
    "description": "Measure the current program structure.",
    "input_schema": {
        "type": "object", "properties": {}, "additionalProperties": False,
    },
    "implementation_py": (
        "def run(ctx, args):\n"
        "    program = ctx.get_program()\n"
        "    return {'ok': True, 'lines': len(program.splitlines())}\n"
    ),
}
MIDDLEWARE = {
    "name": "family_gate",
    "description": "Request a structural family change after a stall.",
    "hook": "before_model",
    "implementation_py": (
        "def before_model(hook_input):\n"
        "    state = hook_input.get('state', {})\n"
        "    if state.get('stalled_evals', 0) >= 2:\n"
        "        return {'note': 'switch structural family'}\n"
        "    return None\n"
    ),
}


def _promotion(blockers=()):
    blockers = list(blockers)
    return {
        "eligible": not blockers,
        "blockers": blockers,
        "observed_repeats": 3,
        "minimum_repeats": 3,
        "positive_sign_consistent": True,
        "minimum_effect": 1e-4,
    }


def _mk_base(root: Path) -> Path:
    base = hs.read_base_spec(INNER_HARNESS)
    pkg = root / "incumbent"
    materialize(base, pkg, meta={"effective": base}, validate_prompt=False)
    return pkg


def _mk_candidate(
    root: Path, k: int, *, skills=None, tools=None, middlewares=None,
    params=None, prompt_suffix: str = "",
) -> Path:
    base = hs.read_base_spec(INNER_HARNESS)
    spec = json.loads(json.dumps(base))
    for s in (skills or []):
        spec.setdefault("new_skills", [])
        spec["new_skills"] = list(spec["new_skills"]) + [s]
        spec["system_prompt"] = (
            str(spec.get("system_prompt") or "")
            + f"\n# Generated Skill: {s['name']}\nEnact it first.\n")
    for group, values in (params or {}).items():
        spec.setdefault(group, {})
        spec[group].update(values)
    for tool in (tools or []):
        spec.setdefault("new_tools", [])
        spec["new_tools"] = list(spec["new_tools"]) + [tool]
        spec["system_prompt"] = (
            str(spec.get("system_prompt") or "").rstrip()
            + "\n\n"
            + generated_component_prompt_block(
                "tool", tool["name"],
                usage_guidance="Call before selecting the next edit.",
                expected_behavior_change="Use measured structure to edit.",
            )
            + "\n"
        )
    for middleware in (middlewares or []):
        spec.setdefault("new_middlewares", [])
        spec["new_middlewares"] = (
            list(spec["new_middlewares"]) + [middleware]
        )
        spec["system_prompt"] = (
            str(spec.get("system_prompt") or "").rstrip()
            + "\n\n"
            + generated_component_prompt_block(
                "middleware", middleware["name"]
            )
            + "\n"
        )
    spec["system_prompt"] = str(spec.get("system_prompt") or "") + prompt_suffix
    cand = root / "tasks" / "t" / f"cand{k:02d}"
    materialize(spec, cand, meta={"effective": spec}, validate_prompt=False)
    return cand


def _mk_usage(
    round_dir: Path, k: int, *, skills=(), tools=(), middlewares=(),
    tid: str = "t", tool_semantic_success: bool = True,
    middleware_fires: bool = True,
) -> None:
    """Record that the executor engaged these generated components.

    Inheritance is keyed on participation read from the rollout audits, so a
    ratchet test has to say what the executor did, not what the slot was
    labelled.
    """
    results = (round_dir / "rollouts" / tid / f"cand{k:02d}" / "20260101-000000"
               / "results")
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{tid}.json").write_text(json.dumps({
        "best_program": "x = 1",
        "tool_audit": {
            n: {
                "invocations": 1,
                "semantic_successful_invocations": int(tool_semantic_success),
                "semantic_rejections": int(not tool_semantic_success),
            }
            for n in tools
        },
        "skill_audit": {n: {"loads_before_first_edit": 1} for n in skills},
        "middleware_audit": {
            n: {"invocations": 1, "fires": int(middleware_fires)}
            for n in middlewares
        },
    }))


class ComponentRatchetTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop("CURVE_COMPONENT_RATCHET", None)
        os.environ.pop("CURVE_COMPONENT_RATCHET_MIN_DELTA", None)

    def _round_dir(self, td):
        # production layout: <task>/rounds/roundNNN — the ledger lives in
        # rounds/, shared across rounds of one task
        rd = Path(td) / "rounds" / "round003"
        rd.mkdir(parents=True, exist_ok=True)
        return rd

    def _run(self, root, candidates, rows, *, improved=False, best_k=None):
        entry = {"package": str(_mk_base(root)), "score": 0.9,
                 "from": "unchanged"}
        rows = [
            {**row, "promotion": row.get("promotion", _promotion())}
            for row in rows
        ]
        rec = apply_component_ratchet(
            round_dir=root, tid="t", round_index=3,
            candidates=candidates, rows=rows,
            improved=improved, best_k=best_k, next_base_entry=entry)
        return rec, entry

    def test_isolated_positive_component_is_inherited_from_a_loser(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            _mk_candidate(root, 4, skills=[SKILL])
            _mk_usage(root, 4, skills=["isolated-winner"])
            rec, entry = self._run(
                root,
                candidates=[{"k": 4, "valid": True,
                             "changed_fields": ["system_prompt",
                                                "new_skills.isolated-winner"]}],
                rows=[{"k": 4, "causal_delta": 0.03, "score": 0.88,
                       "control_score": 0.85, "attribution_valid": True,
                       "valid": True}],
                improved=False)
            self.assertEqual([a["name"] for a in rec["admitted"]],
                             ["isolated-winner"])
            grafted = Path(entry["package"])
            self.assertNotEqual(grafted.name, "incumbent")
            spec = hs.read_base_spec(grafted)
            self.assertIn("isolated-winner",
                          [s["name"] for s in spec.get("new_skills") or []])
            self.assertIn("isolated-winner", spec["system_prompt"])
            self.assertIn("Enact it first.", spec["system_prompt"])
            self.assertTrue(
                (grafted / "skills" / "isolated-winner" / "SKILL.md").is_file())
            ledger_file = root.parent / "component_ledger.jsonl"
            self.assertTrue(ledger_file.is_file())
            rows_ = [json.loads(l) for l in
                     ledger_file.read_text().splitlines()]
            admitted_rows = [r for r in rows_ if r["event"] == "admitted"]
            self.assertEqual(len(admitted_rows), 1, rows_)
            self.assertAlmostEqual(admitted_rows[0]["causal_delta"], 0.03)
            self.assertEqual(admitted_rows[0]["name"], "isolated-winner")

    def test_canonical_tool_declaration_is_preserved_during_graft(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            _mk_candidate(root, 4, tools=[TOOL])
            _mk_usage(root, 4, tools=["structure_probe"])
            rec, entry = self._run(
                root,
                candidates=[{
                    "k": 4, "valid": True,
                    "changed_fields": [
                        "system_prompt", "new_tools.structure_probe"
                    ],
                }],
                rows=[{
                    "k": 4, "causal_delta": 0.03,
                    "attribution_valid": True, "valid": True,
                }],
            )
            self.assertEqual(
                [row["name"] for row in rec["admitted"]],
                ["structure_probe"],
            )
            grafted = hs.read_base_spec(Path(entry["package"]))
            declaration = generated_component_prompt_block(
                "tool", "structure_probe",
                usage_guidance="Call before selecting the next edit.",
                expected_behavior_change="Use measured structure to edit.",
            )
            self.assertIn(declaration, grafted["system_prompt"])
            self.assertEqual(hs.component_prompt_issues(grafted), [])

    def test_canonical_middleware_declaration_is_preserved_during_graft(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            _mk_candidate(root, 4, middlewares=[MIDDLEWARE])
            _mk_usage(root, 4, middlewares=["family_gate"])
            rec, entry = self._run(
                root,
                candidates=[{
                    "k": 4, "valid": True,
                    "changed_fields": [
                        "system_prompt", "new_middlewares.family_gate"
                    ],
                }],
                rows=[{
                    "k": 4, "causal_delta": 0.03,
                    "attribution_valid": True, "valid": True,
                }],
            )
            self.assertEqual(
                [row["name"] for row in rec["admitted"]],
                ["family_gate"],
            )
            grafted = hs.read_base_spec(Path(entry["package"]))
            self.assertIn(
                generated_component_prompt_block(
                    "middleware", "family_gate"
                ),
                grafted["system_prompt"],
            )
            self.assertEqual(hs.component_prompt_issues(grafted), [])

    def test_a_component_the_executor_never_used_is_not_admitted(self):
        # Mounting is not participation.  Admission used to turn on a slot
        # LABEL; it now turns on what the rollout audits recorded.
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            _mk_candidate(root, 2, skills=[SKILL])
            _mk_usage(root, 2)          # rollout ran, engaged nothing
            rec, entry = self._run(
                root,
                candidates=[{"k": 2, "valid": True,
                             "changed_fields": ["system_prompt",
                                                "new_skills.isolated-winner"]}],
                rows=[{"k": 2, "causal_delta": 0.05, "score": 0.9,
                       "control_score": 0.85, "attribution_valid": True,
                       "valid": True}])
            self.assertEqual(rec["admitted"], [])
            self.assertEqual(rec["skipped"][0]["reason"],
                             "not_used_by_executor:isolated-winner")
            self.assertIn("incumbent", entry["package"])

    def test_invocation_only_tool_and_zero_fire_middleware_are_not_used(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            _mk_usage(
                root, 2, tools=["stale_checker"],
                middlewares=["quiet_gate"], tool_semantic_success=False,
                middleware_fires=False,
            )
            self.assertEqual(_used_by_executor(root, "t", 2), set())

            _mk_usage(
                root, 3, tools=["live_checker"],
                middlewares=["live_gate"], tool_semantic_success=True,
                middleware_fires=True,
            )
            self.assertEqual(
                _used_by_executor(root, "t", 3),
                {"live_checker", "live_gate"},
            )

    def test_component_ratchet_honors_repeat_and_sign_promotion_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            _mk_candidate(root, 4, skills=[SKILL])
            _mk_usage(root, 4, skills=["isolated-winner"])
            rec, _ = self._run(
                root,
                candidates=[{
                    "k": 4, "valid": True,
                    "changed_fields": ["new_skills.isolated-winner"],
                }],
                rows=[{
                    "k": 4, "causal_delta": 0.05,
                    "attribution_valid": True, "valid": True,
                    "promotion": _promotion(["paired_repeats_below_3"]),
                }],
            )
            self.assertEqual(rec["admitted"], [])
            self.assertIn(
                "inheritance_promotion_blocked:paired_repeats_below_3",
                [row["reason"] for row in rec["skipped"]],
            )

    def test_missing_promotion_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            _mk_candidate(root, 4, skills=[SKILL])
            _mk_usage(root, 4, skills=["isolated-winner"])
            entry = {"package": str(_mk_base(root)), "score": 0.9,
                     "from": "unchanged"}
            rec = apply_component_ratchet(
                round_dir=root, tid="t", round_index=3,
                candidates=[{
                    "k": 4, "valid": True,
                    "changed_fields": ["new_skills.isolated-winner"],
                }],
                rows=[{
                    "k": 4, "causal_delta": 0.05,
                    "attribution_valid": True, "valid": True,
                }],
                improved=False, best_k=None, next_base_entry=entry,
            )
            self.assertEqual(rec["admitted"], [])
            self.assertIn(
                "promotion_evidence_missing",
                [row["reason"] for row in rec["skipped"]],
            )
            self.assertIn("incumbent", entry["package"])

    def test_substantive_prompt_rewrite_is_not_attributed_to_component(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            _mk_candidate(
                root, 4, skills=[SKILL],
                prompt_suffix=(
                    "\n# Global Search Rewrite\n"
                    "Always replace the whole program before evaluating.\n"
                ),
            )
            _mk_usage(root, 4, skills=["isolated-winner"])
            rec, entry = self._run(
                root,
                candidates=[{
                    "k": 4, "valid": True,
                    "changed_fields": [
                        "system_prompt", "new_skills.isolated-winner"
                    ],
                }],
                rows=[{
                    "k": 4, "causal_delta": 0.05,
                    "attribution_valid": True, "valid": True,
                    "promotion": _promotion(),
                }],
            )
            self.assertEqual(rec["admitted"], [])
            self.assertIn(
                "prompt_not_isolated_component_declaration",
                [row["reason"] for row in rec["skipped"]],
            )
            self.assertIn("incumbent", entry["package"])

    def test_below_threshold_or_invalid_attribution_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            _mk_candidate(root, 4, skills=[SKILL])
            _mk_usage(root, 4, skills=["isolated-winner"])
            cand = {"k": 4, "valid": True,
                    "changed_fields": ["new_skills.isolated-winner"]}
            rec, _ = self._run(
                root, candidates=[cand],
                rows=[{
                    "k": 4, "causal_delta": 0.00005,
                    "attribution_valid": True, "valid": True,
                    "promotion": {**_promotion(), "minimum_effect": 0.0},
                }])
            self.assertTrue(rec["skipped"][0]["reason"].startswith("delta_below"))
            rec2, _ = self._run(
                root, candidates=[cand],
                rows=[{"k": 4, "causal_delta": 0.03,
                       "attribution_valid": False, "valid": True}])
            self.assertEqual(rec2["skipped"][0]["reason"],
                             "attribution_invalid")

    def test_promoted_winner_components_are_not_double_grafted(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            _mk_candidate(root, 4, skills=[SKILL])
            rec, entry = self._run(
                root,
                candidates=[{"k": 4, "valid": True,
                             "changed_fields": ["new_skills.isolated-winner"]}],
                rows=[{"k": 4, "causal_delta": 0.03, "attribution_valid": True,
                       "valid": True}],
                improved=True, best_k=4)
            self.assertEqual(rec["admitted"], [])
            self.assertEqual(rec["skipped"][0]["reason"], "already_in_promoted")

    def test_capacity_is_bounded_by_the_genome_caps(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            s2 = dict(SKILL, name="second-family",
                      description="Another family.")
            s3 = dict(SKILL, name="third-family",
                      description="A third family.")
            _mk_candidate(root, 3, skills=[SKILL])
            _mk_candidate(root, 4, skills=[s2])
            _mk_candidate(root, 5, skills=[s3])
            _mk_usage(root, 3, skills=["isolated-winner"])
            _mk_usage(root, 4, skills=["second-family"])
            _mk_usage(root, 5, skills=["third-family"])
            rec, entry = self._run(
                root,
                candidates=[
                    {"k": 3, "valid": True,
                     "changed_fields": ["new_skills.isolated-winner"]},
                    {"k": 4, "valid": True,
                     "changed_fields": ["new_skills.second-family"]},
                    {"k": 5, "valid": True,
                     "changed_fields": ["new_skills.third-family"]},
                ],
                rows=[{"k": 3, "causal_delta": 0.06,
                       "attribution_valid": True, "valid": True},
                      {"k": 4, "causal_delta": 0.05,
                       "attribution_valid": True, "valid": True},
                      {"k": 5, "causal_delta": 0.01,
                       "attribution_valid": True, "valid": True}])
            admitted = [a["name"] for a in rec["admitted"]]
            self.assertEqual(admitted, ["isolated-winner", "second-family"])
            reasons = {s["name"]: s["reason"] for s in rec["skipped"]
                       if "name" in s}
            self.assertEqual(reasons.get("third-family"), "capacity_full")

    def test_kill_switch(self):
        os.environ["CURVE_COMPONENT_RATCHET"] = "0"
        try:
            with tempfile.TemporaryDirectory() as td:
                root = self._round_dir(td)
                rec, entry = self._run(root, candidates=[], rows=[])
                self.assertIsNone(rec)
        finally:
            os.environ.pop("CURVE_COMPONENT_RATCHET", None)


class ParameterInheritanceTest(unittest.TestCase):
    """The fifth dimension: an isolated operating-point move is heritable."""

    def _round_dir(self, td):
        rd = Path(td) / "rounds" / "round003"
        rd.mkdir(parents=True, exist_ok=True)
        return rd

    def _mk_param_candidate(self, root: Path, k: int, group: str, key: str,
                            value):
        base = hs.read_base_spec(INNER_HARNESS)
        spec = json.loads(json.dumps(base))
        spec.setdefault(group, {})
        spec[group][key] = value
        cand = root / "tasks" / "t" / f"cand{k:02d}"
        materialize(cand.parent and spec, cand, meta={"effective": spec},
                    validate_prompt=False)
        return cand

    def test_isolated_parameter_move_is_grafted(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            self._mk_param_candidate(root, 5, "sampling", "temperature", 0.4)
            entry = {"package": str(_mk_base(root)), "score": 0.9,
                     "from": "unchanged"}
            rec = apply_component_ratchet(
                round_dir=root, tid="t", round_index=3,
                candidates=[{"k": 5, "valid": True,
                             "changed_fields": ["sampling.temperature"]}],
                rows=[{"k": 5, "causal_delta": 0.02, "attribution_valid": True,
                       "valid": True, "promotion": _promotion()}],
                improved=False, best_k=None, next_base_entry=entry)
            self.assertEqual([a["name"] for a in rec["admitted"]],
                             ["temperature"])
            grafted = hs.read_base_spec(Path(entry["package"]))
            self.assertAlmostEqual(grafted["sampling"]["temperature"], 0.4)
            adm = rec["admitted"][0]
            self.assertEqual(adm["field"], "sampling")
            self.assertIn("previous_value", adm)

    def test_a_no_op_parameter_is_not_grafted(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            base_pkg = _mk_base(root)
            current = hs.read_base_spec(base_pkg)["sampling"]["temperature"]
            self._mk_param_candidate(root, 5, "sampling", "temperature",
                                     current)
            entry = {"package": str(base_pkg), "score": 0.9,
                     "from": "unchanged"}
            rec = apply_component_ratchet(
                round_dir=root, tid="t", round_index=3,
                candidates=[{"k": 5, "valid": True,
                             "changed_fields": ["sampling.temperature"]}],
                rows=[{"k": 5, "causal_delta": 0.02, "attribution_valid": True,
                       "valid": True, "promotion": _promotion()}],
                improved=False, best_k=None, next_base_entry=entry)
            self.assertEqual(rec["admitted"], [])
            self.assertEqual(rec["skipped"][0]["reason"], "param_already_set")

    def test_parameter_only_move_cannot_bypass_repeat_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            self._mk_param_candidate(root, 5, "sampling", "temperature", 0.4)
            entry = {"package": str(_mk_base(root)), "score": 0.9,
                     "from": "unchanged"}
            rec = apply_component_ratchet(
                round_dir=root, tid="t", round_index=3,
                candidates=[{"k": 5, "valid": True,
                             "changed_fields": ["sampling.temperature"]}],
                rows=[{
                    "k": 5, "causal_delta": 0.02,
                    "attribution_valid": True, "valid": True,
                    "promotion": _promotion(["paired_repeats_below_3"]),
                }],
                improved=False, best_k=None, next_base_entry=entry,
            )
            self.assertEqual(rec["admitted"], [])
            self.assertEqual(
                rec["skipped"][0]["reason"],
                "inheritance_promotion_blocked:paired_repeats_below_3",
            )
            self.assertIn("incumbent", entry["package"])

    def test_mixed_component_parameter_bundle_is_blocked_as_a_whole(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._round_dir(td)
            _mk_candidate(
                root, 5, skills=[SKILL],
                params={"sampling": {"temperature": 0.4}},
            )
            _mk_usage(root, 5, skills=["isolated-winner"])
            entry = {"package": str(_mk_base(root)), "score": 0.9,
                     "from": "unchanged"}
            rec = apply_component_ratchet(
                round_dir=root, tid="t", round_index=3,
                candidates=[{
                    "k": 5, "valid": True,
                    "changed_fields": [
                        "new_skills.isolated-winner",
                        "sampling.temperature",
                    ],
                }],
                rows=[{
                    "k": 5, "causal_delta": 0.02,
                    "attribution_valid": True, "valid": True,
                    "promotion": _promotion([
                        "activation_inconsistent_across_repeats"
                    ]),
                }],
                improved=False, best_k=None, next_base_entry=entry,
            )
            self.assertEqual(rec["admitted"], [])
            blocked = [
                row for row in rec["skipped"]
                if row["reason"] ==
                "non_atomic_heritable_bundle_requires_exactly_one"
            ]
            self.assertEqual(len(blocked), 1)
            self.assertEqual(
                blocked[0]["names"],
                ["isolated-winner", "sampling.temperature"],
            )
            self.assertIn("incumbent", entry["package"])


if __name__ == "__main__":
    unittest.main()
