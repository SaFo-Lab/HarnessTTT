from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from outer.genome import harness_spec as hs  # noqa: E402
from outer.compiling.materialize import INNER_HARNESS, materialize  # noqa: E402


def _tool(name: str, marker: str) -> dict:
    return {
        "name": name,
        "description": f"Return the {marker} marker.",
        "input_schema": {"type": "object", "properties": {}},
        "implementation_py": (
            "def run(ctx, args):\n"
            f"    return {{'marker': {marker!r}}}\n"
        ),
    }


def _skill(name: str, marker: str) -> dict:
    return {
        "name": name,
        "description": f"Use the {marker} search strategy.",
        "body": f"# {marker}\n\nFollow the {marker} strategy.",
    }


def _middleware(name: str, marker: str) -> dict:
    return {
        "name": name,
        "hook": "before_model",
        "description": f"Emit the {marker} reminder after a stall.",
        "implementation_py": (
            "def before_model(hook_input):\n"
            "    state = hook_input.get('state', {})\n"
            "    if state.get('stalled_evals', 0) >= 2:\n"
            f"        return {marker!r}\n"
            "    return None\n"
        ),
    }


def _patch(prefix: str) -> dict:
    return {
        "schema": hs.SCHEMA_VERSION,
        "new_tools": [_tool(f"{prefix}_probe", prefix)],
        "new_skills": [_skill(f"{prefix}-strategy", prefix)],
        "new_middlewares": [_middleware(f"{prefix}_guard", prefix)],
    }


def _with_owned_component_prompt(spec: dict) -> dict:
    effective = json.loads(json.dumps(spec))
    catalog = hs.h2_component_catalog(effective)
    effective["system_prompt"] = (
        "You are the executor. Follow the task and use mounted components.\n\n"
        "# Available H2 components\n"
        "Tools: " + ", ".join(catalog["tools"]) + "\n"
        "Skills: " + ", ".join(catalog["skills"]) + "\n"
        "Middleware: " + ", ".join(catalog["middlewares"]) + "\n"
        "Tools and skills are selected when relevant; middleware runs automatically."
    )
    return effective


class GeneratedComponentInheritanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base = hs.read_base_spec(INNER_HARNESS)

    def test_generated_components_are_additive_and_attributed_to_their_round(self) -> None:
        round1 = hs.merge_with_base(_patch("alpha"), self.base)
        round2 = hs.merge_with_base(_patch("beta"), round1)

        self.assertEqual(hs.generated_component_inventory(round2), {
            "new_tools": ["alpha_probe", "beta_probe"],
            "new_skills": ["alpha-strategy", "beta-strategy"],
            "new_middlewares": ["alpha_guard", "beta_guard"],
        })
        differs, changed = hs.differs_from_base(round2, round1)
        self.assertTrue(differs)
        self.assertEqual(changed, [
            "new_tools.beta_probe",
            "new_skills.beta-strategy",
            "new_middlewares.beta_guard",
        ])
        for field, expected in (
            ("new_tools", "beta_probe"),
            ("new_skills", "beta-strategy"),
            ("new_middlewares", "beta_guard"),
        ):
            rows = hs.changed_generated_components(round2, round1, field)
            self.assertEqual([row["name"] for row in rows], [expected])

        no_op, no_op_fields = hs.differs_from_base(round2, round2)
        self.assertFalse(no_op)
        self.assertEqual(no_op_fields, [])

    def test_same_name_explicitly_updates_without_dropping_siblings(self) -> None:
        round1 = hs.merge_with_base(_patch("alpha"), self.base)
        round2 = hs.merge_with_base(_patch("beta"), round1)
        updated = hs.merge_with_base({
            "schema": hs.SCHEMA_VERSION,
            "new_tools": [_tool("alpha_probe", "alpha-v2")],
        }, round2)

        self.assertEqual(
            hs.generated_component_inventory(updated)["new_tools"],
            ["alpha_probe", "beta_probe"],
        )
        self.assertIn("alpha-v2", updated["new_tools"][0]["implementation_py"])
        self.assertIn("beta", updated["new_tools"][1]["implementation_py"])
        _, changed = hs.differs_from_base(updated, round2)
        self.assertEqual(changed, ["new_tools.alpha_probe"])

    def test_proposer_can_update_inherited_tool_skill_and_middleware_files(self) -> None:
        round1 = hs.merge_with_base(_patch("alpha"), self.base)
        round2 = hs.merge_with_base(_patch("beta"), round1)
        update = {
            "schema": hs.SCHEMA_VERSION,
            "new_tools": [_tool("alpha_probe", "alpha-tool-v2")],
            "new_skills": [_skill("alpha-strategy", "alpha-skill-v2")],
            "new_middlewares": [
                _middleware("alpha_guard", "alpha-middleware-v2")
            ],
        }
        effective = hs.merge_with_base(update, round2)

        self.assertEqual(hs.generated_component_inventory(effective), {
            "new_tools": ["alpha_probe", "beta_probe"],
            "new_skills": ["alpha-strategy", "beta-strategy"],
            "new_middlewares": ["alpha_guard", "beta_guard"],
        })
        _, changed = hs.differs_from_base(effective, round2)
        self.assertEqual(changed, [
            "new_tools.alpha_probe",
            "new_skills.alpha-strategy",
            "new_middlewares.alpha_guard",
        ])
        lineage = hs.generated_component_lineage(round2, effective)
        for field in hs.GENERATED_COMPONENT_FIELDS:
            self.assertEqual(
                [row["status"] for row in lineage[field]],
                ["updated", "inherited"],
            )

        with tempfile.TemporaryDirectory() as td:
            package = Path(td) / "updated"
            effective = _with_owned_component_prompt(effective)
            materialize(
                effective,
                package,
                meta={"effective": effective, "component_lineage": lineage},
            )
            self.assertIn(
                "alpha-tool-v2",
                (package / "custom_tools" / "alpha_probe.py").read_text(),
            )
            self.assertIn(
                "alpha-skill-v2",
                (package / "skills" / "alpha-strategy" / "SKILL.md").read_text(),
            )
            self.assertIn(
                "alpha-middleware-v2",
                (package / "middlewares" / "alpha_guard.py").read_text(),
            )
            self.assertTrue((package / "custom_tools" / "beta_probe.py").is_file())
            self.assertTrue(
                (package / "skills" / "beta-strategy" / "SKILL.md").is_file()
            )
            self.assertTrue((package / "middlewares" / "beta_guard.py").is_file())

    def test_two_round_materialize_read_round_trip_preserves_every_component(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            round1 = _with_owned_component_prompt(
                hs.merge_with_base(_patch("alpha"), self.base)
            )
            package1 = root / "round1"
            materialize(
                round1,
                package1,
                raw_spec_text="schema: h2spec/1.0",
                meta={
                    "effective": round1,
                    "component_lineage": hs.generated_component_lineage(
                        self.base, round1
                    ),
                },
            )
            recovered1 = hs.read_base_spec(package1)
            self.assertEqual(
                hs.generated_component_inventory(recovered1),
                hs.generated_component_inventory(round1),
            )
            prompt1 = (package1 / "prompt.md").read_text()
            self.assertEqual(prompt1.strip(), round1["system_prompt"])
            self.assertIn("alpha_probe", prompt1)
            self.assertIn("alpha-strategy", prompt1)
            self.assertIn("alpha_guard", prompt1)

            round2 = _with_owned_component_prompt(
                hs.merge_with_base(_patch("beta"), recovered1)
            )
            lineage2 = hs.generated_component_lineage(recovered1, round2)
            package2 = root / "round2"
            materialize(
                round2,
                package2,
                raw_spec_text="schema: h2spec/1.0",
                meta={"effective": round2, "component_lineage": lineage2},
            )
            recovered2 = hs.read_base_spec(package2)

            expected = {
                "new_tools": ["alpha_probe", "beta_probe"],
                "new_skills": ["alpha-strategy", "beta-strategy"],
                "new_middlewares": ["alpha_guard", "beta_guard"],
            }
            self.assertEqual(hs.generated_component_inventory(recovered2), expected)
            prompt2 = (package2 / "prompt.md").read_text()
            for name in (
                "alpha_probe", "beta_probe", "alpha-strategy", "beta-strategy",
                "alpha_guard", "beta_guard",
            ):
                self.assertIn(name, prompt2)
            self.assertIn("probe_solution", prompt2)
            self.assertEqual(prompt2.strip(), round2["system_prompt"])
            self.assertEqual(recovered2["system_prompt"], round2["system_prompt"])
            manifest = json.loads(
                (package2 / "component_manifest.json").read_text()
            )
            self.assertEqual(manifest["inventory"], expected)
            for field in hs.GENERATED_COMPONENT_FIELDS:
                self.assertEqual(
                    [row["status"] for row in manifest["lineage"][field]],
                    ["inherited", "added"],
                )

    def test_declared_component_missing_from_package_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            package = Path(td) / "candidate"
            effective = _with_owned_component_prompt(
                hs.merge_with_base(_patch("alpha"), self.base)
            )
            materialize(effective, package, meta={"effective": effective})
            (package / "custom_tools" / "alpha_probe.py").unlink()

            with self.assertRaisesRegex(ValueError, "has no implementation"):
                hs.read_base_spec(package)


if __name__ == "__main__":
    unittest.main()
