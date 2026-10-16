from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from outer.reviewer.selftest import (  # noqa: E402
    schema_selftest_cases,
    selftest_tool_code,
)
from outer.safety.task_profiles import trusted_component_profile  # noqa: E402
from outer.reviewer.reviewer import review_tool_code  # noqa: E402


class GeneratedToolSelftestTest(unittest.TestCase):
    CP26_PROFILE = trusted_component_profile(
        "eft__math__circle_packing", "constraint_checker"
    )
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
        "additionalProperties": False,
    }

    def test_schema_samples_cover_every_enum_branch(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "mode": {"type": "string", "enum": ["safe", "broken"]},
            },
            "required": ["count", "mode"],
        }
        cases = schema_selftest_cases(schema)
        self.assertIn({"count": 2, "mode": "safe"}, cases)
        self.assertIn({"count": 2, "mode": "broken"}, cases)

        code = (
            "def run(ctx, args):\n"
            "    if args.get('mode') == 'broken':\n"
            "        raise ValueError('broken enum branch')\n"
            "    return {'ok': True}\n"
        )
        ok, detail = selftest_tool_code(code, input_schema=schema)
        self.assertFalse(ok)
        self.assertIn("broken enum branch", detail)

    def test_pattern_sample_is_truly_valid_and_correct_tool_passes(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "token": {"type": "string", "pattern": "^[0-9]+$"},
            },
            "required": ["token"],
            "additionalProperties": False,
        }
        cases = schema_selftest_cases(schema)
        self.assertTrue(cases)
        self.assertTrue(all(case["token"].isdigit() for case in cases))
        code = (
            "def run(ctx, args):\n"
            "    token = args.get('token', '')\n"
            "    if not token.isdigit():\n"
            "        return {'error': 'digits required'}\n"
            "    return {'digits': len(token)}\n"
        )
        ok, detail = selftest_tool_code(code, input_schema=schema)
        self.assertTrue(ok, detail)

    def test_oneof_property_exercises_each_valid_branch(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "integer", "minimum": 1},
                        {"type": "string", "enum": ["safe"]},
                    ],
                },
            },
            "required": ["value"],
            "additionalProperties": False,
        }
        cases = schema_selftest_cases(schema)
        self.assertTrue(any(isinstance(case["value"], int) for case in cases))
        self.assertTrue(any(case["value"] == "safe" for case in cases))

    def test_exclusive_numeric_bounds_generate_only_valid_values(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer", "exclusiveMinimum": 10,
                    "exclusiveMaximum": 12,
                },
                "ratio": {
                    "type": "number", "exclusiveMinimum": 0.9,
                    "exclusiveMaximum": 1.0,
                },
            },
            "required": ["count", "ratio"],
            "additionalProperties": False,
        }
        cases = schema_selftest_cases(schema)
        self.assertTrue(cases)
        self.assertTrue(all(case["count"] == 11 for case in cases))
        self.assertTrue(all(0.9 < case["ratio"] < 1.0 for case in cases))

    def test_schema_references_are_rejected_without_retrieval(self) -> None:
        code = "def run(ctx, args):\n    return {'ok': True}\n"
        references = (
            "https://attacker.invalid/generated-schema.json",
            "#/definitions/missing",
        )
        for reference in references:
            with self.subTest(reference=reference), patch(
                "urllib.request.urlopen"
            ) as retrieve:
                schema = {
                    "type": "object",
                    "properties": {"token": {"$ref": reference}},
                    "required": ["token"],
                    "additionalProperties": False,
                }
                ok, detail = selftest_tool_code(code, input_schema=schema)
                self.assertFalse(ok)
                self.assertIn("references are forbidden", detail)
                retrieve.assert_not_called()

    def test_huge_schema_allocations_fail_before_sample_construction(self) -> None:
        code = "def run(ctx, args):\n    return {'ok': True}\n"
        oversized_properties = (
            {"type": "array", "items": {"type": "number"},
             "minItems": 1_000_000_000},
            {"type": "string", "minLength": 1_000_000_000},
        )
        for property_schema in oversized_properties:
            with self.subTest(property_schema=property_schema):
                schema = {
                    "type": "object",
                    "properties": {"payload": property_schema},
                    "required": ["payload"],
                    "additionalProperties": False,
                }
                ok, detail = selftest_tool_code(code, input_schema=schema)
                self.assertFalse(ok)
                self.assertIn("bounded semantic self-test limit", detail)

    def test_deep_schema_fails_at_bounded_walker(self) -> None:
        nested = {"type": "number"}
        for _ in range(30):
            nested = {"type": "array", "items": nested}
        schema = {
            "type": "object",
            "properties": {"payload": nested},
            "required": ["payload"],
            "additionalProperties": False,
        }
        ok, detail = selftest_tool_code(
            "def run(ctx, args):\n    return {'ok': True}\n",
            input_schema=schema,
        )
        self.assertFalse(ok)
        self.assertIn("bounded semantic self-test structure limits", detail)

    def test_untrusted_regex_chain_is_rejected_before_execution(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "token": {
                    "type": "string",
                    "pattern": "^" + "a*" * 40 + "b$",
                    "minLength": 100,
                },
            },
            "required": ["token"],
            "additionalProperties": False,
        }
        ok, detail = selftest_tool_code(
            "def run(ctx, args):\n    return {'ok': True}\n",
            input_schema=schema,
        )
        self.assertFalse(ok)
        self.assertIn("outside the safe subset", detail)

    def test_json_array_is_exercised_as_the_runtime_will_deliver_it(self) -> None:
        schema = {
            "type": "object",
            "properties": {"weight": {"type": "array"}},
            "required": ["weight"],
        }
        code = (
            "def run(ctx, args):\n"
            "    weight = args.get('weight')\n"
            "    if weight is None:\n"
            "        return {'error': 'missing'}\n"
            "    return {'shape': weight.shape}\n"
        )
        ok, detail = selftest_tool_code(code, input_schema=schema)
        self.assertFalse(ok)
        self.assertIn("has no attribute 'shape'", detail)

    def test_schema_samples_respect_scalar_and_array_upper_bounds(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 0, "maximum": 0},
                "ratio": {"type": "number", "minimum": 0.1, "maximum": 0.2},
                "tag": {"type": "string", "maxLength": 1},
                "items": {
                    "type": "array", "items": {"type": "number"},
                    "maxItems": 1,
                },
            },
            "required": ["count", "ratio", "tag", "items"],
        }
        populated = schema_selftest_cases(schema)[1]
        self.assertEqual(populated["count"], 0)
        self.assertLessEqual(populated["ratio"], 0.2)
        self.assertEqual(len(populated["tag"]), 1)
        self.assertEqual(len(populated["items"]), 1)

    def test_schema_compatible_tool_passes_empty_and_populated_cases(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "number"}}
            },
        }
        code = (
            "def run(ctx, args):\n"
            "    values = args.get('values', [])\n"
            "    return {'total': sum(values)}\n"
        )
        ok, detail = selftest_tool_code(code, input_schema=schema)
        self.assertTrue(ok, detail)

    def test_schema_valid_sample_cannot_pass_via_distinct_error_output(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "centers": {"type": "array", "items": {"type": "number"}},
                "radii": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["centers", "radii"],
        }
        code = (
            "def run(ctx, args):\n"
            "    centers = args.get('centers') or []\n"
            "    radii = args.get('radii') or []\n"
            "    if len(radii) != 26:\n"
            "        return {'error': 'expected 26 radii'}\n"
            "    if len(centers) != 52:\n"
            "        return {'error': 'expected 52 center coordinates'}\n"
            "    return {'pairs_checked': 325}\n"
        )
        ok, detail = selftest_tool_code(code, input_schema=schema)
        self.assertFalse(ok)
        self.assertIn("schema-valid sample", detail)
        self.assertIn("expected 26 radii", detail)

    def test_schema_valid_sample_cannot_pass_with_empty_result(self) -> None:
        schema = {
            "type": "object",
            "properties": {"program": {"type": "string"}},
            "required": ["program"],
        }
        code = (
            "def run(ctx, args):\n"
            "    program = args.get('program')\n"
            "    if program is None:\n"
            "        return {'error': 'program required'}\n"
            "    lines = [line for line in program.splitlines() if line]\n"
            "    return {}\n"
        )
        ok, detail = selftest_tool_code(code, input_schema=schema)
        self.assertFalse(ok)
        self.assertIn("empty result: dict", detail)

    def test_exact_array_cardinality_schema_reaches_success_path(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "centers": {
                    "type": "array", "items": {"type": "number"},
                    "minItems": 52, "maxItems": 52,
                },
                "radii": {
                    "type": "array", "items": {"type": "number"},
                    "minItems": 26, "maxItems": 26,
                },
            },
            "required": ["centers", "radii"],
        }
        code = (
            "def run(ctx, args):\n"
            "    centers = args.get('centers') or []\n"
            "    radii = args.get('radii') or []\n"
            "    if len(centers) != 52 or len(radii) != 26:\n"
            "        return {'error': 'wrong shape'}\n"
            "    return {'pairs_checked': len(radii) * 25 // 2}\n"
        )
        ok, detail = selftest_tool_code(code, input_schema=schema)
        self.assertTrue(ok, detail)

    def test_diagnostic_errors_collection_is_a_valid_result(self) -> None:
        schema = {
            "type": "object",
            "properties": {"program": {"type": "string"}},
            "required": ["program"],
        }
        code = (
            "def run(ctx, args):\n"
            "    program = args.get('program')\n"
            "    if program is None:\n"
            "        return {'error': 'program is required'}\n"
            "    errors = ['missing return'] if 'return' not in program else []\n"
            "    return {'errors': errors, 'line_count': len(program.splitlines())}\n"
        )
        ok, detail = selftest_tool_code(code, input_schema=schema)
        self.assertTrue(ok, detail)

    def test_cp26_adversarial_matrix_rejects_false_feasible_checker(self) -> None:
        code = (
            "def run(ctx, args):\n"
            "    centers = args.get('centers') or []\n"
            "    radii = args.get('radii') or []\n"
            "    return {'ok': True, 'feasible': True, 'n': len(centers), "
            "'sum_radii': sum(radii)}\n"
        )
        ok, detail = selftest_tool_code(
            code,
            input_schema=self.CP26_SCHEMA,
            description="Feasibility constraint checker for n=26 circle packing radii.",
            validation_profile=self.CP26_PROFILE,
        )
        self.assertFalse(ok)
        self.assertIn("adversarial CP26 sample", detail)

    def test_cp26_checker_rejecting_nonfinite_and_bad_shapes_passes(self) -> None:
        code = '''import math

def run(ctx, args):
    centers = args.get("centers")
    radii = args.get("radii")
    if not isinstance(centers, list) or not isinstance(radii, list):
        return {"ok": False, "error": "lists required"}
    if len(centers) != 26 or len(radii) != 26:
        return {"ok": False, "error": "exactly 26 circles required"}
    clean_centers = []
    clean_radii = []
    for center in centers:
        if not isinstance(center, list) or len(center) != 2:
            return {"ok": False, "error": "each center needs two values"}
        x, y = float(center[0]), float(center[1])
        if not math.isfinite(x) or not math.isfinite(y):
            return {"ok": False, "error": "center must be finite"}
        clean_centers.append((x, y))
    for radius in radii:
        value = float(radius)
        if not math.isfinite(value) or value < 0.0:
            return {"ok": False, "error": "radius must be finite and nonnegative"}
        clean_radii.append(value)
    if (any(abs(x) > 1e100 or abs(y) > 1e100 for x, y in clean_centers)
            or any(radius > 1e100 for radius in clean_radii)):
        return {"ok": False, "error": "unsafe geometry magnitude"}
    violations = []
    repair_suggestions = []
    for i, ((x, y), radius) in enumerate(zip(clean_centers, clean_radii)):
        if min(x, y, 1.0 - x, 1.0 - y) - radius < 0.0:
            violations.append(i)
            if radius > 0.5:
                repair_suggestions.append({
                    "kind": "boundary", "circle": i, "repairable": False,
                    "reason": "no center-only repair for radius above 0.5",
                })
            else:
                repair_suggestions.append({
                    "kind": "boundary", "circle": i,
                    "suggested_center_shift": [0.5 - x, 0.5 - y],
                })
        for j in range(i + 1, 26):
            dx = x - clean_centers[j][0]
            dy = y - clean_centers[j][1]
            distance = math.hypot(dx, dy)
            if distance < radius + clean_radii[j]:
                violations.append([i, j])
                if (distance == 0.0 and radius == 0.5
                        and clean_radii[j] == 0.5):
                    repair_suggestions.append({
                        "kind": "overlap", "pair": [i, j],
                        "repairable": False,
                        "reason": "both centers are pinned by the walls",
                    })
                    continue
                ux, uy = ((dx / distance, dy / distance)
                          if distance > 0.0 else (1.0, 0.0))
                step = (radius + clean_radii[j] - distance) / 2.0 + 1e-6
                repair_suggestions.append({
                    "kind": "overlap", "pair": [i, j],
                    "suggested_shift_i": [ux * step, uy * step],
                    "suggested_shift_j": [-ux * step, -uy * step],
                })
    return {"ok": True, "feasible": not violations,
            "violations": violations, "n": 26,
            "repair_suggestions": repair_suggestions}
'''
        ok, detail = selftest_tool_code(
            code,
            input_schema=self.CP26_SCHEMA,
            description="Feasibility constraint checker for 26-circle packing radii.",
            validation_profile=self.CP26_PROFILE,
        )
        self.assertTrue(ok, detail)

    def test_cp26_required_tolerance_and_subscript_feasible_output_pass(self):
        schema = {
            **self.CP26_SCHEMA,
            "properties": {
                **self.CP26_SCHEMA["properties"],
                "tolerance": {
                    "type": "number", "minimum": 0.0, "maximum": 0.0,
                },
            },
            "required": ["centers", "radii", "tolerance"],
        }
        code = '''import math

def run(ctx, args):
    centers = args.get("centers")
    radii = args.get("radii")
    tolerance = float(args["tolerance"])
    if not math.isfinite(tolerance) or tolerance < 0.0:
        return {"ok": False, "error": "bad tolerance"}
    if not isinstance(centers, list) or not isinstance(radii, list):
        return {"ok": False, "error": "lists required"}
    if len(centers) != 26 or len(radii) != 26:
        return {"ok": False, "error": "exactly 26 required"}
    clean_centers = []
    clean_radii = []
    for center in centers:
        if not isinstance(center, list) or len(center) != 2:
            return {"ok": False, "error": "bad center"}
        x, y = float(center[0]), float(center[1])
        if not math.isfinite(x) or not math.isfinite(y):
            return {"ok": False, "error": "nonfinite center"}
        clean_centers.append((x, y))
    for radius in radii:
        value = float(radius)
        if not math.isfinite(value) or value < 0.0:
            return {"ok": False, "error": "bad radius"}
        clean_radii.append(value)
    if (any(abs(x) > 1e100 or abs(y) > 1e100 for x, y in clean_centers)
            or any(radius > 1e100 for radius in clean_radii)):
        return {"ok": False, "error": "unsafe geometry magnitude"}
    violations = []
    repair_suggestions = []
    for i, ((x, y), radius) in enumerate(zip(clean_centers, clean_radii)):
        if min(x, y, 1.0 - x, 1.0 - y) - radius < -tolerance:
            violations.append(["wall", i])
            if radius > 0.5:
                repair_suggestions.append({
                    "kind": "boundary", "circle": i, "repairable": False,
                    "reason": "no center-only repair for radius above 0.5",
                })
            else:
                repair_suggestions.append({
                    "kind": "boundary", "circle": i,
                    "suggested_center_shift": [0.5 - x, 0.5 - y],
                })
        for j in range(i):
            distance = math.hypot(
                x - clean_centers[j][0], y - clean_centers[j][1]
            )
            if distance - radius - clean_radii[j] < -tolerance:
                violations.append(["pair", j, i])
                if (distance == 0.0 and radius == 0.5
                        and clean_radii[j] == 0.5):
                    repair_suggestions.append({
                        "kind": "overlap", "pair": [j, i],
                        "repairable": False,
                        "reason": "both centers are pinned by the walls",
                    })
                    continue
                dx = x - clean_centers[j][0]
                dy = y - clean_centers[j][1]
                ux, uy = ((dx / distance, dy / distance)
                          if distance > 0.0 else (1.0, 0.0))
                step = (radius + clean_radii[j] - distance) / 2.0 + 1e-6
                repair_suggestions.append({
                    "kind": "overlap", "pair": [j, i],
                    "suggested_shift_i": [-ux * step, -uy * step],
                    "suggested_shift_j": [ux * step, uy * step],
                })
    out = {"ok": True, "violations": violations,
           "repair_suggestions": repair_suggestions}
    out["feasible"] = not violations
    return out
'''
        ok, detail = selftest_tool_code(
            code, input_schema=schema,
            description="Audit the supplied public-task layout.",
            validation_profile=self.CP26_PROFILE,
        )
        self.assertTrue(ok, detail)

    def test_cp26_checker_cannot_validate_shapes_then_report_every_layout_feasible(self):
        code = '''import math

def run(ctx, args):
    centers = args.get("centers")
    radii = args.get("radii")
    if not isinstance(centers, list) or not isinstance(radii, list):
        return {"ok": False, "error": "lists required"}
    if len(centers) != 26 or len(radii) != 26:
        return {"ok": False, "error": "exactly 26 required"}
    for center in centers:
        if not isinstance(center, list) or len(center) != 2:
            return {"ok": False, "error": "bad center"}
        if not all(math.isfinite(float(value)) for value in center):
            return {"ok": False, "error": "nonfinite center"}
    for radius in radii:
        if not math.isfinite(float(radius)) or float(radius) < 0:
            return {"ok": False, "error": "bad radius"}
    if (any(abs(float(value)) > 1e100 for row in centers for value in row)
            or any(float(radius) > 1e100 for radius in radii)):
        return {"ok": False, "error": "unsafe geometry magnitude"}
    return {"ok": True, "feasible": True, "violations": []}
'''
        ok, detail = selftest_tool_code(
            code, input_schema=self.CP26_SCHEMA,
            description="Audit this public-task layout.",
            validation_profile=self.CP26_PROFILE,
        )
        self.assertFalse(ok)
        self.assertIn("semantic CP26 sample known_overlap", detail)

    @staticmethod
    def _repair_checker(*, reverse_wall: bool = False,
                        move_pair_together: bool = False) -> str:
        wall_shift = "[-breach, 0.0]" if reverse_wall else "[breach, 0.0]"
        pair_i = "[breach, 0.0]" if move_pair_together else "[-breach / 2.0, 0.0]"
        pair_j = "[breach, 0.0]" if move_pair_together else "[breach / 2.0, 0.0]"
        return f'''import math

def run(ctx, args):
    centers = args.get("centers")
    radii = args.get("radii")
    if not isinstance(centers, list) or not isinstance(radii, list):
        return {{"error": "lists required", "feasible": False}}
    if len(centers) != 26 or len(radii) != 26:
        return {{"error": "exactly 26 required", "feasible": False}}
    try:
        centers = [[float(v) for v in row] for row in centers]
        radii = [float(v) for v in radii]
    except Exception:
        return {{"error": "numeric values required", "feasible": False}}
    if (any(len(row) != 2 or not all(math.isfinite(v) for v in row)
            for row in centers)
            or any(not math.isfinite(r) or r < 0.0 for r in radii)):
        return {{"error": "finite nonnegative geometry required", "feasible": False}}
    if (any(abs(v) > 1e100 for row in centers for v in row)
            or any(r > 1e100 for r in radii)):
        return {{"error": "unsafe geometry magnitude", "feasible": False}}
    violations = []
    repairs = []
    for i, ((x, y), radius) in enumerate(zip(centers, radii)):
        slack = min(x, y, 1.0 - x, 1.0 - y) - radius
        if slack < 0.0:
            breach = -slack
            violations.append(["wall", i])
            if radius > 0.5:
                repairs.append({{"kind": "boundary", "circle": i,
                                "repairable": False,
                                "reason": "no center-only repair"}})
            else:
                repairs.append({{"kind": "boundary", "circle": i,
                                "suggested_center_shift": {wall_shift}}})
        for j in range(i):
            distance = math.hypot(x - centers[j][0], y - centers[j][1])
            margin = distance - radius - radii[j]
            if margin < 0.0:
                breach = -margin
                violations.append(["pair", j, i])
                if distance == 0.0 and radius == 0.5 and radii[j] == 0.5:
                    repairs.append({{"kind": "separation", "pair": [j, i],
                                    "repairable": False,
                                    "reason": "both centers are pinned"}})
                else:
                    repairs.append({{"kind": "separation", "pair": [j, i],
                                    "suggested_shift_i": {pair_i},
                                    "suggested_shift_j": {pair_j}}})
    return {{"feasible": not violations, "violations": violations,
            "repair_suggestions": repairs}}
'''

    def test_cp26_reversed_wall_repair_is_rejected(self) -> None:
        ok, detail = selftest_tool_code(
            self._repair_checker(reverse_wall=True),
            input_schema=self.CP26_SCHEMA,
            validation_profile=self.CP26_PROFILE,
        )
        self.assertFalse(ok)
        self.assertIn("known_wall_violation", detail)
        self.assertIn("does not strictly improve", detail)

    def test_cp26_coincident_repairs_must_separate_both_circles(self) -> None:
        ok, detail = selftest_tool_code(
            self._repair_checker(move_pair_together=True),
            input_schema=self.CP26_SCHEMA,
            validation_profile=self.CP26_PROFILE,
        )
        self.assertFalse(ok)
        self.assertIn("known_overlap", detail)
        self.assertIn("must increase distance", detail)

    def test_cp26_zero_radius_is_valid(self) -> None:
        code = self._repair_checker().replace(
            "r < 0.0 for r in radii",
            "r <= 0.0 for r in radii",
        )
        ok, detail = selftest_tool_code(
            code,
            input_schema=self.CP26_SCHEMA,
            validation_profile=self.CP26_PROFILE,
        )
        self.assertFalse(ok)
        self.assertIn("known_zero_radius", detail)

    def test_nonfinite_tool_output_is_never_accepted(self) -> None:
        code = (
            "def run(ctx, args):\n"
            "    return {'diagnostic': float('nan')}\n"
        )
        ok, detail = selftest_tool_code(
            code, input_schema={"type": "object", "properties": {}}
        )
        self.assertFalse(ok)
        self.assertIn("non-JSON-safe or non-finite", detail)

    def test_dynamic_unsupported_optimality_output_key_is_rejected(self):
        schema = {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array", "items": {"type": "number"},
                    "minItems": 1,
                },
            },
            "required": ["values"],
        }
        code = (
            "def run(ctx, args):\n"
            "    values = args.get('values') or []\n"
            "    total = sum(float(x) for x in values)\n"
            "    key = '_'.join(('global', 'optimum'))\n"
            "    return {key: total, 'count': len(values)}\n"
        )
        outcome = review_tool_code(
            code, input_schema=schema, require_utility=True, max_rounds=0,
        )
        self.assertFalse(outcome.ok)
        self.assertIn(
            "runtime result exposes unsupported", outcome.final_error or ""
        )


if __name__ == "__main__":
    unittest.main()
