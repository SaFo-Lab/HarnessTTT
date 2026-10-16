"""Subprocess self-test for generated tool code.

Runs ``run(MockContext, args)`` in an isolated python with a hard timeout and
resource limits. Schema-derived examples are checked with the declared JSON
Schema before execution and exercise required inputs plus bounded alternatives.
Every generated schema-valid example must reach a non-error result; returning
a different validation error for each sample is not executable evidence.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MAX_SCHEMA_DEPTH = 24
MAX_SCHEMA_NODES = 1024
MAX_SCHEMA_ALTERNATIVES = 64
MAX_SCHEMA_MIN_ITEMS = 512
MAX_SCHEMA_MIN_LENGTH = 4096
MAX_SCHEMA_SAMPLE_COST = 8192

_RUNNER = r'''
import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
sys.path.insert(0, {src_path!r})
from inner.runtime.harness_sdk import MockContext
ns = {{}}
code = open({code_path!r}).read()
exec(compile(code, "generated_tool.py", "exec"), ns)
samples = json.loads({samples_json!r})
success_indices = set(json.loads({success_indices_json!r}))
reject_indices = set(json.loads({reject_indices_json!r}))
reject_labels = json.loads({reject_labels_json!r})
semantic_expectations = json.loads({semantic_expectations_json!r})
unsupported_output_keys = set(json.loads({unsupported_output_keys_json!r}))
require_observable_variation = {require_observable_variation!r}
require_current_candidate_bridge = {require_current_candidate_bridge!r}
outcomes = []

def nested_output_keys(value):
    keys = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).strip().lower())
            keys.update(nested_output_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.update(nested_output_keys(child))
    return keys

def explicit_failure(value):
    if value is None:
        return "empty result: None"
    if isinstance(value, (str, list, tuple, dict)) and len(value) == 0:
        return "empty result: %s" % type(value).__name__
    if isinstance(value, dict):
        # ``errors`` may be the useful diagnostic payload of a checker (for
        # example compiler errors found in a candidate). Singular ``error``
        # and ``exception`` are the generated-tool convention for an API/input
        # failure and therefore cannot satisfy a schema-valid sample.
        for key in ("error", "exception"):
            marker = value.get(key)
            if marker not in (None, "", [], {{}}):
                return "%s=%r" % (key, marker)
        status = str(value.get("status") or "").strip().lower()
        if status in {{"error", "failed", "failure", "invalid_input",
                      "invalid_arguments"}}:
            return "status=%r" % status
    if isinstance(value, str):
        marker = " ".join(value.strip().lower().split())
        prefixes = (
            "error", "invalid input", "invalid argument", "missing required",
            "required argument", "must provide",
        )
        if marker.startswith(prefixes):
            return value[:300]
    return None

def explicit_rejection(value):
    if explicit_failure(value) is not None:
        return True
    if not isinstance(value, dict):
        return False
    if value.get("ok") is False or value.get("feasible") is False \
            or value.get("valid") is False:
        return True
    status = str(value.get("status") or "").strip().lower()
    return status in {{"error", "failed", "failure", "invalid",
                      "invalid_input", "invalid_arguments", "rejected"}}

def nonempty_violation_claim(value):
    if not isinstance(value, dict):
        return None
    for key, child in value.items():
        label = str(key).strip().lower()
        if "violation" in label:
            if isinstance(child, (list, tuple, dict)) and len(child) > 0:
                return "%s=%r" % (key, child)
            if isinstance(child, bool) and child:
                return "%s=True" % key
            if isinstance(child, (int, float)) and not isinstance(child, bool) \
                    and child > 1e-12:
                return "%s=%r" % (key, child)
        nested = nonempty_violation_claim(child)
        if nested:
            return "%s.%s" % (key, nested)
    return None

def finite_vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        vector = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None
    if not all(__import__("math").isfinite(item) for item in vector):
        return None
    return vector

def repair_vectors(value):
    """Return shifts from the one trusted, machine-checkable repair surface."""
    per_index = {{}}
    per_pair = []
    if not isinstance(value, dict):
        return per_index, per_pair, False, []
    claimed = "repair_suggestions" in value
    rows = value.get("repair_suggestions")
    machine_rows = rows if isinstance(rows, list) else []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("kind") or "") == "boundary":
                try:
                    index = int(row.get("circle", row.get("index")))
                except (TypeError, ValueError):
                    continue
                vector = finite_vector(row.get("suggested_center_shift"))
                if vector is not None:
                    per_index[index] = vector
            pair = row.get("pair")
            shift_i = finite_vector(row.get("suggested_shift_i"))
            shift_j = finite_vector(row.get("suggested_shift_j"))
            if isinstance(pair, (list, tuple)) and len(pair) == 2 \
                    and shift_i is not None and shift_j is not None:
                per_pair.append((int(pair[0]), int(pair[1]), shift_i, shift_j))
    return per_index, per_pair, claimed, machine_rows

def explicitly_unrepairable(row):
    if not isinstance(row, dict):
        return False
    status = str(row.get("status") or "").strip().lower()
    reason = str(row.get("reason") or row.get("error") or "").strip()
    return (
        row.get("repairable") is False
        or status in {{"unrepairable", "impossible", "no_center_repair"}}
    ) and bool(reason)

def semantic_oracle(args, out, expectation):
    expected_feasible = bool(expectation.get("feasible"))
    observed = out.get("feasible") if isinstance(out, dict) else None
    if observed is not expected_feasible:
        return "expected feasible=%s; result=%s" % (
            expected_feasible, repr(out)[:600])
    if expected_feasible:
        contradiction = nonempty_violation_claim(out)
        if contradiction:
            return "feasible=True contradicts reported violation %s" % contradiction
    kind = str(expectation.get("kind") or "")
    centers = args.get(expectation.get("centers_field", "centers"))
    radii = args.get(expectation.get("radii_field", "radii"))
    shifts, pair_shifts, claimed, repair_rows = repair_vectors(out)
    if kind == "wall" and claimed:
        index = int(expectation["target"])
        repairable = expectation.get("repairable", True) is not False
        relevant = [
            row for row in repair_rows
            if isinstance(row, dict)
            and str(row.get("kind") or "") == "boundary"
            and int(row.get("circle", row.get("index", -1))) == index
        ]
        if not repairable and any(explicitly_unrepairable(row) for row in relevant):
            return None
        if index not in shifts:
            suffix = (
                " or an explicit repairable=false row with a reason"
                if not repairable else ""
            )
            return "repair surface omitted the known wall-violating circle %d%s" % (
                index, suffix)
        x, y = [float(v) for v in centers[index]]
        radius = float(radii[index])
        before = min(x, y, 1.0 - x, 1.0 - y) - radius
        dx, dy = shifts[index]
        after = min(x + dx, y + dy, 1.0 - x - dx, 1.0 - y - dy) - radius
        if not __import__("math").isfinite(after) or not after > before:
            return ("wall repair for circle %d does not strictly improve target "
                    "slack: before=%r after=%r shift=%r" % (
                        index, before, after, shifts[index]))
    if kind == "coincident" and claimed:
        i, j = [int(value) for value in expectation["pair"]]
        repairable = expectation.get("repairable", True) is not False
        relevant = [
            row for row in repair_rows
            if isinstance(row, dict)
            and isinstance(row.get("pair"), (list, tuple))
            and set(int(value) for value in row.get("pair")) == {{i, j}}
        ]
        if not repairable and any(explicitly_unrepairable(row) for row in relevant):
            return None
        before = __import__("math").hypot(
            float(centers[i][0]) - float(centers[j][0]),
            float(centers[i][1]) - float(centers[j][1]),
        )
        proposals = []
        if i in shifts and j in shifts:
            proposals.append((shifts[i], shifts[j]))
        proposals.extend(
            (shift_i, shift_j) for a, b, shift_i, shift_j in pair_shifts
            if (a, b) == (i, j) or (a, b) == (j, i)
        )
        if not proposals:
            suffix = (
                " or an explicit repairable=false row with a reason"
                if not repairable else ""
            )
            return "repair surface omitted paired shifts for coincident circles%s" % suffix
        improved = False
        for shift_i, shift_j in proposals:
            xi = float(centers[i][0]) + shift_i[0]
            yi = float(centers[i][1]) + shift_i[1]
            xj = float(centers[j][0]) + shift_j[0]
            yj = float(centers[j][1]) + shift_j[1]
            after = __import__("math").hypot(xi - xj, yi - yj)
            wall_i = min(xi, yi, 1.0 - xi, 1.0 - yi) - float(radii[i])
            wall_j = min(xj, yj, 1.0 - xj, 1.0 - yj) - float(radii[j])
            improved = improved or (
                __import__("math").isfinite(after)
                and after > before
                and wall_i >= 0.0 and wall_j >= 0.0
            )
        if not improved:
            return (
                "coincident-pair repair must increase distance without moving "
                "either circle through a wall"
            )
    return None

for sample_index, args in enumerate(samples):
    ctx = MockContext({scratch!r})
    reject_label = reject_labels.get(str(sample_index))
    try:
        out = ns["run"](ctx, args)
    except Exception:
        import traceback
        print(json.dumps({{
            "ok": False,
            "error": (
                "adversarial CP26 sample %s raised; exceptions are invalid, "
                "return a JSON-safe structured rejection instead: %s" % (
                    reject_label, traceback.format_exc(limit=4)[-800:]
                ) if reject_label else
                "schema sample %d args=%r: %s" % (
                    sample_index, args, traceback.format_exc(limit=4)[-1200:]
                )
            ),
        }}))
        raise SystemExit(0)
    ok_types = (str, dict, list, int, float, type(None))
    if not isinstance(out, ok_types):
        print(json.dumps({{
            "ok": False,
            "error": "schema sample %d returned %s; must be str/dict/list/number"
                     % (sample_index, type(out).__name__),
        }}))
        raise SystemExit(0)
    try:
        # A tool result is sent back through JSON.  NaN/Infinity are not JSON
        # values and can silently poison downstream comparisons if accepted.
        json.dumps(out, allow_nan=False)
    except (TypeError, ValueError) as exc:
        print(json.dumps({{
            "ok": False,
            "error": "%s returned a non-JSON-safe or non-finite result: %s" % (
                ("adversarial CP26 sample " + reject_label)
                if reject_label else ("sample %d" % sample_index), exc),
        }}))
        raise SystemExit(0)
    unsupported = sorted(nested_output_keys(out) & unsupported_output_keys)
    if unsupported:
        print(json.dumps({{
            "ok": False,
            "error": "runtime result exposes unsupported aggregate/global "
                     "optimality fields %s" % unsupported,
        }}))
        raise SystemExit(0)
    semantic_failure = explicit_failure(out)
    if sample_index in success_indices and semantic_failure is not None:
        print(json.dumps({{
            "ok": False,
            "error": "schema-valid sample %d args=%r returned an explicit "
                     "failure instead of exercising the capability: %s" % (
                         sample_index, args, semantic_failure
                     ),
        }}))
        raise SystemExit(0)
    if sample_index in reject_indices and not explicit_rejection(out):
        print(json.dumps({{
            "ok": False,
            "error": "adversarial CP26 sample %s was not explicitly "
                     "rejected; result=%s" % (
                         reject_label or sample_index, repr(out)[:600]),
        }}))
        raise SystemExit(0)
    expectation = semantic_expectations.get(str(sample_index))
    if expectation is not None:
        semantic_error = semantic_oracle(args, out, expectation)
        if semantic_error is not None:
            print(json.dumps({{
                "ok": False,
                "error": "semantic CP26 sample %s failed oracle: %s" % (
                    reject_label or sample_index, semantic_error),
            }}))
            raise SystemExit(0)
    outcomes.append({{"ctx_calls": ctx.calls, "empty_args": args == {{}},
                     "ret_type": type(out).__name__,
                     "output_repr": repr(out)[:1000],
                     "semantic_failure": semantic_failure}})

if require_current_candidate_bridge:
    empty_rows = [row for row in outcomes if row.get("empty_args")]
    if not empty_rows or not any(
        "get_candidate_geometry" in row.get("ctx_calls", [])
        for row in empty_rows
    ):
        print(json.dumps({{
            "ok": False,
            "error": "argument-free CP26 sample did not execute "
                     "ctx.get_candidate_geometry(); an unreachable AST call "
                     "does not satisfy the current-candidate contract",
        }}))
        raise SystemExit(0)
    if any(
        row.get("ctx_calls", []).count("get_candidate_geometry") != 1
        for row in empty_rows
    ):
        print(json.dumps({{
            "ok": False,
            "error": "argument-free CP26 path must call "
                     "get_candidate_geometry exactly once",
        }}))
        raise SystemExit(0)

    class BridgeFailureContext(MockContext):
        def get_candidate_geometry(self):
            payload = super().get_candidate_geometry()
            payload.update({{
                "ok": False,
                "error": "synthetic current-candidate execution failure",
            }})
            return payload

    class BridgeExceptionContext(MockContext):
        def get_candidate_geometry(self):
            self.calls.append("get_candidate_geometry")
            raise RuntimeError("synthetic bridge exception")

    for bridge_label, failed_ctx in (
        ("ok=false with stale geometry", BridgeFailureContext({scratch!r})),
        ("exception", BridgeExceptionContext({scratch!r})),
    ):
        try:
            bridge_failure_out = ns["run"](failed_ctx, {{}})
        except Exception:
            import traceback
            print(json.dumps({{
                "ok": False,
                "error": "current-candidate bridge %s path raised instead "
                         "of returning a structured rejection: %s" % (
                             bridge_label, traceback.format_exc(limit=4)[-900:]
                         ),
            }}))
            raise SystemExit(0)
        if failed_ctx.calls.count("get_candidate_geometry") != 1:
            print(json.dumps({{
                "ok": False,
                "error": "bridge %s sample must call get_candidate_geometry "
                         "exactly once" % bridge_label,
            }}))
            raise SystemExit(0)
        try:
            json.dumps(bridge_failure_out, allow_nan=False)
        except (TypeError, ValueError) as exc:
            print(json.dumps({{
                "ok": False,
                "error": "bridge %s result is not JSON-safe: %s" % (
                    bridge_label, exc),
            }}))
            raise SystemExit(0)
        if not explicit_rejection(bridge_failure_out):
            print(json.dumps({{
                "ok": False,
                "error": "current-candidate bridge %s must return a "
                         "structured rejection; result=%s" % (
                             bridge_label, repr(bridge_failure_out)[:600]
                         ),
            }}))
            raise SystemExit(0)
ctx_exercised = any(row["ctx_calls"] for row in outcomes)
outputs_vary = len({{row["output_repr"] for row in outcomes}}) > 1
if require_observable_variation and not ctx_exercised and not outputs_vary:
    print(json.dumps({{
        "ok": False,
        "error": "no schema sample exercised a ctx data/effect path and all "
                 "outputs were identical; constant/placeholder capability",
        "outcomes": outcomes,
    }}))
else:
    print(json.dumps({{"ok": True, "samples": len(samples),
                      "outcomes": outcomes}}))
'''


def _branch_schema(schema: Dict[str, Any], branch: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        key: value for key, value in schema.items()
        if key not in {"oneOf", "anyOf"}
    }
    merged.update(branch)
    return merged


def _string_candidates(schema: Dict[str, Any]) -> List[str]:
    minimum = max(int(schema.get("minLength", 0)), 0)
    maximum = schema.get("maxLength")
    maximum = max(int(maximum), 0) if maximum is not None else None
    seeds: List[str] = []
    for key in ("examples",):
        values = schema.get(key)
        if isinstance(values, list):
            seeds.extend(str(value) for value in values if isinstance(value, str))
    if isinstance(schema.get("example"), str):
        seeds.append(str(schema["example"]))
    seeds.extend(["0", "1", "26", "x", "a", "A", "safe", "sample", "a0", "0a", "_"])
    candidates: List[str] = []
    for seed in seeds:
        value = seed
        if len(value) < minimum:
            fill = value[-1:] or "x"
            value += fill * (minimum - len(value))
        if maximum is not None:
            value = value[:maximum]
        if value not in candidates:
            candidates.append(value)
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        try:
            matching = [value for value in candidates if re.search(pattern, value)]
        except re.error:
            matching = []
        if matching:
            return matching
    return candidates


def _schema_value(schema: Dict[str, Any]) -> Any:
    for keyword in ("oneOf", "anyOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict):
                    return _schema_value(_branch_schema(schema, branch))
    if "const" in schema:
        return schema["const"]
    if "default" in schema:
        return schema["default"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    kind = schema.get("type")
    if kind == "integer":
        value = max(int(schema.get("minimum", 2)), 1)
        if schema.get("exclusiveMinimum") is not None:
            value = max(value, int(schema["exclusiveMinimum"]) + 1)
        if schema.get("maximum") is not None:
            value = min(value, int(schema["maximum"]))
        if schema.get("exclusiveMaximum") is not None:
            upper = float(schema["exclusiveMaximum"])
            value = min(value, int(upper) - 1 if upper.is_integer() else int(upper))
        return value
    if kind == "number":
        value = max(float(schema.get("minimum", 1.0)), 0.5)
        if schema.get("exclusiveMinimum") is not None:
            lower = float(schema["exclusiveMinimum"])
            value = max(value, lower + max(abs(lower) * 1e-6, 1e-6))
        if schema.get("maximum") is not None:
            value = min(value, float(schema["maximum"]))
        if schema.get("exclusiveMaximum") is not None:
            upper = float(schema["exclusiveMaximum"])
            value = min(value, upper - max(abs(upper) * 1e-6, 1e-6))
        return value
    if kind == "boolean":
        return True
    if kind == "string":
        candidates = _string_candidates(schema)
        return candidates[0] if candidates else ""
    if kind == "array":
        item_schema = schema.get("items")
        item = _schema_value(item_schema) if isinstance(item_schema, dict) else 1.0
        minimum = max(int(schema.get("minItems", 0)), 0)
        count = max(minimum, 2)
        if schema.get("maxItems") is not None:
            count = min(count, max(int(schema["maxItems"]), 0))
        return [item for _ in range(count)]
    if kind == "object" or isinstance(schema.get("properties"), dict):
        properties = schema.get("properties") or {}
        return {
            name: _schema_value(prop)
            for name, prop in properties.items()
            if isinstance(prop, dict)
        }
    return None


def _schema_values(schema: Dict[str, Any]) -> List[Any]:
    """Bounded candidate values; the full-schema validator filters them."""

    values: List[Any] = [_schema_value(schema)]
    for keyword in ("enum", "examples"):
        alternatives = schema.get(keyword)
        if isinstance(alternatives, list):
            values.extend(alternatives)
    if "example" in schema:
        values.append(schema["example"])
    for keyword in ("oneOf", "anyOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            values.extend(
                _schema_value(_branch_schema(schema, branch))
                for branch in branches if isinstance(branch, dict)
            )
    kind = schema.get("type")
    if kind == "string":
        values.extend(_string_candidates(schema))
    elif kind == "boolean":
        values.extend([True, False])
    elif kind in {"integer", "number"}:
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if schema.get(key) is not None:
                values.append(schema[key])
    unique: List[Any] = []
    seen = set()
    for value in values:
        try:
            encoded = json.dumps(value, sort_keys=True)
        except (TypeError, ValueError):
            continue
        if encoded not in seen:
            seen.add(encoded)
            unique.append(value)
    return unique


def _reject_schema_references(value: Any, path: str = "$") -> None:
    """Do not let untrusted H1 schemas resolve local/file/network references."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"$ref", "$dynamicRef", "$recursiveRef"}:
                raise ValueError(
                    "input_schema references are forbidden in generated "
                    f"components ({child_path}); inline the bounded schema"
                )
            _reject_schema_references(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_schema_references(child, f"{path}[{index}]")


def _is_bounded_schema_pattern(pattern: str) -> bool:
    """Whitelist one anchored character class; never execute other regexes."""

    if len(pattern) > 256 or not pattern.startswith("^[") or not pattern.endswith("$"):
        return False
    closing = pattern.rfind("]")
    if closing < 3:
        return False
    body = pattern[2:closing]
    quantifier = pattern[closing + 1:-1]
    if not body or any(char in body for char in "[]()"):
        return False
    if quantifier in {"", "+", "*", "?"}:
        return True
    match = re.fullmatch(r"\{(\d{1,4})(?:,(\d{0,4}))?\}", quantifier)
    if match is None:
        return False
    lower = int(match.group(1))
    upper_text = match.group(2)
    upper = int(upper_text) if upper_text else lower
    return lower <= upper <= MAX_SCHEMA_MIN_LENGTH


def _check_schema_generation_limits(
    value: Any, path: str = "$", depth: int = 0,
    stats: Optional[Dict[str, int]] = None,
) -> None:
    """Bound allocations/regex work before deriving any untrusted samples."""

    if stats is None:
        stats = {"nodes": 0, "alternatives": 0}
    stats["nodes"] += 1
    if stats["nodes"] > MAX_SCHEMA_NODES or depth > MAX_SCHEMA_DEPTH:
        raise ValueError(
            "input_schema exceeds bounded semantic self-test structure limits"
        )
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict) and len(properties) > MAX_SCHEMA_ALTERNATIVES:
            raise ValueError(
                f"input_schema has too many properties at {path}: "
                f"{len(properties)} > {MAX_SCHEMA_ALTERNATIVES}"
            )
        for keyword in ("enum", "examples", "oneOf", "anyOf"):
            alternatives = value.get(keyword)
            if isinstance(alternatives, list):
                stats["alternatives"] += len(alternatives)
                if stats["alternatives"] > MAX_SCHEMA_ALTERNATIVES:
                    raise ValueError(
                        "input_schema exceeds bounded semantic self-test "
                        f"alternative limit ({MAX_SCHEMA_ALTERNATIVES})"
                    )
        minimum_items = value.get("minItems")
        if (isinstance(minimum_items, (int, float))
                and not isinstance(minimum_items, bool)
                and minimum_items > MAX_SCHEMA_MIN_ITEMS):
            raise ValueError(
                f"input_schema minItems at {path} exceeds bounded semantic "
                f"self-test limit {MAX_SCHEMA_MIN_ITEMS}"
            )
        minimum_length = value.get("minLength")
        if (isinstance(minimum_length, (int, float))
                and not isinstance(minimum_length, bool)
                and minimum_length > MAX_SCHEMA_MIN_LENGTH):
            raise ValueError(
                f"input_schema minLength at {path} exceeds bounded semantic "
                f"self-test limit {MAX_SCHEMA_MIN_LENGTH}"
            )
        pattern = value.get("pattern")
        if isinstance(pattern, str) and not _is_bounded_schema_pattern(pattern):
            raise ValueError(
                f"input_schema pattern at {path} is outside the safe subset; "
                "use one anchored character class with a bounded quantifier "
                "(for example ^[0-9]+$), enum, const, or example"
            )
        for key, child in value.items():
            _check_schema_generation_limits(
                child, f"{path}.{key}", depth + 1, stats
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_schema_generation_limits(
                child, f"{path}[{index}]", depth + 1, stats
            )


def _schema_sample_cost(schema: Dict[str, Any]) -> int:
    for keyword in ("oneOf", "anyOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list) and branches:
            costs = [
                _schema_sample_cost(_branch_schema(schema, branch))
                for branch in branches if isinstance(branch, dict)
            ]
            if costs:
                return max(costs)
    kind = schema.get("type")
    if kind == "array":
        minimum = max(int(schema.get("minItems", 0)), 0)
        count = max(minimum, 2)
        if schema.get("maxItems") is not None:
            count = min(count, max(int(schema["maxItems"]), 0))
        item_schema = schema.get("items")
        item_cost = (
            _schema_sample_cost(item_schema)
            if isinstance(item_schema, dict) else 1
        )
        return count * item_cost
    if kind == "object" or isinstance(schema.get("properties"), dict):
        properties = schema.get("properties") or {}
        return max(1, sum(
            _schema_sample_cost(prop)
            for prop in properties.values() if isinstance(prop, dict)
        ))
    if kind == "string":
        return max(int(schema.get("minLength", 0)), 1)
    return 1


def schema_selftest_cases(input_schema: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build bounded examples and retain only validator-confirmed valid ones."""

    cases: List[Dict[str, Any]] = [{}]
    if not isinstance(input_schema, dict):
        return cases
    _check_schema_generation_limits(input_schema)
    _reject_schema_references(input_schema)
    try:
        from jsonschema.validators import validator_for
        validator_class = validator_for(input_schema)
        validator_class.check_schema(input_schema)
        validator = validator_class(input_schema)
    except Exception as exc:
        raise ValueError(
            "input_schema validation failed safely during semantic self-test: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    sample_cost = _schema_sample_cost(input_schema)
    if sample_cost > MAX_SCHEMA_SAMPLE_COST:
        raise ValueError(
            "input_schema exceeds bounded semantic self-test sample cost: "
            f"{sample_cost} > {MAX_SCHEMA_SAMPLE_COST}"
        )
    raw_properties = input_schema.get("properties") or {}
    properties = raw_properties if isinstance(raw_properties, dict) else {}
    if properties:
        required = {
            str(name) for name in (input_schema.get("required") or [])
        }
        required_case = {
            name: _schema_value(prop)
            for name, prop in properties.items()
            if name in required and isinstance(prop, dict)
        }
        base = {
            name: _schema_value(prop)
            for name, prop in properties.items()
            if isinstance(prop, dict)
        }
        cases.append(required_case)
        cases.append(base)
        for name, prop in properties.items():
            if isinstance(prop, dict):
                cases.extend(
                    {**base, name: value}
                    for value in _schema_values(prop)[1:]
                )
    unique: List[Dict[str, Any]] = []
    seen = set()
    for case in cases:
        encoded = json.dumps(case, sort_keys=True)
        if encoded not in seen:
            seen.add(encoded)
            unique.append(case)
    try:
        valid = [case for case in unique if validator.is_valid(case)]
    except Exception as exc:
        raise ValueError(
            "input_schema validation failed safely during semantic self-test: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not valid:
        raise ValueError(
            "could not derive a JSON-Schema-valid self-test sample; provide "
            "a valid const/default/enum/example for constrained fields or "
            "simplify unsupported cross-field constraints"
        )
    return valid


def _cp26_constraint_adversarial_cases(
    input_schema: Optional[Dict[str, Any]],
    validation_profile: Optional[Dict[str, Any]],
    template_args: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Malformed-domain matrix for tools claiming exact CP26 feasibility.

    This is deliberately narrow and source-independent: the public task fixes
    26 circles, two finite coordinates per center, and finite nonnegative
    radii.  A checker that advertises that contract must reject malformed
    instances even if its happy-path JSON-Schema samples happen to execute.
    """

    from outer.safety.task_profiles import fixed_circle_checker_contract
    contract = fixed_circle_checker_contract(validation_profile)
    if contract is None:
        return []
    cardinality = int(contract["cardinality"])
    dimension = int(contract["dimension"])
    centers_field = str(contract["centers_field"])
    radii_field = str(contract["radii_field"])
    properties = (
        input_schema.get("properties")
        if isinstance(input_schema, dict) else None
    )
    if not isinstance(properties, dict) \
            or not {centers_field, radii_field} <= set(properties):
        return []

    centers = [
        [0.05 + 0.18 * (index % 5), 0.05 + 0.15 * (index // 5)]
        for index in range(cardinality)
    ]
    if dimension != 2:
        raise ValueError(
            "fixed-circle adversarial self-test currently requires dimension=2"
        )
    radii = [0.01] * cardinality

    def make(c, r, **extra):
        # Preserve every unrelated required property from a schema-valid
        # generated sample.  Otherwise a correct checker with, for example, a
        # required tolerance field would reject all semantic/adversarial rows
        # before the intended geometry branch was exercised.
        case = dict(template_args or {})
        case.update({centers_field: c, radii_field: r})
        case.update(extra)
        return case

    def make_partial(*, centers_only: bool) -> Dict[str, Any]:
        case = dict(template_args or {})
        case.pop(centers_field, None)
        case.pop(radii_field, None)
        case[centers_field if centers_only else radii_field] = (
            centers if centers_only else radii
        )
        return case

    optional_defaults: Dict[str, Any] = {}
    if "expected_n" in properties:
        # A caller-controlled expected_n must never redefine the public task.
        optional_defaults["expected_n"] = 25
    cases = [
        ("wrong_count_25", make(centers[:25], radii[:25], **optional_defaults)),
        ("wrong_count_27", make(centers + [[0.95, 0.95]], radii + [0.01])),
        ("length_mismatch", make(centers, radii[:25])),
        ("centers_without_radii", make_partial(centers_only=True)),
        ("radii_without_centers", make_partial(centers_only=False)),
        ("negative_radius", make(centers, [-0.01] + radii[1:])),
        ("tiny_negative_radius", make(centers, [-1e-13] + radii[1:])),
        ("nan_radius", make(centers, [float("nan")] + radii[1:])),
        ("infinite_radius", make(centers, [float("inf")] + radii[1:])),
        ("malformed_center", make([[0.1]] + centers[1:], radii)),
        ("nan_center", make([[float("nan"), 0.1]] + centers[1:], radii)),
        ("infinite_center", make([[float("inf"), 0.1]] + centers[1:], radii)),
        # Every literal is finite and JSON-valid, but naive subtraction and
        # squared-distance arithmetic overflow. A checker must use scale-safe
        # arithmetic or reject before emitting Inf/NaN.
        ("finite_extreme_geometry", make(
            [[1e308, -1e308]] * cardinality, [1e308] * cardinality,
        )),
    ]
    if "tolerance" in properties:
        cases.extend([
            ("negative_tolerance", make(centers, radii, tolerance=-1.0)),
            ("nan_tolerance", make(centers, radii, tolerance=float("nan"))),
        ])
    return cases


def _cp26_constraint_semantic_cases(
    validation_profile: Optional[Dict[str, Any]],
    template_args: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Schema-valid CP26 layouts whose feasibility has a known answer."""

    from outer.safety.task_profiles import fixed_circle_checker_contract
    contract = fixed_circle_checker_contract(validation_profile)
    if contract is None:
        return []
    cardinality = int(contract["cardinality"])
    centers_field = str(contract["centers_field"])
    radii_field = str(contract["radii_field"])
    import math

    centers = [
        [0.05 + 0.18 * (index % 5), 0.05 + 0.15 * (index // 5)]
        for index in range(cardinality)
    ]
    radii = [0.01] * cardinality
    overlap = [list(center) for center in centers]
    overlap[1] = list(overlap[0])
    outside = [list(center) for center in centers]
    outside[0] = [0.0, 0.5]
    outside_radii = list(radii)
    outside_radii[0] = 0.1
    zero_radii = list(radii)
    zero_radii[0] = 0.0
    tiny_wall = [list(center) for center in centers]
    tiny_wall[0] = [math.nextafter(radii[0], 0.0), centers[0][1]]
    tiny_overlap = [list(center) for center in centers]
    tiny_overlap[1] = [
        tiny_overlap[0][0] + radii[0] + radii[1] - 1e-15,
        tiny_overlap[0][1],
    ]
    tied_corner = [list(center) for center in centers]
    tied_corner[0] = [0.0, 0.0]
    tied_corner_radii = list(radii)
    tied_corner_radii[0] = 0.01

    oversized = [list(center) for center in centers]
    oversized[0] = [0.5, 0.5]
    oversized_radii = [0.0] * cardinality
    oversized_radii[0] = 0.6

    pinned = [list(center) for center in centers]
    pinned[0] = [0.5, 0.5]
    pinned[1] = [0.5, 0.5]
    pinned_radii = [0.0] * cardinality
    pinned_radii[0] = pinned_radii[1] = 0.5

    def make(c, r):
        case = dict(template_args or {})
        case.update({centers_field: c, radii_field: r})
        return case

    return [
        ("known_feasible", make(centers, radii),
         {"feasible": True, "kind": "feasible",
          "centers_field": centers_field, "radii_field": radii_field}),
        ("known_zero_radius", make(centers, zero_radii),
         {"feasible": True, "kind": "zero_radius",
          "centers_field": centers_field, "radii_field": radii_field}),
        ("known_overlap", make(overlap, radii),
         {"feasible": False, "kind": "coincident", "pair": [0, 1],
          "centers_field": centers_field, "radii_field": radii_field}),
        ("known_wall_violation",
         make(outside, outside_radii),
         {"feasible": False, "kind": "wall", "target": 0,
          "centers_field": centers_field, "radii_field": radii_field}),
        ("tied_corner_wall_violation",
         make(tied_corner, tied_corner_radii),
         {"feasible": False, "kind": "wall", "target": 0,
          "centers_field": centers_field, "radii_field": radii_field}),
        ("oversized_radius_center_repair_impossible",
         make(oversized, oversized_radii),
         {"feasible": False, "kind": "wall", "target": 0,
          "repairable": False,
          "centers_field": centers_field, "radii_field": radii_field}),
        ("pinned_coincident_center_repair_impossible",
         make(pinned, pinned_radii),
         {"feasible": False, "kind": "coincident", "pair": [0, 1],
          "repairable": False,
          "centers_field": centers_field, "radii_field": radii_field}),
        ("tiny_wall_violation",
         make(tiny_wall, radii),
         {"feasible": False, "kind": "wall", "target": 0,
          "centers_field": centers_field, "radii_field": radii_field}),
        ("tiny_pair_violation",
         make(tiny_overlap, radii),
         {"feasible": False, "kind": "overlap", "pair": [0, 1],
          "centers_field": centers_field, "radii_field": radii_field}),
    ]


def selftest_tool_code(
    code: str,
    timeout_s: float = 30.0,
    input_schema: Optional[Dict[str, Any]] = None,
    description: str = "",
    validation_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Returns (passed, detail)."""
    from outer.safety.task_profiles import fixed_circle_checker_contract
    if fixed_circle_checker_contract(validation_profile) is not None and re.search(
        r"\bkind\s*[:=]\s*[`'\"]?circle\b", str(description or ""),
        re.IGNORECASE,
    ):
        return False, (
            "executor-visible CP26 descriptor says kind=circle, but the "
            "trusted repair contract requires kind=boundary with circle=<index>"
        )
    src_path = str(Path(__file__).resolve().parents[3])  # .../src
    try:
        samples = schema_selftest_cases(input_schema)
    except ValueError as exc:
        return False, str(exc)
    success_indices = list(range(len(samples)))
    template_args = next(
        (dict(case) for case in samples if isinstance(case, dict)), {}
    )
    adversarial = _cp26_constraint_adversarial_cases(
        input_schema, validation_profile, template_args=template_args,
    )
    reject_indices = list(range(len(samples), len(samples) + len(adversarial)))
    reject_labels = {
        str(index): label
        for index, (label, _case) in zip(reject_indices, adversarial)
    }
    samples.extend(case for _label, case in adversarial)
    semantic_cases = _cp26_constraint_semantic_cases(
        validation_profile, template_args=template_args,
    )
    semantic_start = len(samples)
    semantic_expectations = {
        str(semantic_start + offset): expectation
        for offset, (_label, _case, expectation) in enumerate(semantic_cases)
    }
    reject_labels.update({
        str(semantic_start + offset): label
        for offset, (label, _case, _expectation) in enumerate(semantic_cases)
    })
    samples.extend(case for _label, case, _expectation in semantic_cases)
    properties = (
        input_schema.get("properties")
        if isinstance(input_schema, dict) else None
    )
    require_observable_variation = not bool(properties)
    samples_json = json.dumps(samples)
    success_indices_json = json.dumps(success_indices)
    reject_indices_json = json.dumps(reject_indices)
    reject_labels_json = json.dumps(reject_labels)
    semantic_expectations_json = json.dumps(semantic_expectations)
    # Mirror the AST gate at the actual runtime value boundary.  A proposer
    # can construct a misleading key dynamically (for example
    # '_'.join(('global', 'optimum'))), which no literal-only AST walk can
    # prove; the serialized result is the authoritative capability surface.
    from outer.safety.static_gates import unsupported_optimality_keys
    unsupported_output_keys_json = json.dumps(
        sorted(unsupported_optimality_keys(validation_profile))
    )
    require_current_candidate_bridge = (
        fixed_circle_checker_contract(validation_profile) is not None
        and any(case == {} for case in samples)
    )
    with tempfile.TemporaryDirectory(prefix="tool_selftest_") as td:
        code_path = Path(td) / "generated_tool.py"
        code_path.write_text(code)
        runner = _RUNNER.format(src_path=src_path, code_path=str(code_path),
                                scratch=str(Path(td) / "scratch"),
                                samples_json=samples_json,
                                success_indices_json=success_indices_json,
                                reject_indices_json=reject_indices_json,
                                reject_labels_json=reject_labels_json,
                                semantic_expectations_json=
                                    semantic_expectations_json,
                                unsupported_output_keys_json=
                                    unsupported_output_keys_json,
                                require_observable_variation=(
                                    require_observable_variation
                                ),
                                require_current_candidate_bridge=(
                                    require_current_candidate_bridge
                                ))
        try:
            r = subprocess.run([sys.executable, "-c", runner],
                               capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False, f"self-test timeout ({timeout_s}s)"
        line = (r.stdout or "").strip().splitlines()
        if not line:
            return False, f"self-test produced no verdict; stderr: {(r.stderr or '')[-400:]}"
        try:
            d = json.loads(line[-1])
        except Exception:
            return False, f"unparseable verdict: {line[-1][:200]}"
        return bool(d.get("ok")), d.get("error") or json.dumps(
            {k: d[k] for k in ("samples", "outcomes") if k in d})
