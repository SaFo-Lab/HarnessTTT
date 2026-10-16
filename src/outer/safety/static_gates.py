"""Static safety gates for M_phi-generated tool code (h2spec/1.0).

Fail-closed: any violation returns errors and the candidate is rejected.  The
canonical H1 must edit the workspace and call ``validate_harness`` again; no
post-submit model repairs or byte mutations are allowed.

The contract for generated code:
  * defines exactly one top-level ``def run(ctx, args):``
  * imports only from IMPORT_WHITELIST
  * never touches filesystem/process/network primitives directly — all side
    effects go through the ``ctx`` capability object (see inner/harness_sdk.py)

The gate is a denylist over the AST and cannot see types: ``dataclasses.sys`` is an attribute chain, not an import.
The executor therefore also runs generated tools with a restricted ``__builtins__`` (whitelisted imports only, no
``open``/``exec``/reflection) and hands modules in through a guard that refuses private attributes, the attribute
names below and any module outside the whitelist (``inner.harness.tools.custom_runtime``).
"""
from __future__ import annotations

import ast
import re
from typing import List, Tuple

# the policy is shared with the executor runtime, which executes generated tools in a namespace restricted to it
# (inner.harness.tools.custom_runtime): what the gate forbids statically is also unavailable at run time
from inner.harness.tools.code_policy import (  # noqa: E402
    FORBIDDEN_ATTRIBUTES, FORBIDDEN_MODULES, FORBIDDEN_NAMES, IMPORT_WHITELIST, MAX_CODE_CHARS,
)

MIDDLEWARE_LAST_STEP_KINDS = {
    "seed", "diff", "full_rewrite", "edit_eval", "probe",
}

MIDDLEWARE_STATE_KEYS = {
    "iteration", "current_iteration", "best_so_far", "evaluator_calls",
    "evaluations_remaining", "evals_remaining", "evals_done", "probe_calls",
    "probe_attempts", "probes_remaining", "probe_remaining",
    "probe_available", "probes_since_eval", "edit_calls", "stalled_evals",
    "pending_evaluation",
    "last_edit_staged",
    "family_streak", "families_explored", "last_family",
    "structure_streak", "structures_explored",
    "current_program_valid_syntax", "last_step_kind", "last_error",
    "last_validity", "last_score",
    "active_tool_gate",
}
MIDDLEWARE_CONTEXT_KEYS = MIDDLEWARE_STATE_KEYS | {"state"}
MIDDLEWARE_COLLECTION_STATE_KEYS = {
    "families_explored", "structures_explored",
}

# Core ctx methods that expose an already-existing primitive but do not by
# themselves constitute a new capability. A generated tool may use them as
# part of a real operator; it may not be a one-call relay around them.
_CTX_RELAY_METHODS = {
    "probe", "evaluate", "best_score", "budget_left",
    "get_program", "get_best_program", "log",
}
_CTX_DATA_METHODS = {
    "get_program", "get_best_program", "list_task_inputs",
    "get_candidate_geometry", "read_input_sample", "read_input_df", "scratch_read",
}
_CTX_EFFECT_METHODS = {"stage_edit", "scratch_write"}

# H1-authored tools do not have access to a trusted proof oracle.  In
# particular, a local heuristic, independent per-item cap, or fixed-ray scale
# must not be surfaced to H2 as a certified aggregate/global optimum.  These
# names are deliberately narrow: local/conditional diagnostics remain useful,
# but the measured misleading fields from real rollouts fail closed.
_UNSUPPORTED_OPTIMALITY_KEYS = {
    "ceiling",
    "global_maximal",
    "global_maximum",
    "global_optimal",
    "global_optimum",
    "maximal_sum",
    "maximal_total",
    "maximum_sum",
    "maximum_total",
    "optimal_sum",
    "optimal_total",
    "optimum_sum",
    "optimum_total",
    "sum_max_feasible_radii",
    "total_maximal",
    "total_maximum",
    "total_optimal",
    "total_optimum",
    "true_maximal",
    "true_maximum",
    "true_optimal",
    "true_optimum",
}
_UNSUPPORTED_OPTIMALITY_PHRASES = (
    re.compile(
        r"\b(?:true|certified|proven|provable|global)\s+"
        r"(?:maximal|maximum|optimal|optimum)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bexact\s+(?:(?:aggregate|sum|total|global)\s+)?"
        r"(?:optimal|optimum)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bexact\s+(?:aggregate|sum|total|global)\s+"
        r"(?:maximal|maximum)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:maximal|maximum|optimal|optimum)\s+"
        r"(?:aggregate|sum|total)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:aggregate|sum|total)\s+"
        r"(?:maximal|maximum|optimal|optimum|ceiling)\b",
        re.IGNORECASE,
    ),
)

# The trusted CP26 checker role is a feasibility/repair oracle, not an
# optimizer.  Until a trusted LP oracle exists, these measured rollout claims
# are prohibited even when qualified as "fixed centers": the generated
# coordinate-descent routine did not establish them.
_CP26_UNTRUSTED_OPTIMIZATION_KEYS = {
    "fixed_centers_solved_radii",
    "fixed_centers_max_radii",
    "largest_radii_vector",
    "optimality_gap",
    "radii_assignment_gap",
    "solution_gap",
    "sum_fixed_centers_solved",
}
_CP26_UNTRUSTED_OPTIMIZATION_PHRASES = (
    re.compile(r"\blargest\s+radii\s+vector\b", re.IGNORECASE),
    re.compile(r"\bunique\s+solution\b", re.IGNORECASE),
    re.compile(r"\bmonotone\s+contraction\b", re.IGNORECASE),
    re.compile(r"\bsolve\s+is\s+exact\b", re.IGNORECASE),
    re.compile(
        r"\bexact\s+(?:for\s+the\s+)?fixed[- ]cent(?:er|re)s?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfixed[- ]cent(?:er|re)s?\s+"
        r"(?:solved|maximum|optimal|optimum)\b",
        re.IGNORECASE,
    ),
)


def unsupported_optimality_keys(
    validation_profile: dict | None = None,
) -> set[str]:
    """Runtime output-key denylist for one trusted validation profile."""

    from outer.safety.task_profiles import fixed_circle_checker_contract
    keys = set(_UNSUPPORTED_OPTIMALITY_KEYS)
    if fixed_circle_checker_contract(validation_profile) is not None:
        keys.update(_CP26_UNTRUSTED_OPTIMIZATION_KEYS)
    return keys


def check_tool_code(code: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if len(code) > MAX_CODE_CHARS:
        errors.append(f"code exceeds {MAX_CODE_CHARS} chars")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"syntax error: {e}"]

    run_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
    if len(run_defs) != 1:
        errors.append("must define exactly one top-level `def run(ctx, args):`")
    elif [a.arg for a in run_defs[0].args.args][:2] != ["ctx", "args"]:
        errors.append("run() signature must be (ctx, args)")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) \
                else [node.module or ""]
            for m in mods:
                root = m.split(".")[0]
                if root == "__future__":
                    if not (
                        isinstance(node, ast.ImportFrom)
                        and {alias.name for alias in node.names} == {"annotations"}
                    ):
                        errors.append("only `from __future__ import annotations` is allowed")
                elif root in FORBIDDEN_MODULES:
                    errors.append(f"forbidden import: {m}")
                elif root not in IMPORT_WHITELIST:
                    errors.append(f"import not in whitelist: {m}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"forbidden builtin: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            errors.append(f"forbidden private attribute access: .{node.attr}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            errors.append(f"forbidden I/O-capable attribute: .{node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append("global/nonlocal not allowed")

    return (not errors), sorted(set(errors))


def _literal_mapping_keys(node: ast.AST, mapping_name: str) -> set[str]:
    """Literal keys read from ``mapping.get('x')`` / ``mapping['x']``."""

    keys: set[str] = set()
    for child in ast.walk(node):
        if (isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == mapping_name
                and child.func.attr == "get" and child.args
                and isinstance(child.args[0], ast.Constant)
                and isinstance(child.args[0].value, str)):
            keys.add(child.args[0].value)
        elif (isinstance(child, ast.Subscript)
              and isinstance(child.value, ast.Name)
              and child.value.id == mapping_name):
            key = child.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def check_tool_utility(
    code: str, input_schema: dict | None = None, description: str = "",
    validation_profile: dict | None = None, contract_text: str = "",
) -> Tuple[bool, List[str]]:
    """Reject metadata-shaped tools that add no executor capability.

    This is intentionally a conservative structural gate, not a claim that a
    tool will improve reward. It catches the measured failure class —
    ``quick_probe.py``/score/budget/program relays — while accepting task
    computation over declared args, program/task-input analysis, and bounded
    candidate operators that may use probe/evaluate as one step.
    """

    errors: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [f"cannot assess utility: syntax error: {exc}"]
    run_defs = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    ]
    if len(run_defs) != 1:
        return False, ["cannot assess utility without exactly one run()"]
    run = run_defs[0]

    output_keys = {
        key.value.strip().lower()
        for node in ast.walk(run)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    output_keys.update(
        keyword.arg.strip().lower()
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "dict"
        for keyword in node.keywords
        if isinstance(keyword.arg, str)
    )
    for node in ast.walk(run):
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign)
            else []
        )
        for target in targets:
            if isinstance(target, ast.Subscript) \
                    and isinstance(target.slice, ast.Constant) \
                    and isinstance(target.slice.value, str):
                output_keys.add(target.slice.value.strip().lower())
    from outer.safety.task_profiles import fixed_circle_checker_contract
    circle_contract = fixed_circle_checker_contract(validation_profile)
    denied_keys = unsupported_optimality_keys(validation_profile)
    unsupported_keys = sorted(output_keys & denied_keys)
    visible_claim = str(description or "") + "\n" + str(contract_text or "")
    unsupported_phrases = sorted({
        match.group(0)
        for pattern in _UNSUPPORTED_OPTIMALITY_PHRASES
        for match in pattern.finditer(visible_claim)
    })
    if circle_contract is not None:
        visible_claim += "\n" + str(code or "")
        unsupported_phrases.extend(sorted({
            match.group(0)
            for pattern in _CP26_UNTRUSTED_OPTIMIZATION_PHRASES
            for match in pattern.finditer(visible_claim)
        }))
        unsupported_phrases = sorted(set(unsupported_phrases))
    if unsupported_keys or unsupported_phrases:
        evidence = []
        if unsupported_keys:
            evidence.append(f"output fields={unsupported_keys}")
        if unsupported_phrases:
            evidence.append(f"description phrases={unsupported_phrases}")
        errors.append(
            "unsupported aggregate/global optimality claim ("
            + ", ".join(evidence)
            + "); report the actual conditional quantity (for example "
              "per-item caps, fixed-ray scale, heuristic total, or local "
              "bound) instead of presenting it as a certified optimum"
        )

    def block_terminates(body: list[ast.stmt]) -> bool:
        if not body:
            return False
        statement = body[-1]
        if isinstance(statement, (ast.Return, ast.Raise)):
            return True
        if isinstance(statement, ast.If):
            if (isinstance(statement.test, ast.Constant)
                    and bool(statement.test.value)):
                return block_terminates(statement.body)
            return (
                bool(statement.orelse)
                and block_terminates(statement.body)
                and block_terminates(statement.orelse)
            )
        return False

    def unreachable_lines(body: list[ast.stmt]) -> List[int]:
        lines: List[int] = []
        terminated = False
        for statement in body:
            if terminated:
                lines.append(int(getattr(statement, "lineno", -1)))
                continue
            for child_body in (
                getattr(statement, "body", None),
                getattr(statement, "orelse", None),
                getattr(statement, "finalbody", None),
            ):
                if isinstance(child_body, list):
                    lines.extend(unreachable_lines(child_body))
            terminated = block_terminates([statement])
        return lines

    dead_lines = sorted({line for line in unreachable_lines(run.body) if line > 0})
    if dead_lines:
        errors.append(
            "tool contains unreachable statements after an unconditional "
            f"return/terminating branch at lines {dead_lines}"
        )

    arg_keys = _literal_mapping_keys(run, "args")
    schema_props = set()
    if isinstance(input_schema, dict):
        props = input_schema.get("properties")
        if isinstance(props, dict):
            schema_props = {str(key) for key in props}
    undeclared = sorted(arg_keys - schema_props)
    if undeclared:
        errors.append(
            "tool reads undeclared args fields " + repr(undeclared)
            + "; add those exact properties to input_schema so the executor "
              "can call the capability"
        )

    # Task/role-specific invariants are activated only by orchestrator-owned
    # validation_profile.  H1 controls description and schema text, so neither
    # is authority for deciding whether the CP26 checker contract applies.
    if circle_contract is not None:
        # The bounded semantic self-test proves that the actual return object
        # exposes boolean feasible=True/False on known layouts.  Do not infer
        # this postcondition only from dict-literal AST shape: correct tools
        # may assign out['feasible'] or return dict(feasible=...).
        cardinality = int(circle_contract["cardinality"])
        dimension = int(circle_contract["dimension"])
        centers_field = str(circle_contract["centers_field"])
        radii_field = str(circle_contract["radii_field"])
        if not isinstance(input_schema, dict):
            properties = {}
            required = set()
        else:
            raw_properties = input_schema.get("properties") or {}
            properties = raw_properties if isinstance(raw_properties, dict) else {}
            required_raw = input_schema.get("required") or []
            required = set(required_raw) if isinstance(required_raw, list) else set()
        if required:
            errors.append(
                "trusted fixed-circle checker must allow an argument-free "
                "current-candidate call; centers/radii are paired optional "
                f"fallback fields and other knobs need defaults (found required "
                f"fields {sorted(required)})"
            )
        bridge_calls = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ctx"
            and node.func.attr == "get_candidate_geometry"
            for node in ast.walk(run)
        )
        if not bridge_calls:
            errors.append(
                "trusted fixed-circle checker must call "
                "ctx.get_candidate_geometry() when explicit centers/radii "
                "are omitted, so H2 can inspect its current candidate without "
                "transcribing 78 numbers"
            )
        marker = "sah/trusted-cp26-explicit-or-current-candidate"
        paired_clause = any(
            isinstance(row, dict) and row.get("$comment") == marker
            for row in (
                input_schema.get("allOf") or []
                if isinstance(input_schema, dict) else []
            )
        )
        if not paired_clause:
            errors.append(
                "trusted fixed-circle checker schema is missing the paired "
                "explicit-or-current-candidate source contract; use the "
                "atomic upsert_tool transaction so it can normalize the schema"
            )
        for field in (centers_field, radii_field):
            field_schema = properties.get(field)
            if not isinstance(field_schema, dict) \
                    or field_schema.get("type") != "array" \
                    or field_schema.get("minItems") != cardinality \
                    or field_schema.get("maxItems") != cardinality:
                errors.append(
                    f"trusted fixed-circle checker field {field!r} must declare "
                    f"type=array with minItems=maxItems={cardinality}; do not accept, "
                    "pad, or truncate a different task cardinality"
                )
        centers_schema = properties.get(centers_field)
        center_items = (
            centers_schema.get("items")
            if isinstance(centers_schema, dict) else None
        )
        coordinate_items = (
            center_items.get("items")
            if isinstance(center_items, dict) else None
        )
        if not isinstance(center_items, dict) \
                or center_items.get("type") != "array" \
                or center_items.get("minItems") != dimension \
                or center_items.get("maxItems") != dimension \
                or not isinstance(coordinate_items, dict) \
                or coordinate_items.get("type") != "number":
            errors.append(
                f"trusted fixed-circle checker {centers_field}.items must be "
                f"an exact {dimension}-number array (type=array, "
                f"minItems=maxItems={dimension})"
            )
        radii_schema = properties.get(radii_field)
        radius_items = (
            radii_schema.get("items")
            if isinstance(radii_schema, dict) else None
        )
        if isinstance(radius_items, dict) \
                and radius_items.get("type") == "number":
            lower = radius_items.get("minimum")
            exclusive = radius_items.get("exclusiveMinimum")
            bounded = False
            for value in (lower, exclusive):
                if isinstance(value, (int, float)) \
                        and not isinstance(value, bool) and value >= 0:
                    bounded = True
            if not bounded:
                errors.append(
                    "trusted fixed-circle checker radii.items must declare a "
                    "nonnegative minimum or exclusiveMinimum so negative "
                    "radii cannot be reported feasible"
                )
        else:
            errors.append(
                "trusted fixed-circle checker radii must declare numeric items with "
                "a nonnegative domain bound"
            )

    args_loaded = any(
        isinstance(node, ast.Name) and node.id == "args"
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(run)
    )
    ctx_methods: List[str] = []
    for node in ast.walk(run):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ctx"):
            ctx_methods.append(node.func.attr)

    # Control/data-flow structure is a useful proxy for actually deriving a
    # result. We deliberately do not count assignments or dict construction:
    # ``x = ctx.probe(); return {'score': x}`` must remain a relay.
    structural_nodes = (
        ast.If, ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp,
        ast.GeneratorExp, ast.BinOp, ast.BoolOp, ast.Compare,
    )
    structural_score = sum(
        1 for node in ast.walk(run) if isinstance(node, structural_nodes)
    )
    derived_calls = 0
    for node in ast.walk(run):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            # Calls into helpers or pure builtins/libraries perform work.
            derived_calls += 1
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id in {"ctx", "args"}:
                continue
            derived_calls += 1
    transform_score = structural_score + derived_calls

    has_input = bool(arg_keys or args_loaded or set(ctx_methods) & _CTX_DATA_METHODS)
    has_effect = bool(set(ctx_methods) & _CTX_EFFECT_METHODS)
    if not has_input and not has_effect:
        errors.append(
            "tool consumes no task/candidate data and performs no candidate "
            "effect; a constant, score, budget, probe, or evaluator relay is "
            "not a new capability"
        )
    if transform_score < 2:
        errors.append(
            "tool does not derive a substantive result or transformation "
            "(needs task-specific arithmetic/comparison, branching, iteration, "
            "parsing, or a bounded candidate operator)"
        )
    if (ctx_methods and set(ctx_methods) <= _CTX_RELAY_METHODS
            and not arg_keys and transform_score < 3):
        errors.append(
            "tool is only a thin wrapper around existing ctx primitives; "
            "quick_probe/evaluate/score/budget/program relays are rejected"
        )
    return (not errors), sorted(set(errors))

def check_middleware_code(code: str, hook: str) -> Tuple[bool, List[str]]:
    """Gate a generated middleware hook body. Same import/builtin restrictions
    as tools, but requires exactly one top-level `def <hook>(hook_input):` and
    is even stricter — a middleware runs IN-PROCESS on every model turn."""
    errors: List[str] = []
    if len(code) > MAX_CODE_CHARS:
        errors.append(f"code exceeds {MAX_CODE_CHARS} chars")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"syntax error: {e}"]
    hook_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == hook]
    if len(hook_defs) != 1:
        errors.append(f"must define exactly one top-level `def {hook}(hook_input):`")
    elif [a.arg for a in hook_defs[0].args.args][:1] != ["hook_input"]:
        errors.append(f"{hook}() signature must be (hook_input)")
    def _is_state_view(node: ast.AST) -> bool:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "hook_input" and node.func.attr == "get" \
                and node.args and isinstance(node.args[0], ast.Constant):
            return node.args[0].value == "state"
        if isinstance(node, ast.BoolOp):
            return any(_is_state_view(v) for v in node.values)
        return False

    state_aliases = set()
    if "--USER-HOOK-START--" in code or "--USER-HOOK-END--" in code:
        errors.append("the wrapper sentinels '# --USER-HOOK-START--' / '# --USER-HOOK-END--' may not appear in a hook")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None and _is_state_view(value):
                state_aliases.update(t.id for t in targets if isinstance(t, ast.Name))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                root = m.split(".")[0]
                if root == "__future__":
                    if not (
                        isinstance(node, ast.ImportFrom)
                        and {alias.name for alias in node.names} == {"annotations"}
                    ):
                        errors.append("only `from __future__ import annotations` is allowed")
                elif root in FORBIDDEN_MODULES:
                    errors.append(f"forbidden import: {m}")
                elif root not in IMPORT_WHITELIST:
                    errors.append(f"import not in whitelist: {m}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"forbidden builtin: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            errors.append(f"forbidden private attribute access: .{node.attr}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            errors.append(f"forbidden I/O-capable attribute: .{node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append("global/nonlocal not allowed")
        elif isinstance(node, ast.Subscript) and (
            (isinstance(node.value, ast.Name)
             and node.value.id in ({"hook_input"} | state_aliases))
            or _is_state_view(node.value)
        ):
            errors.append("middleware context must be read with .get(<documented key>)")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "hook_input" and node.attr != "get":
            errors.append(f"unsupported hook_input attribute: .{node.attr}; use documented .get() keys")
        elif (isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and node.func.attr == "get"):
            base = (
                node.func.value.id
                if isinstance(node.func.value, ast.Name) else None
            )
            if base == "hook_input":
                allowed = MIDDLEWARE_CONTEXT_KEYS
            elif base in state_aliases:
                allowed = MIDDLEWARE_STATE_KEYS
            elif _is_state_view(node.func.value):
                base = "hook_input.state"
                allowed = MIDDLEWARE_STATE_KEYS
            else:
                allowed = None
            if allowed is not None:
                if not node.args or not isinstance(node.args[0], ast.Constant) \
                        or not isinstance(node.args[0].value, str):
                    errors.append(f"{base}.get() requires a literal documented key")
                elif node.args[0].value not in allowed:
                    errors.append(f"unsupported middleware context key: {node.args[0].value!r}")
    return (not errors), sorted(set(errors))


def check_middleware_utility(
    code: str, hook: str, description: str = "",
    validation_profile: dict | None = None, contract_text: str = "",
) -> Tuple[bool, List[str]]:
    """Require a state-dependent automatic decision, not constant prose."""

    errors: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [f"cannot assess utility: syntax error: {exc}"]
    hooks = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == hook
    ]
    if len(hooks) != 1:
        return False, [f"cannot assess utility without exactly one {hook}()"]
    fn = hooks[0]

    def _is_state_view(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "hook_input"
            and node.func.attr == "get" and bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "state"
        )

    state_aliases: set[str] = set()
    state_value_names: dict[str, str] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if _is_state_view(value):
            state_aliases.update(
                target.id for target in node.targets
                if isinstance(target, ast.Name)
            )
        if (isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id in state_aliases
                and value.func.attr == "get" and value.args
                and isinstance(value.args[0], ast.Constant)
                and isinstance(value.args[0].value, str)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    state_value_names[target.id] = value.args[0].value

    # Collection-valued state is deliberately typed.  A measured rollout used
    # ``float(families_explored)`` through a generic _num helper, silently
    # converting every real list to the default 0 and defeating the decision
    # boundary.  Reject direct and helper-mediated numeric coercion while
    # allowing meaningful operations such as len(), membership, and iteration.
    numeric_helpers = {
        child.name
        for child in ast.walk(fn)
        if isinstance(child, ast.FunctionDef) and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in {"float", "int"}
            for call in ast.walk(child)
        )
    }
    coerced_collection_keys: set[str] = set()
    for call in (node for node in ast.walk(fn) if isinstance(node, ast.Call)):
        if isinstance(call.func, ast.Name) and call.func.id in {"float", "int"}:
            for arg in call.args:
                if isinstance(arg, ast.Name):
                    key = state_value_names.get(arg.id)
                    if key in MIDDLEWARE_COLLECTION_STATE_KEYS:
                        coerced_collection_keys.add(key)
                if (isinstance(arg, ast.Call)
                        and isinstance(arg.func, ast.Attribute)
                        and arg.func.attr == "get" and arg.args
                        and isinstance(arg.args[0], ast.Constant)
                        and arg.args[0].value in MIDDLEWARE_COLLECTION_STATE_KEYS):
                    coerced_collection_keys.add(str(arg.args[0].value))
        if isinstance(call.func, ast.Name) and call.func.id in numeric_helpers:
            coerced_collection_keys.update(
                str(arg.value) for arg in call.args
                if isinstance(arg, ast.Constant)
                and arg.value in MIDDLEWARE_COLLECTION_STATE_KEYS
            )
    if coerced_collection_keys:
        errors.append(
            "collection-valued middleware state cannot be coerced with "
            f"int/float: {sorted(coerced_collection_keys)}; use len(), "
            "membership, or iteration so the diversity signal is preserved"
        )

    state_keys: set[str] = set()
    gate_tools: set[str] = set()
    unknown_return_keys: set[str] = set()
    actionable_returns = 0
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            base = (
                node.func.value.id
                if isinstance(node.func.value, ast.Name) else None
            )
            key = node.args[0].value
            if base in state_aliases and key in MIDDLEWARE_STATE_KEYS:
                state_keys.add(key)
            elif base == "hook_input" and key in MIDDLEWARE_STATE_KEYS:
                state_keys.add(key)
            elif _is_state_view(node.func.value) and key in MIDDLEWARE_STATE_KEYS:
                state_keys.add(key)
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        if isinstance(value, ast.Constant):
            if isinstance(value.value, str) and value.value.strip():
                actionable_returns += 1
        elif isinstance(value, ast.Dict):
            keys = [
                key.value if isinstance(key, ast.Constant) else None
                for key in value.keys
            ]
            unknown_return_keys.update(
                str(key) for key in keys
                if isinstance(key, str)
                and key not in {
                    "note", "require_tools", "require_structure_change"
                }
            )
            if "note" in keys or "require_tools" in keys:
                actionable_returns += 1
            if "require_tools" in keys:
                index = keys.index("require_tools")
                tool_value = value.values[index]
                if isinstance(tool_value, (ast.List, ast.Tuple)):
                    gate_tools.update(
                        item.value for item in tool_value.elts
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                    )

    if not state_keys:
        errors.append(
            "middleware reads no documented runtime state; constant reminders "
            "are text, not an automatic decision component"
        )
    if not any(isinstance(node, ast.If) for node in ast.walk(fn)):
        errors.append(
            "middleware has no state-dependent branch; intervene only when a "
            "measured condition holds"
        )
    if actionable_returns == 0:
        errors.append(
            "middleware never returns an actionable note or require_tools gate"
        )
    if unknown_return_keys:
        errors.append(
            "middleware hook-result dict has unsupported keys "
            f"{sorted(unknown_return_keys)}; runtime accepts only 'note', "
            "'require_tools', and 'require_structure_change' and otherwise "
            "suppresses the whole intervention"
        )
    invalid_gate = sorted(
        gate_tools - {"probe_solution", "edit_solution", "evaluate_solution"}
    )
    if invalid_gate:
        errors.append(f"middleware gates unsupported tools: {invalid_gate}")
    if gate_tools:
        semantic_text = f"{description}\n{contract_text}\n{code}"
        # ``EVOLVE-BLOCK`` is the trusted name of the editable source region,
        # not the verb "block".  The lexical contradiction check used to pair
        # its suffix with a nearby ``edit_solution`` mention and reject a
        # perfectly consistent gate (iteration-11 cand04).  Preserve string
        # length so any real matched-span witness still indexes useful text.
        contradiction_text = re.sub(
            r"\bEVOLVE-BLOCK\b", "EVOLVE-SCOPE", semantic_text,
            flags=re.IGNORECASE,
        )
        tool_terms = "|".join(
            re.escape(tool) + "|" + re.escape(tool.removesuffix("_solution"))
            for tool in sorted(gate_tools)
        )
        blocking = (
            r"(?:suspend(?:ed|s|ing)?|block(?:ed|s|ing)?|"
            r"disable(?:d|s|ing)?|forbid(?:den|s|ding)?)"
        )
        contradiction = (
            re.search(
                rf"\b{blocking}\b.{{0,80}}\b(?:{tool_terms}|cheap[ -]re-ranking)\b",
                contradiction_text, re.IGNORECASE | re.DOTALL,
            )
            or re.search(
                rf"\b(?:{tool_terms}|cheap[ -]re-ranking)\b.{{0,80}}\b{blocking}\b",
                contradiction_text, re.IGNORECASE | re.DOTALL,
            )
        )
        if contradiction:
            start = max(0, contradiction.start() - 40)
            end = min(len(contradiction_text), contradiction.end() + 40)
            witness = " ".join(
                contradiction_text[start:end].split()
            )[:240]
            errors.append(
                "require_tools forces the listed next tool call; its code or "
                "description instead says that tool/re-ranking is suspended "
                f"or blocked; matched span={witness!r}"
            )

    def state_key(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return state_value_names.get(node.id)
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            base = node.func.value
            if isinstance(base, ast.Name) and (
                base.id in state_aliases or base.id == "hook_input"
            ):
                return node.args[0].value
            if _is_state_view(base):
                return node.args[0].value
        return None

    for node in ast.walk(fn):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        key = next((state_key(value) for value in operands
                    if state_key(value) is not None), None)
        if key == "last_validity":
            compares_bool = any(
                isinstance(value, ast.Constant)
                and isinstance(value.value, bool)
                for value in operands
            )
            identity_non_null = (
                any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops)
                and not any(
                    isinstance(value, ast.Constant) and value.value is None
                    for value in operands
                )
            )
            if compares_bool or identity_non_null:
                errors.append(
                    "last_validity is a numeric float (1.0/0.0), not a bool; "
                    "compare numerically (for example >= 1.0); identity is "
                    "only meaningful for an explicit None check"
                )
        if key == "last_step_kind":
            constants: set[str] = set()
            for value in operands:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    constants.add(value.value)
                elif isinstance(value, (ast.Set, ast.List, ast.Tuple)):
                    constants.update(
                        item.value for item in value.elts
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                    )
            impossible = sorted(constants - MIDDLEWARE_LAST_STEP_KINDS)
            if constants and impossible == sorted(constants):
                errors.append(
                    "last_step_kind branch is unreachable; runtime values are "
                    f"{sorted(MIDDLEWARE_LAST_STEP_KINDS)}, not {impossible}"
                )

    static_ok, _static_errors = check_middleware_code(code, hook)
    if not static_ok:
        errors.append("middleware safety gate must pass before utility self-test")
    if not errors:
        from outer.safety.reviewer.middleware_selftest import (
            selftest_middleware_code,
        )
        ok, detail = selftest_middleware_code(
            code, hook, timeout_s=2.0, description=description,
            contract_text=contract_text,
            validation_profile=validation_profile,
        )
        if not ok:
            errors.append("typed-state truth-table self-test: " + detail)
    return (not errors), sorted(set(errors))
