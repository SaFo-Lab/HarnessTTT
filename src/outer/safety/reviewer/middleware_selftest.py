"""Bounded typed-state reachability test for generated H2 middleware."""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple


MIDDLEWARE_BEHAVIOR_SCHEMA = "sah-middleware-behavior/1.0"


def middleware_state_cases() -> List[dict]:
    base = {
        "__case__": "baseline",
        "iteration": 1,
        "current_iteration": 1,
        "best_so_far": 0.25,
        "evaluator_calls": 1,
        "evals_done": 1,
        "evaluations_remaining": 9,
        "evals_remaining": 9,
        "probe_calls": 0,
        "probe_attempts": 0,
        "probes_remaining": 30,
        "probe_remaining": 30,
        "probe_available": True,
        "probes_since_eval": 0,
        "edit_calls": 1,
        "pending_evaluation": False,
        "last_edit_staged": None,
        "stalled_evals": 0,
        "family_streak": 1,
        "families_explored": ["family-a"],
        "last_family": "family-a",
        "structure_streak": 1,
        "structures_explored": ["family-a"],
        "current_program_valid_syntax": True,
        "last_step_kind": "seed",
        "last_error": None,
        "last_validity": 1.0,
        "last_score": 0.25,
        "active_tool_gate": None,
    }
    cases = [base]
    variants = [
        {"__case__": "stalled_failed_diff",
         "iteration": 12, "current_iteration": 12, "stalled_evals": 4,
         "family_streak": 6, "structure_streak": 6, "evals_done": 5,
         "evaluator_calls": 5, "evals_remaining": 5,
         "evaluations_remaining": 5, "last_step_kind": "diff",
         "last_validity": 0.0, "last_score": 0.0,
         "last_error": "invalid candidate",
         "last_edit_staged": True,
         "current_program_valid_syntax": False},
        {"__case__": "stalled_failed_full_rewrite",
         "iteration": 12, "current_iteration": 12, "stalled_evals": 4,
         "family_streak": 6, "structure_streak": 6, "evals_done": 5,
         "evaluator_calls": 5, "evals_remaining": 5,
         "evaluations_remaining": 5, "last_step_kind": "full_rewrite",
         "last_validity": 0.0, "last_score": 0.0,
         "last_error": "invalid candidate",
         "last_edit_staged": True,
         "current_program_valid_syntax": False},
        {"__case__": "stalled_valid_diff",
         "iteration": 12, "current_iteration": 12, "stalled_evals": 4,
         "family_streak": 6, "structure_streak": 6, "evals_done": 5,
         "evaluator_calls": 5, "evals_remaining": 5,
         "evaluations_remaining": 5, "last_step_kind": "diff",
         "last_validity": 1.0, "last_score": 0.25, "last_error": None,
         "last_edit_staged": True,
         "current_program_valid_syntax": True},
        {"__case__": "stalled_valid_full_rewrite",
         "iteration": 12, "current_iteration": 12, "stalled_evals": 4,
         "family_streak": 6, "structure_streak": 6, "evals_done": 5,
         "evaluator_calls": 5, "evals_remaining": 5,
         "evaluations_remaining": 5, "last_step_kind": "full_rewrite",
         "last_validity": 1.0, "last_score": 0.25, "last_error": None,
         "last_edit_staged": True,
         "current_program_valid_syntax": True},
        # Streak alone is not stagnation: the same family may still be
        # improving. Trusted stagnation gates must require BOTH causal signals.
        {"__case__": "improving_same_family_diff",
         "iteration": 12, "current_iteration": 12, "stalled_evals": 0,
         "family_streak": 6, "structure_streak": 6, "evals_done": 5,
         "evaluator_calls": 5, "evals_remaining": 5,
         "evaluations_remaining": 5, "last_step_kind": "diff",
         "last_validity": 1.0, "last_score": 0.35, "best_so_far": 0.35,
         "last_error": None, "last_edit_staged": True},
        {"__case__": "improving_same_family_full_rewrite",
         "iteration": 12, "current_iteration": 12, "stalled_evals": 0,
         "family_streak": 6, "structure_streak": 6, "evals_done": 5,
         "evaluator_calls": 5, "evals_remaining": 5,
         "evaluations_remaining": 5, "last_step_kind": "full_rewrite",
         "last_validity": 1.0, "last_score": 0.35, "best_so_far": 0.35,
         "last_error": None, "last_edit_staged": True},
        # A run can have several stalled evaluations while already exploring a
        # new family. Re-gating on stalled_evals alone would suppress diversity.
        {"__case__": "stalled_new_family_diff",
         "iteration": 12, "current_iteration": 12, "stalled_evals": 4,
         "family_streak": 1, "structure_streak": 1, "evals_done": 5,
         "evaluator_calls": 5, "evals_remaining": 5,
         "evaluations_remaining": 5, "last_step_kind": "diff",
         "last_validity": 1.0, "last_score": 0.25,
         "last_error": None, "last_edit_staged": True,
         "last_family": "family-new"},
        {"__case__": "stalled_new_family_full_rewrite",
         "iteration": 12, "current_iteration": 12, "stalled_evals": 4,
         "family_streak": 1, "structure_streak": 1, "evals_done": 5,
         "evaluator_calls": 5, "evals_remaining": 5,
         "evaluations_remaining": 5, "last_step_kind": "full_rewrite",
         "last_validity": 1.0, "last_score": 0.25,
         "last_error": None, "last_edit_staged": True,
         "last_family": "family-new"},
        {"__case__": "pending_stalled_diff",
         "stalled_evals": 4, "family_streak": 6, "structure_streak": 6,
         "last_step_kind": "diff", "pending_evaluation": True,
         "last_edit_staged": True},
        {"__case__": "noop_stalled_diff",
         "stalled_evals": 4, "family_streak": 6, "structure_streak": 6,
         "last_step_kind": "diff", "last_edit_staged": False},
        {"__case__": "active_gate_stalled_diff",
         "stalled_evals": 4, "family_streak": 6, "structure_streak": 6,
         "last_step_kind": "diff", "last_edit_staged": True,
         "active_tool_gate": {"required": ["edit_solution"]}},
        {"__case__": "late_full_rewrite_no_probes",
         "iteration": 20, "current_iteration": 20, "evals_remaining": 2,
         "evaluations_remaining": 2, "probes_remaining": 0,
         "probe_remaining": 0, "probe_available": False,
         "last_step_kind": "full_rewrite"},
        {"__case__": "unchanged_reevaluation",
         "last_step_kind": "edit_eval", "last_validity": 0.0,
         "last_score": 0.0, "last_error": "invalid candidate",
         "current_program_valid_syntax": False},
        {"__case__": "after_probe", "last_step_kind": "probe", "probe_calls": 3,
         "probe_attempts": 3, "probes_since_eval": 3},
        {"__case__": "diverse_seed", "last_step_kind": "seed",
         "families_explored": ["a", "b", "c"],
         "structures_explored": ["a", "b", "c"]},
        # Stagnation counters describe the most recent evaluated edit. They
        # can remain nonzero after later non-edit steps; a gate must bind its
        # causal trigger to an actual diff/full_rewrite outcome rather than
        # fire on stale state.
        {"__case__": "stale_stall_after_probe",
         "last_step_kind": "probe", "stalled_evals": 4,
         "family_streak": 6, "structure_streak": 6,
         "last_edit_staged": True, "last_validity": 1.0},
        {"__case__": "stale_stall_after_candidate_geometry",
         "last_step_kind": "candidate_geometry", "stalled_evals": 4,
         "family_streak": 6, "structure_streak": 6,
         "last_edit_staged": True, "last_validity": 1.0},
        {"__case__": "stale_stall_after_seed",
         "last_step_kind": "seed", "stalled_evals": 4,
         "family_streak": 6, "structure_streak": 6,
         "last_edit_staged": True, "last_validity": 1.0},
        {"__case__": "stale_stall_after_edit_eval",
         "last_step_kind": "edit_eval", "stalled_evals": 4,
         "family_streak": 6, "structure_streak": 6,
         "last_edit_staged": True, "last_validity": 1.0},
    ]
    cases.extend({**base, **variant} for variant in variants)
    # Exercise every reachable branch under exhausted budgets too. A gate that
    # is useful with budget left can deadlock H2 when it requires a tool the
    # trusted runtime can no longer execute.
    ordinary = list(cases)
    cases.extend({
        **case,
        "__case__": str(case.get("__case__")) + ":no_probes",
        "probes_remaining": 0,
        "probe_remaining": 0,
        "probe_available": False,
    } for case in ordinary)
    cases.extend({
        **case,
        "__case__": str(case.get("__case__")) + ":no_evals",
        "evaluations_remaining": 0,
        "evals_remaining": 0,
    } for case in ordinary)
    cases.extend({
        **case,
        "__case__": str(case.get("__case__")) + ":no_budget",
        "probes_remaining": 0,
        "probe_remaining": 0,
        "probe_available": False,
        "evaluations_remaining": 0,
        "evals_remaining": 0,
    } for case in ordinary)
    return cases


_RUNNER = r'''import json, re
namespace = {{}}
code = open({code_path!r}, encoding="utf-8").read()
exec(compile(code, {code_path!r}, "exec"), namespace)
hook = namespace[{hook!r}]
cases = json.loads({cases_json!r})
description = {description!r}
contract_text = {contract_text!r}
validation_profile = json.loads({validation_profile_json!r})
outcomes = []
evaluated = []

def invoke(state):
    public = {{key: value for key, value in state.items()
              if not str(key).startswith("__")}}
    return hook({{**public, "state": dict(public)}})

def required_tools(value):
    return value.get("require_tools") if isinstance(value, dict) else None

def structure_required(value):
    return (
        value.get("require_structure_change", False)
        if isinstance(value, dict) else False
    )

def witness(case_index, state, observed, expected):
    compact_state = {{
        key: state.get(key) for key in (
            "__case__", "iteration", "stalled_evals", "family_streak",
            "last_family", "pending_evaluation", "last_edit_staged",
            "last_step_kind", "last_error", "last_validity",
            "evals_remaining", "probes_remaining", "probe_available",
        )
    }}
    return "case=%s state=%s observed=%s expected=%s" % (
        str(state.get("__case__") or ("case-%d" % case_index)),
        json.dumps(compact_state, sort_keys=True), repr(observed)[:500], expected,
    )

def validate_gate(case_index, state, value):
    case_label = str(state.get("__case__") or ("case-%d" % case_index))
    if isinstance(value, dict):
        unknown = sorted(set(value) - {{
            "note", "require_tools", "require_structure_change"
        }})
        if unknown:
            return witness(
                case_index, state, value,
                "result keys limited to note, require_tools, "
                "require_structure_change; unknown=%s" % unknown,
            )
    structural = structure_required(value)
    if not isinstance(structural, bool):
        return witness(
            case_index, state, value,
            "require_structure_change must be boolean",
        )
    required = required_tools(value)
    if not required:
        if structural:
            return witness(
                case_index, state, value,
                "require_structure_change=true requires "
                "require_tools=['edit_solution']",
            )
        return None
    if not isinstance(required, (list, tuple)):
        return witness(case_index, state, value, "require_tools must be a list")
    if any(tool not in (
        "probe_solution", "edit_solution", "evaluate_solution"
    ) for tool in required):
        return witness(
            case_index, state, value,
            "require_tools must contain only probe_solution, edit_solution, "
            "or evaluate_solution",
        )
    if structural and list(required) != ["edit_solution"]:
        return witness(
            case_index, state, value,
            "structure postcondition requires exactly ['edit_solution']",
        )
    if "probe_solution" in required and (
        state.get("probe_available") is False
        or int(state.get("probes_remaining", 0) or 0) <= 0
        or int(state.get("probe_remaining", 0) or 0) <= 0
        or int(state.get("evaluations_remaining", 0) or 0) <= 0
        or int(state.get("evals_remaining", 0) or 0) <= 0
    ):
        return witness(
            case_index, state, value,
            "no probe gate with exhausted probe budget or evaluation budget",
        )
    if any(tool in required for tool in ("edit_solution", "evaluate_solution")) and (
        int(state.get("evaluations_remaining", 0) or 0) <= 0
        or int(state.get("evals_remaining", 0) or 0) <= 0
    ):
        return witness(
            case_index, state, value,
            "no edit/evaluate gate when evaluation budget is exhausted",
        )

    # The trusted runtime wrapper replaces any H1-authored note attached to a
    # gate with a deterministic NEXT-CALL message derived from require_tools.
    # Free-form trigger/history prose is therefore not interpreted as a tool
    # sequence here; reachability and budget checks above remain structural.
    return None

def transition_variants_after_required(state, tool, step):
    """Conservative runtime branches after one gate-satisfying tool call."""
    base = dict(state)
    base["active_tool_gate"] = None
    base["iteration"] = int(state.get("iteration", 0) or 0) + 1
    base["current_iteration"] = int(
        state.get("current_iteration", 0) or 0) + 1
    if tool == "probe_solution":
        base["probe_calls"] = int(state.get("probe_calls", 0) or 0) + 1
        base["probe_attempts"] = int(state.get("probe_attempts", 0) or 0) + 1
        base["probes_remaining"] = max(
            0, int(state.get("probes_remaining", 0) or 0) - 1)
        base["probe_remaining"] = max(
            0, int(state.get("probe_remaining", 0) or 0) - 1)
        base["probe_available"] = base["probes_remaining"] > 0
        base["probes_since_eval"] = int(
            state.get("probes_since_eval", 0) or 0) + 1
        variants = []
        for label, error, validity in (
            ("probe_success", None, 1.0),
            ("probe_failed", "probe found invalid candidate", 0.0),
        ):
            row = dict(base)
            row["last_step_kind"] = "probe"
            row["last_error"] = error
            row["last_validity"] = validity
            variants.append((label, row))
        return variants
    if tool == "edit_solution":
        base["edit_calls"] = int(state.get("edit_calls", 0) or 0) + 1
        same = dict(base)
        same["pending_evaluation"] = True
        same["last_edit_staged"] = True
        same["family_streak"] = int(state.get("family_streak", 0) or 0) + 1
        same["structure_streak"] = int(
            state.get("structure_streak", 0) or 0) + 1
        same["current_program_valid_syntax"] = True

        new = dict(base)
        new["pending_evaluation"] = True
        new["last_edit_staged"] = True
        family = "sequence-new-family-%d" % step
        new["last_family"] = family
        new["family_streak"] = 1
        new["structure_streak"] = 1
        new["families_explored"] = list(dict.fromkeys(
            list(state.get("families_explored") or []) + [family]))
        new["structures_explored"] = list(dict.fromkeys(
            list(state.get("structures_explored") or []) + [family]))
        new["current_program_valid_syntax"] = True

        invalid = dict(same)
        invalid["current_program_valid_syntax"] = False
        # A syntactically valid edit_solution call may still stage no bytes:
        # SEARCH text can fail to match.  The trusted runtime increments the
        # edit counter, clears the satisfied gate, keeps the previous outcome,
        # and leaves pending_evaluation unchanged.  Model that exact branch so
        # an edit gate cannot immediately re-arm forever after no-op edits.
        not_staged = dict(base)
        not_staged["last_edit_staged"] = False
        not_staged["family_streak"] = int(
            state.get("family_streak", 0) or 0) + 1
        not_staged["structure_streak"] = int(
            state.get("structure_streak", 0) or 0) + 1
        return [
            ("same_family_edit", same),
            ("new_family_edit", new),
            ("invalid_same_family_edit", invalid),
            ("edit_not_staged", not_staged),
        ]
    if tool == "evaluate_solution":
        base["evaluator_calls"] = int(state.get("evaluator_calls", 0) or 0) + 1
        base["evals_done"] = int(state.get("evals_done", 0) or 0) + 1
        base["evaluations_remaining"] = max(
            0, int(state.get("evaluations_remaining", 0) or 0) - 1)
        base["evals_remaining"] = max(
            0, int(state.get("evals_remaining", 0) or 0) - 1)
        base["pending_evaluation"] = False
        best = float(state.get("best_so_far", 0.0) or 0.0)

        # A staged edit can have either runtime mode; evaluating without a
        # staged edit produces edit_eval.  Every mode can improve, stall, or
        # fail.  Cross the mode/outcome product instead of assigning one mode
        # to each outcome, which previously hid mode-specific gate cycles.
        modes = (
            ("diff", "full_rewrite")
            if state.get("pending_evaluation") else ("edit_eval",)
        )
        variants = []
        for mode in modes:
            improved = dict(base)
            improved.update({{"last_step_kind": mode, "last_error": None,
                             "last_validity": 1.0,
                             "last_score": best + 0.01,
                             "best_so_far": best + 0.01,
                             "stalled_evals": 0}})
            stalled = dict(base)
            stalled.update({{"last_step_kind": mode, "last_error": None,
                            "last_validity": 1.0, "last_score": best,
                            "stalled_evals": int(
                                state.get("stalled_evals", 0) or 0) + 1}})
            failed = dict(base)
            failed.update({{"last_step_kind": mode,
                           "last_error": "invalid candidate",
                           "last_validity": 0.0, "last_score": 0.0,
                           "current_program_valid_syntax": False,
                           "stalled_evals": int(
                               state.get("stalled_evals", 0) or 0) + 1}})
            variants.extend([
                ("evaluation_improved_" + mode, improved),
                ("evaluation_stalled_" + mode, stalled),
                ("evaluation_failed_" + mode, failed),
            ])
        return variants
    return []

def gate_signature(state, required, structural=False):
    """Bucket state to expose semantic gate cycles despite monotone counters."""
    return (
        tuple(required), bool(structural), state.get("last_step_kind"),
        bool(state.get("last_error")), state.get("last_validity"),
        min(4, int(state.get("family_streak", 0) or 0)),
        min(4, int(state.get("structure_streak", 0) or 0)),
        bool(state.get("probe_available")),
        bool(state.get("pending_evaluation")),
        state.get("last_edit_staged"),
        int(state.get("probes_remaining", 0) or 0) > 0,
        int(state.get("evals_remaining", 0) or 0) > 0,
        min(4, int(state.get("stalled_evals", 0) or 0)),
    )

for case_index, state in enumerate(cases):
    try:
        value = invoke(state)
        actionable = (
            isinstance(value, str) and bool(value.strip())
            or isinstance(value, dict) and (
                bool(str(value.get("note") or "").strip())
                or bool(value.get("require_tools"))
            )
        )
        required = required_tools(value)
        outcomes.append({{"case": state.get("__case__"),
                          "actionable": bool(actionable),
                          "required": list(required or []),
                          "require_structure_change": bool(
                              structure_required(value)),
                          "type": type(value).__name__}})
        gate_error = validate_gate(case_index, state, value)
        if gate_error:
            print(json.dumps({{"ok": False, "error": gate_error}}))
            raise SystemExit(0)
        evaluated.append((
            case_index, state, required, bool(structure_required(value))
        ))
    except Exception as exc:
        print(json.dumps({{"ok": False, "error":
              witness(case_index, state, "runtime error: "
                      + type(exc).__name__ + ": " + str(exc),
                      "hook returns None/string/valid effect dict")}}))
        raise SystemExit(0)

# Only after every direct truth-table row has passed its budget contract, model
# every allowed tool branch.  At most two consecutive forced calls may remain
# active.  This catches singleton, multi-tool, alternating-tool, same-family,
# and outcome-dependent cycles instead of checking only identical re-arms.
for case_index, state, required, structural in evaluated:
    if not required:
        continue
    initial_signature = gate_signature(state, list(required), structural)
    frontier = [(
        dict(state), list(required), structural, [], {{initial_signature}},
    )]
    for step in range(1, 3):
        next_frontier = []
        for sequence_state, sequence_required, sequence_structural, path, seen in frontier:
            for tool in sequence_required:
                for variant, next_state in transition_variants_after_required(
                    sequence_state, tool, step
                ):
                    # The trusted runtime retains a structural edit gate after
                    # same-family, invalid, or no-op edits.  It does not call
                    # the generated hook again to re-arm it. Runtime tests
                    # cover the two-failure auto-lift; only the successful new
                    # family reaches the next hook state here.
                    if sequence_structural and tool == "edit_solution" \
                            and variant != "new_family_edit":
                        continue
                    sequence_value = invoke(next_state)
                    sequence_error = validate_gate(
                        case_index, next_state, sequence_value)
                    if sequence_error:
                        print(json.dumps({{"ok": False, "error": sequence_error}}))
                        raise SystemExit(0)
                    next_required = required_tools(sequence_value) or []
                    if not next_required:
                        continue
                    next_structural = bool(structure_required(sequence_value))
                    signature = gate_signature(
                        next_state, list(next_required), next_structural)
                    next_path = path + [tool + ":" + variant]
                    if signature in seen:
                        print(json.dumps({{"ok": False, "error":
                              witness(case_index, next_state, sequence_value,
                                      "gate cycle: gate must clear after path "
                                      "%s; add outcome/tool/family hysteresis" %
                                      " -> ".join(next_path))}}))
                        raise SystemExit(0)
                    next_frontier.append((
                        next_state, list(next_required), next_structural, next_path,
                        set(seen) | {{signature}},
                    ))
        frontier = next_frontier
        if not frontier:
            break
    if frontier:
        _state, pending, _structural, path, _seen = frontier[0]
        print(json.dumps({{"ok": False, "error":
              witness(case_index, _state, pending,
                      "gate remains active beyond two satisfied calls; path=%s; "
                      "use explicit last_step/outcome/family hysteresis" %
                      " -> ".join(path))}}))
        raise SystemExit(0)
has_action = any(row["actionable"] for row in outcomes)
has_pass_through = any(not row["actionable"] for row in outcomes)
visible_claim = description + "\n" + contract_text
description_requires_gate = bool(re.search(
    r"\b(?:force|forces|forced|gate|gates|required|requires)\b",
    visible_claim, re.IGNORECASE,
))
description_is_stagnation = bool(re.search(
    r"\b(?:stagnat\w*|stall\w*|family)\b", visible_claim, re.IGNORECASE,
))
trusted_stagnation_role = (
    validation_profile.get("portfolio_role") == "stagnation_gate"
    if isinstance(validation_profile, dict) else False
)
required_valid_stall_cases = {{
    "stalled_valid_diff", "stalled_valid_full_rewrite",
}}
covered_valid_stall_cases = {{
    str(row.get("case") or "").split(":", 1)[0]
    for row in outcomes
    if str(row.get("case") or "").split(":", 1)[0]
       in required_valid_stall_cases
    and row.get("required") == ["edit_solution"]
    and bool(row.get("require_structure_change"))
}}
real_stalled_gate = covered_valid_stall_cases == required_valid_stall_cases
must_gate_real_stall = trusted_stagnation_role or (
    description_requires_gate and description_is_stagnation
)
forbidden_stagnation_cases = {{
    "improving_same_family_diff", "improving_same_family_full_rewrite",
    "stalled_new_family_diff", "stalled_new_family_full_rewrite",
    "pending_stalled_diff", "noop_stalled_diff", "active_gate_stalled_diff",
    "stale_stall_after_probe", "stale_stall_after_candidate_geometry",
    "stale_stall_after_seed", "stale_stall_after_edit_eval",
}}
triggered_forbidden_cases = sorted({{
    str(row.get("case") or "").split(":", 1)[0]
    for row in outcomes
    if str(row.get("case") or "").split(":", 1)[0]
       in forbidden_stagnation_cases
    and row.get("actionable")
}})
negative_ok = not triggered_forbidden_cases
claim_ok = not must_gate_real_stall or (real_stalled_gate and negative_ok)
real_stall_row = next((
    row for row in outcomes
    if str(row.get("case") or "").split(":", 1)[0] in
       {{"stalled_failed_diff", "stalled_failed_full_rewrite",
        "stalled_valid_diff", "stalled_valid_full_rewrite"}}
), None)
real_stall_state = next((
    state for state in cases
    if str(state.get("__case__") or "").split(":", 1)[0] in
       {{"stalled_failed_diff", "stalled_failed_full_rewrite",
        "stalled_valid_diff", "stalled_valid_full_rewrite"}}
), cases[0])
if not claim_ok:
    missing = sorted(required_valid_stall_cases - covered_valid_stall_cases)
    final_error = witness(
        0, real_stall_state, real_stall_row,
        "trusted stagnation role requires the conjunction "
        "stalled_evals>=2 AND family_streak>=2, must cover BOTH valid "
        "evaluated edit modes (diff and full_rewrite), and must return "
        "require_tools=['edit_solution'] plus require_structure_change=true. "
        "Missing covered modes: %s; forbidden triggers: %s. " % (
            missing, triggered_forbidden_cases
        ) +
        "Do not gate an improving same-family run, a stalled run that already "
        "changed family, a pending/no-op edit, while another gate is active, "
        "or after seed/probe/candidate_geometry/edit_eval with stale counters. "
        "last_edit_staged=True is historical after evaluation; only "
        "pending_evaluation=True means an edit is awaiting evaluation, and "
        "last_edit_staged=False marks a no-op edit attempt",
    )
elif not has_action:
    final_error = witness(
        0, cases[0], outcomes[0] if outcomes else None,
        "at least one actionable state-dependent return",
    )
elif not has_pass_through:
    final_error = witness(
        0, cases[0], outcomes[0] if outcomes else None,
        "at least one pass-through state; always-on reminders are invalid",
    )
else:
    final_error = None
print(json.dumps({{"ok": has_action and has_pass_through and claim_ok,
                  "outcomes": outcomes,
                  "error": final_error}}))
'''


def selftest_middleware_code(
    code: str, hook: str = "before_model", timeout_s: float = 2.0,
    description: str = "",
    contract_text: str = "",
    validation_profile: dict | None = None,
) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="middleware_selftest_") as td:
        code_path = Path(td) / "generated_middleware.py"
        code_path.write_text(code)
        runner = _RUNNER.format(
            code_path=str(code_path), hook=hook,
            cases_json=json.dumps(middleware_state_cases()),
            description=str(description or ""),
            contract_text=str(contract_text or ""),
            validation_profile_json=json.dumps(validation_profile or {}),
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", runner], capture_output=True,
                text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, f"self-test timeout ({timeout_s}s)"
        lines = (result.stdout or "").strip().splitlines()
        if not lines:
            return False, "self-test produced no verdict: " + (
                result.stderr or ""
            )[-300:]
        try:
            verdict = json.loads(lines[-1])
        except Exception:
            return False, "unparseable self-test verdict: " + lines[-1][:200]
        if verdict.get("ok"):
            return True, json.dumps(verdict.get("outcomes", []))
        return False, str(verdict.get("error") or "unreachable middleware")


def middleware_behavior_fingerprint_record(
    code: str, hook: str = "before_model", description: str = "",
    contract_text: str = "", validation_profile: dict | None = None,
    timeout_s: float = 2.0,
) -> dict:
    """Return a replayable bounded truth-table fingerprint for valid code."""

    ok, detail = selftest_middleware_code(
        code, hook=hook, timeout_s=timeout_s, description=description,
        contract_text=contract_text, validation_profile=validation_profile,
    )
    if not ok:
        return {}
    try:
        outcomes = json.loads(detail)
    except (TypeError, ValueError):
        return {}
    rows = sorted(({
        "case": str(row.get("case") or ""),
        "actionable": bool(row.get("actionable")),
        "required": [str(value) for value in (row.get("required") or [])],
        "require_structure_change": bool(
            row.get("require_structure_change")
        ),
    } for row in outcomes if isinstance(row, dict)), key=lambda row: row["case"])
    payload = {
        "schema": MIDDLEWARE_BEHAVIOR_SCHEMA,
        "hook": str(hook),
        "outcomes": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return {
        "schema": MIDDLEWARE_BEHAVIOR_SCHEMA,
        "fingerprint": "h2beh_" + digest[:24],
        "canonical_payload_sha256": digest,
        "canonical_payload": payload,
    }
