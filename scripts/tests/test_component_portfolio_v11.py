"""v1.1 condition-identical discovery and component-utility contracts."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from outer.proposing.component_portfolio import (
    ALLOWED_SURFACES,
    PORTFOLIO,
    contract_errors,
    h2_compute_contract,
    implementation_change_inventory,
    role_for_slot,
)
from outer.safety.static_gates import (
    check_middleware_utility,
    check_tool_utility,
)
from outer.safety.task_profiles import (
    normalize_tool_schema,
    trusted_component_profile,
)
from outer.workspace.propose_session import ProposeSession
from outer.workspace.h2_workspace import generated_component_prompt_block


CP26_PROFILE = trusted_component_profile(
    "eft__math__circle_packing", "constraint_checker"
)
STAGNATION_PROFILE = trusted_component_profile(
    "eft__math__circle_packing", "stagnation_gate"
)


GEOMETRY_CHECKER = '''import math

def run(ctx, args):
    centers = args.get("centers")
    radii = args.get("radii")
    if centers is None and radii is None:
        geometry = ctx.get_candidate_geometry()
        if not geometry.get("ok"):
            return {"error": geometry.get("error", "geometry unavailable")}
        centers, radii = geometry["centers"], geometry["radii"]
    elif centers is None or radii is None:
        return {"error": "centers and radii must be provided together"}
    if len(centers) != len(radii):
        return {"error": "length mismatch"}
    violations = []
    worst = 0.0
    for i in range(len(centers)):
        x, y = float(centers[i][0]), float(centers[i][1])
        wall_margin = min(x, y, 1.0 - x, 1.0 - y) - float(radii[i])
        worst = min(worst, wall_margin)
        if wall_margin < 0.0:
            violations.append({"kind": "wall", "i": i, "margin": wall_margin})
        for j in range(i + 1, len(centers)):
            dx = x - float(centers[j][0])
            dy = y - float(centers[j][1])
            margin = math.hypot(dx, dy) - float(radii[i]) - float(radii[j])
            worst = min(worst, margin)
            if margin < 0.0:
                violations.append({"kind": "pair", "i": i, "j": j,
                                   "margin": margin})
    return {"feasible": not violations, "worst_margin": worst,
            "violations": violations}
'''

GEOMETRY_SCHEMA = {
    "type": "object",
    "properties": {
        "centers": {"type": "array", "items": {"type": "array"}},
        "radii": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["centers", "radii"],
    "additionalProperties": False,
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
    "additionalProperties": False,
}
CP26_BRIDGE_SCHEMA, _CP26_SCHEMA_NORMALIZATIONS = normalize_tool_schema(
    CP26_PROFILE, CP26_SCHEMA
)


class PortfolioContractTest(unittest.TestCase):
    def test_current_candidate_checker_schema_is_fully_argument_free(self):
        schema = {
            **CP26_SCHEMA,
            "properties": {
                **CP26_SCHEMA["properties"],
                "tolerance": {"type": "number", "minimum": 0},
            },
            "required": ["centers", "radii", "tolerance"],
        }
        normalized, audit = normalize_tool_schema(CP26_PROFILE, schema)
        self.assertEqual(normalized["required"], [])
        self.assertTrue(any("argument" in row for row in audit))

    def test_all_slots_share_one_fixed_open_portfolio_condition(self):
        self.assertEqual(len(PORTFOLIO), 1)
        for round_index in (0, 1, 99):
            self.assertEqual(
                {role_for_slot(k, round_index).key for k in range(8)},
                {"evidence_driven_discovery"},
            )
        self.assertEqual(
            {role.required_kind for role in PORTFOLIO}, {"open_portfolio"}
        )
        self.assertEqual(
            ALLOWED_SURFACES,
            ("prompt", "skill", "sampling", "tool", "middleware", "plumbing"),
        )

    def test_description_or_schema_only_tool_edit_does_not_count(self):
        base = {"new_tools": [{
            "name": "checker", "description": "old", "input_schema": {},
            "implementation_py": GEOMETRY_CHECKER,
        }]}
        effective = {"new_tools": [{
            "name": "checker", "description": "new", "input_schema": GEOMETRY_SCHEMA,
            "implementation_py": GEOMETRY_CHECKER,
        }]}
        self.assertEqual(
            implementation_change_inventory(base, effective)["tools"], []
        )
        self.assertEqual(contract_errors(base, effective, 0), [])

    def test_every_brief_accepts_model_selected_tool_or_middleware(self):
        tool_effective = {"new_tools": [{
            "name": "geometry_check", "description": "constraints",
            "input_schema": GEOMETRY_SCHEMA,
            "implementation_py": GEOMETRY_CHECKER,
        }]}
        for slot in range(8):
            self.assertEqual(contract_errors({}, tool_effective, slot), [])

        old = "def before_model(hook_input):\n    return None\n"
        new = (
            "def before_model(hook_input):\n"
            "    state = hook_input.get('state', {})\n"
            "    if state.get('stalled_evals', 0) >= 2:\n"
            "        return {'note': 'switch family', "
            "'require_tools': ['probe_solution']}\n"
            "    return None\n"
        )
        base = {"new_middlewares": [{"name": "stall_gate", "hook": "before_model",
                                      "implementation_py": old}]}
        effective = {"new_middlewares": [{"name": "stall_gate", "hook": "before_model",
                                           "implementation_py": new}]}
        for slot in range(8):
            self.assertEqual(contract_errors(base, effective, slot), [])

    def test_submit_contract_accepts_evidence_backed_multi_surface_portfolios(self):
        base = {"system_prompt": "Base executor contract."}
        tool = {
            "name": "geometry_check", "description": "constraints",
            "input_schema": GEOMETRY_SCHEMA,
            "implementation_py": GEOMETRY_CHECKER,
        }
        declaration = generated_component_prompt_block(
            "tool", "geometry_check",
            usage_guidance="Call before repairing geometry.",
            expected_behavior_change="Use exact margins for the next edit.",
        )
        effective = {
            **base,
            "system_prompt": base["system_prompt"] + "\n\n" + declaration,
            "new_tools": [tool],
        }
        self.assertEqual(contract_errors(base, effective, 0), [])

        prompt_confounded = {
            **effective,
            "system_prompt": (
                effective["system_prompt"]
                + "\n\n# Global Rewrite\nAlways replace the full program."
            ),
        }
        self.assertEqual(contract_errors(base, prompt_confounded, 0), [])

        parameter_confounded = {
            **effective,
            "sampling": {"temperature": 0.4},
        }
        self.assertEqual(contract_errors(base, parameter_confounded, 0), [])

        second = {
            **tool,
            "name": "second_check",
            "implementation_py": GEOMETRY_CHECKER + "\n# second\n",
        }
        two_components = {
            **effective,
            "new_tools": [tool, second],
        }
        self.assertEqual(contract_errors(base, two_components, 0), [])

    def test_executor_compute_cannot_be_increased(self):
        base = {
            "agent": {"max_iterations": 24},
            "sampling": {"max_tokens": 4096},
        }
        effective = {
            **base,
            "agent": {"max_iterations": 40},
            "new_tools": [{
                "name": "geometry_check", "description": "constraints",
                "input_schema": GEOMETRY_SCHEMA,
                "implementation_py": GEOMETRY_CHECKER,
            }],
        }
        errors = contract_errors(base, effective, 0)
        self.assertTrue(any("compute budget" in error for error in errors))
        self.assertNotEqual(
            h2_compute_contract(base)["sha256"],
            h2_compute_contract(effective)["sha256"],
        )
        contract = h2_compute_contract(
            base, max_evaluator_calls=20, max_probe_calls=30,
            eval_timeout_s=420, trajectory_wall_timeout_s=7200,
            max_context_tokens=49152,
        )
        self.assertEqual(contract["max_executor_model_calls"], 24)
        self.assertEqual(contract["model_max_context_tokens"], 49152)
        self.assertEqual(contract["max_executor_input_tokens"], 24 * 49152)
        self.assertEqual(contract["max_executor_output_tokens"], 24 * 4096)
        self.assertEqual(contract["max_full_evaluations"], 20)
        self.assertEqual(contract["max_probe_cost"], 30)
        self.assertEqual(contract["max_sandbox_seconds"], 7200)

    def test_compute_freeze_survives_portfolio_ablation(self):
        base = {
            "agent": {"max_iterations": 24},
            "sampling": {"max_tokens": 4096},
        }
        effective = {
            **base,
            "sampling": {"max_tokens": 8192},
        }
        with patch.dict(
            "os.environ", {"SAH_H1_COMPONENT_PORTFOLIO": "0"}
        ):
            errors = contract_errors(base, effective, 0)
        self.assertTrue(any("compute budget" in error for error in errors))

    def test_portfolio_ablation_removes_trusted_role_profile(self):
        session = ProposeSession(
            base_spec={}, slot_index=0, portfolio_round=0,
            task_id="eft__math__circle_packing",
        )
        with patch.dict(
            "os.environ", {"SAH_H1_COMPONENT_PORTFOLIO": "0"}
        ):
            self.assertIsNone(session.portfolio_role())
            self.assertIsNone(session.validation_profile())


class ToolUtilityTest(unittest.TestCase):
    def test_geometry_constraint_checker_is_accepted(self):
        self.assertEqual(
            check_tool_utility(GEOMETRY_CHECKER, GEOMETRY_SCHEMA),
            (True, []),
        )

    def test_quick_probe_and_score_budget_relays_are_rejected(self):
        schema = {"type": "object", "properties": {}}
        for code in (
            "def run(ctx, args):\n    return ctx.probe()\n",
            "def run(ctx, args):\n    return {'score': ctx.best_score()}\n",
            "def run(ctx, args):\n    return ctx.budget_left()\n",
        ):
            ok, errors = check_tool_utility(code, schema)
            self.assertFalse(ok)
            self.assertTrue(errors)

    def test_unreachable_substantive_code_cannot_mask_a_placeholder(self):
        code = (
            "def run(ctx, args):\n"
            "    if True:\n"
            "        return {'status': 'placeholder'}\n"
            "    program = ctx.get_program()\n"
            "    lines = [line for line in program.splitlines() if line]\n"
            "    return {'lines': len(lines)}\n"
        )
        ok, errors = check_tool_utility(
            code, {"type": "object", "properties": {}}
        )
        self.assertFalse(ok)
        self.assertTrue(any("unreachable" in error for error in errors))

    def test_args_used_by_code_must_be_reachable_through_schema(self):
        ok, errors = check_tool_utility(
            GEOMETRY_CHECKER,
            {"type": "object", "properties": {},
             "additionalProperties": False},
        )
        self.assertFalse(ok)
        self.assertTrue(any("undeclared args fields" in error for error in errors))

    def test_aggregate_optimality_claims_are_rejected(self):
        misleading = GEOMETRY_CHECKER.replace(
            '"violations": violations',
            '"violations": violations, "maximal_total": sum(radii)',
        )
        ok, errors = check_tool_utility(
            misleading,
            GEOMETRY_SCHEMA,
            description=(
                "Computes the TRUE maximal feasible radii and total for the "
                "global circle-packing problem."
            ),
        )
        self.assertFalse(ok)
        self.assertTrue(any(
            "unsupported aggregate/global optimality claim" in error
            for error in errors
        ))

    def test_conditional_or_local_quantity_is_accepted(self):
        conditional = GEOMETRY_CHECKER.replace(
            '"violations": violations',
            '"violations": violations, "heuristic_total": sum(radii)',
        )
        self.assertEqual(
            check_tool_utility(
                conditional,
                GEOMETRY_SCHEMA,
                description=(
                    "Reports per-circle constraint slack and the supplied "
                    "candidate's heuristic total; it makes no global claim."
                ),
            ),
            (True, []),
        )

    def test_exact_fixed_candidate_diagnostic_is_accepted(self):
        self.assertEqual(
            check_tool_utility(
                GEOMETRY_CHECKER,
                GEOMETRY_SCHEMA,
                description=(
                    "Computes the exact maximum overlap in the supplied "
                    "candidate; no optimization claim."
                ),
            ),
            (True, []),
        )

    def test_cp26_claim_requires_exact_cardinality_and_radius_domain(self):
        ok, errors = check_tool_utility(
            GEOMETRY_CHECKER,
            GEOMETRY_SCHEMA,
            description="Audit a proposed unit-square layout.",
            validation_profile=CP26_PROFILE,
        )
        self.assertFalse(ok)
        self.assertTrue(any("minItems=maxItems=26" in error for error in errors))
        self.assertTrue(any("nonnegative" in error for error in errors))

    def test_cp26_exact_schema_is_accepted_by_utility_gate(self):
        self.assertEqual(
            check_tool_utility(
                GEOMETRY_CHECKER,
                CP26_BRIDGE_SCHEMA,
                description="Constraint checker for n=26 circle packing radii.",
                validation_profile=CP26_PROFILE,
            ),
            (True, []),
        )

    def test_cp26_feasible_output_need_not_be_one_dict_literal(self):
        subscript = GEOMETRY_CHECKER.replace(
            '    return {"feasible": not violations, "worst_margin": worst,\n'
            '            "violations": violations}\n',
            '    out = {"worst_margin": worst, "violations": violations}\n'
            '    out["feasible"] = not violations\n'
            '    return out\n',
        )
        dict_call = GEOMETRY_CHECKER.replace(
            '    return {"feasible": not violations, "worst_margin": worst,\n'
            '            "violations": violations}\n',
            '    return dict(feasible=not violations, worst_margin=worst,\n'
            '                violations=violations)\n',
        )
        for code in (subscript, dict_call):
            self.assertEqual(
                check_tool_utility(
                    code, CP26_BRIDGE_SCHEMA,
                    validation_profile=CP26_PROFILE,
                ),
                (True, []),
            )

    def test_cp26_profile_requires_bridge_and_paired_optional_schema(self):
        ok, errors = check_tool_utility(
            GEOMETRY_CHECKER, CP26_SCHEMA,
            description="Audit this public-task layout.",
            validation_profile=CP26_PROFILE,
        )
        self.assertFalse(ok)
        self.assertTrue(any("argument-free" in error for error in errors))
        self.assertTrue(any("paired" in error for error in errors))

    def test_cp26_fixed_center_solved_claim_is_rejected(self):
        misleading = GEOMETRY_CHECKER.replace(
            '"violations": violations}',
            '"violations": violations, "radii_assignment_gap": 1.0}',
        ).replace(
            "def run(ctx, args):",
            '"""The coordinate update is a monotone contraction with a unique solution."""\n'
            "def run(ctx, args):",
        )
        ok, errors = check_tool_utility(
            misleading, CP26_BRIDGE_SCHEMA,
            validation_profile=CP26_PROFILE,
        )
        self.assertFalse(ok)
        self.assertTrue(any(
            "unsupported aggregate/global optimality claim" in error
            for error in errors
        ))

    def test_non_checker_cp26_role_is_not_forced_into_checker_schema(self):
        state_analyzer_profile = trusted_component_profile(
            "eft__math__circle_packing", "state_analyzer"
        )
        self.assertEqual(
            check_tool_utility(
                GEOMETRY_CHECKER, GEOMETRY_SCHEMA,
                description=(
                    "Schedule a search for the n=26 circle-packing task and "
                    "report diagnostics."
                ),
                validation_profile=state_analyzer_profile,
            ),
            (True, []),
        )


class MiddlewareUtilityTest(unittest.TestCase):
    def test_constant_text_hook_is_rejected(self):
        ok, errors = check_middleware_utility(
            "def before_model(hook_input):\n    return 'try harder'\n",
            "before_model",
        )
        self.assertFalse(ok)
        self.assertTrue(any("runtime state" in error for error in errors))

    def test_state_dependent_gate_is_accepted(self):
        code = (
            "def before_model(hook_input):\n"
            "    state = hook_input.get('state', {})\n"
            "    if state.get('last_error') and "
            "state.get('probe_available', False) and "
            "state.get('probes_remaining', 0) > 0 and "
            "state.get('evals_remaining', 0) > 0 and "
            "state.get('last_step_kind') != 'probe':\n"
            "        return {'note': 'NEXT CALL: probe_solution on the current failed program', "
            "'require_tools': ['probe_solution']}\n"
            "    return None\n"
        )
        self.assertEqual(
            check_middleware_utility(code, "before_model"), (True, [])
        )

    def test_probe_gate_with_exhausted_budget_is_rejected(self):
        code = (
            "def before_model(hook_input):\n"
            "    state = hook_input.get('state', {})\n"
            "    if state.get('stalled_evals', 0) >= 2:\n"
            "        return {'note': 'probe another family', "
            "'require_tools': ['probe_solution']}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(code, "before_model")
        self.assertFalse(ok)
        self.assertTrue(any("exhausted probe budget" in error for error in errors))

    def test_evaluation_gate_with_exhausted_budget_is_rejected(self):
        code = (
            "def before_model(hook_input):\n"
            "    state = hook_input.get('state', {})\n"
            "    if state.get('last_validity') == 0.0:\n"
            "        return {'note': 'verify repair', "
            "'require_tools': ['evaluate_solution']}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(code, "before_model")
        self.assertFalse(ok)
        self.assertTrue(any(
            "evaluation budget" in error for error in errors
        ))

    def test_gate_fallback_is_rejected_when_both_budgets_are_exhausted(self):
        code = (
            "def before_model(hook_input):\n"
            "    state = hook_input.get('state', {})\n"
            "    if state.get('stalled_evals', 0) < 2:\n"
            "        return None\n"
            "    if state.get('probes_remaining', 0) > 0:\n"
            "        return {'note': 'probe', "
            "'require_tools': ['probe_solution']}\n"
            "    return {'note': 'evaluate', "
            "'require_tools': ['evaluate_solution']}\n"
        )
        ok, errors = check_middleware_utility(code, "before_model")
        self.assertFalse(ok)
        self.assertTrue(any(
            "evaluation budget" in error for error in errors
        ))

    def test_collection_diversity_state_cannot_be_numeric_coerced(self):
        code = (
            "def before_model(hook_input):\n"
            "    def _num(state, key):\n"
            "        try:\n"
            "            return float(state.get(key, 0))\n"
            "        except (TypeError, ValueError):\n"
            "            return 0.0\n"
            "    state = hook_input.get('state', {})\n"
            "    families = _num(state, 'families_explored')\n"
            "    if families < 2 and state.get('family_streak', 0) >= 2:\n"
            "        return {'note': 'switch family'}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(code, "before_model")
        self.assertFalse(ok)
        self.assertTrue(any(
            "collection-valued middleware state" in error for error in errors
        ))

    def test_required_tool_cannot_be_described_as_suspended(self):
        code = (
            "def before_model(hook_input):\n"
            "    state = hook_input.get('state', {})\n"
            "    if state.get('stalled_evals', 0) >= 2 and "
            "state.get('probe_available', False):\n"
            "        return {'note': 'change family', "
            "'require_tools': ['probe_solution']}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(
            code,
            "before_model",
            description="Suspend cheap re-ranking with a probe gate.",
        )
        self.assertFalse(ok)
        self.assertTrue(any(
            "require_tools forces" in error for error in errors
        ))
        self.assertTrue(any(
            "matched span=" in error for error in errors
        ))

    def test_evolve_block_identifier_is_not_a_blocking_verb(self):
        code = (
            "def before_model(hook_input):\n"
            "    state = hook_input.get('state', {})\n"
            "    if state.get('last_error') and "
            "state.get('evals_remaining', 0) > 0:\n"
            "        return {'note': 'edit the EVOLVE-BLOCK now', "
            "'require_tools': ['edit_solution']}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(
            code,
            "before_model",
            description="Edit the EVOLVE-BLOCK after a measured failure.",
        )
        self.assertFalse(any(
            "require_tools forces" in error for error in errors
        ), (ok, errors))

    def test_gate_rationale_is_not_parsed_as_an_action_sequence(self):
        code = (
            "def before_model(hook_input):\n"
            "    state = hook_input.get('state', {})\n"
            "    if state.get('last_error') and "
            "state.get('probe_available', False) and "
            "state.get('evals_remaining', 0) > 0 and "
            "state.get('last_step_kind') != 'probe':\n"
            "        return {'note': 'Stage a different construction and then call probe_solution', "
            "'require_tools': ['probe_solution']}\n"
            "    return None\n"
        )
        # The trusted wrapper discards a gated free-form note and emits the
        # require_tools-derived NEXT-CALL message. Historical trigger wording
        # is therefore not interpreted as a second control channel.
        self.assertEqual(
            check_middleware_utility(code, "before_model"), (True, [])
        )

    def test_repeated_probe_gate_is_rejected_as_budget_drain(self):
        code = (
            "def before_model(hook_input):\n"
            "    state = hook_input.get('state', {})\n"
            "    if state.get('stalled_evals', 0) >= 2 and "
            "state.get('probe_available', False) and "
            "state.get('evals_remaining', 0) > 0:\n"
            "        return {'note': 'NEXT CALL: probe_solution current program', "
            "'require_tools': ['probe_solution']}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(code, "before_model")
        self.assertFalse(ok)
        self.assertTrue(any(
            "gate cycle" in error or "beyond two" in error
            for error in errors
        ))

    def test_forced_stagnation_claim_must_gate_real_edited_evaluation(self):
        code = (
            "def before_model(hook_input):\n"
            "    state = hook_input.get('state', {})\n"
            "    if state.get('last_step_kind') == 'edit_eval' and "
            "state.get('family_streak', 0) >= 3 and "
            "state.get('evals_remaining', 0) > 0:\n"
            "        return {'note': 'pivot', 'require_tools': ['edit_solution']}\n"
            "    if state.get('family_streak', 0) >= 3:\n"
            "        return 'consider another family'\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(
            code, "before_model",
            description="Redirects repeated local search into another topology.",
            validation_profile=STAGNATION_PROFILE,
        )
        self.assertFalse(ok)
        self.assertTrue(any(
            "BOTH valid evaluated edit modes" in error
            or "stale_stall_after_edit_eval" in error
            for error in errors
        ), errors)

    def test_stagnation_gate_must_cover_valid_diff_and_full_rewrite(self):
        diff_only = (
            "def before_model(hook_input):\n"
            "    s = hook_input.get('state', {})\n"
            "    if (s.get('last_step_kind') == 'diff' and\n"
            "            s.get('stalled_evals', 0) >= 2 and\n"
            "            s.get('family_streak', 0) >= 2 and\n"
            "            not s.get('pending_evaluation', False) and\n"
            "            s.get('last_edit_staged') is not False and\n"
            "            s.get('evals_remaining', 0) > 0):\n"
            "        return {'require_tools': ['edit_solution'],\n"
            "                'require_structure_change': True}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(
            diff_only, "before_model", validation_profile=STAGNATION_PROFILE,
        )
        self.assertFalse(ok)
        self.assertTrue(any(
            "stalled_valid_full_rewrite" in error for error in errors
        ), errors)

    def test_stagnation_gate_requires_stall_and_family_streak_conjunction(self):
        wrong_or = (
            "def before_model(hook_input):\n"
            "    s = hook_input.get('state', {})\n"
            "    if ((s.get('stalled_evals', 0) >= 2 or\n"
            "         s.get('family_streak', 0) >= 2) and\n"
            "        s.get('last_step_kind') in ('diff', 'full_rewrite') and\n"
            "        not s.get('pending_evaluation', False) and\n"
            "        s.get('last_edit_staged') is not False and\n"
            "        not s.get('active_tool_gate') and\n"
            "        s.get('evals_remaining', 0) > 0):\n"
            "        return {'require_tools': ['edit_solution'],\n"
            "                'require_structure_change': True}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(
            wrong_or, "before_model", validation_profile=STAGNATION_PROFILE,
        )
        self.assertFalse(ok)
        self.assertTrue(any(
            "forbidden triggers" in error
            and "improving_same_family" in error
            and "stalled_new_family" in error
            for error in errors
        ), errors)

    def test_stagnation_gate_rejects_stale_counters_after_non_edit_steps(self):
        stale_trigger = (
            "def before_model(hook_input):\n"
            "    s = hook_input.get('state', {})\n"
            "    if (s.get('stalled_evals', 0) >= 2 and\n"
            "            s.get('family_streak', 0) >= 2 and\n"
            "            not s.get('pending_evaluation', False) and\n"
            "            s.get('last_edit_staged') is not False and\n"
            "            not s.get('active_tool_gate') and\n"
            "            s.get('evals_remaining', 0) > 0):\n"
            "        return {'require_tools': ['edit_solution'],\n"
            "                'require_structure_change': True}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(
            stale_trigger, "before_model", validation_profile=STAGNATION_PROFILE,
        )
        self.assertFalse(ok)
        self.assertTrue(any(
            "stale_stall_after_probe" in error
            and "stale_stall_after_candidate_geometry" in error
            and "stale_stall_after_seed" in error
            and "stale_stall_after_edit_eval" in error
            for error in errors
        ), errors)

    def test_persistent_multi_tool_gate_is_rejected(self):
        code = (
            "def before_model(hook_input):\n"
            "    s = hook_input.get('state', {})\n"
            "    if s.get('stalled_evals', 0) >= 2 and "
            "s.get('probe_available', False) and "
            "s.get('evals_remaining', 0) > 0:\n"
            "        return {'note': 'probe or edit now', "
            "'require_tools': ['probe_solution', 'edit_solution']}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(code, "before_model")
        self.assertFalse(ok)
        self.assertTrue(any(
            "gate cycle" in error or "beyond two" in error
            for error in errors
        ))

    def test_alternating_probe_edit_gate_cycle_is_rejected(self):
        code = (
            "def before_model(hook_input):\n"
            "    s = hook_input.get('state', {})\n"
            "    if (s.get('stalled_evals', 0) < 2 or "
            "not s.get('probe_available', False) or "
            "s.get('evals_remaining', 0) <= 0):\n"
            "        return None\n"
            "    if s.get('last_step_kind') == 'probe':\n"
            "        return {'note': 'edit now', 'require_tools': ['edit_solution']}\n"
            "    return {'note': 'probe now', 'require_tools': ['probe_solution']}\n"
        )
        ok, errors = check_middleware_utility(code, "before_model")
        self.assertFalse(ok)
        self.assertTrue(any(
            "gate cycle" in error or "beyond two" in error
            for error in errors
        ))

    def test_same_family_edit_requires_pending_evaluation_hysteresis(self):
        code = (
            "def before_model(hook_input):\n"
            "    s = hook_input.get('state', {})\n"
            "    if (s.get('stalled_evals', 0) >= 2 and "
            "s.get('family_streak', 0) >= 3 and "
            "s.get('evals_remaining', 0) > 0):\n"
            "        return {'note': 'edit now', 'require_tools': ['edit_solution']}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(
            code, "before_model", validation_profile=STAGNATION_PROFILE
        )
        self.assertFalse(ok)
        self.assertTrue(any("same_family_edit" in error for error in errors))

    def test_valid_stalled_diff_edit_evaluate_cycle_is_rejected(self):
        code = (
            "def before_model(hook_input):\n"
            "    s = hook_input.get('state', {})\n"
            "    if (s.get('pending_evaluation', False) and "
            "s.get('evals_remaining', 0) > 0):\n"
            "        return {'require_tools': ['evaluate_solution']}\n"
            "    if (s.get('stalled_evals', 0) >= 2 and "
            "s.get('last_step_kind') == 'diff' and "
            "s.get('last_validity', 0.0) >= 1.0 and "
            "s.get('last_edit_staged') is not False and "
            "s.get('evals_remaining', 0) > 0):\n"
            "        return {'require_tools': ['edit_solution']}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(
            code, "before_model", validation_profile=STAGNATION_PROFILE
        )
        self.assertFalse(ok)
        self.assertTrue(any(
            "beyond two" in error or "gate cycle" in error
            for error in errors
        ))

    def test_historical_contract_prose_is_not_a_false_action_order(self):
        code = (
            "def before_model(hook_input):\n"
            "    s = hook_input.get('state', {})\n"
            "    if (s.get('last_error') and "
            "s.get('probe_available', False) and "
            "s.get('evals_remaining', 0) > 0 and "
            "s.get('last_step_kind') != 'probe'):\n"
            "        return {'note': 'probe current program', "
            "'require_tools': ['probe_solution']}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(
            code, "before_model",
            contract_text=(
                "After a full rewrite stalls, call probe_solution on the "
                "current program. When the last edit failed, use "
                "probe_solution on that unchanged candidate."
            ),
        )
        self.assertTrue(ok, errors)

    def test_unknown_hook_result_key_is_rejected_like_runtime(self):
        code = (
            "def before_model(hook_input):\n"
            "    s = hook_input.get('state', {})\n"
            "    if s.get('last_error') and s.get('probe_available', False):\n"
            "        return {'note': 'probe', 'require_tools': "
            "['probe_solution'], 'debug': 'stalled'}\n"
            "    return None\n"
        )
        ok, errors = check_middleware_utility(code, "before_model")
        self.assertFalse(ok)
        self.assertTrue(any("unsupported keys" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
