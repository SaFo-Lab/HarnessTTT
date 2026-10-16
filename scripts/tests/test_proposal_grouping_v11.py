from __future__ import annotations

import copy
import unittest
from pathlib import Path

from outer.genome import harness_spec as hs
from outer.rounds.outer_round import (
    _aggregate_condition_reward_groups,
    _aggregate_feedback_repeats,
    _bind_training_advantages,
    _condition_candidate_groups,
    _proposal_condition,
    _trainable_condition_candidates,
)
from outer.proposing.propose import (
    CandidateRecord,
    _declarative_h1_surface,
    _portfolio_tool_names,
    component_behavior_fingerprint_record,
    component_semantic_fingerprint,
    component_semantic_fingerprint_record,
    dedup_group,
)
from outer.proposing.proposer_io import render_feedback


class ProposalConditionTest(unittest.TestCase):
    def _condition(self, **overrides):
        values = {
            "task_id": "cp26",
            "round_index": 3,
            "role": "evidence_driven_discovery",
            "parent_h2_sha256": "parent-a",
            "input_prompt_sha256": "prompt-a",
            "system_prompt_sha256": "system-a",
            "behavior_policy": "trained_phi",
            "behavior_policy_identity": "checkpoint-a",
            "resolved_h1_config": {
                "max_iterations": 12,
                "max_context_tokens": 49152,
                "sampling": {"max_tokens": 4096, "temperature": 1.0},
            },
            "parent_compute_contract": {
                "agent_max_iterations": 36,
                "model_max_output_tokens": 8192,
                "sha256": "budget-a",
            },
            "sampling_mode": "parallel",
        }
        values.update(overrides)
        return _proposal_condition(**values)

    def test_same_conditions_share_group(self):
        self.assertEqual(self._condition(), self._condition())

    def test_declarative_middleware_surface_is_stable_and_behavior_complete(self):
        left = _declarative_h1_surface()
        right = _declarative_h1_surface()
        self.assertEqual(left, right)
        self.assertNotIn(" at 0x", repr(left))

        long_output = next(
            row for row in left["middlewares"]
            if "LongToolOutputMiddleware" in row["import"]
        )
        self.assertEqual(
            long_output["params"]["bypass_tool_names"],
            ["harness_shell", "submit_harness"],
        )

        changed = copy.deepcopy(right)
        changed_long_output = next(
            row for row in changed["middlewares"]
            if "LongToolOutputMiddleware" in row["import"]
        )
        changed_long_output["params"]["bypass_tool_names"] = [
            "submit_harness"
        ]
        self.assertNotEqual(
            self._condition(resolved_h1_config={"declarative": left})[0],
            self._condition(resolved_h1_config={"declarative": changed})[0],
        )

    def test_every_sample_exposes_the_complete_safe_editing_surface(self):
        tool_names = set(_portfolio_tool_names("tool"))
        middleware_names = set(_portfolio_tool_names("middleware"))
        executable_names = set(_portfolio_tool_names("executable"))
        expected = {
            "harness_shell", "write_harness_file", "edit_harness_file",
            "delete_harness_file",
            "upsert_skill", "upsert_tool", "upsert_middleware", "repair_component",
            "validate_harness", "submit_harness",
        }
        self.assertEqual(tool_names, expected)
        self.assertEqual(middleware_names, expected)
        self.assertEqual(executable_names, expected)
        surface = _declarative_h1_surface(
            allowed_tool_names=tool_names,
        )
        self.assertEqual(surface["tool_surface"], "portfolio_full")
        self.assertEqual(
            {row["name"] for row in surface["tools"]}, tool_names
        )
        self.assertEqual(set(surface["tool_schema_sha256"]), tool_names)

    def test_fixed_system_contract_requires_reachable_behavior(self):
        system = (
            Path(__file__).resolve().parents[2]
            / "src/outer/harness/system.md"
        ).read_text()
        self.assertIn("reachable useful path", system)
        self.assertIn("observable success signal", system)
        self.assertIn("Every component must have a distinct evidence-backed role", system)

    def test_role_parent_policy_prompt_and_budget_split_groups(self):
        base_id, _ = self._condition()
        variants = [
            {"role": "tampered_condition_label"},
            {"parent_h2_sha256": "parent-b"},
            {"input_prompt_sha256": "prompt-b"},
            {"behavior_policy": "frozen_base"},
            {"sampling_mode": "sequential"},
            {"resolved_h1_config": {
                "max_iterations": 18,
                "max_context_tokens": 49152,
                "sampling": {"max_tokens": 4096, "temperature": 1.0},
            }},
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(base_id, self._condition(**variant)[0])

    def test_partition_rejects_tampered_group_hash(self):
        group_id, condition = self._condition()
        rows = [
            {"k": 0, "proposal_group_id": group_id,
             "proposal_condition": condition},
            {"k": 1, "proposal_group_id": group_id,
             "proposal_condition": condition},
        ]
        groups = _condition_candidate_groups(rows)
        self.assertEqual([(group_id, rows)], groups)
        rows[1]["proposal_group_id"] = "h1g_tampered"
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            _condition_candidate_groups(rows)

    def test_dedup_is_local_to_the_supplied_condition_group(self):
        effective = {"schema": "h2spec/1.0", "system_prompt": "changed"}
        spec_hash = hs.spec_hash(effective)
        role_a = CandidateRecord(
            k=0, valid=True, effective=effective, spec_hash=spec_hash,
            parent_spec_hash="parent-a",
        )
        role_b = CandidateRecord(
            k=4, valid=True, effective=effective, spec_hash=spec_hash,
            parent_spec_hash="parent-a",
        )
        role_a.proposal_group_id = "h1g_role_a"
        role_b.proposal_group_id = "h1g_role_b"
        dedup_group([role_a], {})
        dedup_group([role_b], {})
        self.assertTrue(role_a.valid)
        self.assertTrue(role_b.valid)

        same_group_duplicate = CandidateRecord(
            k=1, valid=True, effective=effective, spec_hash=spec_hash,
            parent_spec_hash="parent-a",
        )
        same_group_duplicate.proposal_group_id = "h1g_role_a"
        dedup_group([role_a, same_group_duplicate], {})
        self.assertTrue(role_a.valid)
        self.assertFalse(same_group_duplicate.valid)

    def test_semantic_dedup_rejects_rename_literal_and_noop_variants(self):
        schema = {
            "type": "object",
            "properties": {"values": {"type": "array"}},
        }
        first_effective = {
            "new_tools": [{
                "name": "measure_a",
                "implementation_py": (
                    "def run(ctx, args):\n"
                    "    values = args.get('values') or []\n"
                    "    total = 1\n"
                    "    for value in values:\n"
                    "        total += value\n"
                    "    return {'total': total}\n"
                ),
                "input_schema": schema,
            }],
        }
        cosmetic_effective = {
            "new_tools": [{
                "name": "renamed_measure",
                "description": "Different prose must not create novelty.",
                "implementation_py": (
                    "def run(ctx, payload):\n"
                    "    items = payload.get('values') or []\n"
                    "    accumulator = 99\n"
                    "    pass\n"
                    "    accumulator = accumulator\n"
                    "    for item in items:\n"
                    "        accumulator += item\n"
                    "    return {'total': accumulator}\n"
                ),
                "input_schema": {
                    **schema,
                    "description": "Cosmetic schema prose.",
                },
            }],
        }
        different_effective = copy.deepcopy(first_effective)
        different_effective["new_tools"][0]["implementation_py"] = (
            different_effective["new_tools"][0]["implementation_py"].replace(
                "total += value", "total *= value"
            )
        )
        self.assertEqual(
            component_semantic_fingerprint(first_effective),
            component_semantic_fingerprint(cosmetic_effective),
        )
        self.assertNotEqual(
            component_semantic_fingerprint(first_effective),
            component_semantic_fingerprint(different_effective),
        )
        replay = component_semantic_fingerprint_record(first_effective)
        self.assertEqual(replay["schema"], "sah-component-semantic-fingerprint/2.0")
        self.assertEqual(replay["fingerprint"], component_semantic_fingerprint(
            first_effective
        ))
        self.assertIn("canonical_payload", replay)
        self.assertIn("canonical_payload_sha256", replay)
        self.assertNotIn("producer_runtime", replay["canonical_payload"])

        records = [
            CandidateRecord(
                k=0, valid=True, effective=first_effective,
                spec_hash=hs.spec_hash(first_effective),
                parent_spec_hash="parent-a",
            ),
            CandidateRecord(
                k=1, valid=True, effective=cosmetic_effective,
                spec_hash=hs.spec_hash(cosmetic_effective),
                parent_spec_hash="parent-a",
            ),
            CandidateRecord(
                k=2, valid=True, effective=different_effective,
                spec_hash=hs.spec_hash(different_effective),
                parent_spec_hash="parent-a",
            ),
        ]
        dedup_group(records, {})
        self.assertTrue(records[0].valid)
        self.assertFalse(records[1].valid)
        self.assertIn("semantic duplicate", records[1].errors[0])
        self.assertTrue(records[2].valid)
        self.assertEqual(
            records[0].component_semantic_fingerprint_record["fingerprint"],
            records[0].component_semantic_fingerprint,
        )
        self.assertEqual(
            records[1].within_group_semantic_duplicates[0]["k"], 0
        )

    def test_behavior_dedup_rejects_different_code_with_same_truth_table(self):
        condition = (
            "s.get('last_step_kind') in ('diff', 'full_rewrite') and "
            "s.get('stalled_evals', 0) >= 2 and "
            "s.get('family_streak', 0) >= 2 and "
            "not s.get('pending_evaluation', False) and "
            "s.get('last_edit_staged') is not False and "
            "not s.get('active_tool_gate') and "
            "s.get('evals_remaining', 0) > 0"
        )
        inline = (
            "def before_model(hook_input):\n"
            "    s = hook_input.get('state', {})\n"
            f"    if {condition}:\n"
            "        return {'require_tools': ['edit_solution'], "
            "'require_structure_change': True}\n"
            "    return None\n"
        )
        factored = (
            "def before_model(hook_input):\n"
            "    s = hook_input.get('state', {})\n"
            "    edit_outcome = s.get('last_step_kind') in "
            "('diff', 'full_rewrite')\n"
            "    measured_stall = s.get('stalled_evals', 0) >= 2\n"
            "    repeated_family = s.get('family_streak', 0) >= 2\n"
            "    safe = (not s.get('pending_evaluation', False) and "
            "s.get('last_edit_staged') is not False and "
            "not s.get('active_tool_gate') and "
            "s.get('evals_remaining', 0) > 0)\n"
            "    if edit_outcome and measured_stall and repeated_family and safe:\n"
            "        return {'require_tools': ['edit_solution'], "
            "'require_structure_change': True}\n"
            "    return None\n"
        )
        profile = {
            "schema": "sah-trusted-component-profile/1.0",
            "portfolio_role": "stagnation_gate",
        }

        def effective(name, code):
            return {"new_middlewares": [{
                "name": name, "hook": "before_model",
                "description": "Force a structural family switch after stagnation.",
                "implementation_py": code,
            }]}

        first_effective = effective("gate_a", inline)
        second_effective = effective("gate_b", factored)
        first_behavior = component_behavior_fingerprint_record(
            first_effective, {}, profile
        )
        second_behavior = component_behavior_fingerprint_record(
            second_effective, {}, profile
        )
        self.assertTrue(first_behavior)
        self.assertEqual(
            first_behavior["fingerprint"], second_behavior["fingerprint"]
        )
        self.assertNotEqual(
            component_semantic_fingerprint(first_effective),
            component_semantic_fingerprint(second_effective),
        )
        records = [
            CandidateRecord(
                k=0, valid=True, effective=first_effective,
                spec_hash=hs.spec_hash(first_effective),
                parent_spec_hash="parent-a", validation_profile=profile,
            ),
            CandidateRecord(
                k=1, valid=True, effective=second_effective,
                spec_hash=hs.spec_hash(second_effective),
                parent_spec_hash="parent-a", validation_profile=profile,
            ),
        ]
        dedup_group(records, {})
        self.assertTrue(records[0].valid)
        self.assertFalse(records[1].valid)
        self.assertIn("behavioral duplicate", records[1].errors[0])


class ConditionRewardAggregationTest(unittest.TestCase):
    def test_infrastructure_rows_never_enter_advantage_normalization(self):
        candidates = [
            {"k": 0, "training_eligible": True},
            {"k": 1, "training_eligible": True},
            {"k": 2, "training_eligible": True,
             "infrastructure_failure": True,
             "training_block_reason": "provider_context_overflow"},
            {"k": 3, "training_eligible": False},
        ]
        selected = _trainable_condition_candidates(candidates)
        self.assertEqual([row["k"] for row in selected], [0, 1])

        evaluation = {
            "rows": [
                {"k": 0, "advantage": -99.0},
                {"k": 1, "advantage": -98.0},
                {"k": 2, "advantage": 500.0},
                {"k": 3, "advantage": 400.0},
            ],
        }
        training = {
            "rows": [
                {"k": 0, "advantage": -0.25},
                {"k": 1, "advantage": 0.25},
            ],
            "adv_mode": "rloo", "reward_mean": 0.5,
            "reward_std": 0.25,
        }
        _bind_training_advantages(evaluation, training)
        self.assertEqual(
            [row["advantage"] for row in evaluation["rows"]],
            [-0.25, 0.25, 0.0, 0.0],
        )
        self.assertEqual(evaluation["training_slots"], [0, 1])
        self.assertEqual(
            [row["advantage_eligible"] for row in evaluation["rows"]],
            [True, True, False, False],
        )

    def test_single_trainable_condition_has_no_grpo_group(self):
        self.assertEqual(_trainable_condition_candidates([
            {"k": 0, "training_eligible": True},
            {"k": 1, "training_eligible": False},
        ]), [])

    def test_repeat_feedback_keeps_each_repeat_and_aggregates_cost(self):
        rows = [
            {
                "decode_seed": 11, "evals": 2, "probes": 1,
                "llm_calls": 3, "input_tokens": 100,
                "output_tokens": 20, "sandbox_seconds": 1.0,
                "component_invocations": 0, "middleware_fires": 1,
                "stop": "completed", "err": None,
            },
            {
                "decode_seed": 12, "evals": 4, "probes": 3,
                "llm_calls": 5, "input_tokens": 300,
                "output_tokens": 40, "sandbox_seconds": 3.0,
                "component_invocations": 2, "middleware_fires": 0,
                "stop": "iteration_limit", "err": None,
            },
        ]
        aggregate = _aggregate_feedback_repeats(rows)
        self.assertEqual(aggregate["per_repeat"], rows)
        self.assertEqual(aggregate["evals"], 3.0)
        self.assertEqual(aggregate["input_tokens"], 400.0)
        self.assertEqual(aggregate["component_invocations"], 2.0)
        self.assertEqual(aggregate["stop"], "mixed")
        self.assertEqual(
            aggregate["stop_reasons"],
            {"completed": 1, "iteration_limit": 1},
        )

    def test_unknown_repeat_tokens_remain_null(self):
        base = {
            "evals": 1, "probes": 0, "llm_calls": 1,
            "sandbox_seconds": 0.1, "component_invocations": 0,
            "middleware_fires": 0, "stop": "completed", "err": None,
        }
        aggregate = _aggregate_feedback_repeats([
            {**base, "input_tokens": 10, "output_tokens": 2},
            {**base, "input_tokens": None, "output_tokens": None},
        ])
        self.assertIsNone(aggregate["input_tokens"])
        self.assertIsNone(aggregate["output_tokens"])

    def test_feedback_names_inactive_bridge_rejection_and_credit_clamp(self):
        rendered = render_feedback({
            "round": 0,
            "base_score": 0.8,
            "best_score": 0.98,
            "n_stuck_at_base": 0,
            "candidates": [{
                "k": 3, "score": 0.98, "control_score": 0.80,
                "causal_delta": 0.18, "component_credit_delta": 0.0,
                "component_activation_status": "delivered_but_inactive",
                "component_invocations": 1,
                "component_usable_outputs": 0,
                "component_semantic_successes": 0,
                "component_semantic_rejections": 1,
                "component_reject_reasons": {"bridge_stale": 1},
                "middleware_fires": 0,
                "failure_taxonomy": "component_rejected:bridge_stale",
                "changed": ["constraint_checker"],
            }],
        })
        self.assertIn("raw_delta=+0.18", rendered)
        self.assertIn("component_credit_delta=+0", rendered)
        self.assertIn("activation=delivered_but_inactive", rendered)
        self.assertIn('reject_reasons={"bridge_stale":1}', rendered)
        self.assertIn("semantic_ok=0", rendered)

    def test_advantages_remain_local_to_condition_group(self):
        group_a = {
            "rows": [
                {"k": 0, "valid": True, "score": 2.0,
                 "reward": 1.0, "advantage": 0.5},
                {"k": 1, "valid": True, "score": 1.0,
                 "reward": 0.0, "advantage": -0.5},
            ],
            "adv_mode": "rloo", "reward_mean": 0.5, "reward_std": 0.5,
            "best_k": 0, "best_score": 2.0,
        }
        group_b = {
            "rows": [
                {"k": 2, "valid": True, "score": 4.0,
                 "reward": 10.0, "advantage": 0.25},
                {"k": 3, "valid": True, "score": 3.0,
                 "reward": 9.0, "advantage": -0.25},
            ],
            "adv_mode": "rloo", "reward_mean": 9.5, "reward_std": 0.5,
            "best_k": 2, "best_score": 4.0,
        }
        merged = _aggregate_condition_reward_groups(
            task_id="cp26", base_score=0.0,
            condition_groups=[("h1g_a", group_a), ("h1g_b", group_b)],
        )
        self.assertEqual(
            [row["advantage"] for row in merged["rows"]],
            [0.5, -0.5, 0.25, -0.25],
        )
        self.assertEqual(merged["best_k"], 2)
        self.assertEqual(len(merged["condition_groups"]), 2)

    def test_task_promotion_uses_eligible_component_not_raw_lucky_delta(self):
        inactive = {
            "rows": [{
                "k": 0, "valid": True, "score": 1.0,
                "control_score": 0.8, "causal_delta": 0.2,
                "component_credit_delta": 0.0, "reward": 0.0,
                "advantage": 0.0, "attribution_valid": True,
                "promotion_eligible": False,
                "pair_contract": {"samples": [{
                    "candidate_score": 1.0, "control_score": 0.8,
                    "causal_delta": 0.2,
                }]},
            }],
            "adv_mode": "paired_same_seed:rloo+max(a=0.3)",
            "paired_repeats": 3, "min_promotion_repeats": 3,
            "min_effect": 1e-4,
        }
        active = {
            "rows": [{
                "k": 4, "valid": True, "score": 0.9,
                "control_score": 0.87, "causal_delta": 0.03,
                "component_credit_delta": 0.03, "reward": 0.03,
                "advantage": 0.4, "attribution_valid": True,
                "promotion_eligible": True,
                "pair_contract": {"samples": [{
                    "candidate_score": 0.9, "control_score": 0.87,
                    "causal_delta": 0.03,
                }]},
            }],
            "adv_mode": "paired_same_seed:rloo+max(a=0.3)",
            "paired_repeats": 3, "min_promotion_repeats": 3,
            "min_effect": 1e-4,
        }
        merged = _aggregate_condition_reward_groups(
            task_id="cp26", base_score=0.8,
            condition_groups=[("h1g_tool", inactive),
                              ("h1g_middleware", active)],
        )
        self.assertTrue(merged["improved"])
        self.assertEqual(merged["best_k"], 4)
        self.assertEqual(merged["promotion_best_k"], 4)
        self.assertEqual(merged["diagnostic_causal_best_k"], 4)
        self.assertEqual(
            [row["advantage"] for row in merged["rows"]], [0.0, 0.4]
        )


if __name__ == "__main__":
    unittest.main()
