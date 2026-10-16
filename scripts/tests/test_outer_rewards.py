import json
import tempfile
import unittest
from pathlib import Path

from outer.reward.rewards import compute_task_group_paired, load_rollout_score


TASK = "example__task"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


class LoadRolloutScoreTest(unittest.TestCase):
    def test_terminal_failure_does_not_inherit_seed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run0"
            write_json(run / "checkpoints" / f"{TASK}.json", {
                "task_id": TASK,
                "best_score": 7.5,
            })
            write_json(run / "summary.json", [{
                "task_id": TASK,
                "best_score": None,
                "best_program": None,
                "error": "ConfigError: invalid candidate harness",
            }])

            self.assertIsNone(load_rollout_score(root, TASK))


class PairedRewardTest(unittest.TestCase):
    H2_SHA = "c" * 64
    COMPUTE = {
        "agent_max_iterations": 36,
        "model_max_context_tokens": 131072,
        "model_max_output_tokens": 8192,
        "max_executor_model_calls": 36,
        "max_executor_input_tokens": 36 * 131072,
        "max_executor_output_tokens": 36 * 8192,
        "max_full_evaluations": 2,
        "max_probe_cost": 30,
        "max_sandbox_seconds": 10000.0,
        "sha256": "budget-same",
    }

    @classmethod
    def _candidate(cls, k: int, component=None):
        candidate = {
            "k": k, "valid": True, "package_sha256": cls.H2_SHA,
            "parent_compute_contract": dict(cls.COMPUTE),
            "candidate_compute_contract": dict(cls.COMPUTE),
        }
        if component:
            candidate["changed_fields"] = [
                "system_prompt", f"new_tools.{component}"
            ]
        return candidate

    def _score(
        self, root: Path, k: int, score: float, *, eligible=True,
        repeat: int = 0, tool_audit=None, middleware_audit=None,
        timestamp_layer: bool = False,
    ):
        candidate_root = root / TASK / f"cand{k:02d}"
        run = (
            candidate_root / f"repeat{repeat:02d}"
            / f"20260821-1200{repeat:02d}"
            if timestamp_layer else candidate_root / f"run{repeat}"
        )
        decode_seed = 200000 + k + repeat * 8
        result = {
            "task_id": TASK,
            "best_score": score,
            "score_eligible": eligible,
            "decode_seed": decode_seed,
            "ledger": {
                "max_evaluator_calls": 2,
                "max_probe_calls": 30,
                "max_executor_model_calls": 36,
                "max_executor_context_tokens": 131072,
                "max_executor_input_tokens": 36 * 131072,
                "max_executor_output_tokens": 36 * 8192,
                "max_sandbox_seconds": 10000.0,
                "evaluator_calls": 2,
                "probe_calls": 1,
                "llm_calls": 4,
                "llm_input_tokens": 4000,
                "llm_output_tokens": 1000,
                "sandbox_seconds": 8.0,
            },
            "seed_program_provenance": {
                "program_sha256": "a" * 64,
                "registry_sha256": "b" * 64,
            },
            "h2_package_provenance": {
                "sha256": self.H2_SHA,
                "stable_during_rollout": True,
            },
            "tool_audit": tool_audit or {},
            "middleware_audit": middleware_audit or {},
        }
        write_json(run / "summary.json", [result])
        write_json(run / "results" / f"{TASK}.json", result)
        write_json(run / "provenance.json", {
            "model": "frozen-model",
            "task_ids": [TASK],
            "decode_seed": decode_seed,
            "max_evals": 2,
            "seed_program_registry": {"sha256": "b" * 64},
        })

    def test_causal_winner_can_differ_from_absolute_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidates = root / "rollouts"
            controls = root / "controls"
            self._score(candidates, 0, 0.90)
            self._score(controls, 0, 0.80)
            self._score(candidates, 1, 0.95)
            self._score(controls, 1, 0.96)

            group = compute_task_group_paired(
                task_id=TASK,
                candidates=[self._candidate(0), self._candidate(1)],
                rollout_root=candidates,
                control_root=controls,
                base_score=0.85,
                ceiling=1.0,
                min_promotion_repeats=1,
            )

            self.assertEqual(group["best_k"], 0)
            self.assertEqual(group["absolute_best_k"], 1)
            self.assertEqual(group["program_best_k"], 1)
            self.assertEqual(group["program_best_score"], 0.95)
            self.assertAlmostEqual(group["best_causal_delta"], 0.10)
            self.assertTrue(group["improved"])
            self.assertGreater(group["rows"][0]["reward"], 0.0)
            self.assertLess(group["rows"][1]["reward"], 0.0)
            self.assertTrue(group["rows"][0]["pair_contract"]["passed"])

    def test_invoked_rejection_cannot_receive_positive_credit_or_advantage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rejected = {
                "checker": {
                    "mounts": 1, "invocations": 1, "completed": 1,
                    "usable_outputs": 0,
                    "semantic_successful_invocations": 0,
                    "semantic_rejections": 1,
                    "reject_reasons": {"bridge_stale": 1},
                }
            }
            active = {
                "checker": {
                    "mounts": 1, "invocations": 2, "completed": 2,
                    "usable_outputs": 2,
                    "semantic_successful_invocations": 2,
                    "semantic_rejections": 0,
                }
            }
            self._score(
                root / "rollouts", 0, 0.98, tool_audit=rejected
            )
            self._score(root / "controls", 0, 0.80)
            self._score(root / "rollouts", 1, 0.91, tool_audit=active)
            self._score(root / "controls", 1, 0.89)
            group = compute_task_group_paired(
                task_id=TASK,
                candidates=[
                    self._candidate(0, "checker"),
                    self._candidate(1, "checker"),
                ],
                rollout_root=root / "rollouts",
                control_root=root / "controls",
                base_score=0.80,
                ceiling=1.0,
            )
            false_positive, real_positive = group["rows"]
            self.assertAlmostEqual(false_positive["causal_delta"], 0.18)
            self.assertEqual(false_positive["component_credit_delta"], 0.0)
            self.assertEqual(
                false_positive["component_evidence"]["status"],
                "delivered_but_inactive",
            )
            self.assertEqual(
                false_positive["component_evidence"]["reject_reasons"],
                {"bridge_stale": 1},
            )
            self.assertLessEqual(false_positive["reward"], 0.0)
            self.assertLessEqual(false_positive["advantage"], 0.0)
            self.assertFalse(false_positive["promotion_eligible"])
            self.assertAlmostEqual(
                real_positive["component_credit_delta"], 0.02
            )
            self.assertTrue(real_positive["component_evidence"]["activated"])

    def test_component_promotion_requires_three_consistent_active_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            active = {
                "checker": {
                    "mounts": 1, "invocations": 1, "completed": 1,
                    "usable_outputs": 1,
                    "semantic_successful_invocations": 1,
                    "semantic_rejections": 0,
                }
            }
            for repeat, (candidate_score, control_score) in enumerate(
                ((0.91, 0.89), (0.93, 0.90), (0.92, 0.90))
            ):
                self._score(
                    root / "rollouts", 0, candidate_score,
                    repeat=repeat, tool_audit=active,
                )
                self._score(
                    root / "controls", 0, control_score, repeat=repeat
                )
            group = compute_task_group_paired(
                task_id=TASK,
                candidates=[self._candidate(0, "checker")],
                rollout_root=root / "rollouts",
                control_root=root / "controls",
                base_score=0.85,
                ceiling=1.0,
                expected_repeats=3,
            )
            row = group["rows"][0]
            self.assertTrue(
                row["component_evidence"]["activation_consistent"]
            )
            self.assertTrue(
                row["component_evidence"]["positive_credit_eligible"]
            )
            self.assertTrue(row["promotion_eligible"])
            self.assertTrue(row["promotion"]["positive_sign_consistent"])
            self.assertTrue(group["improved"])
            self.assertEqual(group["promotion_best_k"], 0)

    def test_repeat_directories_with_timestamped_runs_are_discovered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for repeat, (candidate_score, control_score) in enumerate(
                ((0.91, 0.89), (0.93, 0.90), (0.92, 0.90))
            ):
                self._score(
                    root / "rollouts", 0, candidate_score,
                    repeat=repeat, timestamp_layer=True,
                )
                self._score(
                    root / "controls", 0, control_score,
                    repeat=repeat, timestamp_layer=True,
                )

            group = compute_task_group_paired(
                task_id=TASK,
                candidates=[self._candidate(0)],
                rollout_root=root / "rollouts",
                control_root=root / "controls",
                base_score=0.85,
                ceiling=1.0,
                expected_repeats=3,
            )

            row = group["rows"][0]
            self.assertEqual(
                row["pair_contract"]["observed_candidate_repeats"], 3
            )
            self.assertEqual(
                row["pair_contract"]["observed_control_repeats"], 3
            )
            self.assertEqual(
                [sample["decode_seed"]
                 for sample in row["pair_contract"]["samples"]],
                [200000, 200008, 200016],
            )

    def test_component_credit_requires_activation_in_every_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            active = {
                "checker": {
                    "mounts": 1, "invocations": 1, "completed": 1,
                    "usable_outputs": 1,
                    "semantic_successful_invocations": 1,
                    "semantic_rejections": 0,
                }
            }
            inactive = {
                "checker": {
                    "mounts": 1, "invocations": 0, "completed": 0,
                    "usable_outputs": 0,
                    "semantic_successful_invocations": 0,
                    "semantic_rejections": 0,
                }
            }
            for repeat, (candidate_score, control_score) in enumerate(
                ((0.92, 0.89), (0.93, 0.90), (0.94, 0.90))
            ):
                self._score(
                    root / "rollouts", 0, candidate_score,
                    repeat=repeat,
                    tool_audit=active if repeat == 0 else inactive,
                )
                self._score(
                    root / "controls", 0, control_score, repeat=repeat
                )
            group = compute_task_group_paired(
                task_id=TASK,
                candidates=[self._candidate(0, "checker")],
                rollout_root=root / "rollouts",
                control_root=root / "controls",
                base_score=0.85,
                ceiling=1.0,
                expected_repeats=3,
            )
            row = group["rows"][0]
            evidence = row["component_evidence"]
            self.assertGreater(row["causal_delta"], 0.0)
            self.assertTrue(evidence["activated"])
            self.assertFalse(evidence["activation_consistent"])
            self.assertFalse(evidence["positive_credit_eligible"])
            self.assertEqual(
                evidence["status"],
                "activation_inconsistent_across_repeats",
            )
            self.assertEqual(evidence["inconsistent_components"], ["checker"])
            self.assertEqual(row["component_credit_delta"], 0.0)
            self.assertLessEqual(row["reward"], 0.0)
            self.assertLessEqual(row["advantage"], 0.0)
            self.assertFalse(row["promotion_eligible"])
            self.assertIn(
                "activation_inconsistent_across_repeats",
                row["promotion"]["blockers"],
            )
            self.assertFalse(group["improved"])

    def test_component_promotion_rejects_mixed_delta_signs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            active = {
                "checker": {
                    "mounts": 1, "invocations": 1, "completed": 1,
                    "usable_outputs": 1,
                    "semantic_successful_invocations": 1,
                    "semantic_rejections": 0,
                }
            }
            for repeat, (candidate_score, control_score) in enumerate(
                ((0.91, 0.89), (0.88, 0.90), (0.94, 0.90))
            ):
                self._score(
                    root / "rollouts", 0, candidate_score,
                    repeat=repeat, tool_audit=active,
                )
                self._score(
                    root / "controls", 0, control_score, repeat=repeat
                )
            group = compute_task_group_paired(
                task_id=TASK,
                candidates=[self._candidate(0, "checker")],
                rollout_root=root / "rollouts",
                control_root=root / "controls",
                base_score=0.85,
                ceiling=1.0,
                expected_repeats=3,
            )
            row = group["rows"][0]
            self.assertFalse(row["promotion_eligible"])
            self.assertIn(
                "paired_delta_sign_inconsistent", row["promotion"]["blockers"]
            )
            self.assertFalse(group["improved"])

    def test_equal_candidate_and_control_has_no_harness_credit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._score(root / "rollouts", 0, 0.95)
            self._score(root / "controls", 0, 0.95)
            group = compute_task_group_paired(
                task_id=TASK,
                candidates=[self._candidate(0)],
                rollout_root=root / "rollouts",
                control_root=root / "controls",
                base_score=0.85,
                ceiling=1.0,
            )
            self.assertEqual(group["rows"][0]["causal_delta"], 0.0)
            self.assertEqual(group["rows"][0]["advantage"], 0.0)
            self.assertFalse(group["improved"])

    def test_missing_control_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._score(root / "rollouts", 0, 0.95)
            with self.assertRaisesRegex(ValueError, "refusing unpaired"):
                compute_task_group_paired(
                    task_id=TASK,
                    candidates=[self._candidate(0)],
                    rollout_root=root / "rollouts",
                    control_root=root / "controls",
                    base_score=0.85,
                )

    def test_mismatched_decode_seed_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._score(root / "rollouts", 0, 0.95)
            self._score(root / "controls", 0, 0.90)
            result_path = (
                root / "controls" / TASK / "cand00" / "run0"
                / "results" / f"{TASK}.json"
            )
            result = json.loads(result_path.read_text())
            result["decode_seed"] = 999
            write_json(result_path, result)
            with self.assertRaisesRegex(ValueError, "decode-seed sets differ"):
                compute_task_group_paired(
                    task_id=TASK,
                    candidates=[self._candidate(0)],
                    rollout_root=root / "rollouts",
                    control_root=root / "controls",
                    base_score=0.85,
                )

    def test_repeated_pairs_average_effect_by_decode_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._score(root / "rollouts", 0, 0.90, repeat=0)
            self._score(root / "controls", 0, 0.80, repeat=0)
            self._score(root / "rollouts", 0, 0.70, repeat=1)
            self._score(root / "controls", 0, 0.80, repeat=1)
            group = compute_task_group_paired(
                task_id=TASK,
                candidates=[self._candidate(0)],
                rollout_root=root / "rollouts",
                control_root=root / "controls",
                base_score=0.70,
                ceiling=1.0,
                expected_repeats=2,
            )
            row = group["rows"][0]
            self.assertAlmostEqual(row["score"], 0.80)
            self.assertAlmostEqual(row["control_score"], 0.80)
            self.assertAlmostEqual(row["causal_delta"], 0.0)
            self.assertEqual(len(row["pair_contract"]["samples"]), 2)
            self.assertEqual(group["program_best_score"], 0.90)
            self.assertFalse(group["improved"])

    def test_candidate_h2_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._score(root / "rollouts", 0, 0.95)
            self._score(root / "controls", 0, 0.90)
            with self.assertRaisesRegex(
                ValueError, "candidate H2 is not the declared materialized proposal"
            ):
                compute_task_group_paired(
                    task_id=TASK,
                    candidates=[{
                        "k": 0,
                        "valid": True,
                        "package_sha256": "d" * 64,
                        "parent_compute_contract": dict(self.COMPUTE),
                        "candidate_compute_contract": dict(self.COMPUTE),
                    }],
                    rollout_root=root / "rollouts",
                    control_root=root / "controls",
                    base_score=0.85,
                )

    def test_missing_candidate_h2_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._score(root / "rollouts", 0, 0.95)
            self._score(root / "controls", 0, 0.90)
            with self.assertRaisesRegex(
                ValueError, "materialized candidate H2 hash missing"
            ):
                compute_task_group_paired(
                    task_id=TASK,
                    candidates=[{
                        "k": 0, "valid": True,
                        "parent_compute_contract": dict(self.COMPUTE),
                        "candidate_compute_contract": dict(self.COMPUTE),
                    }],
                    rollout_root=root / "rollouts",
                    control_root=root / "controls",
                    base_score=0.85,
                )

    def test_candidate_control_compute_signature_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = self._candidate(0)
            candidate["candidate_compute_contract"] = {
                **self.COMPUTE, "agent_max_iterations": 80,
                "sha256": "budget-more-compute",
            }
            with self.assertRaisesRegex(ValueError, "compute signatures differ"):
                compute_task_group_paired(
                    task_id=TASK,
                    candidates=[candidate],
                    rollout_root=root / "rollouts",
                    control_root=root / "controls",
                    base_score=0.85,
                )

    def test_missing_token_usage_fails_paired_compute_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._score(root / "rollouts", 0, 0.95)
            self._score(root / "controls", 0, 0.90)
            result_path = (
                root / "rollouts" / TASK / "cand00" / "run0"
                / "results" / f"{TASK}.json"
            )
            payload = json.loads(result_path.read_text())
            payload["ledger"]["llm_input_tokens"] = None
            result_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(
                ValueError, "llm_input_tokens missing under metered compute contract"
            ):
                compute_task_group_paired(
                    task_id=TASK,
                    candidates=[self._candidate(0)],
                    rollout_root=root / "rollouts",
                    control_root=root / "controls",
                    base_score=0.85,
                )

    def test_runtime_cap_mismatch_fails_paired_compute_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._score(root / "rollouts", 0, 0.95)
            self._score(root / "controls", 0, 0.90)
            result_path = (
                root / "controls" / TASK / "cand00" / "run0"
                / "results" / f"{TASK}.json"
            )
            payload = json.loads(result_path.read_text())
            payload["ledger"]["max_probe_calls"] = 31
            result_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "max_probe_calls"):
                compute_task_group_paired(
                    task_id=TASK,
                    candidates=[self._candidate(0)],
                    rollout_root=root / "rollouts",
                    control_root=root / "controls",
                    base_score=0.85,
                )

    def test_checkpoint_is_allowed_when_terminal_row_never_materialized(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "run0" / "checkpoints" / f"{TASK}.json", {
                "task_id": TASK,
                "best_score": 7.5,
            })

            self.assertEqual(load_rollout_score(root, TASK), 7.5)

    def test_terminal_scores_remain_best_across_multiple_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "run0" / "summary.json", {
                "task_id": TASK,
                "best_score": 6.0,
            })
            write_json(root / "run1" / "summary.json", [{
                "task_id": TASK,
                "best_score": 8.0,
            }])

            self.assertEqual(load_rollout_score(root, TASK), 8.0)

    def test_middleware_ineligible_score_is_never_rewarded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "run0" / "summary.json", [{
                "task_id": TASK,
                "best_score": 9.0,
                "score_eligible": False,
                "error": "MiddlewareParticipationError: hook not mounted",
            }])
            write_json(root / "run0" / "checkpoints" / f"{TASK}.json", {
                "task_id": TASK,
                "best_score": 9.0,
            })

            self.assertIsNone(load_rollout_score(root, TASK))


if __name__ == "__main__":
    unittest.main()
