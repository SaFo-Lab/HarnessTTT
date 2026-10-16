from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trainer.grpo_to_replay import (  # noqa: E402
    _h1_tool_schemas,
    _normalizer,
    _policy_trainable,
    _tool_schemas_for_row,
    convert_row,
)
from outer.proposing import proposer_io  # noqa: E402


class H1TrainingContractTest(unittest.TestCase):
    def test_contract_hash_binds_typed_middleware_runtime(self) -> None:
        baseline = proposer_io.proposer_contract_hash()
        original = Path.read_bytes
        for target in (
            "middleware_selftest.py", "harness_spec.py", "harness_sdk.py",
        ):
            def changed(path: Path, target_name: str = target) -> bytes:
                payload = original(path)
                if path.name == target_name:
                    return payload + b"\n# simulated semantic change\n"
                return payload

            with self.subTest(target=target), patch.object(
                Path, "read_bytes", changed
            ):
                modified = proposer_io.proposer_contract_hash()
            self.assertNotEqual(baseline, modified)

    def test_training_tools_are_exactly_the_inference_mounts(self) -> None:
        names = [row["function"]["name"] for row in _h1_tool_schemas()]
        self.assertEqual(names, [
            "harness_shell", "write_harness_file", "edit_harness_file",
            "delete_harness_file", "upsert_skill", "upsert_tool", "upsert_middleware",
            "repair_component", "validate_harness", "submit_harness",
        ])
        self.assertNotIn("submit_spec", names)
        self.assertNotIn("validate_spec", names)

    def test_replay_uses_recorded_complete_portfolio_surface(self) -> None:
        mounted = _h1_tool_schemas()
        full_surface = [
            "harness_shell", "write_harness_file", "edit_harness_file",
            "delete_harness_file", "upsert_skill", "upsert_tool", "upsert_middleware",
            "repair_component", "validate_harness", "submit_harness",
        ]
        row = {
            "proposal_condition": {
                "resolved_h1_config": {"tools": full_surface}
            },
            "trajectory": [{
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "upsert_tool"}],
            }],
        }
        selected = _tool_schemas_for_row(row, mounted)
        self.assertEqual(
            [tool["function"]["name"] for tool in selected],
            full_surface,
        )

        row["proposal_condition"]["resolved_h1_config"]["tools"] = (
            full_surface[1:]
        )
        row["trajectory"][0]["content"][0]["name"] = "harness_shell"
        with self.assertRaisesRegex(ValueError, "outside its recorded"):
            _tool_schemas_for_row(row, mounted)

    def test_candidates_share_group_key_but_keep_unique_sample_ids(self) -> None:
        def row(k: int) -> dict:
            return {
                "round": 4,
                "task_id": "task-a",
                "proposal_group_id": "h1g_same_condition",
                "behavior_policy": "trained_phi",
                "training_eligible": True,
                "k": k,
                "advantage": 1.0 if k == 0 else -1.0,
                "reward": 0.5 if k == 0 else -0.5,
                "valid": True,
                "spec_hash": f"hash-{k}",
                "system": "system",
                "user": "user",
                "response": "response",
                "trajectory": [
                    {"role": "user", "content": "inspect"},
                    {"role": "assistant", "content": f"candidate {k}"},
                ],
            }

        normalize = lambda trajectory: trajectory
        first = convert_row(row(0), [], normalize)
        second = convert_row(row(1), [], normalize)
        self.assertEqual(first["metadata"]["seed"], "h1g_same_condition")
        self.assertEqual(first["metadata"]["seed"], second["metadata"]["seed"])
        self.assertNotEqual(
            first["metadata"]["sample_id"], second["metadata"]["sample_id"]
        )
        self.assertEqual(first["messages"][-1], {"role": "user", "content": "ok"})

    def test_frozen_and_nonexchangeable_rows_are_evaluation_only(self) -> None:
        self.assertTrue(_policy_trainable({
            "behavior_policy": "trained_phi", "training_eligible": True,
        }))
        self.assertFalse(_policy_trainable({
            "behavior_policy": "frozen_base", "training_eligible": True,
        }))
        self.assertFalse(_policy_trainable({
            "behavior_policy": "trained_phi", "training_eligible": False,
        }))
        self.assertFalse(_policy_trainable({
            "behavior_policy": "trained_phi", "training_eligible": True,
            "infrastructure_failure": True,
        }))
        self.assertFalse(_policy_trainable({
            "behavior_policy": "trained_phi", "training_eligible": True,
            "training_block_reason": "context_exhausted",
        }))

    def test_nexau_content_blocks_become_qwen_text_and_tool_xml(self) -> None:
        row = {
            "round": 1,
            "task_id": "task-a",
            "proposal_group_id": "h1g_content_blocks",
            "behavior_policy": "trained_phi",
            "training_eligible": True,
            "k": 0,
            "advantage": 1.0,
            "reward": 0.5,
            "valid": True,
            "spec_hash": "hash-0",
            "system": "fallback-system",
            "user": "fallback-user",
            "response": "fallback-response",
            "trajectory": [
                {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                {"role": "user", "content": [{"type": "text", "text": "inspect"}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "I will inspect."},
                    {"type": "tool_use", "id": "call-1", "name": "harness_shell",
                     "input": {"command": "cat agent.yaml"}},
                ]},
                {"role": "tool", "content": [
                    {"type": "tool_result", "tool_use_id": "call-1",
                     "content": "agent contents", "is_error": False},
                ]},
            ],
        }
        replay = convert_row(row, _h1_tool_schemas(), _normalizer())
        self.assertTrue(all(isinstance(m["content"], str) for m in replay["messages"]))
        self.assertIn("<function=harness_shell>", replay["messages"][2]["content"])
        self.assertIn("<parameter=command>", replay["messages"][2]["content"])
        self.assertEqual(replay["messages"][3]["content"], "agent contents")
        self.assertNotIn("tool_calls", replay["messages"][2])


if __name__ == "__main__":
    unittest.main()
