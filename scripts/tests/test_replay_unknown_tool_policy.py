"""SAH_REPLAY_UNKNOWN_TOOL_CALLS controls how hallucinated H1 tool names are replayed."""
import os
import unittest
from unittest import mock

from trainer.grpo_to_replay import _tool_schemas_for_row


def _row():
    return {
        "proposal_condition": {"resolved_h1_config": {"tools": ["harness_shell", "submit_harness"]}},
        "trajectory": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "cat", "input": {"command": "cat agent.yaml"}}]},
            {"role": "tool", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Tool 'cat' not found", "is_error": True}]},
        ],
    }


MOUNTED = [{"type": "function", "function": {"name": "harness_shell"}}, {"type": "function", "function": {"name": "submit_harness"}}]


class UnknownToolPolicyTest(unittest.TestCase):
    def test_default_is_fail_closed(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SAH_REPLAY_UNKNOWN_TOOL_CALLS", None)
            with self.assertRaises(ValueError):
                _tool_schemas_for_row(_row(), MOUNTED)

    def test_keep_retains_row_with_recorded_surface(self):
        with mock.patch.dict(os.environ, {"SAH_REPLAY_UNKNOWN_TOOL_CALLS": "keep"}):
            tools = _tool_schemas_for_row(_row(), MOUNTED)
        self.assertEqual([t["function"]["name"] for t in tools], ["harness_shell", "submit_harness"])


if __name__ == "__main__":
    unittest.main()
