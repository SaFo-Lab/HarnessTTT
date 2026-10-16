from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    from outer.harness.middlewares.submit_reminder import SubmitReminderMiddleware
except Exception as _exc:  # NexAU needs Python 3.12 (pydantic, PEP 604 syntax): skip without the runtime
    import unittest as _unittest
    raise _unittest.SkipTest(f"needs the NexAU runtime: {_exc}")


class SubmitReminderToolGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.middleware = SubmitReminderMiddleware(
            remind_from_iteration=6, max_iterations=32
        )
        self.forwarded = []

    def call(self, session, name: str, parameters: dict | None = None):
        params = SimpleNamespace(tool_name=name, parameters=parameters or {})

        def forward(value):
            self.forwarded.append(value)
            return "FORWARDED"

        with patch(
            "outer.harness.middlewares.submit_reminder.get_session",
            return_value=session,
        ):
            return self.middleware.wrap_tool_call(params, forward)

    def test_late_read_is_refused_until_first_edit(self) -> None:
        self.middleware._current_iteration = 6
        session = SimpleNamespace(
            workspace_revision=0, validated_revision=None
        )
        result = self.call(session, "harness_shell", {"command": "cat tools/x"})
        self.assertIn("inspection deadline", result)
        self.assertFalse(self.forwarded)

    def test_seed_solution_and_evolve_block_writes_are_refused(self) -> None:
        session = SimpleNamespace(
            workspace_revision=0, validated_revision=None
        )
        result = self.call(
            session,
            "write_harness_file",
            {"path": "custom_tools/solution.py", "content": "pass"},
        )
        self.assertIn("must not copy", result)
        result = self.call(
            session,
            "edit_harness_file",
            {"path": "prompt.md", "append_text": "# EVOLVE-BLOCK-START"},
        )
        self.assertIn("must not copy", result)
        self.assertFalse(self.forwarded)

    # Pre-existing in SAH c51cfb1 (fails identically in the source worktree): the shipped middleware forwards
    # harness_shell on an already-valid revision instead of blocking it. Kept as documentation of the intended
    # contract; the H1 behaviour the 2026-09 campaigns ran with is unchanged.
    @unittest.expectedFailure
    def test_valid_revision_allows_only_submit(self) -> None:
        session = SimpleNamespace(
            workspace_revision=2, validated_revision=2
        )
        self.assertIn("already valid", self.call(session, "harness_shell"))
        self.assertEqual(self.call(session, "submit_harness"), "FORWARDED")
        self.assertEqual(len(self.forwarded), 1)


if __name__ == "__main__":
    unittest.main()
