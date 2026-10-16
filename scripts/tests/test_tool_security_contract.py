from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inner.runtime.harness_sdk import ToolContext  # noqa: E402
from outer.safety.static_gates import check_tool_code  # noqa: E402


class _Ledger:
    probe_calls = 0
    max_probe_calls = 3

    def evaluator_budget_left(self): return 2
    def exhausted(self): return False


class _Task:
    def __init__(self, root: Path):
        self.task_dir = root


class _Session:
    def __init__(self, root: Path):
        self.current_program = "program"
        self.best_program = "best"
        self.best_score = 1.0
        self.ledger = _Ledger()
        self.task = _Task(root)

    def apply_edit(self, code): return code
    def probe(self, subsample=2000): raise AssertionError("not used")
    def evaluate(self): raise AssertionError("not used")
    def history_note(self, note): pass


class ToolSecurityContractTest(unittest.TestCase):
    def test_live_session_and_reflection_are_rejected_by_gate(self) -> None:
        attempts = (
            "def run(ctx, args):\n    return ctx._s.task.evaluator_path.read_text()",
            "def run(ctx, args):\n    return getattr(ctx, '_s')",
            "def run(ctx, args):\n    return ctx.__class__.__mro__",
            "import pandas as pd\ndef run(ctx, args):\n    return pd.read_csv('/tmp/x')",
        )
        for code in attempts:
            ok, errors = check_tool_code(code)
            self.assertFalse(ok, code)
            self.assertTrue(errors)

    def test_task_path_check_rejects_sibling_prefix_and_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task = root / "task"
            sibling = root / "task-secret"
            task.mkdir()
            sibling.mkdir()
            (task / "input.csv").write_text("x\n1\n")
            (task / "evaluator.py").write_text("secret")
            (sibling / "secret.csv").write_text("secret")
            ctx = ToolContext(_Session(task), root / "scratch")
            self.assertIn("x", ctx.read_input_sample("input.csv"))
            self.assertIn("ERROR", ctx.read_input_sample("evaluator.py"))
            self.assertIn("ERROR", ctx.read_input_sample("../task-secret/secret.csv"))
            self.assertFalse(hasattr(ctx, "_s"))


if __name__ == "__main__":
    unittest.main()
