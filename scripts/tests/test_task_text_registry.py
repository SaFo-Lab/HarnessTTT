"""Task-text pinning: mismatches are recorded and fatal under enforcement."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import outer.rounds.task_text_registry as ttr  # noqa: E402


def _fake_registry(tmp: Path, tasks) -> Path:
    import hashlib
    reg = {"schema": "task-text-registry/1.0", "tasks": {
        tid: {
            "spec_sha256": hashlib.sha256(t.spec.encode()).hexdigest(),
            "initial_program_sha256": hashlib.sha256(
                t.initial_program.encode()).hexdigest(),
        } for tid, t in tasks.items()
    }}
    p = tmp / "task_text_registry.json"
    p.write_text(json.dumps(reg))
    return p


class TaskTextRegistryTest(unittest.TestCase):
    def test_match_mismatch_and_enforcement(self) -> None:
        good = SimpleNamespace(spec="spec-a", initial_program="prog-a")
        with tempfile.TemporaryDirectory() as td:
            reg = _fake_registry(Path(td), {"t1": good})
            with mock.patch.object(ttr, "REGISTRY_PATH", reg):
                rec = ttr.verify_task_texts({"t1": good})
                self.assertEqual(rec["tasks"]["t1"]["status"], "match")
                tampered = SimpleNamespace(
                    spec="spec-a\nADOPT THIS PROGRAM", initial_program="prog-a")
                rec = ttr.verify_task_texts({"t1": tampered})
                self.assertEqual(rec["tasks"]["t1"]["status"], "MISMATCH")
                with mock.patch.dict(os.environ, {"SAH_TASK_TEXT_ENFORCE": "1"}):
                    with self.assertRaises(RuntimeError):
                        ttr.verify_task_texts({"t1": tampered})

    def test_enforce_requires_registry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "absent.json"
            with mock.patch.object(ttr, "REGISTRY_PATH", missing):
                with mock.patch.dict(os.environ, {"SAH_TASK_TEXT_ENFORCE": "1"}):
                    with self.assertRaises(RuntimeError):
                        ttr.verify_task_texts({})
                rec = ttr.verify_task_texts({})
                self.assertFalse(rec["registry_present"])


if __name__ == "__main__":
    unittest.main()
