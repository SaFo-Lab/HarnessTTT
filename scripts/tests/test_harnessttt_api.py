"""Smoke tests for the thin ``harnessttt`` API: import, task loading, optimize() against the fake Slurm, run_harness argv."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAKE = ROOT / "scripts" / "tests" / "fake_slurm"
DATASET = ROOT / "dataset" / "self_adapt_harness"

from site_env import ensure_site_env  # noqa: E402

ensure_site_env()
import harnessttt  # noqa: E402  (conftest puts src/ on sys.path)


class ImportTest(unittest.TestCase):
    def test_public_surface(self):
        for name in ("optimize", "load_task", "run_harness", "run_harness_argv", "Task", "Result", "INITIAL_HARNESS", "examples"):
            self.assertTrue(hasattr(harnessttt, name), name)
        self.assertTrue((harnessttt.INITIAL_HARNESS / "agent.yaml").is_file())
        self.assertTrue((harnessttt.INITIAL_HARNESS / "system.md").is_file())
        self.assertEqual(harnessttt.ROOT, ROOT)


@unittest.skipUnless((DATASET / "MANIFEST_eft.json").is_file(), "dataset snapshot not present (git-ignored payload)")
class TaskTest(unittest.TestCase):
    def test_circle_packing_task(self):
        task = harnessttt.examples.circle_packing.init_task()
        self.assertEqual(task.id, "eft__math__circle_packing")
        self.assertIn("# EVOLVE-BLOCK-START", task.initial_program)
        self.assertIn("# EVOLVE-BLOCK-END", task.initial_program)
        self.assertTrue(task.description.strip())
        self.assertTrue(task.evaluator_path.is_file())
        self.assertEqual(task.direction, "maximize")
        self.assertEqual(task.metric, "sum_radii")
        self.assertAlmostEqual(task.native_metric(1.0), 2.635)
        self.assertEqual(harnessttt.load_task("cp26").id, task.id)


class OptimizeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="htt_api_"))
        os.environ["FAKE_SLURM_DIR"] = str(self.tmp / "slurm")
        os.environ["FAKE_STEP_SECONDS"] = "0.05"
        for key in ("FAKE_FAIL_ROUNDS", "FAKE_ZERO_ADV_ROUNDS", "FAKE_FAIL_TRAIN"):
            os.environ.pop(key, None)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def common(self, **over):
        base = dict(campaign="t", name="api_smoke", slurm_bin=str(FAKE), sah_root=str(ROOT), run_root=str(self.tmp / "runs"),
                    freeze=False, require_clean_tree=False, poll_seconds=0.1, heartbeat_minutes=1000, quiet=True)
        base.update(over)
        return base

    def test_other_seed_harness_is_rejected_before_any_submission(self):
        with self.assertRaises(NotImplementedError):
            harnessttt.optimize("hadamard", seed_harness=self.tmp, rounds=1, **self.common())
        self.assertFalse((self.tmp / "runs").exists())

    def test_two_different_base_models_are_rejected(self):
        with self.assertRaises(ValueError):
            harnessttt.optimize("hadamard", executor_lm="/a", proposer_lm="/b", rounds=1, **self.common())

    def test_dry_run_writes_the_round_1_script_only(self):
        res = harnessttt.optimize("hadamard", rounds=1, proposals_per_round=8, max_metric_calls=20, dry_run=True, **self.common())
        self.assertIsNotNone(res.dry_run_script)
        text = Path(res.dry_run_script).read_text()
        self.assertIn("SAH9B_K=8", text)
        self.assertIn("SAH9B_MAX_EVALS=20", text)
        self.assertIn("SAH9B_BASES_FILE= ", text)       # round 1 proposes from the built-in initial harness
        self.assertFalse((self.tmp / "slurm").exists(), "dry run must not submit")
        self.assertEqual(res.rounds_completed, 0)

    def test_one_round_against_fake_slurm_returns_the_retained_harness_and_score(self):
        res = harnessttt.optimize("hadamard", rounds=1, proposals_per_round=8, max_metric_calls=20, **self.common())
        self.assertTrue(res.done)
        self.assertEqual(res.rounds_completed, 1)
        self.assertEqual(res.task_id, "eft__math__hadamard_maximal_det")
        self.assertTrue((res.run_dir / "rounds" / "round001" / "round_summary.json").is_file())
        self.assertTrue(res.best_programs_json.is_file())
        self.assertIsNotNone(res.best_score)
        self.assertIsNotNone(res.best_metric)
        self.assertIsNotNone(res.best_harness_path)
        self.assertTrue(res.best_program and res.best_program.strip())
        out = res.write_best_program(self.tmp / "best.py")
        self.assertEqual(out.read_text(), res.best_program)
        progress = json.loads(res.progress_json.read_text())
        self.assertTrue(progress["done"])
        self.assertEqual(progress["k"], 8)
        self.assertEqual(progress["max_round"], 1)
        self.assertFalse((res.run_dir / "training" / "round001" / "model_state.json").exists(), "no training after the last round")


class RunHarnessArgvTest(unittest.TestCase):
    def test_argv_maps_to_run_baseline_flags(self):
        task = harnessttt.Task(id="eft__math__circle_packing", description="d", initial_program_path=Path("p.py"),
                               evaluator_path=Path("e.py"), task_dir=Path("."), domain="x")
        argv = harnessttt.run_harness_argv("/pkg", task, base_url="https://api.openai.com/v1", model="gpt-5.5",
                                           api_key="k", max_evals=20, out_dir="runs/transfer", eval_python="/usr/bin/python3")
        self.assertEqual(argv[:3], [sys.executable, "-m", "inner.cli.run_baseline"])
        for flag, value in (("--ids", "eft__math__circle_packing"), ("--base-url", "https://api.openai.com/v1"), ("--model", "gpt-5.5"),
                            ("--api-key", "k"), ("--max-evals", "20"), ("--eval-python", "/usr/bin/python3"), ("--out", str(Path("runs/transfer").resolve())),
                            ("--harness-dir", str(Path("/pkg").resolve()))):
            self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        self.assertIn("--require-trajectory", argv)
        self.assertNotIn("--harness-dir", harnessttt.run_harness_argv(None, task, base_url="http://localhost:8800/v1", model="m"))

    def test_hosted_apis_get_no_vllm_only_fields_and_local_vllm_keeps_them(self):
        task = harnessttt.Task(id="t", description="d", initial_program_path=Path("p.py"), evaluator_path=Path("e.py"),
                               task_dir=Path("."), domain="x")
        hosted = harnessttt.run_harness_argv(None, task, base_url="https://api.openai.com/v1", model="gpt-5.5")
        local = harnessttt.run_harness_argv(None, task, base_url="http://127.0.0.1:8800/v1", model="qwen3.5-9b")
        self.assertIn("--no-vllm-extras", hosted)
        self.assertNotIn("--no-vllm-extras", local)
        self.assertNotIn("--no-vllm-extras", harnessttt.run_harness_argv(None, task, base_url="https://api.openai.com/v1", model="m", vllm_extras=True))
        self.assertTrue(Path(hosted[hosted.index("--out") + 1]).is_absolute())


class ServedNameTest(unittest.TestCase):
    def test_revision_hash_directories_are_served_under_the_model_name(self):
        from harnessttt.api import _served_name
        self.assertEqual(_served_name("/models/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"), "qwen3.5-9b")
        self.assertEqual(_served_name("/models/Qwen3.5-9B"), "qwen3.5-9b")


class LoadTaskErrorTest(unittest.TestCase):
    def test_unknown_task_raises_value_error(self):
        with self.assertRaises(ValueError):
            harnessttt.load_task("no_such_task_xyz")


if __name__ == "__main__":
    unittest.main()
