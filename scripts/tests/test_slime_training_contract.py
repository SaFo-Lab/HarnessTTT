"""The trainer is slime; SAH owns the data contract and imports nothing else.

Migrating off the previous wrapper repository is only durable if the tree
cannot silently regain a dependency on it, so these tests pin the contract
rather than the prose.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
TRAINING = ROOT / "src" / "trainer"
PIPELINE = ROOT / "scripts" / "pipeline"


class NoExternalTrainerImportTest(unittest.TestCase):
    def test_training_modules_import_only_slime_and_sah(self):
        for path in sorted(TRAINING.glob("*.py")):
            tree = ast.parse(path.read_text())
            mods = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    mods.append(node.module)
                elif isinstance(node, ast.Import):
                    mods += [a.name for a in node.names]
            for m in mods:
                root = m.split(".")[0]
                self.assertNotIn(
                    root, {"Weave", "common", "actions"},
                    f"{path.name} imports {m!r} from another training repo")

    def test_no_sys_path_injection_into_another_repo(self):
        for path in sorted(TRAINING.glob("*.py")):
            src = path.read_text()
            self.assertNotIn("WEAVE_ROOT", src, f"{path.name} still resolves WEAVE_ROOT")


class ReplayContractTest(unittest.TestCase):
    """The rollout function slime calls must keep its documented signature."""

    def test_rollout_function_signature_and_field_use(self):
        src = (TRAINING / "slime_replay_rollout.py").read_text()
        tree = ast.parse(src)
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "generate_rollout")
        self.assertEqual([a.arg for a in fn.args.args],
                         ["args", "rollout_id", "data_buffer", "evaluation"])
        # the advantage the outer loop computed must be what slime trains on
        self.assertIn("sample.reward = advantage", src)
        self.assertIn('sample.metadata["raw_reward"] = raw_reward', src)
        self.assertIn("Sample.Status.COMPLETED", src)

    def test_launcher_points_slime_at_the_in_tree_rollout(self):
        sh = (PIPELINE / "train_worker.sh").read_text()
        self.assertIn(
            "--rollout-function-path trainer.slime_replay_rollout.generate_rollout", sh)
        self.assertIn("--advantage-estimator grpo", sh)
        self.assertIn("--disable-rewards-normalization", sh)
        self.assertIn("--use-lora", sh)
        self.assertIn("--train-backend fsdp", sh)
        self.assertIn('TRAINER="$SAH_ROOT/src/trainer"', sh)
        self.assertIn('"$TRAINER/direct_ray_driver.py"', sh)
        self.assertIn('"$TRAINER/bound_replay.py"', sh)
        self.assertIn('"$TRAINER/pad_replay.py"', sh)
        self.assertIn('"$TRAINER/merge_lora.sh"', sh)
        for forbidden in ("Weave_v2", "Weave.actions", "OpenClaw-RL", "clean_sft2"):
            self.assertNotIn(forbidden, sh)
        for name in ("direct_ray_driver.py", "bound_replay.py", "pad_replay.py", "merge_lora.sh",
                     "slime_replay_rollout.py", "qwen35_mask.py", "tfpatch/sitecustomize.py"):
            self.assertTrue((TRAINING / name).is_file(), name)
        # the slime patch the launcher's needles depend on ships with the tree and matches the lock
        import hashlib
        lock = json.loads((ROOT / "config/dependencies.lock.json").read_text())
        patch = ROOT / lock["slime"]["patch"]["path"]
        self.assertTrue(patch.is_file(), patch)
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), lock["slime"]["patch"]["sha256"])

    def test_slurm_templates_take_qos_from_env(self):
        for name in ("round.sbatch", "train.sbatch"):
            text = (PIPELINE / "templates" / name).read_text()
            self.assertIn("@@SBATCH_QOS@@", text)          # expands to "#SBATCH --qos=<HTT_SLURM_QOS>" or to a comment when unset
            self.assertIn("@@SBATCH_ACCOUNT@@", text)
            self.assertNotIn("Weave_v2", text)
            self.assertNotRegex(text, r"--qos=(normal|short|high|interactive)")
        # no site-specific account / partition / QOS is baked into the tree: env.sh only passes through what
        # config/local.env sets; preflight.sh and the runner enforce a *_low QOS only under HARNESSTTT_REQUIRE_LOW_QOS=1
        env = (ROOT / "config/env.sh").read_text()
        self.assertRegex(env, r'HTT_SLURM_QOS="\$\{HTT_SLURM_QOS:-\}"')
        self.assertRegex(env, r'HTT_SLURM_ACCOUNT="\$\{HTT_SLURM_ACCOUNT:-\}"')
        preflight = (ROOT / "scripts/setup/preflight.sh").read_text()
        self.assertIn('*_low', preflight)


class ConversionTest(unittest.TestCase):
    def test_batch_converts_to_trainable_replay_rows(self):
        import tempfile
        batch = [{
            "round": 0, "task_id": "t", "k": i,
            "proposal_group_id": "h1g_same_condition",
            "behavior_policy": "trained_phi", "training_eligible": True,
            "system": "sys", "user": "usr",
            "response": "submit", "trajectory": [
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "submit_harness", "arguments": "{}"}}]},
                {"role": "tool", "content": "SUBMITTED"},
            ],
            "reward": 0.5, "advantage": 0.5 if i else -0.5,
            "valid": True, "score": 0.9, "spec_hash": "sha256:x",
        } for i in range(2)]
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td) / "round000"
            rd.mkdir()
            (rd / "grpo_batch.jsonl").write_text(
                "\n".join(json.dumps(r) for r in batch) + "\n")
            out = Path(td) / "replay.jsonl"
            r = subprocess.run(
                [sys.executable, str(TRAINING / "grpo_to_replay.py"),
                 "--rounds", str(rd), "--out", str(out)],
                capture_output=True, text=True,
                env={"PYTHONPATH": os.pathsep.join(p for p in (str(ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p),
                     "PATH": "/usr/bin:/bin"})
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(sorted(row), ["messages", "metadata", "tools"])
            self.assertIn("advantage", row["metadata"])
            self.assertIn("reward", row["metadata"])
            # tool calls must be materialised as Qwen3.5 output XML, not structs
            for m in row["messages"]:
                self.assertNotIn("tool_calls", m)


if __name__ == "__main__":
    unittest.main()
