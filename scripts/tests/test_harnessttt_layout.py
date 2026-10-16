"""HarnessTTT tree contract: pinned dependencies, self-contained paths, consistent pipeline templates."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PIPELINE = ROOT / "scripts" / "pipeline"
LOCK = json.loads((ROOT / "config" / "dependencies.lock.json").read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LockTest(unittest.TestCase):
    def test_slime_is_the_official_repo_pinned_with_a_checked_patch(self):
        self.assertEqual(LOCK["slime"]["url"], "https://github.com/THUDM/slime.git")
        self.assertRegex(LOCK["slime"]["commit"], r"^[0-9a-f]{40}$")
        patch = ROOT / LOCK["slime"]["patch"]["path"]
        self.assertEqual(sha256(patch), LOCK["slime"]["patch"]["sha256"])
        self.assertEqual(patch.read_text().count("\ndiff --git ") + 1,
                         LOCK["slime"]["patch"]["files_modified"] + LOCK["slime"]["patch"]["files_added"])
        self.assertGreaterEqual(len(LOCK["slime"]["required_needles"]), 7)

    def test_nexau_patch_and_config_hashes_match_the_lock(self):
        self.assertEqual(sha256(ROOT / LOCK["nexau"]["patch"]["path"]), LOCK["nexau"]["patch"]["sha256"])
        for name, digest in LOCK["config"].items():
            self.assertEqual(sha256(ROOT / "config" / name), digest, name)


class SelfContainedSourceTest(unittest.TestCase):
    def test_no_absolute_site_paths_or_old_package_names_in_src(self):
        for path in SRC.rglob("*.py"):
            text = path.read_text()
            self.assertNotIn("/lustre/", text, path)
            self.assertNotRegex(text, r"^\s*(from|import) training\b", path)

    def test_repo_relative_defaults_point_into_this_tree(self):
        eft = (SRC / "inner/tasks/eft_task.py").read_text()
        self.assertIn('parents[3] / "dataset" / "self_adapt_harness"', eft)
        reg = (SRC / "outer/rounds/task_text_registry.py").read_text()
        self.assertIn('parents[3] / "config" / "task_text_registry.json"', reg)
        outer = (SRC / "outer/rounds/outer_round.py").read_text()
        self.assertIn('os.environ.get("SAH_FINCH_TARGETS")', outer)
        self.assertIn('"config" / "finch_targets.json"', outer)
        self.assertIn('REPO / "config" / "baseline_h2_20ev.json"', outer)
        self.assertIn("eft__math__hadamard_maximal_det", json.loads((ROOT / "config/baseline_h2_20ev.json").read_text())["baseline"])
        for name in ("finch_targets.json", "task_text_registry.json", "baseline_h2_20ev.json", "env.sh", "local.env.example"):
            self.assertTrue((ROOT / "config" / name).is_file(), name)


class PipelineTemplateTest(unittest.TestCase):
    def test_every_template_placeholder_is_filled_by_chain(self):
        chain = (PIPELINE / "chain.sh").read_text()
        filled = set(re.findall(r"@@([A-Z0-9_]+)@@", chain))
        for name in ("round.sbatch", "train.sbatch"):
            used = set(re.findall(r"@@([A-Z0-9_]+)@@", (PIPELINE / "templates" / name).read_text()))
            self.assertLessEqual(used, filled, f"{name}: unfilled {sorted(used - filled)}")

    def test_pipeline_scripts_reference_files_that_exist(self):
        worker = (PIPELINE / "worker_round.sh").read_text()
        for name in ("vllm_protocol.py", "audit_trajectories.py", "build_bases.py"):
            self.assertIn(name, worker)
            self.assertTrue((PIPELINE / name).is_file(), name)
        for module in ("outer.rounds.outer_round", "outer.reward.trajectory_budget", "inner.cli.run_baseline"):
            self.assertIn(module, worker)
            self.assertTrue((SRC / (module.replace(".", "/") + ".py")).is_file(), module)
        self.assertIn('SAH_OVER_CAP_AS_INVALID="${SAH_OVER_CAP_AS_INVALID:-1}"', worker)
        # every variable the worker reads under `set -u` must be assigned at the top (a comment appended to the header line once
        # swallowed K= and MAX_EVALS=, killing the job seconds after start)
        for assign in ('RUN_DIR="$(cd "${SAH9B_RUN_DIR:?}" && pwd -P)"', 'TASKS="${SAH9B_TASKS:?}"', 'K="${SAH9B_K:-4}"',
                       'MAX_EVALS="${SAH9B_MAX_EVALS:-5}"', 'N_GPUS="${SLURM_GPUS_ON_NODE:?no GPU allocation}"',
                       'PROPOSE_PAR="${SAH9B_PROPOSE_PAR:-8}"', 'ROLLOUT_PAR="${SAH9B_ROLLOUT_PAR:-8}"', 'ROUND="${SAH9B_ROUND:-1}"'):
            self.assertRegex(worker, r"(?m)^(?:[^#\n]*; )?" + re.escape(assign), assign)
        self.assertNotIn("experiments/", worker)
        self.assertIn("src/trainer/grpo_to_replay.py", (PIPELINE / "chain.sh").read_text())

    def test_shell_scripts_parse(self):
        for path in list(PIPELINE.glob("*.sh")) + list((PIPELINE / "templates").glob("*.sbatch")) + \
                list((ROOT / "scripts/setup").glob("*.sh")) + [ROOT / "config/env.sh", SRC / "trainer/merge_lora.sh"]:
            r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{path}: {r.stderr}")


if __name__ == "__main__":
    unittest.main()
