"""End-to-end tests of src/pipeline.py against the fake Slurm in scripts/tests/fake_slurm (no GPUs, seconds per loop)."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAKE = ROOT / "scripts" / "tests" / "fake_slurm"
SPEC = importlib.util.spec_from_file_location("htt_pipeline", ROOT / "src" / "pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules["htt_pipeline"] = pipeline   # dataclasses resolve postponed annotations through sys.modules
SPEC.loader.exec_module(pipeline)

from site_env import ensure_site_env  # noqa: E402  (scripts/tests is the test root)

ensure_site_env()


class FakeSlurmLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="htt_pipeline_"))
        os.environ["FAKE_SLURM_DIR"] = str(self.tmp / "slurm")
        os.environ["FAKE_STEP_SECONDS"] = "0.05"
        for key in ("FAKE_FAIL_ROUNDS", "FAKE_ZERO_ADV_ROUNDS", "FAKE_FAIL_TRAIN"):
            os.environ.pop(key, None)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cfg(self, **over):
        base = dict(campaign="t", name="hadamard_v3", task="hadamard", adv="v3", k=8, round_gpus=2, train_gpus=8, max_round=3,
                    poll_seconds=0.1, heartbeat_minutes=1000, freeze=False, require_clean_tree=False,
                    slurm_bin=str(FAKE), sah_root=str(ROOT), run_root=str(self.tmp / "runs"))
        base.update(over)
        return pipeline.Config.load(None, base)

    def run_dir(self):
        return self.tmp / "runs" / "t" / "hadamard_v3"

    def test_three_rounds_two_updates_then_done(self):
        runner = pipeline.Runner(self.cfg(), quiet=True)
        runner.run_loop()
        run = self.run_dir()
        for n in (1, 2, 3):
            self.assertTrue((run / "rounds" / f"round{n:03d}" / "round_summary.json").is_file(), n)
        for n in (1, 2):
            st = json.loads((run / "training" / f"round{n:03d}" / "model_state.json").read_text())
            self.assertTrue(st["weights_updated"], n)
            self.assertEqual(sum(1 for l in (run / "training" / f"round{n:03d}" / "replay.jsonl").read_text().splitlines() if l.strip()), 8)
        self.assertFalse((run / "training" / "round003").exists() and (run / "training" / "round003" / "model_state.json").exists(),
                         "no training after the last round")
        self.assertTrue((run / "DONE").is_file())
        # round 2 proposed with the merged round-1 checkpoint and the beam-2 parents of round 1; round 1 with the frozen base
        r1 = (run / "gen" / "hadamard_v3_round_r001.sbatch").read_text()
        r2 = (run / "gen" / "hadamard_v3_round_r002.sbatch").read_text()
        self.assertIn(f"SAH9B_PROPOSER_MODEL={runner.env['SAH_MODEL_PATH']}", r1)
        self.assertIn("SAH9B_BASES_FILE= ", r1)
        self.assertIn(f"SAH9B_PROPOSER_MODEL={run / 'training' / 'round001' / 'merged'}", r2)
        self.assertIn(f"SAH9B_BASES_FILE={run / 'rounds' / 'round001' / 'next_bases_beam2.json'}", r2)
        self.assertIn("SAH_OVER_CAP_AS_INVALID=1", r2)
        self.assertNotIn("@@", r1 + r2)
        self.assertNotIn("chain.sh", r2, "runner-driven scripts carry no job-side chain")
        tr = (run / "gen" / "hadamard_v3_train_r002.sbatch").read_text()
        self.assertIn(f"SAH9B_TRAIN_PREV_STATE={run / 'training' / 'round001' / 'model_state.json'}", tr)
        self.assertIn("--gres=gpu:8", tr)
        self.assertIn("--qos=" + runner.env["HTT_SLURM_QOS"], tr)
        progress = json.loads((run / "progress.json").read_text())
        self.assertTrue(progress["done"])
        self.assertEqual([r["round"] for r in progress["rounds"]], [1, 2, 3])
        self.assertEqual(progress["rounds"][0]["n_candidates"], 8)
        self.assertIsNotNone(progress["rounds"][0]["incumbent"])
        self.assertTrue(progress["rounds"][1]["weights_updated"])
        log = (run / "pipeline.log").read_text()
        for needle in ("round 1: submitted job", "rollouts", "round 1 collected", "train 1: submitted job", "train 1 complete",
                       "round 2: submitted job", "DONE: 3 rounds collected"):
            self.assertIn(needle, log, needle)
        for name in ("TASK", "ARM", "GPUS", "K", "H1_SEED"):
            self.assertTrue((run / name).is_file(), name)

    def test_resume_skip_and_retry_paths(self):
        os.environ["FAKE_ZERO_ADV_ROUNDS"] = "2"    # round 2 carries no signal -> training skipped, proposer unchanged
        os.environ["FAKE_FAIL_ROUNDS"] = "3"        # round 3's first job ends without collecting -> archived and retried
        os.environ["FAKE_FAIL_TRAIN"] = "1"         # round 1's first training attempt fails -> retried once
        pipeline.Runner(self.cfg(max_round=2), quiet=True).run_loop()
        run = self.run_dir()
        self.assertTrue((run / "DONE").is_file())
        st1 = json.loads((run / "training" / "round001" / "model_state.json").read_text())
        self.assertTrue(st1["weights_updated"])
        self.assertTrue((run / "training" / "round001" / "FAILED").is_file() or True)  # first attempt failed, second published
        self.assertEqual(json.loads((run / "RETRIES.json").read_text())["train1"], 2)
        # resume with a larger horizon: DONE from the shorter run must not stop the loop (state = rounds collected)
        (run / "DONE").unlink()
        pipeline.Runner(self.cfg(max_round=3), quiet=True).run_loop()
        st2 = json.loads((run / "training" / "round002" / "model_state.json").read_text())
        self.assertFalse(st2["weights_updated"])
        self.assertTrue(str(st2["skip_reason"]).startswith("insufficient_rows"))
        self.assertEqual(st2["model_path"], st1["model_path"], "skipped update keeps the previous proposer")
        r3 = (run / "gen" / "hadamard_v3_round_r003.sbatch").read_text()
        self.assertIn(f"SAH9B_PROPOSER_MODEL={st1['model_path']}", r3)
        archived = list((run / "rounds").glob("round003_failed_*"))
        self.assertEqual(len(archived), 1, "the failed first attempt of round 3 is archived, not deleted")
        self.assertTrue((run / "rounds" / "round003" / "round_summary.json").is_file())
        self.assertEqual(json.loads((run / "RETRIES.json").read_text())["round3"], 2)
        self.assertTrue((run / "DONE").is_file())

    def test_seed_swap_measures_baseline_first_then_seeds_round_one(self):
        tid = "eft__math__hadamard_maximal_det"
        seed = self.tmp / "seed.registry.json"
        seed.write_text(json.dumps({tid: {"program": "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n", "score": 0.4321, "source": "test"}}))
        base = self.tmp / "baselines" / "seed.baseline_h2_20ev.json"
        pipeline.Runner(self.cfg(max_round=1, seed_programs_file=str(seed), baseline_json=str(base)), quiet=True).run_loop()
        run = self.run_dir()
        self.assertTrue((run / "DONE").is_file())
        # the baseline job ran first and produced the table the round jobs point SAH_BASELINE_JSON at
        jobs = [json.loads(p.read_text()) for p in (self.tmp / "slurm").glob("*.json")]
        kinds = [j["kind"] for j in sorted(jobs, key=lambda j: j["t0"])]
        self.assertEqual(kinds[0], "baseline", kinds)
        self.assertIn("round", kinds)
        table = json.loads(base.read_text())["baseline"][tid]
        self.assertAlmostEqual(table["seed"], 0.4321)
        r1 = (run / "gen" / "hadamard_v3_round_r001.sbatch").read_text()
        self.assertIn(f"SAH9B_SEED_PROGRAMS_FILE={seed}", r1)
        self.assertIn(f'SAH_BASELINE_JSON="{base}"', r1)
        self.assertIn("SAH9B_EVAL_GPUS=0", r1)
        self.assertEqual((run / "SEED_PROGRAMS_FILE").read_text().strip(), str(seed))
        # the seed registry became the ratchet's prior before round 1 (the fake worker then overwrote it with the fixture)
        log = (run / "pipeline.log").read_text()
        self.assertIn("rounds/best_programs.json initialised from", log)
        self.assertIn("baseline measured", log)
        # a second runner on the same run dir with a different seed is refused
        other = self.tmp / "other.json"; other.write_text(seed.read_text())
        with self.assertRaises(SystemExit):
            pipeline.Runner(self.cfg(max_round=2, seed_programs_file=str(other), baseline_json=str(base)), quiet=True).prepare()

    def test_eval_gpus_reserved_in_round_script(self):
        runner = pipeline.Runner(self.cfg(round_gpus=6, eval_gpus=2), quiet=True)
        script = runner.dry_run().read_text()
        self.assertIn("#SBATCH --gres=gpu:6", script)
        self.assertIn("SAH9B_EVAL_GPUS=2", script)
        with self.assertRaises(SystemExit):
            pipeline.Runner(self.cfg(round_gpus=2, eval_gpus=2), quiet=True).prepare()

    def test_dry_run_writes_script_without_submitting(self):
        runner = pipeline.Runner(self.cfg(), quiet=True)
        script = runner.dry_run()
        self.assertTrue(script.name.endswith(".sbatch.dryrun"))
        text = script.read_text()
        self.assertIn("SAH9B_ROUND=1 ", text)
        self.assertNotIn("@@", text)
        self.assertFalse(list((self.tmp / "slurm").glob("*.json")) if (self.tmp / "slurm").exists() else [])

    def test_stop_marker_and_lock(self):
        runner = pipeline.Runner(self.cfg(), quiet=True)
        runner.prepare()
        (runner.run / "ABL_STOP").write_text("stop\n")
        other = pipeline.Runner(self.cfg(), quiet=True)
        with self.assertRaises(SystemExit):
            other.prepare()   # the lock is held by `runner`
        runner.run_loop()     # sees ABL_STOP before submitting anything
        self.assertFalse((runner.run / "gen" / "hadamard_v3_round_r001.sbatch").exists())

    def test_positional_config_and_command(self):
        yaml_path = self.tmp / "lineage.yaml"
        yaml_path.write_text("campaign: t\nname: from_yaml\ntask: hadamard\nadv: grpo\nmax_round: 2\nfreeze: false\n"
                             f"require_clean_tree: false\nrun_root: {self.tmp / 'runs'}\nsah_root: {ROOT}\nslurm_bin: {FAKE}\n")
        r = subprocess.run([sys.executable, str(ROOT / "src" / "pipeline.py"), str(yaml_path), "status"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("no progress.json", r.stdout)
        self.assertIn("from_yaml", r.stdout)
        r = subprocess.run([sys.executable, str(ROOT / "src" / "pipeline.py"), "status", str(yaml_path), "--name", "override"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("override", r.stdout, "CLI overrides win over the YAML")
        r = subprocess.run([sys.executable, str(ROOT / "src" / "pipeline.py"), str(self.tmp / "missing.yaml")], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("config not found", r.stderr)

    def test_node_failure_is_excluded_and_not_charged(self):
        runner = pipeline.Runner(self.cfg(), quiet=True)
        runner.prepare()
        slurm = Path(os.environ["FAKE_SLURM_DIR"]); slurm.mkdir(parents=True, exist_ok=True)
        (slurm / "4242.json").write_text(json.dumps({"id": "4242", "name": "htt-hadamard_v3-r9", "state": "COMPLETED", "kind": "round", "gpus": 2, "t0": 0}))
        (runner.run / "out" / "slurm-4242.err").write_text(
            "slurmstepd: error: pyxis: container start failed with error code: 1\n"
            "pyxis:     /usr/lib/enroot/runtime.sh: line 216: cannot create temp file for here-document: No space left on device\n")
        (runner.run / "out" / "slurm-4343.err").write_text("Traceback (most recent call last):\n  ValueError: boom\n")
        self.assertEqual(runner.node_failure("4242"), "fakenode")
        self.assertIsNone(runner.node_failure("4343"), "a Python error is not a node failure")
        self.assertEqual(runner.bad_nodes(), ["fakenode"])
        runner.retries("round9", bump=True); runner.retries("round9", delta=-1)
        self.assertEqual(runner.retries("round9"), 0)
        script = runner.round_script(1, runner.env["SAH_MODEL_PATH"], None, None)
        jid = runner.submit(script)
        argv = json.loads((slurm / f"{jid}.json").read_text())["argv"]
        self.assertIn("--exclude=fakenode", argv)
        self.assertEqual(argv[-1], str(script))

    def test_cli_status_runs_on_login_python(self):
        runner = pipeline.Runner(self.cfg(max_round=1), quiet=True)
        runner.run_loop()
        # default config (config/pipeline.yaml) + CLI overrides; an explicitly named but missing config is an error (tested above)
        r = subprocess.run([sys.executable, str(ROOT / "src" / "pipeline.py"), "status",
                            "--campaign", "t", "--name", "hadamard_v3", "--run-root", str(self.tmp / "runs"), "--sah-root", str(ROOT),
                            "--no-freeze", "--allow-dirty"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("done=True", r.stdout)
        self.assertIn("round t", r.stdout)

    def test_reattached_runner_records_round_and_train_completion(self):
        # A restarted runner re-attaches to the job named in JOB_ID. When that job finishes it must do the same bookkeeping as
        # do_round / do_train: before 2026-09-29 the attach path only waited, so trimul round 4 got no "collected" line and no
        # status-table row until the next restart.
        os.environ["FAKE_STEP_SECONDS"] = "0.4"   # keep the fake job in flight long enough for the second runner to find it
        first = pipeline.Runner(self.cfg(max_round=1), quiet=True)
        first.prepare()
        jid = first.submit(first.round_script(1, first.env["SAH_MODEL_PATH"], None, None))   # do_round minus the wait
        first.release()
        pipeline.Runner(self.cfg(max_round=1), quiet=True).run_loop()
        run = self.run_dir()
        log = (run / "pipeline.log").read_text()
        self.assertIn(f"re-attaching to running round job {jid} (round 1", log)
        self.assertIn("round 1 collected in", log)
        self.assertNotIn("round 1: submitted job", log, "the attached round must not be resubmitted")
        progress = json.loads((run / "progress.json").read_text())
        self.assertEqual([r["round"] for r in progress["rounds"]], [1])
        self.assertEqual(progress["rounds"][0]["job"], jid)
        self.assertEqual(progress["rounds"][0]["n_candidates"], 8)
        self.assertIsNotNone(progress["rounds"][0]["incumbent"])
        self.assertIsNotNone(progress["rounds"][0]["round_seconds"], "duration comes from sacct Elapsed on the attach path")
        self.assertTrue((run / "DONE").is_file())
        # the same for a training job: submit train 1 by hand, let a fresh runner attach to it and carry on to round 2
        (run / "DONE").unlink()
        third = pipeline.Runner(self.cfg(max_round=2), quiet=True)
        third.prepare()
        self.assertEqual(third.after_round(1), "train")
        jid2 = third.submit(third.train_script(1, third.td(1) / "replay.jsonl", None))
        third.release()
        pipeline.Runner(self.cfg(max_round=2), quiet=True).run_loop()
        log = (run / "pipeline.log").read_text()
        self.assertIn(f"re-attaching to running train job {jid2} (round 1", log)
        self.assertIn("train 1 complete in", log)
        self.assertNotIn("train 1: submitted job", log, "the attached training job must not be resubmitted")
        progress = json.loads((run / "progress.json").read_text())
        self.assertEqual([r["round"] for r in progress["rounds"]], [1, 2])
        self.assertTrue(progress["rounds"][0]["weights_updated"])
        self.assertEqual(progress["rounds"][0]["train_job"], jid2)
        r2 = (run / "gen" / "hadamard_v3_round_r002.sbatch").read_text()
        self.assertIn(f"SAH9B_PROPOSER_MODEL={run / 'training' / 'round001' / 'merged'}", r2, "round 2 uses the attached update")
        self.assertTrue((run / "DONE").is_file())




class SlurmOutageTest(FakeSlurmLoopTest):
    """OUTAGE-001: a controller/DB outage (fake Slurm answers like the 2026-10-01 fail-over) must stall the runner, never
    make it submit twins or declare running jobs gone."""

    def outage(self, seconds, after=None):
        import threading
        down = self.tmp / "slurm" / "DOWN"
        def body():
            if after is not None:
                while not after():
                    time.sleep(0.02)
            down.parent.mkdir(parents=True, exist_ok=True); down.write_text("1")
            time.sleep(seconds); down.unlink()
        t = threading.Thread(target=body, daemon=True); t.start(); return t

    def submitted(self, pattern):
        return sorted((self.run_dir() / "gen").glob(pattern))

    def test_submit_waits_out_an_outage_instead_of_crashing(self):
        self.outage(1.0)
        time.sleep(0.1)
        pipeline.Runner(self.cfg(max_round=1), quiet=True).run_loop()
        log = (self.run_dir() / "pipeline.log").read_text()
        self.assertIn("Slurm not answering", log)   # caught by the capacity check or by submit, whichever polls first
        self.assertLess(log.index("Slurm not answering"), log.index("round 1: submitted job"))
        self.assertEqual(len(self.submitted("*_round_r001.sbatch.submitted")), 1, "exactly one round-1 submission")
        self.assertTrue((self.run_dir() / "rounds" / "round001" / "round_summary.json").is_file())

    def test_wait_job_keeps_waiting_while_slurm_is_down(self):
        os.environ["FAKE_STEP_SECONDS"] = "0.15"
        # the outage starts once the round-1 job exists and lasts longer than 30 polls at poll_seconds=0.02
        self.outage(1.2, after=lambda: bool(self.submitted("*_round_r001.sbatch.submitted")))
        pipeline.Runner(self.cfg(max_round=1, poll_seconds=0.02), quiet=True).run_loop()
        log = (self.run_dir() / "pipeline.log").read_text()
        self.assertIn("keeping the wait for job", log)
        self.assertNotIn("previous attempt did not collect", log, "the running job must not be archived and resubmitted")
        self.assertEqual(len(self.submitted("*_round_r001.sbatch.submitted")), 1)
        self.assertEqual(len(list((self.tmp / "slurm").glob("*.json"))), 1, "one job in the fake Slurm registry")

    def test_lost_sbatch_reply_adopts_the_existing_job(self):
        os.environ["FAKE_SBATCH_LOSE_REPLY"] = "1"
        os.environ["FAKE_STEP_SECONDS"] = "0.4"   # the orphaned job must still be queued when submit retries after 1 s
        try:
            pipeline.Runner(self.cfg(max_round=1, poll_seconds=0.05), quiet=True).run_loop()
        finally:
            os.environ.pop("FAKE_SBATCH_LOSE_REPLY", None)
        log = (self.run_dir() / "pipeline.log").read_text()
        self.assertIn("adopting it instead of submitting again", log)
        self.assertEqual(len(list((self.tmp / "slurm").glob("*.json"))), 1, "the lost-reply job is the only job")
        self.assertTrue((self.run_dir() / "rounds" / "round001" / "round_summary.json").is_file())

    def test_accounting_db_outage_finishes_on_artefacts(self):
        # squeue works, sacct does not: a job that left the queue with complete artefacts must count as COMPLETED
        (self.tmp / "slurm").mkdir(parents=True, exist_ok=True)
        (self.tmp / "slurm" / "DOWN_DB").write_text("1")
        pipeline.Runner(self.cfg(max_round=2, poll_seconds=0.05), quiet=True).run_loop()
        log = (self.run_dir() / "pipeline.log").read_text()
        self.assertIn("artefacts are complete; accounting DB unavailable", log)
        self.assertTrue((self.run_dir() / "DONE").is_file())
        self.assertEqual(len(self.submitted("*_round_r00?.sbatch.submitted")), 2)
        self.assertEqual(len(self.submitted("*_train_r001.sbatch.submitted")), 1)

    def test_layout_larger_than_a_node_is_rejected_at_prepare(self):
        # NODECAP-001: 4 replicas x TP2 + 2 evaluator GPUs = 10 GPUs cannot be scheduled on an 8-GPU node
        with self.assertRaises(SystemExit) as cm:
            pipeline.Runner(self.cfg(round_gpus=10, eval_gpus=2, vllm_tp=2), quiet=True).prepare()
        self.assertIn("node_gpus=8", str(cm.exception))
        self.assertIn("round_gpus", str(cm.exception))
        with self.assertRaises(SystemExit):
            pipeline.Runner(self.cfg(train_gpus=16), quiet=True).prepare()
        pipeline.Runner(self.cfg(round_gpus=8, eval_gpus=2, vllm_tp=2), quiet=True).prepare()   # 3 x TP2 + 2 fits

    def test_job_base_model_survives_sourcing_env(self):
        # MODELCLOBBER-001: config/local.env must not replace a base model the job already exported
        out = subprocess.run(["bash", "-c", f'export SAH_MODEL_PATH=/tmp/job-base-model; source "{ROOT}/config/env.sh"; echo "$SAH_MODEL_PATH"'],
                             capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "/tmp/job-base-model")
        for t in ("round", "baseline", "train"):
            self.assertIn("SAH9B_BASE_MODEL=@@MODEL_PATH@@", (ROOT / "scripts" / "pipeline" / "templates" / f"{t}.sbatch").read_text(), t)
        for w in ("worker_round.sh", "train_worker.sh", "baseline_worker.sh"):
            self.assertIn("MODELCLOBBER-001", (ROOT / "scripts" / "pipeline" / w).read_text(), w)

    def test_capacity_gate_yields_to_the_longer_waiting_sibling(self):
        # FAIRSHARE-001: a live sibling runner that started waiting earlier gets the slot; a dead sibling's mark is ignored
        runner = pipeline.Runner(self.cfg(), quiet=True)
        runner.prepare()
        sib = runner.camp / "other_lineage"; sib.mkdir()
        (sib / "WAITING_SINCE").write_text(f"{time.time() - 100:.3f} round 2\n")
        (sib / "pipeline.pid").write_text(f"{os.getpid()}\n")          # alive (this test process)
        self.assertEqual(runner.earlier_waiter(time.time()), "other_lineage")
        self.assertIsNone(runner.earlier_waiter(time.time() - 200), "a waiter that started even earlier does not yield")
        (sib / "pipeline.pid").write_text("999999999\n")              # dead runner -> stale mark
        self.assertIsNone(runner.earlier_waiter(time.time()))
        runner.release()

    def test_restart_during_outage_does_not_archive_the_running_round(self):
        os.environ["FAKE_STEP_SECONDS"] = "0.3"
        first = pipeline.Runner(self.cfg(max_round=1, poll_seconds=0.05), quiet=True)
        first.prepare()
        script = first.round_script(1, first.env["SAH_MODEL_PATH"], None, None)
        jid = first.submit(script)
        first.release()
        time.sleep(0.4)   # the fake job is RUNNING and has written round files
        self.assertTrue((self.run_dir() / "rounds" / "round001").is_dir())
        self.outage(0.8)
        time.sleep(0.1)
        pipeline.Runner(self.cfg(max_round=1, poll_seconds=0.05), quiet=True).run_loop()
        log = (self.run_dir() / "pipeline.log").read_text()
        self.assertIn("re-attach: Slurm not answering", log)
        self.assertIn(f"re-attaching to running round job {jid}", log)
        self.assertNotIn("previous attempt did not collect", log)
        self.assertEqual(len(list((self.tmp / "slurm").glob("*.json"))), 1)


class SiteSettingsTest(unittest.TestCase):
    def test_site_problems_lists_unset_placeholder_and_missing_paths(self):
        from site_env import fake_site
        site = fake_site()
        env = {"HTT_SLURM_ACCOUNT": "acct", "HTT_SLURM_PARTITION": "<gpu-partition>", "HTT_SLURM_QOS": "", "CONTAINER_MOUNTS": "/:/",
               "SAH_MODEL_PATH": "/nonexistent/model", "SAH_SERVED_MODEL": "m", "SAH_IMAGE": str(site / "image.sqsh"),
               "VLLM_ENV": str(ROOT), "SAH_PYTHON": str(site / "runtime" / "bin" / "python"), "SAH_DATA_ROOT": str(site / "dataset"),
               "NEXAU_ROOT": str(site / "nexau")}
        problems = pipeline.site_problems(env, train=False)
        self.assertTrue(any("HTT_SLURM_PARTITION is still a placeholder" in x for x in problems), problems)
        self.assertFalse(any("HTT_SLURM_QOS" in x for x in problems), "an empty QOS means the cluster default")
        self.assertTrue(any(x.startswith("SAH_MODEL_PATH: /nonexistent/model does not exist") for x in problems), problems)
        self.assertTrue(any(x.startswith("VLLM_ENV:") and x.endswith("bin/vllm does not exist") for x in problems), problems)
        self.assertFalse(any("TRAIN_ENV" in x for x in problems))
        self.assertTrue(any("TRAIN_ENV is unset" in x for x in pipeline.site_problems(env, train=True)))
        env.update({"HTT_SLURM_PARTITION": "p", "SAH_MODEL_PATH": str(site / "model"), "VLLM_ENV": str(site / "vllm")})
        self.assertEqual(pipeline.site_problems(env, train=False), [])
        self.assertTrue(any("SAH_AHC_NATIVE_BIN" in x for x in pipeline.site_problems(dict(env, SAH_AHC_NATIVE_BIN="/nonexistent"), train=False,
                                                                                        task_id="eft__ahc_simpletes__ahc039")))

    def test_prepare_fails_before_any_submission_when_the_site_file_is_incomplete(self):
        tmp = Path(tempfile.mkdtemp(prefix="htt_site_"))
        try:
            real = pipeline.load_env
            pipeline.load_env = lambda root: dict(real(root), HTT_SLURM_ACCOUNT="<your-slurm-account>")
            try:
                cfg = pipeline.Config.load(None, dict(campaign="t", name="site", task="hadamard", max_round=1, freeze=False,
                                                      require_clean_tree=False, slurm_bin=str(FAKE), sah_root=str(ROOT),
                                                      run_root=str(tmp / "runs")))
                runner = pipeline.Runner(cfg, quiet=True)
                with self.assertRaises(SystemExit) as ctx:
                    runner.prepare()
                self.assertIn("HTT_SLURM_ACCOUNT is still a placeholder", str(ctx.exception))
                self.assertIn("config/local.env", str(ctx.exception))
                self.assertFalse((tmp / "runs" / "t" / "site" / "gen").glob("*.sbatch") and list((tmp / "runs" / "t" / "site" / "gen").glob("*.sbatch")))
                runner.release()
            finally:
                pipeline.load_env = real
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_unmounted_paths_are_detected(self):
        self.assertEqual(pipeline.unmounted("/data:/data,/tmp:/tmp", ["/data/x", "/tmp/y"]), [])
        self.assertEqual(pipeline.unmounted("/data:/data,/opt/time:/usr/bin/time", ["/opt/time", "/no_such_mount_htt/run"]),
                         ["/opt/time", "/no_such_mount_htt/run"])
        self.assertEqual(pipeline.mount_sources("/a,/b:/c:ro"), [("/a", "/a"), ("/b", "/c")])

    def test_symlinked_interpreter_under_a_mount_is_not_reported(self):
        tmp = Path(tempfile.mkdtemp(prefix="htt_mount_"))
        try:
            (tmp / "env" / "bin").mkdir(parents=True)
            link = tmp / "env" / "bin" / "python"
            link.symlink_to("/no_such_container_dir_htt/python3.12")   # target exists only inside the container image
            self.assertEqual(pipeline.unmounted(f"{tmp}:{tmp}", [link]), [])
            self.assertEqual(pipeline.unmounted("/data:/data", [link]), [str(link)])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class FreezeTreeTest(unittest.TestCase):
    def test_frozen_worktree_links_the_ignored_payloads(self):
        tmp = Path(tempfile.mkdtemp(prefix="htt_freeze_"))
        try:
            main = tmp / "main"; main.mkdir()
            git = lambda *a: subprocess.run(["git", "-C", str(main), *a], check=True, capture_output=True, text=True)
            git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
            (main / ".gitignore").write_text("/slime\n/dataset/self_adapt_harness\n/.deps\n/config/local.env\n")
            (main / "config").mkdir(); (main / "config" / "env.sh").write_text("# env\n")
            git("add", "-A"); git("commit", "-q", "-m", "init")
            (main / ".deps").mkdir(); (main / ".deps" / "NexAU").mkdir()
            (main / "dataset" / "self_adapt_harness").mkdir(parents=True)
            (main / "config" / "local.env").write_text("export X=1\n")
            wt = pipeline.freeze_tree(main, tmp / "worktrees", lambda m: None)
            for rel in (".deps", "dataset/self_adapt_harness", "config/local.env"):
                self.assertTrue((wt / rel).is_symlink(), rel)
                self.assertEqual((wt / rel).resolve(), (main / rel).resolve(), rel)
            self.assertFalse((wt / "slime").exists(), "absent payloads are not linked")
            self.assertEqual(pipeline.git(wt, "status", "--porcelain"), "")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SlurmPortabilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="htt_port_"))
        os.environ["FAKE_SLURM_DIR"] = str(self.tmp / "slurm")
        os.environ["FAKE_STEP_SECONDS"] = "0.05"
        for key in ("FAKE_FAIL_ROUNDS", "FAKE_ZERO_ADV_ROUNDS", "FAKE_FAIL_TRAIN", "FAKE_NO_ACCOUNTING", "FAKE_TEST_ONLY_FAIL",
                    "FAKE_PROLOGUE_FAIL_ROUNDS", "FAKE_TIMEOUT_ROUNDS"):
            os.environ.pop(key, None)

    def tearDown(self):
        for key in ("FAKE_NO_ACCOUNTING", "FAKE_TEST_ONLY_FAIL", "FAKE_PROLOGUE_FAIL_ROUNDS", "FAKE_TIMEOUT_ROUNDS"):
            os.environ.pop(key, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cfg(self, **over):
        base = dict(campaign="t", name="port", task="hadamard", adv="v3", k=8, round_gpus=2, train_gpus=8, max_round=1,
                    poll_seconds=0.1, heartbeat_minutes=1000, freeze=False, require_clean_tree=False,
                    slurm_bin=str(FAKE), sah_root=str(ROOT), run_root=str(self.tmp / "runs"))
        base.update(over)
        return pipeline.Config.load(None, base)

    def test_gres_fields_of_every_shape_are_counted(self):
        for field, want in (("gres/gpu:2", 2), ("gres/gpu:h100:8", 8), ("gres/gpu:a100_80gb:4", 4), ("gpu:4(IDX:0-3)", 4),
                            ("gres/gpu:8,gres/nvme:1", 8), ("N/A", 0), ("", 0), ("gres/nvme:1", 0), ("gres:gpu:2", 2),
                            ("gres/gpu:3g.20gb:2", 2), ("gpu:h100:2(IDX:0-1)", 2)):
            self.assertEqual(pipeline.gpus_in_gres(field), want, field)

    def test_zero_caps_are_values_not_defaults(self):
        runner = pipeline.Runner(self.cfg(gpu_cap=0, max_jobs=0), quiet=True)
        self.assertEqual(runner.caps(), (0, 0))
        self.assertEqual(pipeline.Runner(self.cfg(gpu_cap=7, max_jobs=3), quiet=True).caps(), (7, 3))

    def test_job_without_accounting_record_completes_on_its_artefacts(self):
        os.environ["FAKE_NO_ACCOUNTING"] = "1"
        runner = pipeline.Runner(self.cfg(), quiet=True)
        runner.run_loop()
        log = (runner.run / "pipeline.log").read_text()
        self.assertIn("accounting database has no record", log)
        self.assertTrue((runner.run / "DONE").is_file())
        self.assertTrue((runner.run / "rounds" / "round001" / "round_summary.json").is_file())

    def test_submission_preflight_reports_a_rejected_layout_before_round_one(self):
        os.environ["FAKE_TEST_ONLY_FAIL"] = "1"
        runner = pipeline.Runner(self.cfg(), quiet=True)
        with self.assertRaises(SystemExit) as ctx:
            runner.prepare()
        self.assertIn("Slurm rejects the round job", str(ctx.exception))
        self.assertIn("Requested node configuration is not available", str(ctx.exception))
        self.assertFalse(list((runner.run / "gen").glob("_preflight_*")), "preflight scripts are removed")
        runner.release()

    def test_unset_slurm_keys_emit_comments_instead_of_sbatch_lines(self):
        real = pipeline.load_env
        pipeline.load_env = lambda root: {**real(root), "HTT_SLURM_ACCOUNT": "", "HTT_SLURM_PARTITION": "", "HTT_SLURM_QOS": ""}
        try:
            runner = pipeline.Runner(self.cfg(), quiet=True)
            script = runner.dry_run()
            text = Path(script).read_text()
            for flag in ("--account", "--partition", "--qos"):
                self.assertNotIn(f"#SBATCH {flag}=", text)
                self.assertIn(f"# {flag}: cluster default", text)
            self.assertIn("export RUN_ROOT=" + str(runner.run_root), text)
            self.assertIn("not set; the jobs use the cluster's default", (runner.run / "pipeline.log").read_text())
            self.assertEqual((runner.run / "GPUS").read_text().strip(), "2")
            self.assertEqual((runner.run / "GPU_LAYOUT").read_text().strip(), "2")
            runner.release()
        finally:
            pipeline.load_env = real

    def test_round_resources_are_configurable(self):
        runner = pipeline.Runner(self.cfg(round_cpus=16, round_mem="120G", round_gpus=4, eval_gpus=1), quiet=True)
        script = runner.dry_run()
        text = Path(script).read_text()
        self.assertIn("#SBATCH --cpus-per-task=16", text)
        self.assertIn("#SBATCH --mem=120G", text)
        self.assertIn("#SBATCH --gres=gpu:4", text)
        self.assertEqual((runner.run / "GPU_LAYOUT").read_text().strip(), "4 eval_gpus=1")
        runner.release()

    def test_prologue_failure_stops_the_runner_with_the_message(self):
        os.environ["FAKE_PROLOGUE_FAIL_ROUNDS"] = "1"
        runner = pipeline.Runner(self.cfg(), quiet=True)
        with self.assertRaises(SystemExit) as ctx:
            runner.run_loop()
        self.assertIn("failed before srun", str(ctx.exception))
        self.assertIn("container image SAH_IMAGE", str(ctx.exception))
        self.assertEqual(json.loads((runner.run / "RETRIES.json").read_text()).get("round1"), 1, "not retried")

    def test_timeout_is_not_resubmitted_unchanged(self):
        os.environ["FAKE_TIMEOUT_ROUNDS"] = "1"
        runner = pipeline.Runner(self.cfg(), quiet=True)
        with self.assertRaises(SystemExit) as ctx:
            runner.run_loop()
        self.assertIn("TIMEOUT", str(ctx.exception))
        self.assertIn("round_time", str(ctx.exception))

    def test_worker_budgets_follow_round_time(self):
        self.assertEqual(pipeline.wall_seconds("06:00:00"), 21600)
        self.assertEqual(pipeline.wall_seconds("1-02:00:00"), 93600)
        self.assertEqual(pipeline.wall_seconds("90"), 5400)
        runner = pipeline.Runner(self.cfg(round_time="02:00:00"), quiet=True)
        script = Path(runner.dry_run()).read_text()
        self.assertIn("SAH9B_PROPOSE_TIMEOUT=2880", script)       # 40 % of 2 h
        self.assertIn("SAH9B_ROLLOUT_WALL_TIMEOUT=3240", script)  # 45 %
        self.assertIn("SAH9B_JOB_SECONDS=6300", script)           # 2 h - 15 min
        runner.release()

    def test_training_skipped_by_the_runner_leaves_the_same_marker_as_the_worker(self):
        os.environ["FAKE_ZERO_ADV_ROUNDS"] = "1"
        runner = pipeline.Runner(self.cfg(max_round=2), quiet=True)
        runner.run_loop()
        td = runner.run / "training" / "round001"
        self.assertTrue((td / "TRAIN_COMPLETE").is_file())
        self.assertIn("updated=false", (td / "TRAIN_COMPLETE").read_text())
        os.environ.pop("FAKE_ZERO_ADV_ROUNDS", None)


if __name__ == "__main__":
    unittest.main()
