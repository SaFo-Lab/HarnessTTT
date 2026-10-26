"""The dataset recipe (config/dataset/*) is self-consistent and free of site paths; the task loader honours the manifest's
optional fields; the builder's verify step works on a tree built from the recipe's rules."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "config" / "dataset"
SRC = ROOT / "src"


class RecipeTest(unittest.TestCase):
    def test_recipe_files_exist_and_carry_no_site_paths(self):
        for name in ("sources.json", "MANIFEST_eft.json", "CHECKSUMS.json", "README.snapshot.md"):
            self.assertTrue((CFG / name).is_file(), name)
        for path in [p for p in CFG.rglob("*") if p.is_file()]:
            text = path.read_text(errors="replace")
            self.assertNotIn("/lustre/", text, path)
            self.assertNotIn("yingzim", text, path)

    def test_sources_cover_every_manifest_task_directory(self):
        sources = json.loads((CFG / "sources.json").read_text())
        manifest = json.loads((CFG / "MANIFEST_eft.json").read_text())
        produced = [dst for repo in sources["repos"] for _src, dst in repo["copy"]]
        overlay = sources["overlay"]["dir"]
        produced += [p.relative_to(ROOT / overlay).as_posix() for p in (ROOT / overlay).rglob("*") if p.is_file()]

        def covered(rel):   # a task directory is produced when a copied subtree contains it or files are laid into it
            rel = rel.rstrip("/")
            return any(rel == d or rel.startswith(d.rstrip("/") + "/") or d.startswith(rel + "/") for d in produced)

        for rec in manifest["tasks"]:
            self.assertTrue(covered(rec["rel_path"]), rec["task_id"])
        for rec in manifest["cross_referenced_tasks"]:
            self.assertTrue(covered(rec["use_path"]), rec["task_id"])
        for d in sources["downloads"]:
            self.assertRegex(d["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(d["url"].startswith("https://"))
        for repo in sources["repos"]:
            self.assertRegex(repo["commit"], r"^[0-9a-f]{40}$")
            self.assertIn(repo["tier"], ("core", "gpu", "extra"))

    def test_checksums_only_list_files_the_recipe_produces(self):
        sources = json.loads((CFG / "sources.json").read_text())
        checks = json.loads((CFG / "CHECKSUMS.json").read_text())["files"]
        produced = [dst for repo in sources["repos"] for _src, dst in repo["copy"]] + [d["path"] for d in sources["downloads"]]
        overlay = ROOT / sources["overlay"]["dir"]
        produced += [p.relative_to(overlay).as_posix() for p in overlay.rglob("*") if p.is_file()]
        self.assertGreater(len(checks), 300)
        for rel, rec in checks.items():
            self.assertTrue(any(rel == d or rel.startswith(d.rstrip("/") + "/") for d in produced), rel)
            self.assertIn(rec["tier"], ("core", "gpu", "extra"))
        self.assertTrue(any(rel.startswith("raw/eft-aac2e79/benchmarks/math/circle_packing/") for rel in checks))
        self.assertIn("runtime/ahc_native/bin/docker", checks)

    def test_checksums_exclude_generated_task_data(self):
        import fnmatch
        sources = json.loads((CFG / "sources.json").read_text())
        checks = json.loads((CFG / "CHECKSUMS.json").read_text())["files"]
        for pattern in sources["generated_not_checksummed"]:
            hits = [rel for rel in checks if fnmatch.fnmatch(rel, pattern) or rel.startswith(pattern.rstrip("*"))]
            self.assertEqual(hits, [], pattern)

    def test_manifest_paths_are_variables_not_machines(self):
        manifest = json.loads((CFG / "MANIFEST_eft.json").read_text())
        for rec in manifest["tasks"] + manifest["cross_referenced_tasks"]:
            for key in ("python_exe",):
                if rec.get(key):
                    self.assertTrue(rec[key].startswith("${"), (rec["task_id"], key))
            for entry in rec.get("extra_pythonpath", []):
                self.assertFalse(entry.startswith("/"), (rec["task_id"], entry))


class LoaderTest(unittest.TestCase):
    """A two-task fake snapshot exercises the manifest fields in a subprocess (eft_task binds SAH_DATASET_ROOT at import)."""

    def test_loader_honours_optional_manifest_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "snap"
            task = root / "raw/x/taskA"; task.mkdir(parents=True)
            (task / "initial_program.py").write_text("# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n")
            (task / "evaluator.py").write_text("def evaluate(p):\n    return {'combined_score': 1.0}\n")
            (task / "config.yaml").write_text("prompt:\n  system_message: spec A\nevaluator:\n  timeout: 100\n")
            cross = root / "raw/y/taskB"; cross.mkdir(parents=True)
            (cross / "init_program.py").write_text("y = 2\n"); (cross / "evaluator.py").write_text("pass\n"); (cross / "spec.txt").write_text("spec B")
            (root / "runtime/skydiscover_min").mkdir(parents=True); (root / "runtime/simpletes_min").mkdir(parents=True)
            (root / "rel/overlay").mkdir(parents=True)
            manifest = {"tasks": [{"task_id": "a", "rel_path": "raw/x/taskA", "evaluator_rel": "evaluator.py", "cost_tier": "t0",
                                   "plan_family": "p", "domain": "d", "per_eval_timeout_s": 42, "python_exe": "${HTT_T_PY}",
                                   "extra_pythonpath": ["rel/overlay", "${HTT_T_UNSET}/x"], "env": {"A": "1"}}],
                        "cross_referenced_tasks": [{"task_id": "b", "use_path": "raw/y/taskB", "cost_tier": "tier1", "plan_family": "pf",
                                                    "domain": "dom", "per_eval_timeout_s": 7, "deps": ["numpy", "z"],
                                                    "extra_pythonpath": ["${HTT_T_SET}"], "env": {"B": "2"}}]}
            (root / "MANIFEST_eft.json").write_text(json.dumps(manifest))
            code = textwrap.dedent("""
                import json
                from inner.tasks.eft_task import load_tasks
                out = {}
                for t in load_tasks(include_adrs=False):
                    out[t.task_id] = dict(timeout=t.per_eval_timeout_s, python_exe=t.python_exe, extra=t.extra_pythonpath, env=t.env,
                                          cost_tier=t.cost_tier, plan_family=t.plan_family, domain=t.domain, deps=t.deps, spec=t.spec)
                print(json.dumps(out))
            """)
            env = dict(os.environ, SAH_DATASET_ROOT=str(root), PYTHONPATH=str(SRC), HTT_T_PY=sys.executable, HTT_T_SET="/tmp/htt_set")
            env.pop("HTT_T_UNSET", None)
            out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
            got = json.loads(out)
            a, b = got["a"], got["b"]
            self.assertEqual(a["timeout"], 42.0)                              # manifest overrides config.yaml's 100
            self.assertEqual(a["python_exe"], sys.executable)                 # ${VAR} expanded
            self.assertEqual(a["extra"], [str(root / "rel/overlay")])         # relative -> dataset root; unset variable dropped
            self.assertEqual(a["env"], {"A": "1"})
            self.assertEqual(a["spec"], "spec A")
            self.assertEqual((b["cost_tier"], b["plan_family"], b["domain"]), ("tier1", "pf", "dom"))
            self.assertEqual(b["timeout"], 7.0)
            self.assertEqual(b["deps"], ["numpy", "z"])
            self.assertEqual(b["extra"], ["/tmp/htt_set"])
            self.assertEqual(b["env"], {"B": "2"})
            self.assertEqual(b["spec"], "spec B")


class BuilderVerifyTest(unittest.TestCase):
    def test_verify_reports_missing_and_mismatched_files(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "snap"
            checks = json.loads((CFG / "CHECKSUMS.json").read_text())["files"]
            rel = "runtime/ahc_native/bin/docker"
            (dest / rel).parent.mkdir(parents=True)
            (dest / rel).write_text("not the shim")
            out = subprocess.run([sys.executable, str(ROOT / "scripts/setup/setup_dataset.py"), "--verify", "--dest", str(dest), "--tiers", "core"],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
            self.assertIn("differs   " + rel, out.stdout)
            n_core = sum(1 for r in checks.values() if r["tier"] == "core")
            self.assertIn(f"{n_core - 1} missing", out.stdout)


if __name__ == "__main__":
    unittest.main()
