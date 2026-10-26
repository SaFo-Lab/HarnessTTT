#!/usr/bin/env python3
"""HarnessTTT evolve loop runner (submitting host; Python >= 3.9 with PyYAML).

    python3 src/pipeline.py config/pipelines/hadamard.yaml          start or resume the lineage described by that YAML
    python3 src/pipeline.py                                            same, with the default config/pipeline.yaml
    python3 src/pipeline.py config/pipelines/hadamard.yaml status   progress table of that run (progress.json + run dir)
    python3 src/pipeline.py config/pipelines/hadamard.yaml stop     write ABL_STOP: stop between jobs (a running job finishes)
    python3 src/pipeline.py <config> --dry-run                         write the next sbatch script into <run>/gen without submitting
    (any YAML key can be overridden on the command line: --max-round 3 --name test ...)

One process drives one lineage (run dir):

    round r  (2 GPUs)   submit templates/round.sbatch -> wait, report vLLM / propose / rollouts / collect progress
    after r             build beam-2 parents (build_bases.py) + training replay (trainer/grpo_to_replay.py)
    train r  (8 GPUs)   submit templates/train.sbatch  -> wait, report bounding / training / merge
    round r+1           proposer = merged checkpoint from training/roundNNN/model_state.json ... until max_round

State is derived from the run dir, so a killed runner resumes where it left off and re-attaches to a running job.  Progress
goes to stdout, <run>/pipeline.log and <run>/progress.json.  Caps (HARNESSTTT_GPU_CAP / HARNESSTTT_MAX_JOBS, or a GPU_CAP
file in the campaign dir) are enforced before every submission; the low-QOS account/partition come from config/env.sh.
By default the jobs execute from an immutable git worktree of the committed HEAD (freeze), so the tree stays editable.
"""
from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_ROOT / "src")) if str(MAIN_ROOT / "src") not in sys.path else None
from inner.tasks.aliases import DISPLAY, TASK_IDS, fmt_metric  # noqa: E402,F401  (one alias/display table for every tool)


ADVS = ("v3", "v2", "grpo", "legacy", "anchored")
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING"}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "DEADLINE", "BOOT_FAIL", "REVOKED"}


def _user() -> str:
    """The submitting user as Slurm knows them (USER may be unset under cron/systemd launches)."""
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "")


def gpus_in_gres(field: str) -> int:
    """GPUs in one squeue ``%b`` field: ``gres/gpu:2``, ``gres/gpu:h100:8``, ``gpu:4(IDX:0-3)``, ``gres/gpu:8,gres/nvme:1``; ``N/A`` -> 0."""
    total = 0
    for item in field.split(","):
        m = re.search(r"(?:^|[/:])gpu(?::[^:,()\s]+)?:(\d+)(?:\(|\s*$)", item.strip())
        if m:
            total += int(m.group(1))
    return total


_UNKNOWN_WAIT_SECONDS = 900.0   # a job gone from squeue without an accounting record and without artefacts: how long to keep waiting
RESOURCE_STATES = {"TIMEOUT", "OUT_OF_MEMORY", "DEADLINE"}   # resubmitting the same script would only burn the same GPU hours again
# `sbatch --test-only` answers that describe the cluster's current state, not the script (the real submission would just wait)
_TEST_ONLY_TRANSIENT_RE = re.compile(r"Required node not available|down, drained or reserved|violates accounting/QOS policy|"
                                     r"MaxSubmitJobs|AssocMaxJobs|QOSMax|ReqNodeNotAvail|Nodes required for job are DOWN", re.I)


def wall_seconds(limit: str) -> int:
    """Slurm time limit in seconds: 'MM', 'MM:SS', 'HH:MM:SS', 'D-HH', 'D-HH:MM', 'D-HH:MM:SS'; UNLIMITED -> ten years."""
    text = str(limit).strip()
    if not text or text.upper() in ("UNLIMITED", "INFINITE", "NONE"):
        return 10 * 365 * 86400
    days, _, rest = text.rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    if days:                                   # D-HH[:MM[:SS]]
        parts = (parts + [0, 0])[:3]
        secs = parts[0] * 3600 + parts[1] * 60 + parts[2] + int(days) * 86400
    elif len(parts) == 3:                      # HH:MM:SS
        secs = parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:                      # MM:SS
        secs = parts[0] * 60 + parts[1]
    else:                                      # MM
        secs = parts[0] * 60
    return secs


def err_tail(path: Path) -> Tuple[str, Optional[str]]:
    """Last lines of a job's stderr file and, if present, the job script's own '[sbatch] ...' prologue message."""
    if not path.is_file():
        return "", None
    lines = [l.strip() for l in path.read_text(errors="replace").splitlines() if l.strip()]
    marker = next((l for l in reversed(lines) if l.startswith("[sbatch] ")), None)
    return " / ".join(lines[-3:])[:300], marker


class SlurmUnavailable(RuntimeError):
    """squeue/sacct/sbatch did not answer (controller fail-over, slurmdbd outage): callers keep waiting instead of guessing."""


# OUTAGE-001: the Slurm controller failed over ("Slurm backup controller in standby mode"); squeue timed
# out, sacct answered with database errors and sbatch refused submissions.  The runner read the empty squeue answer as free
# capacity, submitted into the outage and crashed on the sbatch error; a longer outage would have ended wait_job's 30-poll
# miss cap for RUNNING jobs and resubmitted them.  Every Slurm client call now goes through Runner._slurm_run, which turns
# these answers into SlurmUnavailable, and every caller waits it out.
_SLURM_DOWN_RE = re.compile(r"backup controller in standby|Unable to contact slurm controller|Socket timed out|"
                            r"Zero Bytes were transmitted|Connection refused|Protocol authentication error|"
                            r"temporarily unable|Slurm database|slurmdbd|talking to the database|Connection timed out|"
                            r"Communication connection failure|Transport endpoint is not connected", re.I)
_SLURM_CALL_TIMEOUT = 300.0


# ----------------------------------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Config:
    campaign: str = "default"
    name: str = "hadamard"
    task: str = "hadamard"
    adv: str = "v3"
    k: int = 8
    round_gpus: int = 2
    train_gpus: int = 8
    max_round: int = 15
    h1_seed: int = 7311
    max_evals: int = 20
    eval_timeout: int = 600
    token_budget: int = 18000
    lr: str = "1e-5"
    kl: str = "0.1"
    over_cap_as_invalid: bool = True
    round_time: str = "06:00:00"
    train_time: str = "02:30:00"
    poll_seconds: float = 60.0
    heartbeat_minutes: float = 10.0
    freeze: bool = True
    require_clean_tree: bool = True
    max_round_retries: int = 2
    max_train_retries: int = 1
    job_prefix: str = "htt"
    gpu_cap: Optional[int] = None
    max_jobs: Optional[int] = None
    run_root: Optional[str] = None
    sah_root: Optional[str] = None
    slurm_bin: Optional[str] = None
    brief_dir: Optional[str] = None
    # seed-program swap (experiments that evolve from another initial program):
    #   seed_programs_file  JSON {task_id: {"program": text, "score": evaluated combined_score, ...}} -- the round-1 rollouts
    #                       start from this program and it becomes the ratchet's incumbent (copied to rounds/best_programs.json)
    #   baseline_json       baseline_h2_20ev-format table for this seed (SAH_BASELINE_JSON of every job); when the file does not
    #                       exist yet the runner first submits a baseline job (frozen harness, max_evals, baseline_seeds runs)
    seed_programs_file: Optional[str] = None
    baseline_json: Optional[str] = None
    baseline_seeds: int = 2
    baseline_time: str = "04:00:00"
    # GPUs of a round job reserved for the task evaluator (GPU kernel tasks): vLLM replicas use round_gpus - eval_gpus,
    # evaluations are serialised per reserved GPU (SAH_EVAL_GPU_POOL in the worker)
    eval_gpus: int = 0
    # per-lineage base model: proposer and frozen executor; None = the tree's env (SAH_MODEL_PATH / SAH_SERVED_MODEL).
    # vllm_tp = GPUs per vLLM replica (27B bf16 needs 2 on H100 80GB); round_gpus = replicas * vllm_tp + eval_gpus.
    # max_model_len = vLLM --max-model-len = SAH_MAX_CONTEXT_TOKENS handed to both agent packages (CONTEXT-001).
    model_path: Optional[str] = None
    served_model: Optional[str] = None
    vllm_tp: int = 1
    max_model_len: int = 131072
    node_gpus: int = 8          # GPUs per node; every job here is single-node (NODECAP-001)
    # resources of a training job: slime gathers the full state dict on every rank when it saves the merged checkpoint
    # (8 x 55 GB for a 27B), so the job takes the whole node's memory by default (--mem=0); set e.g. "1000G" where mem=0 is refused
    train_cpus: int = 32
    train_mem: str = "0"
    # resources of a round / baseline job; None = 12 CPUs and 75-100 GB per GPU (round_gpus 1/2/4 -> 12/24/48 CPUs, 100/200/300 GB)
    round_cpus: Optional[int] = None
    round_mem: Optional[str] = None

    @classmethod
    def load(cls, path: Optional[Path], overrides: Dict[str, Any]) -> "Config":
        data: Dict[str, Any] = {}
        if path is not None:
            try:
                import yaml  # PyYAML: the only third-party module the submitting host needs
            except ImportError:
                raise SystemExit("PyYAML is required to read pipeline configs: `pip install pyyaml` into this Python "
                                 "(or run src/pipeline.py with one that has it, e.g. the HarnessTTT venv)") from None
            loaded = yaml.safe_load(path.read_text()) or {}
            if not isinstance(loaded, dict):
                raise SystemExit(f"{path}: expected a mapping at the top level")
            data.update(loaded)
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise SystemExit(f"unknown pipeline config keys: {unknown} (known: {sorted(known)})")
        cfg = cls(**data)
        if cfg.adv not in ADVS:
            raise SystemExit(f"adv must be one of {ADVS}, got {cfg.adv!r}")
        for name in ("lr", "kl", "train_mem", "round_mem"):
            if getattr(cfg, name) is not None:
                setattr(cfg, name, str(getattr(cfg, name)))
        return cfg

    @property
    def task_id(self) -> str:
        return TASK_IDS.get(self.task, self.task)


def load_env(sah_root: Path) -> Dict[str, str]:
    """Source config/env.sh (+ config/local.env) the way the launchers do and return the resulting environment."""
    env_sh = sah_root / "config" / "env.sh"
    if not env_sh.is_file():
        raise SystemExit(f"missing {env_sh}")
    out = subprocess.run(["bash", "-c", f'set -a; source "{env_sh}" && env -0'], capture_output=True, check=True)
    env: Dict[str, str] = {}
    for item in out.stdout.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            env[key.decode()] = value.decode(errors="replace")
    return env


# site settings every job needs (config/local.env); a missing or placeholder value fails the run before anything is submitted.
# Slurm account / partition / QOS are optional (their #SBATCH lines are emitted only when set); SLURM_KEYS only reject placeholders.
SITE_KEYS_JOB = ("CONTAINER_MOUNTS", "SAH_MODEL_PATH", "SAH_SERVED_MODEL", "SAH_IMAGE", "VLLM_ENV", "SAH_PYTHON", "SAH_DATA_ROOT", "NEXAU_ROOT")
SITE_KEYS_TRAIN = ("TRAIN_ENV", "SLIME_ROOT", "MEGATRON_ROOT")
SLURM_KEYS = ("HTT_SLURM_ACCOUNT", "HTT_SLURM_PARTITION", "HTT_SLURM_QOS")
# what the job scripts open before srun (the same checks the templates make, so a typo fails here and not in the queue):
# (env key, relative entry, kind) with kind "dir" | "file" | "entry" (exists, possibly a symlink into the container image)
SITE_PATHS_JOB = (("SAH_MODEL_PATH", "", "dir"), ("SAH_IMAGE", "", "file"), ("VLLM_ENV", "bin/vllm", "entry"), ("SAH_PYTHON", "", "entry"),
                  ("SAH_DATA_ROOT", "MANIFEST_eft.json", "file"), ("NEXAU_ROOT", "", "dir"))
SITE_PATHS_TRAIN = (("TRAIN_ENV", "bin/python", "entry"), ("SLIME_ROOT", "", "dir"), ("MEGATRON_ROOT", "", "dir"))


def site_problems(env: Dict[str, str], *, train: bool = True, task_id: str = "") -> List[str]:
    """Settings of config/local.env that are unset, still the example's ``<placeholder>``, or point at nothing."""
    problems: List[str] = []
    keys = SITE_KEYS_JOB + (SITE_KEYS_TRAIN if train else ())
    for key in keys:
        value = (env.get(key) or "").strip()
        if not value:
            problems.append(f"{key} is unset")
        elif "<" in value or ">" in value:
            problems.append(f"{key} is still a placeholder: {value}")
    for key in SLURM_KEYS:
        value = (env.get(key) or "").strip()
        if "<" in value or ">" in value:
            problems.append(f"{key} is still a placeholder: {value} (leave it empty to use the cluster default)")
    checks = list(SITE_PATHS_JOB) + (list(SITE_PATHS_TRAIN) if train else [])
    if "ahc" in task_id:
        checks.append(("SAH_AHC_NATIVE_BIN", "docker", "entry"))
    for key, rel, kind in checks:
        value = (env.get(key) or "").strip()
        if not value or "<" in value:
            continue
        path = Path(value) / rel if rel else Path(value)
        ok = path.is_dir() if kind == "dir" else path.is_file() if kind == "file" else os.path.lexists(path)
        if not ok:
            problems.append(f"{key}: {path} does not exist")
    return problems


def mount_sources(container_mounts: str) -> List[Tuple[str, str]]:
    """``src:dst[:flags]`` entries of CONTAINER_MOUNTS as (src, dst); a bare ``path`` mounts to itself."""
    out: List[Tuple[str, str]] = []
    for item in (container_mounts or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        out.append((parts[0], parts[1] if len(parts) > 1 and parts[1] else parts[0]))
    return out


def unmounted(container_mounts: str, paths: List[Any]) -> List[str]:
    """Paths the container would not see under the same name: no mount source covers them with src == dst.

    A path is checked as given (absolute, symlinks kept) and fully resolved; either form under a mount counts, because
    a venv's ``bin/python`` may be a symlink to an interpreter that exists only inside the container image.
    """
    mounts = mount_sources(container_mounts)
    bases = []
    for src, dst in mounts:
        if src == dst:
            bases.extend({Path(os.path.abspath(src)), Path(src).resolve()})
    missing: List[str] = []
    for raw in paths:
        if not raw:
            continue
        forms = {Path(os.path.abspath(str(raw))), Path(str(raw)).resolve()}
        if not any(form == base or base in form.parents for form in forms for base in bases):
            missing.append(str(raw))
    return missing


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def freeze_tree(main_root: Path, worktrees: Path, log) -> Path:
    """Create (or reuse) a detached worktree of the committed HEAD and link the ignored payloads into it."""
    if git(main_root, "status", "--porcelain"):
        raise SystemExit(f"{main_root} has uncommitted changes; commit first (jobs execute from the committed tree)")
    sha = git(main_root, "rev-parse", "HEAD")
    wt = worktrees / f"htt_{sha[:10]}"
    if not (wt / ".git").exists():
        wt.parent.mkdir(parents=True, exist_ok=True)
        git(main_root, "worktree", "add", "--detach", str(wt), sha)
        log(f"frozen worktree created: {wt} (commit {sha[:10]})")
    # git-ignored payloads and the site file: the worktree sees the main tree's copies (config/local.env is not tracked)
    for rel in ("slime", "dataset/self_adapt_harness", ".deps", "config/local.env"):
        src, dst = main_root / rel, wt / rel
        if src.exists() and not dst.exists() and not dst.is_symlink():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src.resolve(), dst)
    dirty = git(wt, "status", "--porcelain")
    if dirty:
        raise SystemExit(f"frozen worktree {wt} is not clean:\n{dirty}")
    return wt


def fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ----------------------------------------------------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------------------------------------------------
class Runner:
    def __init__(self, cfg: Config, quiet: bool = False):
        self.cfg = cfg
        self.quiet = quiet
        self.main_root = Path(cfg.sah_root).resolve() if cfg.sah_root else MAIN_ROOT
        self.env = load_env(self.main_root)
        self.apply_model_override()
        # physical path: SAH resolves candidate/workspace paths and checks them against the run dir it was given, so a
        # run dir reached through a symlink (e.g. .../users/<u> -> .../projects/.../users/<u>) breaks propose
        self.run_root = Path(cfg.run_root or self.env.get("RUN_ROOT") or (self.main_root / "runs")).resolve()
        self.camp = self.run_root / cfg.campaign
        self.run = self.camp / cfg.name
        self.sah_root = self.main_root
        self._log_fh = None
        self._lock_fh = None
        self.rounds_info: Dict[int, Dict[str, Any]] = {}
        self.current: Dict[str, Any] = {}

    # ---- basics -------------------------------------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        if not self.quiet:
            print(line, flush=True)
        if self._log_fh is None:
            self.run.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.run / "pipeline.log", "a")
        self._log_fh.write(line + "\n")
        self._log_fh.flush()

    def release(self) -> None:
        """Drop the run-dir lock and close the log, so another runner (a restart) can take the run over."""
        for fh in (self._lock_fh, self._log_fh):
            if fh is not None:
                fh.close()
        self._lock_fh = self._log_fh = None

    def prepare(self) -> None:
        """Freeze the tree (optional), reload env from the frozen tree, write the run-dir contract, take the lock."""
        self.run.mkdir(parents=True, exist_ok=True)
        (self.run / "out").mkdir(exist_ok=True)
        (self.run / "gen").mkdir(exist_ok=True)
        (self.run / "rounds").mkdir(exist_ok=True)
        (self.run / "training").mkdir(exist_ok=True)
        self._lock_fh = open(self.run / "pipeline.lock", "w")
        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pid = (self.run / "pipeline.pid").read_text().strip() if (self.run / "pipeline.pid").is_file() else "?"
            raise SystemExit(f"another pipeline.py (pid {pid}) drives {self.run}; stop it first")
        (self.run / "pipeline.pid").write_text(f"{os.getpid()}\n")
        problems = site_problems(self.env, train=self.cfg.max_round > 1, task_id=self.cfg.task_id)
        if problems:
            raise SystemExit("site settings incomplete; edit config/local.env (template: config/local.env.example):\n  - "
                             + "\n  - ".join(problems))
        unset = [k for k in SLURM_KEYS if not (self.env.get(k) or "").strip()]
        if unset:
            self.log(f"note: {', '.join(unset)} not set; the jobs use the cluster's default account/partition/QOS")
        qos = (self.env.get("HTT_SLURM_QOS") or "").strip()
        if self.env.get("HARNESSTTT_REQUIRE_LOW_QOS") == "1" and not qos.endswith("_low"):
            raise SystemExit(f"HARNESSTTT_REQUIRE_LOW_QOS=1 but HTT_SLURM_QOS={qos!r} is not a *_low QOS (site policy)")
        if self.cfg.freeze:
            self.sah_root = freeze_tree(self.main_root, self.camp / "worktrees", self.log)
            self.env = load_env(self.sah_root)
            self.apply_model_override()
        elif self.cfg.require_clean_tree and git(self.main_root, "status", "--porcelain"):
            raise SystemExit(f"{self.main_root} has uncommitted changes; commit (or run with freeze) before submitting jobs")
        # the jobs run inside a container: everything they open must be visible under the same path (CONTAINER_MOUNTS)
        e = self.env
        for path in unmounted(e.get("CONTAINER_MOUNTS", ""), [self.sah_root, self.main_root, self.run_root, e.get("SAH_MODEL_PATH"),
                                                               e.get("SAH_DATA_ROOT"), e.get("SAH_IMAGE"), e.get("VLLM_ENV"),
                                                               e.get("SAH_PYTHON"), e.get("NEXAU_ROOT"), e.get("SLIME_ROOT")]):
            self.log(f"warning: {path} is not under any CONTAINER_MOUNTS source (same path inside and outside); jobs will not see it")
        for name, value in (("TASK", self.cfg.task), ("ARM", self.cfg.adv), ("K", self.cfg.k), ("H1_SEED", self.cfg.h1_seed)):
            path = self.run / name
            if not path.is_file():
                path.write_text(f"{value}\n")
            elif path.read_text().strip() != str(value):
                raise SystemExit(f"{path} holds {path.read_text().strip()!r} but the config says {value!r}; "
                                 f"a run dir is bound to one task/arm/K")
        # the GPU layout of a round job is a resource setting (it changes nothing about the experiment): record changes.
        # GPUS stays a bare integer (chain.sh / restart_from.sh read it); the full layout goes to GPU_LAYOUT.
        layout_path = self.run / "GPU_LAYOUT"
        layout_now = f"{self.cfg.round_gpus}" + (f" eval_gpus={self.cfg.eval_gpus}" if self.cfg.eval_gpus else "") + \
            (f" vllm_tp={self.cfg.vllm_tp}" if self.cfg.vllm_tp > 1 else "")
        if layout_path.is_file() and layout_path.read_text().strip() != layout_now:
            self.log(f"round GPU layout changed: {layout_path.read_text().strip()!r} -> {layout_now!r} (applies from the next submitted job)")
        layout_path.write_text(layout_now + "\n")
        (self.run / "GPUS").write_text(f"{self.cfg.round_gpus}\n")
        brief = Path(self.cfg.brief_dir) if self.cfg.brief_dir else self.camp / "briefs_none"
        brief.mkdir(parents=True, exist_ok=True)
        self.brief_dir = brief
        if self.cfg.eval_gpus < 0 or self.cfg.eval_gpus >= self.cfg.round_gpus:
            raise SystemExit(f"eval_gpus must leave at least one vLLM replica: eval_gpus {self.cfg.eval_gpus}, round_gpus {self.cfg.round_gpus}")
        if self.cfg.vllm_tp < 1 or (self.cfg.round_gpus - self.cfg.eval_gpus) % self.cfg.vllm_tp:
            raise SystemExit(f"round_gpus - eval_gpus must be a multiple of vllm_tp: round_gpus {self.cfg.round_gpus}, "
                             f"eval_gpus {self.cfg.eval_gpus}, vllm_tp {self.cfg.vllm_tp}")
        # NODECAP-001: the first 27B lineages asked for 10 GPUs (4 replicas x TP2 + 2 evaluator GPUs) on
        # one node; the cluster's nodes have 8 GPUs, and sbatch answered "Requested node configuration is not available" only when capacity
        # finally allowed the submission, crashing both runners hours after launch. Every job here is single-node.
        too_big = {k: v for k, v in (("round_gpus", self.cfg.round_gpus), ("train_gpus", self.cfg.train_gpus),
                                      ("baseline (vllm_tp + eval_gpus)", self.cfg.vllm_tp + self.cfg.eval_gpus)) if v > self.cfg.node_gpus}
        if too_big:
            raise SystemExit(f"single-node jobs cannot exceed node_gpus={self.cfg.node_gpus}: {too_big}")
        if self.cfg.model_path and not Path(self.cfg.model_path).is_dir():
            raise SystemExit(f"model_path is not a directory: {self.cfg.model_path}")
        # seed-program swap: the registry entry is the round-1 rollout seed AND the ratchet's incumbent (strict_single rejects
        # an inherited seed without a prior registry entry: "unexpected_inherited_seed"), so it is copied in before round 1
        self.seed_file: Optional[Path] = None
        if self.cfg.seed_programs_file:
            self.seed_file = Path(self.cfg.seed_programs_file).resolve()
            if not self.seed_file.is_file():
                raise SystemExit(f"seed_programs_file does not exist: {self.seed_file}")
            ent = json.loads(self.seed_file.read_text()).get(self.cfg.task_id)
            if not isinstance(ent, dict) or not ent.get("program") or ent.get("score") is None:
                raise SystemExit(f"{self.seed_file}: needs {{{self.cfg.task_id!r}: {{'program': ..., 'score': <evaluated combined_score>}}}}")
            reg = self.run / "rounds" / "best_programs.json"
            if not reg.is_file():
                shutil.copy(self.seed_file, reg)
                self.log(f"seed program: rounds/best_programs.json initialised from {self.seed_file} "
                         f"(score {float(ent['score']):.6f}, {len(ent['program'])} chars, source {ent.get('source', '?')})")
        seed_mark = self.run / "SEED_PROGRAMS_FILE"
        seed_value = str(self.seed_file) if self.seed_file else "task_initial"
        if not seed_mark.is_file():
            seed_mark.write_text(seed_value + "\n")
        elif seed_mark.read_text().strip() != seed_value:
            raise SystemExit(f"{seed_mark} holds {seed_mark.read_text().strip()!r} but the config says {seed_value!r}; "
                             f"a run dir is bound to one seed program")
        self.baseline_json: Optional[Path] = Path(self.cfg.baseline_json).resolve() if self.cfg.baseline_json else None
        self.load_progress()
        self.log(f"{self.cfg.job_prefix} {self.cfg.name}: run dir {self.run} | tree {self.sah_root} "
                 f"(commit {git(self.sah_root, 'rev-parse', '--short', 'HEAD')}) | task {self.cfg.task_id} adv {self.cfg.adv} "
                 f"K {self.cfg.k} max_round {self.cfg.max_round} | round {self.cfg.round_gpus} GPUs, train {self.cfg.train_gpus} GPUs"
                 f" | model {self.env['SAH_MODEL_PATH']} tp {self.cfg.vllm_tp} max_model_len {self.cfg.max_model_len}")

        self.preflight_submission()

    # ---- paths & state -------------------------------------------------------------------------------------------------
    def apply_model_override(self) -> None:
        """A lineage's model_path/served_model replace the tree's SAH_MODEL_PATH/SAH_SERVED_MODEL everywhere the runner uses them."""
        if self.cfg.model_path:
            self.env["SAH_MODEL_PATH"] = self.cfg.model_path
        if self.cfg.served_model:
            self.env["SAH_SERVED_MODEL"] = self.cfg.served_model

    def rd(self, n: int) -> Path:
        return self.run / "rounds" / f"round{n:03d}"

    def td(self, n: int) -> Path:
        return self.run / "training" / f"round{n:03d}"

    def collected(self, n: int) -> bool:
        return (self.rd(n) / "next_bases.json").is_file() and (self.rd(n) / "round_summary.json").is_file()

    def last_collected(self) -> int:
        n = 0
        for p in (self.run / "rounds").glob("round[0-9][0-9][0-9]"):
            m = int(p.name[5:])
            if self.collected(m):
                n = max(n, m)
        return n

    def train_state(self, n: int) -> Optional[Dict[str, Any]]:
        p = self.td(n) / "model_state.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def stop_requested(self) -> bool:
        return (self.run / "ABL_STOP").exists() or (self.camp / "ABL_STOP").exists()

    # ---- slurm ---------------------------------------------------------------------------------------------------------
    def _slurm(self, name: str) -> str:
        return str(Path(self.cfg.slurm_bin) / name) if self.cfg.slurm_bin else name

    def bad_nodes(self) -> List[str]:
        p = self.run / "BAD_NODES"
        return [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []

    _NODE_FAILURE_RE = re.compile(r"pyxis: container start failed|No space left on device|couldn't start container|"
                                  r"Failed to invoke spank plugin stack|task_init\(\) failed", re.I)

    def fatal_job_outcome(self, kind: str, n: int, jid: str, final: str, err_path: Path, *, retry_key: str,
                          limit: str, wall_marker: Optional[Path] = None) -> None:
        """Stop the runner (SystemExit) when resubmitting the same script could not help: the job script failed before
        srun ('[sbatch] ...' prologue message), or the job ran into its wall time / memory (sacct TIMEOUT/OUT_OF_MEMORY/
        DEADLINE, the worker's WALL_LIMIT marker written on Slurm's TERM signal, or an elapsed time at the limit).  Such an
        attempt is refunded: it is a configuration problem, not an infrastructure retry."""
        _tail, marker = err_tail(err_path)
        label = f"{kind} {n}" if kind != "baseline" else "baseline"
        if marker:
            self.retries(retry_key, delta=-1)
            self.set_phase(f"{label}: configuration error", f"job {jid}")
            raise SystemExit(f"{label}: job {jid} failed before srun: {marker} (see {err_path}); configuration error, not retried -- "
                             f"fix config/local.env or the tree and re-run (the runner resumes)")
        state = final.split()[0].split("+")[0] if final else ""
        limit_s = wall_seconds(limit)
        elapsed = self.job_elapsed_seconds(jid)
        at_limit = (wall_marker is not None and wall_marker.is_file()) or (elapsed is not None and elapsed >= limit_s - 1000)
        if state in RESOURCE_STATES or at_limit:
            knobs = {"round": "round_time / round_mem (or lower k, max_evals)", "train": "train_time / train_mem (or lower token_budget, k)",
                     "baseline": "baseline_time / round_mem"}[kind]
            why = state if state in RESOURCE_STATES else f"the wall limit {limit} (the worker was signalled before the limit)"
            self.retries(retry_key, delta=-1)
            self.set_phase(f"{label}: wall limit / resources", f"job {jid}")
            raise SystemExit(f"{label}: job {jid} ended at {why}; resubmitting the same script would spend the same GPU hours "
                             f"again -- raise {knobs} in the pipeline config and re-run (the runner resumes)")

    def node_failure(self, jid: str) -> Optional[str]:
        """A job that died at container start (full node-local disk, enroot/pyxis error) is a node problem, not a round
        problem: record the node in <run>/BAD_NODES so later submissions exclude it and let the caller refund the retry."""
        errs = [p for p in (self.run / "out").glob(f"*-{jid}.err") if p.is_file()]
        text = "\n".join(p.read_text(errors="replace")[-20000:] for p in errs)
        if not text or not self._NODE_FAILURE_RE.search(text):
            return None
        try:
            out = self._slurm_run(["sacct", "-n", "-X", "-j", jid, "-o", "NodeList"])
            node = (out.stdout.split() or ["?"])[0]
        except SlurmUnavailable:
            node = "?"
        if node not in ("?", "None") and node not in self.bad_nodes():
            with open(self.run / "BAD_NODES", "a") as fh:
                fh.write(node + "\n")
        return node

    def _slurm_run(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a Slurm client (cmd[0] = squeue/sacct/sbatch); an outage answer raises SlurmUnavailable (OUTAGE-001)."""
        argv = [self._slurm(cmd[0]), *cmd[1:]]
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=_SLURM_CALL_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise SlurmUnavailable(f"{cmd[0]} gave no answer within {_SLURM_CALL_TIMEOUT:.0f}s")
        except OSError as e:
            raise SystemExit(f"Slurm client {argv[0]} cannot be run ({e}); is Slurm installed on this host (or slurm_bin set correctly)?") from None
        if _SLURM_DOWN_RE.search(out.stderr) and (out.returncode != 0 or not out.stdout.strip()):
            raise SlurmUnavailable(f"{cmd[0]}: {out.stderr.strip().splitlines()[-1][:200]}")
        return out

    def queued_job_named(self, name: str) -> Optional[str]:
        """Id of the user's queued/running job with this exact name (a submission whose reply was lost is still real)."""
        out = self._slurm_run(["squeue", "-u", _user(), "-h", "-o", "%i|%j|%T"])
        if out.returncode != 0:
            raise SlurmUnavailable(f"squeue -u: {out.stderr.strip()[:200]}")
        for line in out.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 3 and parts[1] == name and parts[2] in ACTIVE:
                return parts[0]
        return None

    def submit(self, script: Path) -> str:
        m = re.search(r"^#SBATCH --job-name=(\S+)", script.read_text(), re.M)
        name = m.group(1) if m else ""
        cmd = ["sbatch", "--parsable"]
        if self.bad_nodes():
            cmd.append("--exclude=" + ",".join(self.bad_nodes()))
        cmd.append(str(script))
        said = False
        while True:
            try:
                # OUTAGE-001: sbatch may have created the job although its reply never arrived; never submit a twin
                jid = self.queued_job_named(name) if name else None
                if jid is not None:
                    self.log(f"submit: {name} is already queued as job {jid}; adopting it instead of submitting again")
                else:
                    out = self._slurm_run(cmd)
                    if out.returncode != 0:
                        raise SystemExit(f"sbatch rejected {script.name} ({out.returncode}): {out.stderr.strip()[-300:]} -- check the "
                                         f"account/partition/QOS and the GPU/CPU/memory layout in the pipeline config, then re-run")
                    jid = out.stdout.strip().split(";")[0]
            except SlurmUnavailable as e:
                if not said:
                    self.log(f"submit {name}: Slurm not answering ({e}); retrying every {max(2 * self.cfg.poll_seconds, 1.0):.0f}s")
                    said = True
                self.set_phase("slurm unavailable", f"submit {name}")
                time.sleep(max(2 * self.cfg.poll_seconds, 1.0))
                continue
            if said:
                self.log(f"submit {name}: Slurm answers again")
            (self.run / "JOB_ID").write_text(jid + "\n")
            return jid

    def squeue_job(self, jid: str) -> Optional[Dict[str, str]]:
        out = self._slurm_run(["squeue", "-h", "-j", jid, "-o", "%T|%M|%N|%r|%j"])
        line = out.stdout.strip().splitlines()
        if out.returncode != 0 or not line:
            return None   # not in the queue: finished and aged out ("Invalid job id specified") or never submitted
        parts = (line[0].split("|") + [""] * 5)[:5]
        return {"state": parts[0], "elapsed": parts[1], "node": parts[2], "reason": parts[3], "name": parts[4]}

    def job_elapsed_seconds(self, jid: str) -> Optional[int]:
        """Wall time of a finished job from sacct ('[D-]HH:MM:SS'); None if sacct has no answer."""
        try:
            out = self._slurm_run(["sacct", "-n", "-X", "-j", jid, "-o", "Elapsed"])
            txt = (out.stdout.split() or [""])[0]
            days, _, hms = txt.rpartition("-")
            h, m, sec = (int(x) for x in hms.split(":"))
            return (int(days) * 86400 if days else 0) + h * 3600 + m * 60 + sec
        except (OSError, ValueError, SlurmUnavailable):
            return None

    def sacct_state(self, jid: str) -> str:
        for _ in range(6):
            out = self._slurm_run(["sacct", "-n", "-X", "-j", jid, "-o", "State"])
            tok = out.stdout.split()
            if tok:
                return tok[0]
            time.sleep(min(5.0, self.cfg.poll_seconds))
        return "UNKNOWN"

    def user_load(self) -> Tuple[int, int]:
        out = self._slurm_run(["squeue", "-u", _user(), "-h", "-o", "%b"])
        if out.returncode != 0:
            raise SlurmUnavailable(f"squeue -u: {out.stderr.strip()[:200]}")   # an error is not an empty queue
        jobs, gpus = 0, 0
        for line in out.stdout.splitlines():
            if line.strip():
                jobs += 1
                gpus += gpus_in_gres(line)
        return jobs, gpus

    def caps(self) -> Tuple[int, int]:
        cap_file = self.camp / "GPU_CAP"
        gpu_cap = int(cap_file.read_text().strip()) if cap_file.is_file() else \
            int(self.cfg.gpu_cap if self.cfg.gpu_cap is not None else (self.env.get("HARNESSTTT_GPU_CAP") or 48))
        max_jobs = int(self.cfg.max_jobs if self.cfg.max_jobs is not None else (self.env.get("HARNESSTTT_MAX_JOBS") or 15))
        return gpu_cap, max_jobs

    def earlier_waiter(self, since: float) -> Optional[str]:
        """Name of a sibling lineage (same campaign) whose live runner has been waiting for capacity longer than `since`.
        FAIRSHARE-001: two runners sharing the 48-GPU budget poll independently, and the one whose job has
        just ended re-takes the freed GPUs within one poll, so the other starved for hours (pruned lineage, 08:33-11:38).
        A runner records when it started waiting (<run>/WAITING_SINCE); whoever waited longest gets the next slot."""
        for sibling in sorted(self.camp.iterdir()) if self.camp.is_dir() else []:
            mark = sibling / "WAITING_SINCE"
            if sibling == self.run or not mark.is_file():
                continue
            try:
                ts = float(mark.read_text().split()[0])
                pid = int((sibling / "pipeline.pid").read_text().strip())
                os.kill(pid, 0)   # ESRCH -> that runner is gone; its mark is stale
            except (OSError, ValueError, IndexError):
                continue
            if ts < since:
                return sibling.name
        return None

    def wait_for_capacity(self, need: int, what: str) -> None:
        said = yielded = False
        mark = self.run / "WAITING_SINCE"
        since = time.time()
        mark.write_text(f"{since:.3f} {what}\n")
        try:
            while True:
                gpu_cap, max_jobs = self.caps()
                try:
                    jobs, gpus = self.user_load()
                except SlurmUnavailable as e:
                    if not said:
                        self.log(f"{what}: Slurm not answering ({e}); waiting before the capacity check")
                        said = True
                    self.set_phase("slurm unavailable", f"capacity check for {what}")
                    time.sleep(max(2 * self.cfg.poll_seconds, 1.0))
                    continue
                if jobs < max_jobs and gpus + need <= gpu_cap:
                    other = self.earlier_waiter(since)
                    if other:
                        if not yielded:
                            self.log(f"{what}: capacity for {need} GPUs is free but {other} has waited longer; yielding (FAIRSHARE-001)")
                            yielded = True
                        self.set_phase("yielding capacity", f"to {other}")
                        time.sleep(max(self.cfg.poll_seconds, 1.0))
                        continue
                    if said or yielded:
                        self.log(f"{what}: capacity available again (jobs {jobs}, gpus {gpus} + {need} <= {gpu_cap})")
                    return
                if not said:
                    self.log(f"{what}: waiting for capacity (jobs {jobs}/{max_jobs}, gpus {gpus} + {need} > cap {gpu_cap})")
                    said = True
                self.set_phase("waiting for capacity", f"jobs {jobs}/{max_jobs} gpus {gpus}+{need}/{gpu_cap}")
                time.sleep(max(2 * self.cfg.poll_seconds, 1.0))
        finally:
            mark.unlink(missing_ok=True)

    # ---- templates -----------------------------------------------------------------------------------------------------
    def job_resources(self, gpus: int) -> Tuple[int, str]:
        """CPUs and memory of a round / baseline job: the config's round_cpus / round_mem, else 12 CPUs and 75-100 GB per GPU."""
        cpus = self.cfg.round_cpus if self.cfg.round_cpus is not None else {1: 12, 2: 24, 4: 48}.get(gpus, 12 * gpus)
        mem = self.cfg.round_mem if self.cfg.round_mem is not None else {1: "100G", 2: "200G", 4: "300G"}.get(gpus, f"{75 * gpus}G")
        return int(cpus), str(mem)

    def sbatch_lines(self) -> Dict[str, str]:
        """``#SBATCH --account/--partition/--qos`` lines, or a comment when the site leaves the value to the cluster default."""
        out: Dict[str, str] = {}
        for key, flag in (("HTT_SLURM_ACCOUNT", "account"), ("HTT_SLURM_PARTITION", "partition"), ("HTT_SLURM_QOS", "qos")):
            value = (self.env.get(key) or "").strip()
            out[f"SBATCH_{flag.upper()}"] = f"#SBATCH --{flag}={value}" if value else f"# --{flag}: cluster default ({key} not set)"
        return out

    def common_mapping(self) -> Dict[str, Any]:
        g = self.cfg.round_gpus
        cpus, mem = self.job_resources(g)
        e = self.env
        return {
            **self.sbatch_lines(), "RUN_ROOT": self.run_root,
            "RUN": self.run, "SAH_ROOT": self.sah_root, "PIPELINE_DIR": self.sah_root / "scripts" / "pipeline",
            "ENV_FILE": self.sah_root / "config" / "env.sh", "TASK_ID": self.cfg.task_id, "K": self.cfg.k, "ADV": self.cfg.adv,
            "H1_SEED": self.cfg.h1_seed, "MAX_ROUND": self.cfg.max_round, "BRIEF_DIR": self.brief_dir, "GPUS": g, "CPUS": cpus,
            "MEM": mem, "TRAIN_GPUS": self.cfg.train_gpus, "TRAIN_CPUS": self.cfg.train_cpus, "TRAIN_MEM": self.cfg.train_mem,
            "TOKEN_BUDGET": self.cfg.token_budget, "LR": self.cfg.lr,
            "KL": self.cfg.kl, "ACCOUNT": e.get("HTT_SLURM_ACCOUNT", ""), "PARTITION": e.get("HTT_SLURM_PARTITION", ""), "QOS": e.get("HTT_SLURM_QOS", ""),
            "CONTAINER_MOUNTS": e["CONTAINER_MOUNTS"], "OVERCAP": 1 if self.cfg.over_cap_as_invalid else 0,
            "MAX_EVALS": self.cfg.max_evals, "EVAL_TIMEOUT": self.cfg.eval_timeout,
            "EPILOGUE": "# driven by src/pipeline.py (no job-side chain)",
            "BASELINE_JSON": self.baseline_json or "", "EVAL_GPUS": self.cfg.eval_gpus,
            "MODEL_PATH": e["SAH_MODEL_PATH"], "SERVED_MODEL": e["SAH_SERVED_MODEL"], "VLLM_TP": self.cfg.vllm_tp,
            "MAX_MODEL_LEN": self.cfg.max_model_len,
            **self.round_timeouts(),
        }

    ROUND_FIXED_OVERHEAD_S = 1200   # measured: ~7 min vLLM start-up + ~2.5 min pool switch + <=7 min collect
    ROUND_SIGNAL_WINDOW_S = 900     # round.sbatch --signal=TERM@900

    def round_timeouts(self) -> Dict[str, int]:
        """Worker-side wall budgets of a round job derived from round_time: after the TERM window and the fixed overhead,
        proposing gets 42 % and the rollouts 50 % of what is left (the template used to hard-code 3 h + 3 h, more than a 6 h job)."""
        secs = wall_seconds(self.cfg.round_time)
        usable = max(600, secs - self.ROUND_SIGNAL_WINDOW_S - self.ROUND_FIXED_OVERHEAD_S)
        job = max(600, secs - self.ROUND_SIGNAL_WINDOW_S)
        return {"PROPOSE_TIMEOUT": max(300, int(usable * 0.42)), "ROLLOUT_TIMEOUT": max(300, int(usable * 0.50)),
                "JOB_SECONDS": job, "MIN_ROUND_SECONDS": min(7200, job // 2)}

    def fill(self, template: str, mapping: Dict[str, Any]) -> str:
        text = (self.sah_root / "scripts" / "pipeline" / "templates" / template).read_text()
        for key, value in mapping.items():
            text = text.replace(f"@@{key}@@", str(value))
        left = sorted(set(re.findall(r"@@([A-Z0-9_]+)@@", text)))
        if left:
            raise RuntimeError(f"{template}: unfilled placeholders {left}")
        return text

    def round_script_text(self, n: int, proposer: str, bases: Optional[Path], seed_programs: Optional[Path]) -> str:
        mapping = dict(self.common_mapping(), JOBNAME=f"{self.cfg.job_prefix}-{self.cfg.name}-r{n}", ROUND=n,
                       BASES=bases or "", SEED_PROGRAMS=seed_programs or "", PROPOSER_MODEL=proposer, TIME=self.cfg.round_time)
        return self.fill("round.sbatch", mapping)

    def round_script(self, n: int, proposer: str, bases: Optional[Path], seed_programs: Optional[Path]) -> Path:
        out = self.run / "gen" / f"{self.cfg.name}_round_r{n:03d}.sbatch"
        out.write_text(self.round_script_text(n, proposer, bases, seed_programs))
        return out

    def train_script_text(self, n: int, replay: Path, prev_state: Optional[Path]) -> str:
        mapping = dict(self.common_mapping(), JOBNAME=f"{self.cfg.job_prefix}-{self.cfg.name}-train-r{n}", TAG=self.cfg.name,
                       ROUND=n, ROUND3=f"{n:03d}", REPLAY=replay, TRAIN_DIR=self.td(n), PREV_STATE=prev_state or "",
                       TIME=self.cfg.train_time)
        return self.fill("train.sbatch", mapping)

    def train_script(self, n: int, replay: Path, prev_state: Optional[Path]) -> Path:
        out = self.run / "gen" / f"{self.cfg.name}_train_r{n:03d}.sbatch"
        out.write_text(self.train_script_text(n, replay, prev_state))
        return out

    def preflight_submission(self) -> None:
        """``sbatch --test-only`` on the round-1 script, the training script (when the loop trains) and the baseline script
        (when a seed swap needs one): an unsatisfiable layout, account, partition or QOS is reported now, not after round 1
        ran for an hour (NODECAP-001).  Answers about the cluster's current state (nodes down, limits reached) only warn."""
        probes = [("round", self.round_script_text(1, self.env["SAH_MODEL_PATH"], None, None))]
        if self.cfg.max_round > 1:
            probes.append(("train", self.train_script_text(1, self.td(1) / "replay.jsonl", None)))
        if self.baseline_json and not self.baseline_ready():
            probes.append(("baseline", self.baseline_script_text()))
        for kind, text in probes:
            path = self.run / "gen" / f"_preflight_{kind}.sbatch"
            path.write_text(text)
            try:
                out = self._slurm_run(["sbatch", "--test-only", str(path)])
            except SlurmUnavailable as e:
                self.log(f"preflight: Slurm not answering ({e}); skipping the --test-only check of the {kind} job")
                return
            finally:
                if path.exists():
                    path.unlink()
            if out.returncode != 0:
                text_out = (out.stderr.strip() or out.stdout.strip() or "sbatch --test-only failed").splitlines()[-1][:300]
                if _TEST_ONLY_TRANSIENT_RE.search(text_out):
                    self.log(f"preflight: the {kind} job cannot start right now ({text_out}); if this is a QOS/association limit on the "
                             f"job's size or time, lower round_time/train_time or the GPU counts -- otherwise the submission waits")
                    continue
                raise SystemExit(f"Slurm rejects the {kind} job as configured: {text_out}\n  adjust round_gpus/train_gpus, round_cpus/round_mem, "
                                 f"train_cpus/train_mem (pipeline config) or HTT_SLURM_ACCOUNT/PARTITION/QOS (config/local.env)")
        self.log("preflight: sbatch --test-only accepts the " + ", ".join(k for k, _ in probes) + " job script(s)")

    # ---- baseline job (seed-program swap) ---------------------------------------------------------------------------------
    def baseline_script_text(self) -> str:
        g = self.cfg.vllm_tp + self.cfg.eval_gpus
        cpus, mem = self.job_resources(g)
        mapping = dict(self.common_mapping(), JOBNAME=f"{self.cfg.job_prefix}-{self.cfg.name}-baseline", GPUS=g, CPUS=cpus,
                       MEM=mem, TIME=self.cfg.baseline_time, SEED_PROGRAMS=self.seed_file or "", BASELINE_OUT=self.baseline_json,
                       BASELINE_SEEDS=self.cfg.baseline_seeds, BASELINE_SECONDS=wall_seconds(self.cfg.baseline_time))
        return self.fill("baseline.sbatch", mapping)

    def baseline_script(self) -> Path:
        out = self.run / "gen" / f"{self.cfg.name}_baseline.sbatch"
        out.write_text(self.baseline_script_text())
        return out

    def baseline_progress(self) -> Tuple[str, str]:
        p = self.run / "baseline" / "progress.txt"
        if p.is_file():
            return "baseline", p.read_text(errors="replace").strip()[:160]
        return "starting", "job started, no baseline files yet"

    def baseline_ready(self) -> bool:
        if not self.baseline_json or not self.baseline_json.is_file():
            return False
        try:
            return self.cfg.task_id in json.loads(self.baseline_json.read_text()).get("baseline", {})
        except ValueError:
            return False

    def ensure_baseline(self) -> None:
        """A swapped seed program needs its own frozen-harness baseline (the round-1 base_score of the rewards and the
        ratchet): measure it once with the built-in harness before round 1, in the same container as the rollouts."""
        if not self.baseline_json or self.baseline_ready():
            return
        while True:
            if self.stop_requested():
                self.log("ABL_STOP present: not submitting the baseline job")
                self.set_phase("stopped", "ABL_STOP")
                raise SystemExit(0)
            attempt = self.retries("baseline")
            if attempt > self.cfg.max_round_retries:
                raise SystemExit(f"baseline job failed {attempt} times; see {self.run / 'baseline'}, {self.run / 'out'} and {self.run / 'RETRIES.json'}")
            g = self.cfg.vllm_tp + self.cfg.eval_gpus
            self.wait_for_capacity(g, "baseline")
            script = self.baseline_script()
            t0 = time.time()
            jid = self.submit(script)
            self.retries("baseline", bump=True)
            self.log(f"baseline: submitted job {jid} ({g} GPUs, frozen harness, {self.cfg.max_evals} evals x "
                     f"{self.cfg.baseline_seeds} seeds, seed program {self.seed_file or 'task initial'}) -> {script}")
            final = self.wait_job(jid, "baseline", 0)
            if self.baseline_ready():
                b = json.loads(self.baseline_json.read_text())["baseline"][self.cfg.task_id]
                self.log(f"baseline measured in {fmt_hms(time.time() - t0)}: seed {self.display(b['seed'])}, "
                         f"h2_best {self.display(b['h2_best'])} -> {self.baseline_json}")
                self.snapshot()
                return
            node = self.node_failure(jid)
            if node:
                self.retries("baseline", delta=-1)
                self.log(f"baseline: node-level failure on {node}; excluded, attempt not charged")
                continue
            out = self.run / "out" / f"slurm-{jid}.out"
            if out.is_file() and "Traceback (most recent call last)" in out.read_text(errors="replace"):
                self.set_phase("baseline: code error", f"job {jid}")
                raise SystemExit(f"baseline: job {jid} died with a Python traceback (see {out}); fix and re-run")
            if final.startswith("CANCELLED"):
                raise SystemExit(f"baseline: job {jid} was cancelled; re-run to continue")
            self.fatal_job_outcome("baseline", 0, jid, final, self.run / "out" / f"slurm-{jid}.err", retry_key="baseline",
                                   limit=self.cfg.baseline_time)
            self.log(f"baseline: job {jid} ended ({final}) without writing {self.baseline_json}; retrying")

    # ---- progress ------------------------------------------------------------------------------------------------------
    def set_phase(self, phase: str, detail: str = "", **extra: Any) -> None:
        self.current = {"phase": phase, "detail": detail, "updated_at": now(), **extra}
        self.snapshot()

    def snapshot(self, done: bool = False) -> None:
        payload = {
            "run": str(self.run), "tree": str(self.sah_root), "task": self.cfg.task_id, "adv": self.cfg.adv, "k": self.cfg.k,
            "max_round": self.cfg.max_round, "updated_at": now(), "done": done,
            "last_collected": self.last_collected(), "current": self.current,
            "rounds": [self.rounds_info[k] for k in sorted(self.rounds_info)],
        }
        tmp = self.run / "progress.json.tmp"
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, self.run / "progress.json")

    def load_progress(self) -> None:
        p = self.run / "progress.json"
        if p.is_file():
            try:
                for row in json.loads(p.read_text()).get("rounds", []):
                    self.rounds_info[int(row["round"])] = row
            except (ValueError, KeyError):
                pass
        for n in range(1, self.last_collected() + 1):
            self.summarize_round(n)
            st = self.train_state(n)
            if st is not None:
                self.rounds_info.setdefault(n, {"round": n}).update(
                    trained=True, weights_updated=bool(st.get("weights_updated")), skip_reason=st.get("skip_reason"),
                    model_path=st.get("model_path"))

    def display(self, score: Optional[float]) -> str:
        if score is None:
            return "-"
        kind, c, unit = DISPLAY.get(self.cfg.task_id, ("mul", 1.0, "score"))
        if kind == "mul" and c == 1.0:
            return f"{score:.4f}"
        return f"{score:.4f} ({fmt_metric(kind, c, score):.6g} {unit})"

    def round_progress(self, n: int, since: float = 0.0) -> Tuple[str, str]:
        rd = self.rd(n)
        phases = (self.run / "phases") if n == 1 else (rd / "phases")
        tid = self.cfg.task_id
        if not rd.exists():
            return "starting", ""
        if (rd / "round_summary.json").is_file():
            return "collected", ""
        if (rd / "collect.log").is_file():
            return "collect", "rewards, advantages, ratchet"
        if (rd / "h2_slot_plan.json").is_file():
            try:
                total = len(json.loads((rd / "h2_slot_plan.json").read_text()).get("slots", []))
            except ValueError:
                total = 0
            done = len(list((rd / "rollouts").glob(f"{tid}/cand*/*/summary.json"))) if (rd / "rollouts").is_dir() else 0
            return "rollouts", f"{done}/{total} executor trajectories finished"
        if (rd / "trajectory_budget.log").is_file():
            return "rollouts", "preparing executor pool"
        if (rd / "propose.log").is_file():
            cands = len(list((rd / "tasks" / tid).glob("cand*"))) if (rd / "tasks" / tid).is_dir() else 0
            return "propose", f"H1 proposing: {cands}/{self.cfg.k} candidates materialised"
        probes = sum(1 for f in phases.glob("protocol-*/validation.json") if f.stat().st_mtime >= since) if phases.is_dir() else 0
        if probes:
            replicas = max(1, (self.cfg.round_gpus - self.cfg.eval_gpus) // max(1, self.cfg.vllm_tp))
            return "warmup", f"{probes}/{replicas} replicas passed the protocol probe"
        if (rd / "PORT_BASE").is_file():
            return "vllm", f"vLLM replicas starting (port base {(rd / 'PORT_BASE').read_text().strip()})"
        return "starting", "job started, no round files yet"

    def train_progress(self, n: int) -> Tuple[str, str]:
        td = self.td(n)
        if (td / "TRAIN_COMPLETE").is_file():
            return "complete", ""
        if (td / "FAILED").is_file():
            return "failed", (td / "FAILED").read_text().strip()[:120]
        if (td / "merge.log").is_file():
            return "merge", "merging LoRA into the base checkpoint"
        if (td / "TRAINED").is_file():
            return "merge", "training finished, merging"
        if (td / "train.log").is_file():
            lines = [l for l in (td / "train.log").read_text(errors="replace").splitlines() if l.strip()]
            last = lines[-1][:100] if lines else ""
            return "training", f"train.log {len(lines)} lines | {last}"
        if (td / "ray_status.log").is_file():
            return "training", "ray head up, starting slime"
        if (td / "bound_replay.log").is_file():
            return "bounding", "bounding replay to the token budget"
        return "starting", "job started"

    def artefacts_complete(self, kind: str, n: int) -> bool:
        """The job's final artefacts exist: a round that collected, a training job that published model_state, a baseline
        table with the task's row. Each is written as the worker's last step, so together with "not queued any more" it
        proves the job finished without needing the accounting database."""
        if kind == "round":
            return self.collected(n)
        if kind == "train":
            return self.train_state(n) is not None and (self.td(n) / "TRAIN_COMPLETE").is_file()
        return self.baseline_ready()

    def wait_job(self, jid: str, kind: str, n: int) -> str:
        """Poll a job until it leaves the queue; print progress lines on change and at the heartbeat interval."""
        t0 = time.time()
        last_key, last_print = None, 0.0
        misses, blind, gone_blind = 0, 0, 0
        unknown_since: Optional[float] = None
        while True:
            try:
                info = self.squeue_job(jid)
            except SlurmUnavailable as e:
                # OUTAGE-001: no answer at all is neither "running" nor "gone"; the job keeps running on its node meanwhile
                blind += 1
                if blind == 1 or (time.time() - last_print) >= self.cfg.heartbeat_minutes * 60:
                    self.log(f"{kind} {n}: Slurm not answering ({e}); keeping the wait for job {jid} ({blind} unanswered polls)")
                    last_print = time.time()
                self.set_phase(f"{kind} {n}: slurm unavailable", f"job {jid}, {blind} unanswered polls", job={"id": jid})
                time.sleep(self.cfg.poll_seconds)
                continue
            final = ""
            if info is None:
                # A transient squeue miss is not a finished job: one empty squeue answer made the runner
                # declare a RUNNING round job gone, archive its round dir and resubmit, so two jobs ran the same round.
                # Only a terminal sacct state (or a long streak of unanswered polls) ends the wait.
                try:
                    final = self.sacct_state(jid)
                except SlurmUnavailable as e:
                    # the controller says "not queued"; only the accounting DB is down (round 13 of
                    # trimul_lb_k32 had collected, yet the runner waited for sacct). Complete artefacts settle it.
                    if self.artefacts_complete(kind, n):
                        self.log(f"{kind} {n}: job {jid} is no longer queued and its artefacts are complete; accounting DB "
                                 f"unavailable ({e}) - treating it as COMPLETED after {fmt_hms(time.time() - t0)}")
                        return "COMPLETED"
                    gone_blind += 1
                    if gone_blind >= 60:
                        self.log(f"{kind} {n}: job {jid} not queued for {gone_blind} polls, no artefacts, accounting DB still "
                                 f"unavailable; giving up the wait (UNKNOWN)")
                        return "UNKNOWN"
                    if gone_blind == 1 or (time.time() - last_print) >= self.cfg.heartbeat_minutes * 60:
                        self.log(f"{kind} {n}: job {jid} not queued, artefacts incomplete, accounting DB unavailable ({e}); waiting")
                        last_print = time.time()
                    self.set_phase(f"{kind} {n}: accounting DB unavailable", f"job {jid} not queued, {gone_blind} polls", job={"id": jid})
                    time.sleep(self.cfg.poll_seconds)
                    continue
            if blind:
                self.log(f"{kind} {n}: Slurm answers again after {blind} unanswered polls; job {jid} is "
                         f"{info['state'] if info else final}")
                blind = 0
            if info is None:
                misses += 1
                if final == "UNKNOWN":
                    # sacct answers but has no record of the job (no job accounting on this cluster, or the record is not
                    # visible): complete artefacts prove the job finished; otherwise wait a bounded time, not hours
                    if self.artefacts_complete(kind, n):
                        self.log(f"{kind} {n}: job {jid} left the queue and its artefacts are complete; the accounting database "
                                 f"has no record of it - treating it as COMPLETED after {fmt_hms(time.time() - t0)}")
                        return "COMPLETED"
                    unknown_since = unknown_since or time.time()
                    if time.time() - unknown_since >= _UNKNOWN_WAIT_SECONDS:
                        self.log(f"{kind} {n}: job {jid} not queued for {fmt_hms(time.time() - unknown_since)}, no accounting record "
                                 f"and no artefacts; ending the wait (UNKNOWN)")
                        return "UNKNOWN"
                # an sacct that works but has no record (UNKNOWN) gets far longer than a contradictory RUNNING answer
                cap = 240 if final == "UNKNOWN" else 30
                if final.split()[0].split("+")[0] in TERMINAL or final.startswith("CANCELLED") or misses >= cap:
                    self.log(f"{kind} {n}: job {jid} left the queue ({final}) after {fmt_hms(time.time() - t0)}")
                    return final
                if misses == 1:
                    self.log(f"{kind} {n}: job {jid} missing from squeue but sacct says {final}; keeping the wait")
                time.sleep(self.cfg.poll_seconds)
                continue
            misses = 0
            if kind == "round":
                phase, detail = self.round_progress(n, since=t0)
            elif kind == "baseline":
                phase, detail = self.baseline_progress()
            else:
                phase, detail = self.train_progress(n)
            # the training log grows every minute; print those ticks only at the heartbeat, not on every change
            key = (info["state"], info["reason"], phase, "" if phase == "training" else detail)
            due = (time.time() - last_print) >= self.cfg.heartbeat_minutes * 60
            if key != last_key or due:
                where = f" {info['node']}" if info["node"] else ""
                why = f" ({info['reason']})" if info["state"] == "PENDING" and info["reason"] not in ("", "None") else ""
                self.log(f"{kind} {n}: job {jid} {info['state']}{why} {info['elapsed']}{where} | {phase}" + (f": {detail}" if detail else ""))
                last_key, last_print = key, time.time()
            self.set_phase(f"{kind} {n}: {phase}", detail, job={"id": jid, "state": info["state"], "elapsed": info["elapsed"]})
            time.sleep(self.cfg.poll_seconds)

    def summarize_round(self, n: int) -> Dict[str, Any]:
        rd, tid = self.rd(n), self.cfg.task_id
        info = self.rounds_info.setdefault(n, {"round": n})
        try:
            group = json.loads((rd / "round_summary.json").read_text())["groups"][tid]
            rows = group.get("rows", [])
            valid = [r for r in rows if r.get("valid")]
            scores = [float(r["score"]) for r in valid if r.get("score") is not None]
            after = json.loads((rd / "best_programs_after.json").read_text()).get(tid, {}).get("score")
            prev = None
            if n > 1 and (self.rd(n - 1) / "best_programs_after.json").is_file():
                prev = json.loads((self.rd(n - 1) / "best_programs_after.json").read_text()).get(tid, {}).get("score")
            info.update(
                n_candidates=len(rows), n_valid=len(valid), best_k=group.get("best_k"),
                best_candidate=max(scores) if scores else None, incumbent=float(after) if after is not None else None,
                # the ratchet's own verdict; the prev/after comparison is only a fallback for rounds written before it existed
                # (for a seeded round 1 the registry already holds the seed, so "after != None" would read as accepted)
                accepted=bool(group["accepted_improvement"]) if "accepted_improvement" in group else
                    bool(after is not None and (prev is None or float(after) > float(prev) + 1e-12)),
                nonzero_advantages=sum(1 for r in rows if abs(float(r.get("advantage") or 0.0)) > 1e-6),
                over_cap=group.get("over_cap_candidates"), training_suppressed=bool(group.get("training_suppressed")),
            )
        except (OSError, ValueError, KeyError) as exc:
            info["summary_error"] = str(exc)[:200]
        return info

    def record_round_collected(self, n: int, jid: str, seconds: Optional[int]) -> Dict[str, Any]:
        info = self.summarize_round(n)
        info.update(job=jid, round_seconds=seconds, collected_at=now())
        self.log(f"round {n} collected in {fmt_hms(seconds) if seconds is not None else '?'}: valid {info.get('n_valid')}/{info.get('n_candidates')}, "
                 f"best candidate {self.display(info.get('best_candidate'))} (k={info.get('best_k')}), "
                 f"incumbent {self.display(info.get('incumbent'))} ({'accepted' if info.get('accepted') else 'kept'}), "
                 f"non-zero advantages {info.get('nonzero_advantages')}" +
                 (f", over-cap {info['over_cap']}" if info.get('over_cap') else "") +
                 (", TRAINING SUPPRESSED" if info.get('training_suppressed') else ""))
        self.snapshot()
        return info

    def record_train_complete(self, n: int, jid: str, seconds: Optional[int], st: Dict[str, Any]) -> None:
        self.rounds_info.setdefault(n, {"round": n}).update(
            trained=True, weights_updated=bool(st.get("weights_updated")), skip_reason=st.get("skip_reason"),
            model_path=st.get("model_path"), train_job=jid, train_seconds=seconds)
        verdict = "weights updated" if st.get("weights_updated") else f"skipped ({st.get('skip_reason')})"
        self.log(f"train {n} complete in {fmt_hms(seconds) if seconds is not None else '?'}: {verdict}, proposer -> {st.get('model_path')}")
        self.snapshot()

    # ---- steps ---------------------------------------------------------------------------------------------------------
    def archive_round(self, n: int, why: str) -> None:
        rd = self.rd(n)
        if rd.exists():
            dst = rd.with_name(f"{rd.name}_failed_{time.strftime('%Y%m%d-%H%M%S')}")
            os.rename(rd, dst)
            self.log(f"round {n}: archived incomplete round dir -> {dst.name} ({why})")

    def retries(self, key: str, bump: bool = False, delta: int = 0) -> int:
        p = self.run / "RETRIES.json"
        data = json.loads(p.read_text()) if p.is_file() else {}
        if bump or delta:
            data[key] = max(0, int(data.get(key, 0)) + (1 if bump else 0) + delta)
            p.write_text(json.dumps(data, indent=1))
        return int(data.get(key, 0))

    def do_round(self, n: int, proposer: str, bases: Optional[Path], seed_programs: Optional[Path]) -> None:
        while True:
            if self.stop_requested():
                self.log(f"ABL_STOP present: not submitting round {n}")
                self.set_phase("stopped", "ABL_STOP")
                raise SystemExit(0)
            if self.rd(n).exists() and not self.collected(n):
                self.archive_round(n, "previous attempt did not collect")
            attempt = self.retries(f"round{n}")
            if attempt > self.cfg.max_round_retries:
                raise SystemExit(f"round {n} failed {attempt} times; see {self.run / 'out'} (job logs) and {self.run / 'RETRIES.json'}")
            self.wait_for_capacity(self.cfg.round_gpus, f"round {n}")
            script = self.round_script(n, proposer, bases, seed_programs)
            t0 = time.time()
            jid = self.submit(script)
            self.retries(f"round{n}", bump=True)
            self.log(f"round {n}: submitted job {jid} ({self.cfg.round_gpus} GPUs, proposer = "
                     f"{'frozen base' if proposer == self.env.get('SAH_MODEL_PATH') else proposer}) -> {script}")
            final = self.wait_job(jid, "round", n)
            if final == "UNKNOWN" and not self.collected(n):
                # m-5: no accounting record and no artefacts yet -- the worker may still be running under the same name
                # (squeue answered with an error, not with "gone"); re-attach instead of archiving a live round
                name = re.search(r"^#SBATCH --job-name=(\S+)", script.read_text(), re.M)
                try:
                    twin = self.queued_job_named(name.group(1)) if name else None
                except SlurmUnavailable:
                    twin = None
                if twin:
                    self.log(f"round {n}: job {twin} named {name.group(1)} is still queued/running; re-attaching instead of resubmitting")
                    (self.run / "JOB_ID").write_text(twin + "\n")
                    final = self.wait_job(twin, "round", n)
                    jid = twin
            if self.collected(n):
                self.record_round_collected(n, jid, int(time.time() - t0))
                return
            tail, traceback = "", False
            out = self.run / "out" / f"slurm-{jid}.out"
            if out.is_file():
                text = out.read_text(errors="replace")
                traceback = "Traceback (most recent call last)" in text
                tail = " | last log: " + " / ".join(text.splitlines()[-2:])[:300]
            err_text, _marker = err_tail(self.run / "out" / f"slurm-{jid}.err")
            if err_text:
                tail += " | stderr: " + err_text
            infra = self.run / "INFRA_FAILURES.log"
            if infra.is_file():
                tail += " | infra: " + infra.read_text().splitlines()[-1][:200]
            self.log(f"round {n}: job {jid} ended ({final}) without collecting{tail}")
            self.fatal_job_outcome("round", n, jid, final, self.run / "out" / f"slurm-{jid}.err", retry_key=f"round{n}",
                                   limit=self.cfg.round_time, wall_marker=self.rd(n) / "WALL_LIMIT")
            node = self.node_failure(jid)
            if node:
                self.retries(f"round{n}", delta=-1)
                self.log(f"round {n}: node-level failure on {node} (container start / node-local disk); the node is excluded "
                         f"from later submissions and this attempt is not charged to the retry budget")
                continue
            if traceback:
                self.set_phase(f"round {n}: code error", f"job {jid}")
                raise SystemExit(f"round {n}: the job died with a Python traceback (see {out}); this is a code/config error, "
                                 f"not an infrastructure failure -- fix it and re-run (the runner resumes)")
            if final.startswith("CANCELLED"):
                self.set_phase(f"round {n}: cancelled", f"job {jid}")
                raise SystemExit(f"round {n}: job {jid} was cancelled; not resubmitting automatically -- re-run to continue")

    def after_round(self, n: int) -> str:
        """Beam parents + training replay for round n. Returns 'train' or 'skip'."""
        rd, td = self.rd(n), self.td(n)
        td.mkdir(parents=True, exist_ok=True)
        pipeline_dir = self.sah_root / "scripts" / "pipeline"
        env = dict(os.environ, SAH9B_BASES_CALLER="chain")
        r = subprocess.run([sys.executable, str(pipeline_dir / "build_bases.py"), str(self.run), str(n), "2"],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0 or not (rd / "next_bases_beam2.json").is_file():
            raise SystemExit(f"build_bases failed for round {n}: {r.stdout[-400:]} {r.stderr[-400:]}")
        beam = json.loads((rd / "next_bases_beam2.json").read_text()).get(self.cfg.task_id, {}).get("beam") or []
        env = dict(os.environ, SAH_REPLAY_UNKNOWN_TOOL_CALLS="keep",
                   PYTHONPATH=os.pathsep.join(p for p in (str(self.sah_root / "src"), os.environ.get("PYTHONPATH", "")) if p))
        replay = td / "replay.jsonl"
        r = subprocess.run([sys.executable, str(self.sah_root / "src" / "trainer" / "grpo_to_replay.py"),
                            "--rounds", str(rd), "--out", str(replay)], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise SystemExit(f"grpo_to_replay failed for round {n}: {r.stderr[-600:]}")
        rows = sum(1 for line in replay.read_text().splitlines() if line.strip()) if replay.is_file() else 0
        self.log(f"round {n}: beam parents built ({len(beam)}) | training replay {rows} rows")
        if rows < 4:
            prev = self.train_state(n - 1) if n > 1 else None
            state = {"schema": 2, "tag": self.cfg.name, "round": n,
                     "model_path": (prev or {}).get("model_path") or self.env["SAH_MODEL_PATH"],
                     "adapter_path": (prev or {}).get("adapter_path"), "weights_updated": False,
                     "skip_reason": f"insufficient_rows_{rows}", "published_at": now()}
            (td / "model_state.json").write_text(json.dumps(state, indent=2) + "\n")
            (td / "TRAIN_COMPLETE").write_text(f"completed_utc={now()}\nupdated=false\nreason={state['skip_reason']}\n")
            self.rounds_info.setdefault(n, {"round": n}).update(trained=True, weights_updated=False,
                                                                 skip_reason=state["skip_reason"], model_path=state["model_path"])
            self.log(f"round {n}: only {rows} trainable rows -> training skipped, proposer unchanged")
            self.snapshot()
            return "skip"
        return "train"

    def do_train(self, n: int) -> None:
        td = self.td(n)
        replay = td / "replay.jsonl"
        prev = self.td(n - 1) / "model_state.json" if n > 1 else None
        if prev is not None and not prev.is_file():
            prev = None
        while True:
            if self.stop_requested():
                self.log(f"ABL_STOP present: not submitting training for round {n}")
                self.set_phase("stopped", "ABL_STOP")
                raise SystemExit(0)
            attempt = self.retries(f"train{n}")
            if attempt > self.cfg.max_train_retries:
                raise SystemExit(f"training for round {n} failed {attempt} times; see {td}, {self.run / 'out'} and {self.run / 'RETRIES.json'}")
            self.wait_for_capacity(self.cfg.train_gpus, f"train {n}")
            script = self.train_script(n, replay, prev)
            t0 = time.time()
            jid = self.submit(script)
            self.retries(f"train{n}", bump=True)
            resume = json.loads(prev.read_text()).get("adapter_path") if prev else None
            self.log(f"train {n}: submitted job {jid} ({self.cfg.train_gpus} GPUs, resume adapter: {resume or 'none'}) -> {script}")
            final = self.wait_job(jid, "train", n)
            st = self.train_state(n)
            if st is not None and (td / "TRAIN_COMPLETE").is_file():
                self.record_train_complete(n, jid, int(time.time() - t0), st)
                return
            failed = (td / "FAILED").read_text().strip() if (td / "FAILED").is_file() else final
            err_text, _marker = err_tail(self.run / "out" / f"train-r{n:03d}-{jid}.err")
            self.log(f"train {n}: job {jid} ended without publishing model_state ({failed}){' | stderr: ' + err_text if err_text else ''}")
            self.fatal_job_outcome("train", n, jid, final, self.run / "out" / f"train-r{n:03d}-{jid}.err", retry_key=f"train{n}",
                                   limit=self.cfg.train_time)
            self.log(f"train {n}: retrying")
            node = self.node_failure(jid)
            if node:
                self.retries(f"train{n}", delta=-1)
                self.log(f"train {n}: node-level failure on {node}; node excluded from later submissions, attempt not charged")

    def attach_running_job(self) -> None:
        p = self.run / "JOB_ID"
        if not p.is_file():
            return
        jid = p.read_text().strip()
        while True:
            try:
                info = self.squeue_job(jid)
                break
            except SlurmUnavailable as e:
                # OUTAGE-001: a restarted runner must not mistake an outage for "no job running" (it would archive the
                # round dir of the still-running job and submit a twin); it waits here until Slurm answers.
                self.log(f"re-attach: Slurm not answering ({e}); waiting before deciding whether job {jid} still runs")
                self.set_phase("slurm unavailable", f"re-attach check for job {jid}")
                time.sleep(max(2 * self.cfg.poll_seconds, 1.0))
        if info is None or info["state"] not in ACTIVE:
            return
        if info["name"].endswith("-baseline"):
            # a restarted runner must not submit a second baseline job while the first one is still running
            self.log(f"re-attaching to running baseline job {jid} ({info['state']})")
            self.wait_job(jid, "baseline", 0)
            return
        m = re.search(r"-(train-)?r(\d+)$", info["name"])
        if not m:
            self.log(f"job {jid} ({info['name']}) is active but is not one of ours; not attaching")
            return
        kind, n = ("train" if m.group(1) else "round"), int(m.group(2))
        self.log(f"re-attaching to running {kind} job {jid} (round {n}, {info['state']})")
        final = self.wait_job(jid, kind, n)
        # Same bookkeeping as do_round/do_train: previously a round that finished under a re-attached runner got no
        # "collected" line and no status-table row until the next restart (trimul r4), because only do_round summarised.
        # A job that ended without its artefacts is left to run_loop, which resubmits it through the normal retry path.
        if kind == "round":
            if self.collected(n):
                self.record_round_collected(n, jid, self.job_elapsed_seconds(jid))
            else:
                self.log(f"round {n}: re-attached job {jid} ended ({final}) without collecting; the loop will resubmit it")
        else:
            st = self.train_state(n)
            if st is not None and (self.td(n) / "TRAIN_COMPLETE").is_file():
                self.record_train_complete(n, jid, self.job_elapsed_seconds(jid), st)
            else:
                self.log(f"train {n}: re-attached job {jid} ended ({final}) without publishing model_state; the loop will retry it")

    def run_loop(self) -> None:
        self.prepare()
        self.attach_running_job()
        while True:
            if self.stop_requested():
                self.log("ABL_STOP present: loop stopped between jobs")
                self.set_phase("stopped", "ABL_STOP")
                return
            n = self.last_collected()
            if n >= self.cfg.max_round:
                best = self.rounds_info.get(n, {}).get("incumbent")
                (self.run / "DONE").write_text(f"finished {now()} rounds={n}\n")
                self.log(f"DONE: {n} rounds collected (max_round {self.cfg.max_round}); final incumbent {self.display(best)}")
                self.set_phase("done", f"{n} rounds")
                self.snapshot(done=True)
                return
            if n == 0:
                self.ensure_baseline()
                self.set_phase("round 1", "submitting")
                self.do_round(1, self.env["SAH_MODEL_PATH"], None, self.seed_file)
                continue
            st = self.train_state(n)
            if st is None:
                self.set_phase(f"after round {n}", "beam + replay")
                if self.after_round(n) == "train":
                    self.do_train(n)
                continue
            model = st.get("model_path") or self.env["SAH_MODEL_PATH"]
            if not Path(model).is_dir():
                raise SystemExit(f"proposer checkpoint for round {n + 1} does not exist: {model}")
            bases = self.rd(n) / "next_bases_beam2.json"
            if not bases.is_file():
                self.after_round(n)  # rebuild the beam if the state was written by hand
            self.do_round(n + 1, model, bases, self.run / "rounds" / "best_programs.json")

    def dry_run(self) -> Path:
        self.prepare()
        n = self.last_collected()
        if n >= self.cfg.max_round:
            raise SystemExit("nothing to submit: max_round reached")
        if n == 0:
            script = self.round_script(1, self.env["SAH_MODEL_PATH"], None, self.seed_file)
        elif self.train_state(n) is None:
            script = self.train_script(n, self.td(n) / "replay.jsonl", self.td(n - 1) / "model_state.json" if n > 1 else None)
        else:
            script = self.round_script(n + 1, self.train_state(n)["model_path"], self.rd(n) / "next_bases_beam2.json",
                                       self.run / "rounds" / "best_programs.json")
        dry = script.with_suffix(".sbatch.dryrun")
        os.replace(script, dry)
        self.log(f"dry run: next script written to {dry} (not submitted)")
        return dry

    def status(self) -> None:
        p = self.run / "progress.json"
        if not p.is_file():
            print(f"no progress.json in {self.run} (run dir absent or the loop never started)")
            return
        data = json.loads(p.read_text())
        kind, c, unit = DISPLAY.get(data.get("task", ""), ("mul", 1.0, "score"))
        print(f"{self.cfg.job_prefix} {self.cfg.name}  task {data.get('task')}  adv {data.get('adv')}  K {data.get('k')}  "
              f"max_round {data.get('max_round')}  updated {data.get('updated_at')}  done={data.get('done')}")
        print(f"run {data.get('run')}")
        print(f"tree {data.get('tree')}")
        cur = data.get("current") or {}
        job = cur.get("job") or {}
        line = f"current: {cur.get('phase', '-')} {cur.get('detail', '')}"
        if job:
            line += f" | job {job.get('id')} {job.get('state')} {job.get('elapsed')}"
        print(line)

        def num(value, digits=4):
            return "-" if value is None else f"{value:.{digits}f}"

        def scaled(value):
            return "" if value is None or (kind == "mul" and c == 1.0) else f"{fmt_metric(kind, c, value):.6g}"

        def dur(value):
            return fmt_hms(value) if value else "-"

        head = ("round", "valid", "best cand", "incumbent", "" if (kind == "mul" and c == 1.0) else unit, "acc", "adv!=0", "train", "round t", "train t")
        print("")
        print("{:>5} {:>6} {:>10} {:>10} {:>10} {:>4} {:>6} {:>14} {:>8} {:>8}".format(*head))
        for row in data.get("rounds", []):
            trained = "-"
            if row.get("trained"):
                trained = "updated" if row.get("weights_updated") else "skip:" + str(row.get("skip_reason") or "")[:8]
            valid = f"{row.get('n_valid', '-')}/{row.get('n_candidates', '-')}"
            print("{:>5} {:>6} {:>10} {:>10} {:>10} {:>4} {:>6} {:>14} {:>8} {:>8}".format(
                row["round"], valid, num(row.get("best_candidate")), num(row.get("incumbent")), scaled(row.get("incumbent")),
                "yes" if row.get("accepted") else "no", str(row.get("nonzero_advantages", "-")), trained,
                dur(row.get("round_seconds")), dur(row.get("train_seconds"))))
        log = self.run / "pipeline.log"
        if log.is_file():
            print("")
            print("last log lines:")
            for line in log.read_text(errors="replace").splitlines()[-5:]:
                print("  " + line[:200])


# ----------------------------------------------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="python3 src/pipeline.py [CONFIG.yaml] [run|status|stop] [--dry-run] [overrides]")
    ap.add_argument("positional", nargs="*", metavar="CONFIG.yaml | run | status | stop",
                    help="a pipeline YAML (default config/pipeline.yaml) and/or a command (default run)")
    ap.add_argument("--config", default=None, help="pipeline YAML (same as the positional form)")
    ap.add_argument("--dry-run", action="store_true", help="write the next sbatch script without submitting")
    ap.add_argument("--quiet", action="store_true", help="log to <run>/pipeline.log only")
    for key in ("campaign", "name", "task", "adv", "run_root", "sah_root", "slurm_bin", "brief_dir", "lr", "kl", "round_time", "train_time",
                "train_mem", "round_mem", "model_path", "served_model", "seed_programs_file", "baseline_json"):
        ap.add_argument(f"--{key.replace('_', '-')}", dest=key)
    for key in ("k", "round_gpus", "train_gpus", "train_cpus", "round_cpus", "max_round", "h1_seed", "max_evals", "eval_timeout", "token_budget",
                "max_round_retries", "max_train_retries", "gpu_cap", "max_jobs", "eval_gpus", "vllm_tp", "max_model_len"):
        ap.add_argument(f"--{key.replace('_', '-')}", dest=key, type=int)
    for key in ("poll_seconds", "heartbeat_minutes"):
        ap.add_argument(f"--{key.replace('_', '-')}", dest=key, type=float)
    ap.add_argument("--no-freeze", dest="freeze", action="store_const", const=False, help="run jobs from the live tree (must be clean)")
    ap.add_argument("--allow-dirty", dest="require_clean_tree", action="store_const", const=False)
    args = ap.parse_args(argv)
    command, config = "run", args.config
    for token in args.positional:
        if token in ("run", "status", "stop"):
            command = token
        elif token.endswith((".yaml", ".yml")) or Path(token).is_file():
            if config is not None and config != token:
                ap.error(f"two configs given: {config} and {token}")
            config = token
        else:
            ap.error(f"unexpected argument {token!r}: expected a pipeline YAML or one of run|status|stop")
    args.command = command
    if config is None:
        args.config = str(MAIN_ROOT / "config" / "pipeline.yaml")
    else:
        args.config = config
        if not Path(config).is_file():
            ap.error(f"config not found: {config}")
    overrides = {f.name: getattr(args, f.name, None) for f in fields(Config)}
    cfg_path = Path(args.config) if Path(args.config).is_file() else None
    if cfg_path is None:
        print(f"[pipeline] {args.config} not found; using built-in defaults + CLI overrides", file=sys.stderr)
    cfg = Config.load(cfg_path, overrides)
    runner = Runner(cfg, quiet=args.quiet)
    if args.command == "status":
        runner.status()
        return 0
    if args.command == "stop":
        runner.run.mkdir(parents=True, exist_ok=True)
        (runner.run / "ABL_STOP").write_text(f"stop requested {now()} by pipeline.py\n")
        print(f"ABL_STOP written to {runner.run}; the loop stops between jobs (a running job finishes)")
        return 0
    try:
        if args.dry_run:
            runner.dry_run()
        else:
            runner.run_loop()
    except KeyboardInterrupt:
        runner.log("runner interrupted; a submitted job keeps running -- re-run to resume (the runner re-attaches)")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
