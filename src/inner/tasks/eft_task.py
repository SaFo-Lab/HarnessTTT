"""EFT held-out task registry for the inner (M0 + H2) harness.

Loads the discovery tasks of the dataset snapshot (``dataset/self_adapt_harness/``,
built by ``scripts/setup/setup_dataset.py``) into a uniform :class:`EFTTask` view the
harness can drive: seed program, evaluator entry, task prompt, the runtime
shim needed to import the evaluator, and per-task budget hints.

Two on-disk formats are unified:
  * ``source="eft"``      — OpenEvolve/skydiscover layout: ``initial_program.py`` +
    ``evaluator.py`` (or ``evaluator/evaluator.py``) + ``config.yaml`` whose
    ``prompt.system_message`` is the task spec. Needs the ``skydiscover_min`` shim.
  * ``source="simpletes"`` — the 3 held-out tasks (hadamard, ahc039, ahc058)
    that EFT references but ships from SimpleTES: ``init_program.py`` +
    ``evaluator.py`` + ``<task>.txt`` (the spec). Needs the ``simpletes_min`` shim.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Prepared dataset root (dataset/README.md).  Override with SAH_DATASET_ROOT to point at a snapshot elsewhere.
DATASET_ROOT = Path(
    os.environ.get(
        "SAH_DATASET_ROOT",
        str(Path(__file__).resolve().parents[3] / "dataset" / "self_adapt_harness"),
    )
)
EFT_SHIM = DATASET_ROOT / "runtime" / "skydiscover_min"
SIMPLETES_SHIM = DATASET_ROOT / "runtime" / "simpletes_min"


@dataclass
class EFTTask:
    task_id: str
    source: str  # "eft" | "simpletes"
    cost_tier: str
    plan_family: str
    domain: str
    task_dir: Path
    initial_program_path: Path
    evaluator_path: Path
    spec: str  # the task description shown to the model (system_message or .txt)
    diff_based: bool
    per_eval_timeout_s: float
    max_iterations_hint: int
    deps: List[str] = field(default_factory=list)
    shim_path: Path = EFT_SHIM
    language: str = "python"
    # optional interpreter for the evaluator subprocess (manifest field "python_exe"): lets a task pin the exact
    # torch/triton stack its scores must be comparable with (gpu_mode__trimul_lb: torch 2.7 / triton 3.3 like kernelbot)
    python_exe: Optional[str] = None
    # extra PYTHONPATH entries and environment variables of the evaluator subprocess (manifest fields "extra_pythonpath",
    # "env"); paths are absolute, relative to the dataset root, or hold ${VARIABLES} that the loader expands
    extra_pythonpath: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    @property
    def initial_program(self) -> str:
        return self.initial_program_path.read_text()


def _expand(value: Optional[str]) -> Optional[str]:
    """Expand ``${VAR}`` references in a manifest path; None when a referenced variable is unset (the entry is optional)."""
    if not value:
        return None
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        return None
    return expanded


def _manifest_paths(values: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for raw in values or []:
        path = _expand(raw)
        if path is None:
            continue
        if not os.path.isabs(path):
            path = str(DATASET_ROOT / path)
        out.append(path)
    return out


def _manifest_env(values: Optional[Dict[str, str]]) -> Dict[str, str]:
    return {str(k): os.path.expandvars(str(v)) for k, v in (values or {}).items()}


# --------------------------------------------------------------------------- #
def _load_eft_task(rec: Dict, default_timeout: float = 360.0) -> Optional[EFTTask]:
    task_dir = DATASET_ROOT / rec["rel_path"]
    if not task_dir.exists():
        return None
    ip = task_dir / (rec.get("initial_program") or "initial_program.py")
    ev = task_dir / rec["evaluator_rel"]
    if not ip.exists() or not ev.exists():
        return None

    spec, diff_based, timeout, max_iter = "", True, default_timeout, 100
    cfg_path = task_dir / "config.yaml"
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            spec = (cfg.get("prompt") or {}).get("system_message", "") or ""
            diff_based = bool(cfg.get("diff_based_generation", True))
            timeout = float((cfg.get("evaluator") or {}).get("timeout", default_timeout))
            max_iter = int(cfg.get("max_iterations", 100))
        except Exception:
            pass
    if rec.get("per_eval_timeout_s") is not None:   # a manifest record may override the task's config.yaml timeout
        timeout = float(rec["per_eval_timeout_s"])
    return EFTTask(
        task_id=rec["task_id"],
        source="eft",
        cost_tier=rec["cost_tier"],
        plan_family=rec["plan_family"],
        domain=rec["domain"],
        task_dir=task_dir,
        initial_program_path=ip,
        evaluator_path=ev,
        spec=spec,
        diff_based=diff_based,
        per_eval_timeout_s=timeout,
        max_iterations_hint=max_iter,
        deps=rec.get("deps", []),
        shim_path=EFT_SHIM,
        language=rec.get("language", "python"),
        python_exe=_expand(rec.get("python_exe")),
        extra_pythonpath=_manifest_paths(rec.get("extra_pythonpath")),
        env=_manifest_env(rec.get("env")),
    )


def _load_simpletes_task(cross: Dict) -> Optional[EFTTask]:
    task_dir = DATASET_ROOT / cross["use_path"]
    if not task_dir.exists():
        return None
    ip = task_dir / "init_program.py"
    ev = task_dir / "evaluator.py"
    txt = next(iter(task_dir.glob("*.txt")), None)
    if not (ip.exists() and ev.exists() and txt):
        return None
    is_ahc = "ahc" in cross["task_id"]
    # the cross-reference record may carry the same optional fields as a task record; the defaults are the
    # original three SimpleTES tasks (hadamard = construction, ahc = heuristic programming)
    return EFTTask(
        task_id=cross["task_id"],
        source="simpletes",
        cost_tier=cross.get("cost_tier") or ("tier2_heavy" if is_ahc else "tier0_cpu_nosetup"),
        plan_family=cross.get("plan_family") or ("program_optimization" if is_ahc else "combinatorial_construction"),
        domain=cross.get("domain") or ("algorithm_engineering" if is_ahc else "combinatorial_construction"),
        task_dir=task_dir,
        initial_program_path=ip,
        evaluator_path=ev,
        spec=txt.read_text(),
        diff_based=bool(cross.get("diff_based", False)),  # SimpleTES tasks are single-construction; full-rewrite is safer
        per_eval_timeout_s=float(cross.get("per_eval_timeout_s", 360.0)),
        max_iterations_hint=int(cross.get("max_iterations_hint", 100)),
        deps=list(cross.get("deps") or ["numpy", "simpletes_pkg"]),
        shim_path=SIMPLETES_SHIM,
        language=cross.get("language", "python"),
        python_exe=_expand(cross.get("python_exe")),
        extra_pythonpath=_manifest_paths(cross.get("extra_pythonpath")),
        env=_manifest_env(cross.get("env")),
    )


# ADRS System-Performance tasks (imported into raw/eft-aac2e79/benchmarks/ADRS/).
# combined_score is higher-is-better and IS the table value directly.
_ADRS_RECS = [
    dict(task_id="adrs__prism", rel_path="raw/eft-aac2e79/benchmarks/ADRS/prism",
         evaluator_rel="evaluator/evaluator.py", initial_program="initial_program.py",
         cost_tier="tier1_cpu_lightsetup", plan_family="program_optimization",
         domain="system_performance", deps=["numpy"]),
    dict(task_id="adrs__txn_scheduling", rel_path="raw/eft-aac2e79/benchmarks/ADRS/txn_scheduling",
         evaluator_rel="evaluator/evaluator.py", initial_program="initial_program.py",
         cost_tier="tier1_cpu_lightsetup", plan_family="program_optimization",
         domain="system_performance", deps=["numpy"]),
    dict(task_id="adrs__llm_sql", rel_path="raw/eft-aac2e79/benchmarks/ADRS/llm_sql",
         evaluator_rel="evaluator/evaluator.py", initial_program="initial_program.py",
         cost_tier="tier1_cpu_lightsetup", plan_family="program_optimization",
         domain="system_performance", deps=["pandas", "bundled_datasets"]),
    dict(task_id="adrs__eplb", rel_path="raw/eft-aac2e79/benchmarks/ADRS/eplb",
         evaluator_rel="evaluator/evaluator.py", initial_program="initial_program.py",
         cost_tier="tier1_cpu_lightsetup", plan_family="program_optimization",
         domain="system_performance", deps=["torch", "expert-load.json"]),
]


def load_tasks(
    *, include_simpletes_crossref: bool = True, include_adrs: bool = True,
    cost_tiers: Optional[List[str]] = None,
) -> List[EFTTask]:
    """Load the EFT held-out task suite (up to 22 tasks).

    cost_tiers: if given, keep only tasks in these tiers
    (e.g. ``["tier0_cpu_nosetup", "tier1_cpu_lightsetup"]`` for CPU-only runs).
    """
    manifest = json.loads((DATASET_ROOT / "MANIFEST_eft.json").read_text())
    tasks: List[EFTTask] = []
    for rec in manifest["tasks"]:
        if not rec.get("scorable", True):
            continue
        t = _load_eft_task(rec)
        if t is not None:
            tasks.append(t)
    if include_adrs:
        for rec in _ADRS_RECS:
            t = _load_eft_task(rec)
            if t is not None:
                t.source = "adrs"
                tasks.append(t)
    if include_simpletes_crossref:
        for cross in manifest.get("cross_referenced_tasks", []):
            t = _load_simpletes_task(cross)
            if t is not None:
                tasks.append(t)
    if cost_tiers is not None:
        tasks = [t for t in tasks if t.cost_tier in cost_tiers]
    return tasks


def get_task(task_id: str) -> EFTTask:
    for t in load_tasks():
        if t.task_id == task_id:
            return t
    raise KeyError(f"unknown task_id: {task_id}")


if __name__ == "__main__":
    for t in load_tasks():
        print(f"{t.cost_tier:22} {t.source:9} {t.task_id:38} diff={int(t.diff_based)} "
              f"spec={len(t.spec):5}B eval={t.evaluator_path.name}")
