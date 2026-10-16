"""Isolated evaluation of a candidate program against an EFT task.

Runs the task evaluator in a subprocess with a hard wall-clock timeout, so a
crashing / hanging / import-polluting candidate cannot take down the harness
Returns a normalized :class:`EvalOutcome`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from inner.tasks.eft_task import EFTTask

_WORKER = Path(__file__).with_name("_eval_worker.py")


@dataclass
class EvalOutcome:
    combined_score: float
    validity: float
    error: Optional[str]
    wall_s: float
    timed_out: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.error is None and self.validity >= 1.0


def _acquire_eval_gpu(env: Dict[str, str]):
    """Reserve one GPU of ``SAH_EVAL_GPU_POOL`` (comma-separated CUDA indices) for this evaluation.

    The round job keeps these GPUs free of vLLM replicas; concurrent rollouts must not benchmark kernels on the same GPU
    at the same time, so each evaluation holds an exclusive file lock on one pool GPU for its whole duration (the wait for
    a free GPU is not charged to the evaluator timeout).  Returns the open lock file (close it to release) or None when
    no pool is configured.  ``CUDA_VISIBLE_DEVICES`` of the evaluator subprocess is set to the reserved GPU.
    """
    pool = [p.strip() for p in os.environ.get("SAH_EVAL_GPU_POOL", "").split(",") if p.strip()]
    if not pool:
        return None
    import fcntl
    lock_dir = Path(os.environ.get("SAH_EVAL_GPU_LOCK_DIR", "/tmp"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    if os.environ.get("SAH_EVAL_LOCK_PHASE", "whole") == "benchmark":
        # BENCHLOCK-001 (2026-10-01): only the timed benchmarks need an idle GPU. Kernel compilation and the correctness tests
        # run unlocked on a pool GPU chosen per process, and the evaluator itself takes the GPU's flock around its benchmark
        # phase (shared_eval*.py honour SAH_EVAL_BENCH_LOCK_FILE). Under whole-evaluation locking an autotuning kernel held a
        # pool GPU for ~10 min of ptxas per candidate, which made K=32 rounds take days.
        gpu = pool[os.getpid() % len(pool)]
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["SAH_EVAL_BENCH_LOCK_FILE"] = str(lock_dir / f"sah_eval_gpu_{job}_{gpu}.lock")
        return None
    handles = [open(lock_dir / f"sah_eval_gpu_{job}_{g}.lock", "w") for g in pool]
    while True:  # try every pool GPU without blocking, then sleep briefly (a blocking flock on one GPU would ignore free ones)
        for gpu, fh in zip(pool, handles):
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                continue
            for other in handles:
                if other is not fh:
                    other.close()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            return fh
        time.sleep(1.0)


def evaluate_program(
    task: EFTTask,
    program: str,
    *,
    timeout_s: Optional[float] = None,
    python_exe: str = sys.executable,
    workdir: Optional[Path] = None,
    subsample: Optional[int] = None,
) -> EvalOutcome:
    """Write ``program`` to a temp file and score it with ``task``'s evaluator."""
    timeout = timeout_s if timeout_s is not None else task.per_eval_timeout_s
    if getattr(task, "python_exe", None):
        python_exe = task.python_exe   # task-pinned interpreter (see EFTTask.python_exe)
    hard_timeout = timeout + 30.0  # margin over the evaluator's own internal timeout
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="eft_eval_"))
    tmp.mkdir(parents=True, exist_ok=True)
    prog_path = tmp / "candidate.py"
    res_path = tmp / "result.json"
    req_path = tmp / "request.json"
    prog_path.write_text(program)
    req_path.write_text(json.dumps({
        "evaluator_path": str(task.evaluator_path),
        "program_path": str(prog_path),
        "shim_path": str(task.shim_path),
        "result_path": str(res_path),
        "subsample": subsample,
    }))

    env = dict(os.environ)
    env.update(getattr(task, "env", None) or {})   # task-pinned evaluator settings (manifest "env")
    # let the evaluator import its runtime shim + the candidate's deps (+ the task's extra PYTHONPATH entries)
    entries = [str(task.shim_path), str(task.task_dir)] + [str(x) for x in (getattr(task, "extra_pythonpath", None) or [])]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)

    gpu_lock = _acquire_eval_gpu(env)  # GPU kernel tasks: one evaluation at a time per reserved GPU (timings need an idle GPU)
    t0 = time.time()
    timed_out = False
    try:
        subprocess.run(
            [python_exe, str(_WORKER), str(req_path)],
            timeout=hard_timeout, env=env, cwd=str(tmp),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        if gpu_lock is not None:
            gpu_lock.close()  # releases the flock
    dt = time.time() - t0

    if timed_out:
        return EvalOutcome(0.0, 0.0, f"Timeout after {hard_timeout:.0f}s", dt, timed_out=True)
    if not res_path.exists():
        return EvalOutcome(0.0, 0.0, "worker produced no result (crash?)", dt)
    try:
        d = json.loads(res_path.read_text())
    except Exception as e:
        return EvalOutcome(0.0, 0.0, f"bad worker result: {e}", dt)
    return EvalOutcome(
        combined_score=float(d.get("combined_score", 0.0)),
        validity=float(d.get("validity", 0.0)),
        error=d.get("error"),
        wall_s=dt,
        metrics=d.get("metrics", {}),
    )
