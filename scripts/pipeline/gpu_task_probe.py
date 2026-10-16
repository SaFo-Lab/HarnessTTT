#!/usr/bin/env python3
"""Score candidate initial programs of the GPU kernel tasks with the tasks' own evaluators (inside the SAH container, on a GPU).

    python gpu_task_probe.py OUT.json TASK_ID=PROGRAM.py [TASK_ID=PROGRAM.py ...]

Uses inner.evaluation.evaluate_program exactly like an executor rollout (evaluator subprocess, shim on PYTHONPATH, timeout
from the task config), so the numbers are the ones the pipeline would see.  Records score, metrics, error and wall time.
"""
import json, os, sys, time
from pathlib import Path

from inner.evaluation.eval_runner import evaluate_program
from inner.tasks.eft_task import get_task

out = Path(sys.argv[1]); rows = []
for spec in sys.argv[2:]:
    tid, prog = spec.split("=", 1)
    task = get_task(tid)
    text = Path(prog).read_text()
    t0 = time.time()
    res = evaluate_program(task, text, timeout_s=task.per_eval_timeout_s, python_exe=sys.executable)
    row = {"task_id": tid, "program": prog, "combined_score": res.combined_score, "validity": res.validity, "error": res.error,
           "timed_out": res.timed_out, "wall_s": round(res.wall_s, 1), "metrics": res.metrics, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all")}
    rows.append(row)
    print(json.dumps(row), flush=True)
    out.write_text(json.dumps({"node": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"), "rows": rows}, indent=1))
print("done ->", out)
