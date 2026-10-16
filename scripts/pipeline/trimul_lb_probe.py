#!/usr/bin/env python3
"""Run the GPU MODE leaderboard-protocol evaluator (shared_eval_lb) on kernel files and record the results.

    python trimul_lb_probe.py OUT.json PASSES LABEL=PROGRAM.py [LABEL=PROGRAM.py ...]

Run from the gpu_mode benchmark directory with trimul_lb + the skydiscover shim on PYTHONPATH (reference.py must be the
trimul_lb one).  Passes are interleaved across programs; every pass is a full evaluate() call (18 correctness tests,
warm-up, 7 leaderboard benchmarks with recheck).
"""
import json, os, statistics, sys, time
from pathlib import Path

import torch
import shared_eval_lb as lb

out = Path(sys.argv[1]); passes = int(sys.argv[2]); programs = [s.split("=", 1) for s in sys.argv[3:]]
try:
    import triton; triton_v = triton.__version__
except Exception:
    triton_v = "n/a"
res = {"node": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID"), "python": sys.executable, "torch": torch.__version__,
       "triton": triton_v, "gpu": torch.cuda.get_device_name(0), "protocol": lb.PROTOCOL, "popcorn_seed": lb.POPCORN_SEED, "programs": {}}
print(f"[env] python={sys.executable} torch={torch.__version__} triton={triton_v} gpu={res['gpu']}", flush=True)
for label, path in programs:
    res["programs"][label] = {"path": path, "passes": []}
for p in range(passes):
    for label, path in programs:
        t0 = time.time()
        r = lb.evaluate(path)
        row = {"pass": p, "metrics": r.metrics, "artifacts": r.artifacts, "wall_s": round(time.time() - t0, 1)}
        res["programs"][label]["passes"].append(row)
        cases = [r.artifacts.get(f"bench_{i}_mean_us") for i in range(7)]
        print(f"[pass {p}] {label}: geom_mean_us={r.metrics.get('geom_mean_us') and round(r.metrics['geom_mean_us'], 1)} "
              f"score={round(r.metrics.get('combined_score', 0), 4)} correctness={r.metrics.get('correctness')} cases_us={cases} "
              f"wall={row['wall_s']}s" + (f" ERROR={r.artifacts.get('error')}" if r.artifacts.get('error') else ""), flush=True)
        out.write_text(json.dumps(res, indent=1))
for label, r in res["programs"].items():
    vals = [x["metrics"]["geom_mean_us"] for x in r["passes"] if x["metrics"].get("geom_mean_us")]
    r["median_geom_mean_us"] = statistics.median(vals) if vals else None
    print(f"[summary] {label}: median geom_mean_us={r['median_geom_mean_us']} over {len(vals)} passes", flush=True)
out.write_text(json.dumps(res, indent=1)); print("done ->", out)
