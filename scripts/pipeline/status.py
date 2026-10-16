#!/usr/bin/env python3
"""Status table for HarnessTTT online-training arms: per arm, the best accepted program score after each round (raw evaluator
metric), training state (updates / failures) and the current Slurm job.

usage: status.py --root <dir with one run dir per arm> [--arms a b ...] [--rmax 15] [--ref label=<frozen lineage run dir> ...]
Display units and task aliases come from src/pipeline.py (DISPLAY / TASK_IDS): cp26 -> sum_radii, hadamard -> det_ratio,
kernel tasks -> µs; any other task -> raw combined_score."""
import argparse, json, os, subprocess, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from pipeline import DISPLAY as _DISPLAY, TASK_IDS as ALIAS, fmt_metric  # noqa: E402  (one alias/display table for every tool)


def _conv(kind, c):
    return lambda score: fmt_metric(kind, c, score)


DISPLAY = {tid: (_conv(kind, c), unit) for tid, (kind, c, unit) in _DISPLAY.items()}


def best_after(run, tid, conv, r):
    f = f"{run}/rounds/round{r:03d}/best_programs_after.json"
    if not os.path.isfile(f):
        return None
    e = json.load(open(f)).get(tid) or {}
    return conv(float(e["score"])) if e.get("score") is not None else None


def jobstate(jid):
    if not jid:
        return "-"
    out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%T %M"], capture_output=True, text=True).stdout.strip()
    if out:
        return out
    out = subprocess.run(["sacct", "-n", "-X", "-j", jid, "-o", "State"], capture_output=True, text=True).stdout.strip().split()
    return (out[0] if out else "?") + " (ended)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.environ.get("SAH9B_ABL_ROOT"), required=os.environ.get("SAH9B_ABL_ROOT") is None)
    ap.add_argument("--arms", nargs="*", help="run dir names (default: every subdir with a TASK file)")
    ap.add_argument("--rmax", type=int, default=15)
    ap.add_argument("--ref", nargs="*", default=[], help="label=<run dir> of a frozen-proposer lineage shown as reference")
    a = ap.parse_args()
    arms = a.arms or sorted(d for d in os.listdir(a.root) if os.path.isfile(f"{a.root}/{d}/TASK"))
    by_task = {}
    for arm in arms:
        task = open(f"{a.root}/{arm}/TASK").read().strip()
        by_task.setdefault(ALIAS.get(task, task), []).append(arm)
    refs = [r.split("=", 1) for r in a.ref]
    print(f"HarnessTTT status  {time.strftime('%Y-%m-%d %H:%M %Z')}  root={a.root}")
    for tid, tarms in by_task.items():
        conv, unit = DISPLAY.get(tid, (lambda c: c, "score"))
        print(f"\n[{tid}] {unit}; columns = best accepted program after round r")
        print("arm            " + " ".join(f"r{r:<5d}" for r in range(1, a.rmax + 1)) + "  rounds trained  job")

        def row(name, vals, trained="-", job="-"):
            cells = " ".join((f"{v:.3f} " if v is not None else "  -   ") for v in vals)
            print(f"{name:14s} {cells}  {sum(v is not None for v in vals):>2d}     {trained:<7s}  {job}")

        for label, rdir in refs:
            rt = open(f"{rdir}/TASK").read().strip() if os.path.isfile(f"{rdir}/TASK") else None
            if rt is None or ALIAS.get(rt, rt) == tid:
                row(label, [best_after(rdir, tid, conv, r) for r in range(1, a.rmax + 1)])
        for arm in tarms:
            run = f"{a.root}/{arm}"
            vals = [best_after(run, tid, conv, r) for r in range(1, a.rmax + 1)]
            tr = sorted(d for d in os.listdir(f"{run}/training") if d.startswith("round")) if os.path.isdir(f"{run}/training") else []
            done = sum(1 for d in tr if os.path.isfile(f"{run}/training/{d}/TRAIN_COMPLETE"))
            failed = sum(1 for d in tr if os.path.isfile(f"{run}/training/{d}/FAILED"))
            jid = open(f"{run}/JOB_ID").read().strip() if os.path.isfile(f"{run}/JOB_ID") else ""
            row(arm, vals, f"{done}" + (f"/F{failed}" if failed else ""), f"{jid} {jobstate(jid)}" + (" DONE" if os.path.exists(f"{run}/DONE") else ""))
        for arm in tarms:
            try:
                last = open(f"{a.root}/{arm}/chain.log").read().strip().splitlines()[-1][:160]
            except Exception:
                last = ""
            if last:
                print(f"  {arm} chain: {last}")


if __name__ == "__main__":
    main()
