#!/usr/bin/env python3
"""Build the parent beam for round N+1 of one lineage from round N's collect output (stdlib only, py3.9+).
usage: build_bases.py <run_dir> <N> [width=2]   -> writes <run_dir>/rounds/roundNNN/next_bases_beam2.json
BEAM-001 rule: head = promoted/retained incumbent; then the best component-carrying candidate within tol; then by score.
The collector's published beam (CURVE_INHERIT_BEAM>1) is reused when present. Cross-lineage migration: if the sibling
lineage (A <-> B) holds a better incumbent harness, it is inserted as parent 2."""
import glob, json, os, sys
run_dir = os.path.abspath(sys.argv[1]).rstrip("/"); N = int(sys.argv[2]); width = int(sys.argv[3]) if len(sys.argv) > 3 else 2
R, tag = os.path.dirname(run_dir), os.path.basename(run_dir); rd = f"{run_dir}/rounds/round{N:03d}"
if os.path.exists(f"{run_dir}/INJOB_STOP") and os.environ.get("SLURM_JOB_ID") and os.environ.get("SAH9B_BASES_CALLER") != "chain":
    print(f"{run_dir}/INJOB_STOP present: not building in-job bases; the sbatch self-chain will submit the next round on the new source tree"); sys.exit(3)
nbs = json.load(open(f"{rd}/next_bases.json")); groups = json.load(open(f"{rd}/round_summary.json"))["groups"]
tol = 0.02; out = {}
for tid, nb in nbs.items():
    if nb.get("beam"):
        out[tid] = nb; print(f"{tag} r{N}: collector beam reused ({len(nb['beam'])} parents)"); continue
    head = {"package": nb["package"], "score": float(nb["score"]), "from": "incumbent"}; pool = []
    for row in groups[tid]["rows"]:
        if row.get("score") is None or not row.get("valid", True): continue
        d = f"{rd}/tasks/{tid}/cand{int(row['k']):02d}"
        if d == head["package"] or not os.path.isdir(d): continue
        has_c = bool([f for f in (row.get("changed_fields") or []) if not str(f).startswith("system_prompt") and not str(f).startswith("sampling.")])
        pool.append({"package": d, "score": float(row["score"]), "from": f"round{N:03d}/cand{int(row['k']):02d}", "has_component": has_c})
    pool.sort(key=lambda e: -e["score"]); beam = [head]
    near = [e for e in pool if e["has_component"] and head["score"] - e["score"] <= tol]
    if near: beam.append(near[0])
    for e in pool:
        if len(beam) >= width: break
        if e["package"] not in {b["package"] for b in beam}: beam.append(e)
    out[tid] = dict(nb, beam=beam[:width], beam_policy={"width": width, "tolerance": tol, "rule": "head=promoted/retained; then best component-carrying candidate within tolerance; then by score", "built_by": "scripts/pipeline/build_bases.py"})
    print(f"{tag} r{N} beam:", [(b["from"], round(b["score"], 6)) for b in out[tid]["beam"]])
sib = tag[:-2] if tag.endswith("_b") else tag + "_b"
sib_rounds = [d for d in sorted(glob.glob(f"{R}/{sib}/rounds/round[0-9][0-9][0-9]")) if os.path.isfile(f"{d}/next_bases.json")]
if sib_rounds:
    sib_nb = json.load(open(f"{sib_rounds[-1]}/next_bases.json"))
    for tid, nb in out.items():
        if tid not in sib_nb: continue
        sh = {"package": sib_nb[tid]["package"], "score": float(sib_nb[tid]["score"]), "from": f"{sib}/{os.path.basename(sib_rounds[-1])} head (migration)"}
        beam = list(nb.get("beam") or [{"package": nb["package"], "score": float(nb["score"]), "from": "incumbent"}])
        if sh["score"] > beam[0]["score"] and sh["package"] not in {b["package"] for b in beam} and os.path.isdir(sh["package"]):
            beam.insert(1, sh); nb["beam"] = beam
            nb.setdefault("beam_policy", {})["migration"] = f"sibling lineage {sib} incumbent {sh['score']:.6f} > own {beam[0]['score']:.6f}; inserted as parent 2"
            print(f"{tag} r{N}: migrated sibling parent {sh['from']} score {sh['score']:.6f} (own head {beam[0]['score']:.6f}); beam now {len(beam)}")
json.dump(out, open(f"{rd}/next_bases_beam2.json", "w"), indent=1)
# --- cross-lineage PROGRAM migration: the harness beam above only moves the H2 package; the executor's starting program
#     comes from this lineage's rounds/best_programs.json. If the sibling lineage's ratchet-accepted program is strictly
#     better, adopt it (backup kept). Runs only at a round transition, so the next round's seed snapshot and the collector's
#     registry hash stay consistent. Disable with <run_dir>/NO_PROGRAM_MIGRATION.
import time, shutil
# --- eval-time guard shared by rollback + migration: never (re)adopt a program that sits at the evaluator's limit ---
def _eval_time_of(ent):
    try:
        return float(json.load(open(ent["result_path"]))["best_metrics"]["eval_time"])
    except Exception:
        return None
_LIMITS = {"eft__math__hadamard_maximal_det": 350.0}
def too_slow(ent, frac=0.8):
    et = _eval_time_of(ent)
    if et is None: return False
    lim = _LIMITS.get(str(ent.get("task_id", "")), float(os.environ.get("SAH9B_EVAL_TIMEOUT", "600")))
    return et > frac * lim
# --- operator registry rollback (explicit, logged): <run_dir>/REGISTRY_ROLLBACK.json =
#     {"task_id":..., "replace_sha256": <sha of the entry to replace>, "entry": {<full registry entry>}, "note": ...}
#     Applied only at a round transition and only while the head still is the named fragile program; marker renamed afterwards.
own_p = f"{run_dir}/rounds/best_programs.json"
rb_p = f"{run_dir}/REGISTRY_ROLLBACK.json"
if os.path.isfile(rb_p) and os.path.isfile(own_p):
    rb = json.load(open(rb_p)); own = json.load(open(own_p)); tid = rb["task_id"]; head = own.get(tid) or {}
    stamp = time.strftime("%Y%m%dT%H%M%S")
    if head.get("program_sha256") == rb["replace_sha256"]:
        shutil.copy(own_p, own_p + f".pre_rollback_{stamp}")
        new = dict(rb["entry"]); new["rolled_back_from"] = {"sha256": head.get("program_sha256"), "score": head.get("score"), "round": head.get("round"), "k": head.get("k"), "at": stamp, "note": rb.get("note")}
        own[tid] = new; json.dump(own, open(own_p + ".tmp", "w"), indent=1); os.replace(own_p + ".tmp", own_p)
        os.replace(rb_p, f"{run_dir}/REGISTRY_ROLLBACK.applied_{stamp}")
        print(f"{tag} r{N}: REGISTRY ROLLBACK applied for {tid}: {head.get('score')} (r{head.get('round')} k{head.get('k')}, sha {str(head.get('program_sha256'))[:12]}) -> {new.get('score')} (r{new.get('round')} k{new.get('k')}); note: {rb.get('note')}")
    else:
        os.replace(rb_p, f"{run_dir}/REGISTRY_ROLLBACK.skipped_{stamp}")
        print(f"{tag} r{N}: REGISTRY ROLLBACK skipped for {tid}: head sha {str(head.get('program_sha256'))[:12]} != {rb['replace_sha256'][:12]} (registry moved on)")
own_p = f"{run_dir}/rounds/best_programs.json"; sib_p = f"{R}/{sib}/rounds/best_programs.json"
if os.path.isfile(sib_p) and not os.path.exists(f"{run_dir}/NO_PROGRAM_MIGRATION"):
    own = json.load(open(own_p)) if os.path.isfile(own_p) else {}
    sibr = json.load(open(sib_p)); changed = False
    for tid, ent in sibr.items():
        try: s_sib = float(ent.get("score")); s_own = float((own.get(tid) or {}).get("score", "-inf"))
        except (TypeError, ValueError): continue
        if s_sib > s_own + 1e-12 and ent.get("program") and not too_slow(dict(ent, task_id=tid)):
            new = dict(ent); new["migrated_from"] = {"lineage": sib, "round": ent.get("round"), "k": ent.get("k"), "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "replaced_score": s_own}
            own[tid] = new; changed = True
            print(f"{tag} r{N}: PROGRAM migrated from {sib} (score {s_sib:.6f} > own {s_own:.6f}, round {ent.get('round')} k {ent.get('k')})")
        elif s_sib > s_own + 1e-12 and ent.get("program"):
            print(f"{tag} r{N}: PROGRAM migration from {sib} refused: eval_time {_eval_time_of(ent)} s exceeds 0.8 x limit (score {s_sib:.6f} > own {s_own:.6f})")
    if changed:
        if os.path.isfile(own_p): shutil.copy(own_p, own_p + f".pre_migration_{time.strftime('%Y%m%dT%H%M%S')}")
        json.dump(own, open(own_p + ".tmp", "w"), indent=1); os.replace(own_p + ".tmp", own_p)
