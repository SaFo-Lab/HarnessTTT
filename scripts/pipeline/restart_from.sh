#!/bin/bash
# Clean restart of one arm from round N (lineage state = after round N-1), e.g. after an infrastructure contamination.
# Archives rounds >= N and training >= N into <run>/{rounds,training}_contaminated_<stamp>/ (never deletes), restores the registry files
# from the round-(N-1) snapshots, truncates the component ledger to rounds < N, clears the markers and re-enters the chain at
# after_train N-1 (which submits round N with the round-(N-1) merged proposer).
#   usage: SAH9B_ABL_ROOT=<root> [SAH9B_RESTART_REASON="..."] [SAH9B_LEDGER_PREFIX=<ledger.jsonl the arm was seeded from>] \
#          bash scripts/pipeline/restart_from.sh <arm> <N>
set -euo pipefail
L=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P); A="${SAH9B_ABL_ROOT:?set SAH9B_ABL_ROOT}"
tag=${1:?arm}; N=${2:?round}; run=$A/$tag; stamp=$(date +%Y%m%d-%H%M%S); prefix="${SAH9B_JOB_PREFIX:-htt}"
log(){ echo "[$(date -Is)] [$tag restart_from $N] $*" | tee -a "$run/chain.log"; }
squeue -u "${USER:-$(id -un)}" -h -o "%j" | grep -q "^${prefix}-${tag}-" && { log "a job of $tag is still running/pending: refusing"; exit 1; }
prev=$(printf 'round%03d' $((N-1)))
for f in "$run/rounds/$prev/best_programs_after.json" "$run/rounds/$prev/task_feedback_after.json" "$run/rounds/$prev/next_bases_beam2.json" "$run/training/$prev/model_state.json"; do test -s "$f" || { log "missing $f"; exit 2; }; done
"${HARNESSTTT_HOST_PYTHON:-python3}" - "$run" "$N" "$stamp" "${SAH9B_LEDGER_PREFIX:-}" "${SAH9B_RESTART_REASON:-operator restart}" <<'PY'
import json, os, re, shutil, sys
run, N, stamp, prefix_path, reason = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
moved = {"rounds": [], "training": []}
for kind in ("rounds", "training"):
    arch = f"{run}/{kind}_contaminated_{stamp}"
    for d in sorted(os.listdir(f"{run}/{kind}")):
        m = re.match(r"round(\d{3})(?:_.*)?$", d)
        if m and int(m.group(1)) >= N and os.path.isdir(f"{run}/{kind}/{d}"):
            os.makedirs(arch, exist_ok=True); os.rename(f"{run}/{kind}/{d}", f"{arch}/{d}"); moved[kind].append(d)
arch = f"{run}/rounds_contaminated_{stamp}"; os.makedirs(arch, exist_ok=True)
prev = f"round{N-1:03d}"
for name, src in (("best_programs.json", f"{run}/rounds/{prev}/best_programs_after.json"), ("task_feedback.json", f"{run}/rounds/{prev}/task_feedback_after.json")):
    dst = f"{run}/rounds/{name}"
    if os.path.isfile(dst): shutil.copy2(dst, f"{arch}/{name}.before_restart")
    shutil.copy2(src, dst)
# component ledger (audit log): keep the seeded prefix (if any) + this arm's own rows for rounds < N
led = f"{run}/rounds/component_ledger.jsonl"; rows = open(led).read().splitlines() if os.path.isfile(led) else []
prefix_rows = open(prefix_path).read().splitlines() if prefix_path else []
if prefix_rows:
    assert rows[:len(prefix_rows)] == prefix_rows, "ledger prefix differs from the seeded source"
own = [r for r in rows[len(prefix_rows):] if json.loads(r).get("round", 10**9) < N]
if rows:
    shutil.copy2(led, f"{arch}/component_ledger.jsonl.before_restart")
    open(led, "w").write("\n".join(prefix_rows + own) + ("\n" if (prefix_rows + own) else ""))
for f in ("ABL_STOP", "JOB_ID", "STALLED", "DONE"):
    p = f"{run}/{f}"
    if os.path.exists(p): os.replace(p, f"{arch}/{f}.before_restart")
json.dump({"restart_from_round": N, "stamp": stamp, "moved": moved, "registry_restored_from": prev,
           "ledger_rows": {"before": len(rows), "after": len(prefix_rows) + len(own)}, "reason": reason},
          open(f"{run}/RESTART_{stamp}.json", "w"), indent=1)
print(f"archived rounds={moved['rounds']} training={moved['training']}; registry restored from {prev}; ledger {len(rows)} -> {len(prefix_rows)+len(own)} rows")
PY
log "state reset to after $prev (archive stamp $stamp); re-entering chain at after_train $((N-1))"
exec bash "$L/chain.sh" "$run" after_train $((N-1))
