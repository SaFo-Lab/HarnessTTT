#!/bin/bash
# Event monitor for HarnessTTT chains: one line per state change per arm (poll 60 s); infrastructure warnings once per (job, message).
#   SAH9B_ABL_ROOT=<dir holding one run dir per arm> SAH9B_ABL_ARMS="cp26_v3 hadamard_v3" bash scripts/pipeline/monitor.sh
A="${SAH9B_ABL_ROOT:?set SAH9B_ABL_ROOT}"; ARMS="${SAH9B_ABL_ARMS:?set SAH9B_ABL_ARMS (space-separated run dir names)}"
declare -A P_STATE P_CHAIN P_WARN
while true; do
  for t in $ARMS; do
    run=$A/$t; [ -e "$run/DONE" ] && continue
    jid=$(cat "$run/JOB_ID" 2>/dev/null); [ -n "$jid" ] || continue
    st=$(squeue -h -j "$jid" -o "%T %j" 2>/dev/null | head -1); [ -z "$st" ] && st="$(sacct -n -X -j "$jid" -o State 2>/dev/null | awk 'NR==1{print $1}') (ended)"
    rd=$(ls -d "$run"/rounds/round[0-9][0-9][0-9] 2>/dev/null | tail -1); tr=$(ls -d "$run"/training/round[0-9][0-9][0-9] 2>/dev/null | tail -1)
    done_n=$(find "$rd/rollouts" -mindepth 4 -maxdepth 4 -name summary.json 2>/dev/null | wc -l); started=$(find "$rd/rollouts" -mindepth 3 -maxdepth 3 -type d 2>/dev/null | wc -l)
    trained=$( [ -s "$tr/TRAIN_COMPLETE" ] && echo yes || ( [ -s "$tr/FAILED" ] && echo FAILED || echo no ) )
    key="$st|$rd|$started|$done_n|$tr|$trained"
    if [ "${P_STATE[$t]}" != "$key" ]; then echo "$t job=$jid $st round=$(basename "$rd") rollouts=$done_n/$started training=$(basename "$tr" 2>/dev/null):$trained"; P_STATE[$t]="$key"; fi
    last=$(tail -n 1 "$run/chain.log" 2>/dev/null); if [ -n "$last" ] && [ "${P_CHAIN[$t]}" != "$last" ]; then echo "$t chain: ${last:0:220}"; P_CHAIN[$t]="$last"; fi
    f="$run/out/slurm-$jid.out"; w=$(grep -m1 -o "INFRASTRUCTURE FAILURE[^)]*)\|could not be revived\|could not bring up[^:]*\|FOREIGN server[^:]*" "$f" 2>/dev/null | head -1)
    if [ -n "$w" ] && [ "${P_WARN[$t]}" != "$jid|$w" ]; then echo "$t WARN $w (job $jid)"; P_WARN[$t]="$jid|$w"; fi
  done
  sleep 60
done
