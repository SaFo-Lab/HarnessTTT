#!/bin/bash
# HarnessTTT job-side driver of the online-training loop (one lineage/arm = one run dir):
#   chain.sh <run_dir> after_round <N> : round N collected -> beam bases + training replay -> submit TRAIN job (8 GPUs)
#   chain.sh <run_dir> after_train <N> : model_state for round N published -> submit ROUND job N+1 (proposer = merged checkpoint)
#   chain.sh <run_dir> start 1         : submit ROUND job 1 with the frozen proposer (initial H2 package, initial programs)
# Run-dir contract: TASK (cp26|hadamard|<eft task id>), ARM (v3|v2|grpo|legacy|anchored -> SAH_ADV of the collect step), GPUS (round-job
#   GPUs, default 2), optional K (8), H1_SEED (7311); rounds/roundNNN/, rounds/best_programs.json, training/roundNNN/, out/, gen/
#   (generated sbatch files), chain.log, JOB_ID, ABL_STOP (stop marker, also honoured in the parent dir), DONE, OVERCAP001_FROM_ROUND.
# Env (config/env.sh + config/local.env are sourced first): SAH9B_RUN_ROOT (root holding GPU_CAP; default RUN_ROOT), SAH9B_SAH_ROOT (the
#   committed, clean HarnessTTT tree the jobs execute from; default HARNESSTTT_ROOT), SAH9B_ABL_ROOT (default: parent of <run_dir>),
#   SAH9B_MAX_ROUND (15), SAH9B_TOKEN_BUDGET (18000), SAH9B_TRAIN_LR (1e-5), SAH9B_TRAIN_KL (0.1), SAH9B_BRIEF_DIR (empty dir -> no
#   operator brief), SAH9B_OVER_CAP_AS_INVALID (1), HARNESSTTT_GPU_CAP (48), HARNESSTTT_MAX_JOBS (15), SAH9B_JOB_PREFIX (htt),
#   SAH9B_SBATCH (sbatch; point it at a stub for dry runs), HTT_SLURM_ACCOUNT / HTT_SLURM_PARTITION / HTT_SLURM_QOS, CONTAINER_MOUNTS.
set -euo pipefail
run=$(cd "${1:?run_dir}" && pwd -P); phase=${2:?after_round|after_train}; N=${3:?round}   # physical path (see worker_round.sh)
L=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ENV_FILE="${HARNESSTTT_ENV:-$(cd "$L/../../config" && pwd -P)/env.sh}"; source "$ENV_FILE"
R="${SAH9B_RUN_ROOT:-$RUN_ROOT}"; A="${SAH9B_ABL_ROOT:-$(dirname "$run")}"; SAH_ROOT="${SAH9B_SAH_ROOT:-$HARNESSTTT_ROOT}"
SBATCH_BIN="${SAH9B_SBATCH:-sbatch}"; JOB_PREFIX="${SAH9B_JOB_PREFIX:-htt}"; HOST_PY="${HARNESSTTT_HOST_PYTHON:-python3}"
tag=$(basename "$run"); task=$(cat "$run/TASK"); arm=$(cat "$run/ARM"); GPUS=$(cat "$run/GPUS" 2>/dev/null || echo 2); K=$(cat "$run/K" 2>/dev/null || echo 8)
H1_SEED=$(cat "$run/H1_SEED" 2>/dev/null || echo 7311); MAXR=${SAH9B_MAX_ROUND:-15}; BRIEF_DIR="${SAH9B_BRIEF_DIR:-$R/briefs_none}"
log(){ echo "[$(date -Is)] [$tag $phase $N] $*" | tee -a "$run/chain.log"; }
[ -e "$A/ABL_STOP" ] || [ -e "$run/ABL_STOP" ] && { log "ABL_STOP present: not chaining"; exit 0; }
case "$task" in cp26) TID=eft__math__circle_packing;; hadamard) TID=eft__math__hadamard_maximal_det;; ac1) TID=eft__math__first_autocorr_ineq;;
  ac2) TID=eft__math__second_autocorr_ineq;; erdos) TID=eft__math__erdos_min_overlap;; trimul) TID=gpu_mode__trimul;; trimul_lb) TID=gpu_mode__trimul_lb;;
  mla_decode) TID=gpu_mode__mla_decode;; eft__*|gpu_mode__*|adrs__*) TID=$task;; *) log "unknown task $task"; exit 2;; esac
case "$arm" in v3|v2|grpo|legacy|anchored) ADV=$arm;; *) log "unknown arm $arm"; exit 2;; esac
case "$GPUS" in 1) CPUS=12; MEM=100G;; 2) CPUS=24; MEM=200G;; 4) CPUS=48; MEM=300G;; *) CPUS=$((GPUS*12)); MEM=$((GPUS*75))G;; esac
mkdir -p "$run/out" "$run/gen" "$BRIEF_DIR"
gpu_ok(){ local n g cap; n=$(squeue -u "$USER" -h | wc -l); g=$(squeue -u "$USER" -h -o "%b" | awk -F: '{s+=$NF} END{print s+0}'); cap=$(cat "$R/GPU_CAP" 2>/dev/null || echo "${HARNESSTTT_GPU_CAP:-48}")
  [ -n "${SLURM_JOB_ID:-}" ] && n=$((n-1))   # this job is finishing
  if [ "$n" -ge "${HARNESSTTT_MAX_JOBS:-15}" ] || [ $((g + $1)) -gt "$cap" ]; then log "cap check failed: jobs=$n gpus=$g +$1 > cap $cap (resubmit by hand: bash $L/chain.sh $run $phase $N)"; return 1; fi; return 0; }
fill(){ sed -e "s|@@RUN@@|$run|g" -e "s|@@SAH_ROOT@@|$SAH_ROOT|g" -e "s|@@PIPELINE_DIR@@|$L|g" -e "s|@@ENV_FILE@@|$ENV_FILE|g" -e "s|@@TASK_ID@@|$TID|g" -e "s|@@K@@|$K|g" -e "s|@@ADV@@|$ADV|g" \
  -e "s|@@H1_SEED@@|$H1_SEED|g" -e "s|@@MAX_ROUND@@|$MAXR|g" -e "s|@@BRIEF_DIR@@|$BRIEF_DIR|g" -e "s|@@GPUS@@|$GPUS|g" -e "s|@@CPUS@@|$CPUS|g" -e "s|@@MEM@@|$MEM|g" \
  -e "s|@@TOKEN_BUDGET@@|${SAH9B_TOKEN_BUDGET:-18000}|g" -e "s|@@LR@@|${SAH9B_TRAIN_LR:-1e-5}|g" -e "s|@@KL@@|${SAH9B_TRAIN_KL:-0.1}|g" \
  -e "s|@@ACCOUNT@@|${HTT_SLURM_ACCOUNT:?}|g" -e "s|@@PARTITION@@|${HTT_SLURM_PARTITION:?}|g" -e "s|@@QOS@@|${HTT_SLURM_QOS:?}|g" -e "s|@@CONTAINER_MOUNTS@@|${CONTAINER_MOUNTS:?}|g" \
  -e "s|@@OVERCAP@@|${SAH9B_OVER_CAP_AS_INVALID:-1}|g" -e "s|@@MAX_EVALS@@|${SAH9B_MAX_EVALS:-20}|g" -e "s|@@EVAL_TIMEOUT@@|${SAH9B_EVAL_TIMEOUT:-600}|g" \
  -e "s|@@TRAIN_GPUS@@|${SAH9B_TRAIN_GPUS:-8}|g" -e "s|@@TRAIN_CPUS@@|${SAH9B_TRAIN_CPUS:-32}|g" -e "s|@@TRAIN_MEM@@|${SAH9B_TRAIN_MEM:-0}|g" \
  -e "s|@@BASELINE_JSON@@|${SAH9B_BASELINE_JSON:-}|g" -e "s|@@EVAL_GPUS@@|${SAH9B_EVAL_GPUS:-0}|g" \
       -e "s|@@MODEL_PATH@@|${SAH_MODEL_PATH}|g" -e "s|@@SERVED_MODEL@@|${SAH_SERVED_MODEL}|g" -e "s|@@VLLM_TP@@|${SAH9B_VLLM_TP:-1}|g" -e "s|@@MAX_MODEL_LEN@@|${SAH9B_MAX_MODEL_LEN:-131072}|g" "$@"; }
epilogue(){ awk -v epi="$1" '$0=="@@EPILOGUE@@"{print epi; next}{print}'; }   # last template line -> the job-side chain call
rd=$(printf '%s/rounds/round%03d' "$run" "$N")
if [ "$phase" = after_round ]; then
  test -s "$rd/next_bases.json" || { log "round $N not collected (no next_bases.json)"; exit 3; }
  [ "$N" -ge "$MAXR" ] && { log "round $N reached SAH9B_MAX_ROUND=$MAXR: chain complete"; echo "finished $(date -Is)" > "$run/DONE"; exit 0; }
  SAH9B_BASES_CALLER=chain "$HOST_PY" "$L/build_bases.py" "$run" "$N" 2 2>&1 | tail -1 | tee -a "$run/chain.log"; test -s "$rd/next_bases_beam2.json"
  T=$(printf '%s/training/round%03d' "$run" "$N"); mkdir -p "$T"
  # H1 trajectories that called a tool outside their recorded surface are the policy's own behaviour: keep them (dropping a row would
  # change the group the advantage was normalised over). Always regenerate: a refused conversion can leave a partial file behind.
  SAH_REPLAY_UNKNOWN_TOOL_CALLS=keep PYTHONPATH="$SAH_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$HOST_PY" "$SAH_ROOT/src/trainer/grpo_to_replay.py" --rounds "$rd" --out "$T/replay.jsonl" 2>&1 | tail -2 | tee -a "$run/chain.log"
  rows=$(wc -l < "$T/replay.jsonl" 2>/dev/null || echo 0)
  prev=""; [ "$N" -gt 1 ] && prev=$(printf '%s/training/round%03d/model_state.json' "$run" "$((N-1))")
  if [ "$rows" -lt 4 ]; then
    log "only $rows trainable rows in round $N: skipping training, proposer unchanged"
    "$HOST_PY" - "$T/model_state.json" "$prev" "$N" "$tag" "${SAH_MODEL_PATH:-}" <<'PY'
import json, sys, time
out, prev, n, tag, base = sys.argv[1:]
p = json.load(open(prev)) if prev else {"model_path": base, "adapter_path": None}
json.dump({"schema": 2, "tag": tag, "round": int(n), "model_path": p["model_path"], "adapter_path": p.get("adapter_path"), "weights_updated": False,
           "skip_reason": "insufficient_rows", "published_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(out, "w"), indent=2)
PY
    exec bash "$L/chain.sh" "$run" after_train "$N"
  fi
  gpu_ok "${SAH9B_TRAIN_GPUS:-8}" || exit 0
  out=$run/gen/${tag}_train_r$(printf '%03d' "$N").sbatch
  fill -e "s|@@JOBNAME@@|${JOB_PREFIX}-${tag}-train-r${N}|g" -e "s|@@TAG@@|$tag|g" -e "s|@@ROUND@@|$N|g" -e "s|@@ROUND3@@|$(printf '%03d' "$N")|g" \
       -e "s|@@REPLAY@@|$T/replay.jsonl|g" -e "s|@@TRAIN_DIR@@|$T|g" -e "s|@@PREV_STATE@@|$prev|g" -e "s|@@TIME@@|${SAH9B_TRAIN_TIME:-02:30:00}|g" "$L/templates/train.sbatch" \
    | epilogue "bash $L/chain.sh $run after_train $N >> $run/chain.log 2>&1" > "$out"
  grep -q "@@" "$out" && { log "unfilled placeholder in $out"; exit 6; }
  jid=$("$SBATCH_BIN" --parsable "$out"); echo "$jid" > "$run/JOB_ID"; log "train job for round $N -> $jid ($out)"
elif [ "$phase" = after_train ]; then
  T=$(printf '%s/training/round%03d' "$run" "$N"); test -s "$T/model_state.json" || { log "no model_state for round $N (training failed?)"; exit 4; }
  model=$("$HOST_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' "$T/model_state.json"); test -d "$model"
  NN=$((N+1)); [ "$NN" -le "$MAXR" ] || { log "chain complete at round $N"; echo "finished $(date -Is)" > "$run/DONE"; exit 0; }
  gpu_ok "$GPUS" || exit 0
  out=$run/gen/${tag}_round_r$(printf '%03d' "$NN").sbatch
  fill -e "s|@@JOBNAME@@|${JOB_PREFIX}-${tag}-r${NN}|g" -e "s|@@ROUND@@|$NN|g" -e "s|@@BASES@@|$rd/next_bases_beam2.json|g" \
       -e "s|@@SEED_PROGRAMS@@|$run/rounds/best_programs.json|g" -e "s|@@PROPOSER_MODEL@@|$model|g" -e "s|@@TIME@@|${SAH9B_ROUND_TIME:-06:00:00}|g" "$L/templates/round.sbatch" \
    | epilogue "bash $L/chain.sh $run after_round $NN >> $run/chain.log 2>&1" > "$out"
  grep -q "@@" "$out" && { log "unfilled placeholder in $out"; exit 6; }
  jid=$("$SBATCH_BIN" --parsable "$out"); echo "$jid" > "$run/JOB_ID"; log "round job $NN (proposer=$model, SAH_ADV=$ADV, over_cap_as_invalid=${SAH9B_OVER_CAP_AS_INVALID:-1}) -> $jid ($out)"
  if [ "${SAH9B_OVER_CAP_AS_INVALID:-1}" = 1 ] && [ ! -e "$run/OVERCAP001_FROM_ROUND" ]; then echo "$NN" > "$run/OVERCAP001_FROM_ROUND"; log "OVERCAP-001 applies to $tag from round $NN"; fi
elif [ "$phase" = start ]; then
  [ "$N" = 1 ] || { log "start submits round 1 only (got $N)"; exit 2; }
  [ -e "$run/rounds/round001/round.json" ] && { log "round 1 already exists in $run"; exit 0; }
  : "${SAH_MODEL_PATH:?config/local.env must set SAH_MODEL_PATH}"; test -d "$SAH_MODEL_PATH"; mkdir -p "$run/rounds" "$run/training"
  gpu_ok "$GPUS" || exit 0
  out=$run/gen/${tag}_round_r001.sbatch
  fill -e "s|@@JOBNAME@@|${JOB_PREFIX}-${tag}-r1|g" -e "s|@@ROUND@@|1|g" -e "s|@@BASES@@||g" -e "s|@@SEED_PROGRAMS@@||g" -e "s|@@PROPOSER_MODEL@@|$SAH_MODEL_PATH|g" -e "s|@@TIME@@|${SAH9B_ROUND_TIME:-06:00:00}|g" "$L/templates/round.sbatch" \
    | epilogue "bash $L/chain.sh $run after_round 1 >> $run/chain.log 2>&1" > "$out"
  grep -q "@@" "$out" && { log "unfilled placeholder in $out"; exit 6; }
  jid=$("$SBATCH_BIN" --parsable "$out"); echo "$jid" > "$run/JOB_ID"; log "round job 1 (frozen proposer=$SAH_MODEL_PATH, SAH_ADV=$ADV, K=$K, over_cap_as_invalid=${SAH9B_OVER_CAP_AS_INVALID:-1}) -> $jid ($out)"
  if [ "${SAH9B_OVER_CAP_AS_INVALID:-1}" = 1 ] && [ ! -e "$run/OVERCAP001_FROM_ROUND" ]; then echo 1 > "$run/OVERCAP001_FROM_ROUND"; fi
else log "unknown phase $phase"; exit 2; fi
