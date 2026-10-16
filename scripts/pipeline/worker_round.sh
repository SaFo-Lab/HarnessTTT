#!/usr/bin/env bash
# HarnessTTT: one SAH outer round (propose -> fixed H2 slots -> frozen-9B rollouts -> collect) inside the SAH container.
# [v14 tri-state model-root readiness; v13 per-job vLLM port range; v12 two-phase pools (trained proposer for propose, frozen base
#  for rollouts); v11 replica liveness guard; v7 stall watchdog]
# Env contract (set by scripts/pipeline/templates/round.sbatch): SAH9B_RUN_DIR, SAH9B_TASKS, SAH9B_ROUND, SAH9B_K, SAH9B_MAX_EVALS,
#   SAH9B_PROPOSER_MODEL (trained merged checkpoint; default = frozen base), SAH9B_BASES_FILE, SAH9B_SEED_PROGRAMS_FILE,
#   SAH9B_ADV (v3|grpo|v2|...), SAH_OVER_CAP_AS_INVALID (default 1); site paths come from config/env.sh + config/local.env.
set -euo pipefail
umask 027
export GIT_OPTIONAL_LOCKS=0   # v9: nine jobs share one worktree; never block on optional index refresh locks
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_JOB_BASE_MODEL="${SAH_MODEL_PATH:-}"; _JOB_SERVED_MODEL="${SAH_SERVED_MODEL:-}"
source "${HARNESSTTT_ENV:-$PIPELINE_DIR/../../config/env.sh}"
# MODELCLOBBER-001: config/local.env used to export SAH_MODEL_PATH unconditionally, so re-sourcing env.sh here replaced
# the lineage's base model (Qwen3.8-27B) with the site default (9B) after the sbatch had exported it: in the first 27B rounds the
# executor pool, the LoRA base and every trained proposer were the 9B while the baselines ran on the 27B. The job-level value is
# restored after the source and checked against the explicit contract SAH9B_BASE_MODEL.
if [ -n "${_JOB_BASE_MODEL:-}" ]; then export SAH_MODEL_PATH="$_JOB_BASE_MODEL"; fi
if [ -n "${_JOB_SERVED_MODEL:-}" ]; then export SAH_SERVED_MODEL="$_JOB_SERVED_MODEL"; fi
if [ -n "${SAH9B_BASE_MODEL:-}" ] && [ "${SAH_MODEL_PATH:-}" != "$SAH9B_BASE_MODEL" ]; then
  echo "MODELCLOBBER-001: SAH_MODEL_PATH=${SAH_MODEL_PATH:-unset} but this job was submitted for base model $SAH9B_BASE_MODEL" >&2; exit 66
fi
echo "[$(date -Is)] base model (executor pool / LoRA base) = $SAH_MODEL_PATH (served as ${SAH_SERVED_MODEL:-?})"
export SAH_ROOT="${SAH9B_SAH_ROOT:-$HARNESSTTT_ROOT}"   # the committed, clean HarnessTTT tree this job executes from
# RUN_DIR is canonicalised (pwd -P): propose checks resolved workspace paths against the run dir it was given.
RUN_DIR="$(cd "${SAH9B_RUN_DIR:?}" && pwd -P)"; TASKS="${SAH9B_TASKS:?}"; K="${SAH9B_K:-4}"; MAX_EVALS="${SAH9B_MAX_EVALS:-5}"
N_GPUS="${SLURM_GPUS_ON_NODE:?no GPU allocation}"; PROPOSE_PAR="${SAH9B_PROPOSE_PAR:-8}"; ROLLOUT_PAR="${SAH9B_ROLLOUT_PAR:-8}"
# GPU kernel tasks: SAH9B_EVAL_GPUS GPUs are reserved for the task evaluator (never for vLLM); rollouts inherit SAH_EVAL_GPU_POOL
# and eval_runner serialises evaluations per pool GPU (a kernel benchmark needs an otherwise idle GPU for its timings).
EVAL_GPUS="${SAH9B_EVAL_GPUS:-0}"; TP="${SAH9B_VLLM_TP:-1}"; MAX_MODEL_LEN="${SAH9B_MAX_MODEL_LEN:-131072}"
# TENSORPARALLEL-001: a replica spans SAH9B_VLLM_TP contiguous GPUs (27B bf16 = 55.6 GB of weights does not leave a
# usable KV cache on one H100; TP=1 is the 9B one-GPU-per-replica layout). The evaluator GPUs come after the replica GPUs.
N_REPLICAS=$(( (N_GPUS - EVAL_GPUS) / TP ))
[ "$N_REPLICAS" -ge 1 ] && [ $(( N_REPLICAS * TP + EVAL_GPUS )) -eq "$N_GPUS" ] || { echo "GPU layout mismatch (gpus=$N_GPUS eval_gpus=$EVAL_GPUS tp=$TP)" >&2; exit 65; }
if [ "$EVAL_GPUS" -gt 0 ]; then export SAH_EVAL_GPU_POOL="$(seq -s, $((N_REPLICAS * TP)) $((N_GPUS - 1)))"; fi
replica_gpus(){ seq -s, $(( $1 * TP )) $(( $1 * TP + TP - 1 )); }
# EAGER-001 / MULTITP-001 (from the 27B campaigns): --enforce-eager only for TP=1 (a 64-layer 27B pays for it on every decode step);
# with TP>1 disable vLLM's FlashInfer fused all-reduce/RMSNorm pass, whose workspace creation races between independent TP replicas.
if [ "$TP" -gt 1 ]; then VLLM_EXTRA=(--tensor-parallel-size "$TP" --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'); else VLLM_EXTRA=(--enforce-eager); fi
# GPU-IDLE-001: rollouts are CPU-bound (evaluator) and H1/H2 decode is latency-bound, so 4 replicas per round sat
# ~90% idle. Size each replica's concurrency to the parallelism it must carry (K sessions over N_REPLICAS replicas)
# instead of adding GPUs: fewer, busier replicas.  Override with SAH9B_VLLM_MAX_SEQS.
PAR_MAX=$(( PROPOSE_PAR > ROLLOUT_PAR ? PROPOSE_PAR : ROLLOUT_PAR )); VLLM_MAX_SEQS="${SAH9B_VLLM_MAX_SEQS:-$(( (PAR_MAX + N_REPLICAS - 1) / N_REPLICAS ))}"
[ "$VLLM_MAX_SEQS" -ge 4 ] || VLLM_MAX_SEQS=4
ROUND="${SAH9B_ROUND:-1}"; ROUND_DIR="$RUN_DIR/rounds/round$(printf %03d "$ROUND")"
if [ "$ROUND" = 1 ]; then PHASE="$RUN_DIR/phases"; VLOG="$RUN_DIR/vllm"; else PHASE="$ROUND_DIR/phases"; VLOG="$ROUND_DIR/vllm"; fi
[ -e "$ROUND_DIR/round.json" ] && { echo "refusing to overwrite an existing round: $ROUND_DIR" >&2; exit 42; }
mkdir -p "$ROUND_DIR" "$PHASE" "$VLOG"
log(){ echo "[$(date -Is)] $*"; }
# --- per-job vLLM port range: vLLM binds its API socket with SO_REUSEPORT, so two jobs of this experiment on
#     one node that both used 127.0.0.1:8800/8801 silently shared the ports and the kernel split their H1 requests across
#     BOTH proposers (a job's own servers answered as few as 4 of ~138 turns). Pick a free range per job (plain bind test,
#     which fails against reuse-port sockets too) and require, for readiness, OUR pid alive AND the served model root ==
#     the model this job launched.
PORT_BASE=$("$SAH_PYTHON" - "${SLURM_JOB_ID:-0}" "$N_REPLICAS" <<'PY'
import socket, sys
job, n = int(sys.argv[1]), int(sys.argv[2])
def free(p):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", p)); return True
    except OSError:
        return False
    finally:
        s.close()
start = 20000 + (job * 37) % 30000
for base in range(start, 60000, 16):
    if all(free(base + i) for i in range(n)):
        print(base); break
else:
    sys.exit("no free port range")
PY
)
[ -n "$PORT_BASE" ] || { echo "could not find a free port range" >&2; exit 67; }
echo "$PORT_BASE" > "$ROUND_DIR/PORT_BASE"; log "vLLM port range: $PORT_BASE..$((PORT_BASE + N_REPLICAS - 1)) (job ${SLURM_JOB_ID:-?})"
# GET /v1/models must come from OUR server: the reported model root must equal the model this job launched
# 0 = OUR server (root matches); 1 = reachable and parsed but a DIFFERENT root (foreign); 2 = unreachable / not ready yet.
# (A 2 s timeout that treated "check failed but port reachable" as foreign aborted a healthy job on its slow first response.)
models_root_ok(){ "$SAH_PYTHON" - "$1" "$2" <<'PY' >/dev/null 2>&1
import json, os, sys, urllib.request
port, expected = sys.argv[1], sys.argv[2]
try:
    d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10))
    roots = [m.get("root") or "" for m in d.get("data", [])]
except Exception:
    sys.exit(2)
ok = any(r == expected or os.path.realpath(r) == os.path.realpath(expected) for r in roots)
sys.exit(0 if ok else 1)
PY
}

# --- STALL WATCHDOG (v7): self-cancel only when NOTHING is alive: heartbeat stale AND no new vLLM requests AND GPUs idle ---
#     Liveness = any rollout/round file change (heartbeat loops), OR the vLLM access logs gained chat-completion requests
#     since the previous minute (agents in long CPU evaluations still issue requests between tool calls). scancel is not
#     available inside the container, so the fallback is a hard kill of this script; the sbatch step then ends the job.
MAIN_PID=$$; HEARTBEAT="$RUN_DIR/HEARTBEAT"; STALL_MINUTES="${SAH9B_STALL_MINUTES:-45}"
beat(){ date -Is > "$HEARTBEAT"; if [ -n "${1:-}" ]; then log "$1"; fi; }
mkdir -p "$RUN_DIR"; beat "worker start"
watchdog(){
  local idle=0 util stale reqs prev_reqs=-1
  while sleep 60; do
    reqs=$(cat "$VLOG"/replica-*.log 2>/dev/null | grep -c "POST /v1/chat/completions" || true); reqs=${reqs:-0}
    if [ "$reqs" -gt "$prev_reqs" ] && [ "$prev_reqs" -ge 0 ]; then date -Is > "$HEARTBEAT"; fi
    prev_reqs=$reqs
    stale=$(( ($(date +%s) - $(stat -c %Y "$HEARTBEAT" 2>/dev/null || echo 0)) / 60 ))
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{if($1>m)m=$1} END{print m+0}')
    if [ "$stale" -ge "$STALL_MINUTES" ] && [ "${util:-0}" -lt 5 ]; then idle=$((idle+1)); else idle=0; fi
    if [ "$idle" -ge 3 ]; then
      { echo "stalled_at=$(date -Is)"; echo "heartbeat_stale_min=$stale"; echo "max_gpu_util=$util"; echo "vllm_requests=$reqs"; echo "job=$SLURM_JOB_ID"; } > "$RUN_DIR/STALLED"
      echo "[watchdog] no heartbeat/requests for ${stale} min and GPUs idle (max ${util}%) -> ending job $SLURM_JOB_ID" >&2
      command -v scancel >/dev/null 2>&1 && scancel "$SLURM_JOB_ID" || true
      kill -TERM "$MAIN_PID" 2>/dev/null || true      # fires the cleanup trap -> vLLM servers are killed
      sleep 30; kill -KILL -- -"$MAIN_PID" 2>/dev/null || true; kill -KILL "$MAIN_PID" 2>/dev/null || true; exit 1
    fi
  done
}
watchdog & WATCHDOG_PID=$!

# --- provenance: refuse a dirty source tree (same rule outer_round enforces) ---
git -C "$SAH_ROOT" rev-parse HEAD > "$RUN_DIR/SAH_COMMIT"
git -C "$SAH_ROOT" diff --quiet --no-ext-diff HEAD -- || { echo "worktree has tracked modifications" >&2; exit 65; }
test -z "$(git -C "$SAH_ROOT" ls-files --others --exclude-standard)" || { echo "worktree has untracked files" >&2; exit 65; }
log "source commit $(cat "$RUN_DIR/SAH_COMMIT") root=$SAH_ROOT round=$ROUND tasks=[$TASKS] K=$K MAX_EVALS=$MAX_EVALS gpus=$N_GPUS replicas=$N_REPLICAS tp=$TP max_model_len=$MAX_MODEL_LEN model=$SAH_MODEL_PATH eval_gpus=$EVAL_GPUS vllm_max_seqs=$VLLM_MAX_SEQS adv=${SAH9B_ADV:-v2} over_cap_as_invalid=${SAH_OVER_CAP_AS_INVALID:-1} bases=${SAH9B_BASES_FILE:-static} seed_programs=${SAH9B_SEED_PROGRAMS_FILE:-initial}"

# --- environment (mirrors the evaluator preflight + evolve smoke) ---
export PATH="$SAH_AHC_NATIVE_BIN:$PATH"
export AHC_CASE_WORKERS=8 AHC_PIN_CORES=0 AHC_DOCKER_TIMEOUT=900 SIMPLETES_WORKER_ID=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=3"   # v10: erdos r8 had 29/29 seed-eval timeouts under 29 concurrent JAX evals
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$NEXAU_ROOT:$SAH_ROOT/src:$SAH_ROOT"
export SAH_DATASET_ROOT="$SAH_DATA_ROOT"
export SAH_TASK_TEXT_REGISTRY="${SAH_TASK_TEXT_REGISTRY:-$SAH_ROOT/config/task_text_registry.json}" SAH_TASK_TEXT_ENFORCE=1
export LOGICAL_ROUND_INDEX=$((ROUND - 1)) H2_SEED_BASE="${SAH9B_H2_SEED_BASE:-200000}"
if [ "${SAH9B_FIXED_SLOTS:-1}" = 1 ]; then export SAH_FIXED_INFERENCE_SLOTS=1; FIXED_FLAG="--fixed-slots"; else export SAH_FIXED_INFERENCE_SLOTS=0; FIXED_FLAG=""; fi
export SAH_PROGRAM_RATCHET_MODE=strict_single SAH_REQUIRE_STRICT_RATCHET=1 SAH_ADV="${SAH9B_ADV:-v2}" SAH_PROPOSAL_GATE="${SAH9B_PROPOSAL_GATE:-off}"
export SAH_OVER_CAP_AS_INVALID="${SAH_OVER_CAP_AS_INVALID:-1}"   # OVERCAP-001: an over-cap program is an INVALID proposal (reward -1), not a refused round
export EVAL_TIMEOUT="${SAH9B_EVAL_TIMEOUT:-180}" ROLLOUT_WALL_TIMEOUT="${SAH9B_ROLLOUT_WALL_TIMEOUT:-2400}"
export MODEL_PATH="$SAH_MODEL_PATH" SERVED_MODEL="$SAH_SERVED_MODEL" OPENAI_API_KEY=EMPTY SAH_MAX_CONTEXT_TOKENS="$MAX_MODEL_LEN"   # CONTEXT-001: server limit == agent limit
# v12: the proposer (M_phi) may be a trained checkpoint; the executor (M0) is always the frozen base. MPHI_PATH is recorded in round.json provenance.
PROPOSER_MODEL="${SAH9B_PROPOSER_MODEL:-$SAH_MODEL_PATH}"; test -d "$PROPOSER_MODEL"; export MPHI_PATH="$PROPOSER_MODEL"
POOL_MODEL="$PROPOSER_MODEL"   # model served by the current replica pool (propose phase first)
export TMPDIR="/tmp/harnessttt-$SLURM_JOB_ID"; mkdir -p "$TMPDIR"
unset ANTHROPIC_API_KEY AZURE_API_KEY AZURE_OPENAI_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN GEMINI_API_KEY GOOGLE_API_KEY WANDB_API_KEY 2>/dev/null || true
test -x "$SAH_PYTHON"; test -x "$VLLM_ENV/bin/vllm"; test -d "$SAH_MODEL_PATH"
test "$(command -v docker)" = "$SAH_AHC_NATIVE_BIN/docker"
nvidia-smi -L | tee "$PHASE/nvidia-smi-L.txt"
"$SAH_PYTHON" -c "from nexau import Agent; import numpy, scipy, yaml; print('runtime imports ok')"


# --- v11: replica liveness guard. All vLLM EngineCore processes of two jobs on two different
#     nodes were killed externally; v10 kept looping and burned rounds 2-8 as empty rounds (proposer: "Connection refused").
#     v11 probes every replica with a 1-token completion before each round and after propose/rollouts, relaunches dead
#     replicas (GPU memory is released when the EngineCore dies), redoes the damaged round in-job (<= 2 retries per round),
#     and otherwise stops the loop instead of recording empty rounds. Damaged round dirs are renamed *_infra_failed_<stamp>.
INFRA_RE='EngineDeadError|EngineCore encountered an issue|Connection refused|APIConnectionError|Connection error\.'
declare -A INFRA_RETRIES=()
replica_alive(){ "$SAH_PYTHON" - "$((PORT_BASE + $1))" "$SERVED_MODEL" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
body = json.dumps({"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=180).read()
PY
}
start_replica(){ local g=$1 port=$((PORT_BASE + $1))
  CUDA_VISIBLE_DEVICES="$(replica_gpus "$g")" setsid env -u VLLM_ENV "$VLLM_ENV/bin/vllm" serve "$POOL_MODEL" \
    --host 127.0.0.1 --port "$port" --served-model-name "$SERVED_MODEL" \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$VLLM_MAX_SEQS" --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 "${VLLM_EXTRA[@]}" --language-model-only \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    >> "$VLOG/replica-$g.log" 2>&1 &
  VPIDS[$g]=$!; }
wait_replica_ready(){ local g=$1 port=$((PORT_BASE + $1)) ok=0 _i rc foreign=0
  for _i in $(seq 1 240); do
    kill -0 "${VPIDS[$g]}" 2>/dev/null || { tail -n 40 "$VLOG/replica-$g.log" >&2; log "replica $g died during (re)start"; return 1; }
    rc=0; models_root_ok "$port" "$POOL_MODEL" || rc=$?
    if [ "$rc" = 0 ]; then ok=1; break; fi
    if [ "$rc" = 1 ]; then foreign=$((foreign + 1)); [ "$foreign" -ge 3 ] && { log "FOREIGN server answering on :$port (model root != $POOL_MODEL, 3 consecutive checks): refusing to continue"; return 1; }; else foreign=0; fi
    sleep 5
  done
  [ "$ok" = 1 ] || { log "replica $g not ready after 20 min"; return 1; }; return 0; }
probe_replica(){ local g=$1 suffix=$2
  timeout --foreground --kill-after=30s 900s "$SAH_PYTHON" "$PIPELINE_DIR/vllm_protocol.py" \
    --base-url "http://127.0.0.1:$((PORT_BASE + g))/v1" --model "$SERVED_MODEL" --timeout 600 \
    --out "$PHASE/protocol-$g$suffix" > "$PHASE/protocol-$g$suffix.log" 2>&1 || return 1
  test "$("$SAH_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["passed"])' "$PHASE/protocol-$g$suffix/validation.json")" = "True"; }
ensure_replicas(){ local g dead=0 stamp   # 0 = every replica answers a completion (dead ones were relaunched + re-probed)
  for g in $(seq 0 $((N_REPLICAS - 1))); do
    replica_alive "$g" && continue
    stamp=$(date +%Y%m%d-%H%M%S)
    log "replica $g on :$((PORT_BASE + g)) is DEAD (pid ${VPIDS[$g]:-?}); last log lines: $(tail -n 3 "$VLOG/replica-$g.log" 2>/dev/null | cut -c1-160 | tr '\n' '|')"
    p="${VPIDS[$g]:-}"; if [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; then kill -KILL -- "-$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null || true; fi; sleep 15
    cp "$VLOG/replica-$g.log" "$VLOG/replica-$g.died-$stamp.log" 2>/dev/null || true; : > "$VLOG/replica-$g.log"
    start_replica "$g"
    if wait_replica_ready "$g" && probe_replica "$g" "-restart-$stamp"; then beat "replica $g RELAUNCHED on :$((PORT_BASE + g)) and passed the protocol probe"; else log "replica $g could not be revived"; dead=1; fi
  done
  return $dead; }
switch_pool(){ # v12: tear down the current pool and serve $1 on every GPU (protocol-probed) -- used to swap proposer -> frozen executor
  local model=$1 g p stamp; stamp=$(date +%Y%m%d-%H%M%S); log "switching replica pool to $model"
  for p in "${VPIDS[@]}"; do [ -n "$p" ] && { kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null || true; }; done; sleep 10
  for p in "${VPIDS[@]}"; do [ -n "$p" ] && { kill -KILL -- "-$p" 2>/dev/null || true; }; done; sleep 5
  POOL_MODEL="$model"
  for g in $(seq 0 $((N_REPLICAS - 1))); do cp "$VLOG/replica-$g.log" "$VLOG/replica-$g.propose-$stamp.log" 2>/dev/null || true; : > "$VLOG/replica-$g.log"; start_replica "$g"; done
  for g in $(seq 0 $((N_REPLICAS - 1))); do wait_replica_ready "$g" || { log "executor replica $g not ready"; return 1; }; done
  local -a pp=(); for g in $(seq 0 $((N_REPLICAS - 1))); do ( probe_replica "$g" "-exec-$stamp" ) & pp+=("$!"); done
  local rc=0; for p in "${pp[@]}"; do wait "$p" || rc=1; done; [ "$rc" = 0 ] || { log "executor pool failed the protocol probe"; return 1; }
  beat "executor pool ($model) ready on $N_REPLICAS replicas"; return 0; }
infra_fail(){ local why=$1 stamp; stamp=$(date +%Y%m%d-%H%M%S)   # archive the damaged round dir; 0 = caller may redo the round
  INFRA_RETRIES[$ROUND]=$(( ${INFRA_RETRIES[$ROUND]:-0} + 1 ))
  log "INFRASTRUCTURE FAILURE (round $ROUND, attempt ${INFRA_RETRIES[$ROUND]}): $why"
  printf '%s round=%s attempt=%s %s\n' "$(date -Is)" "$ROUND" "${INFRA_RETRIES[$ROUND]}" "$why" >> "$RUN_DIR/INFRA_FAILURES.log"
  [ -d "$ROUND_DIR" ] && mv "$ROUND_DIR" "${ROUND_DIR}_infra_failed_$stamp" && log "archived $ROUND_DIR -> ${ROUND_DIR}_infra_failed_$stamp"
  [ "${INFRA_RETRIES[$ROUND]}" -le 2 ] || { log "round $ROUND failed ${INFRA_RETRIES[$ROUND]} times on infrastructure: stopping the in-job loop"; return 1; }
  mkdir -p "$ROUND_DIR" "$VLOG"   # the vllm log dir moved with the archived round; relaunches need it back
  ensure_replicas || { log "replicas cannot be revived: stopping the in-job loop"; return 1; }
  mkdir -p "$ROUND_DIR"; beat "=== redoing round $ROUND in-job after infrastructure failure (bases=${SAH9B_BASES_FILE:-static}) ==="; return 0; }

# --- serve one replica per TP GPUs (round 1: proposer == executor weights) ---
export VLLM_USE_FLASHINFER_SAMPLER=0
declare -a VPIDS=()
log "propose pool serves PROPOSER_MODEL=$PROPOSER_MODEL (executor pool = frozen base $SAH_MODEL_PATH)"
for g in $(seq 0 $((N_REPLICAS - 1))); do
  port=$((PORT_BASE + g))
  CUDA_VISIBLE_DEVICES="$(replica_gpus "$g")" setsid env -u VLLM_ENV "$VLLM_ENV/bin/vllm" serve "$POOL_MODEL" \
    --host 127.0.0.1 --port "$port" --served-model-name "$SERVED_MODEL" \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$VLLM_MAX_SEQS" --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 "${VLLM_EXTRA[@]}" --language-model-only \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    > "$VLOG/replica-$g.log" 2>&1 &
  VPIDS+=("$!")
done
cleanup(){ set +e; kill "$WATCHDOG_PID" 2>/dev/null; for p in "${VPIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; done; sleep 5; for p in "${VPIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done; }
trap cleanup EXIT INT TERM
for g in $(seq 0 $((N_REPLICAS - 1))); do
  port=$((PORT_BASE + g)); ok=0; foreign=0
  for _ in $(seq 1 240); do
    kill -0 "${VPIDS[$g]}" 2>/dev/null || { tail -n 60 "$VLOG/replica-$g.log" >&2; echo "replica $g died" >&2; exit 1; }
    rc=0; models_root_ok "$port" "$POOL_MODEL" || rc=$?
    if [ "$rc" = 0 ]; then ok=1; break; fi
    if [ "$rc" = 1 ]; then foreign=$((foreign + 1)); [ "$foreign" -ge 3 ] && { echo "FOREIGN server answering on :$port (model root != $POOL_MODEL, 3 consecutive checks)" >&2; exit 66; }; else foreign=0; fi
    sleep 5
  done
  [ "$ok" = 1 ] || { tail -n 60 "$VLOG/replica-$g.log" >&2; echo "replica $g not ready" >&2; exit 1; }
  beat "replica $g ready on :$port"
done
# exact-protocol warmup gate on every replica (plain turn + parsed tool call + post-tool turn)
declare -a PPIDS=()
for g in $(seq 0 $((N_REPLICAS - 1))); do
  timeout --foreground --kill-after=30s 900s "$SAH_PYTHON" "$PIPELINE_DIR/vllm_protocol.py" \
    --base-url "http://127.0.0.1:$((PORT_BASE + g))/v1" --model "$SERVED_MODEL" --timeout 600 \
    --out "$PHASE/protocol-$g" > "$PHASE/protocol-$g.log" 2>&1 &
  PPIDS+=("$!")
done
for p in "${PPIDS[@]}"; do wait "$p" || true; done   # never bare `wait`: the vLLM servers are background children too
for g in $(seq 0 $((N_REPLICAS - 1))); do
  test "$("$SAH_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["passed"])' "$PHASE/protocol-$g/validation.json")" = "True"
done
beat "all $N_REPLICAS replicas passed the protocol probe"

# --- lineage init: a fresh run dir entering at ROUND>1 needs the task-local program registry the strict ratchet
#     requires at collect (canary 7242776 failed there); initialise it from the same seed-programs file propose snapshots.
if [ "$ROUND" != 1 ] && [ -n "${SAH9B_SEED_PROGRAMS_FILE:-}" ] && [ ! -e "$RUN_DIR/rounds/best_programs.json" ]; then
  mkdir -p "$RUN_DIR/rounds"; cp "$SAH9B_SEED_PROGRAMS_FILE" "$RUN_DIR/rounds/best_programs.json"; log "initialised $RUN_DIR/rounds/best_programs.json from $SAH9B_SEED_PROGRAMS_FILE"
fi
# --- v8: run up to SAH9B_ROUNDS_PER_JOB rounds back-to-back in this job (vLLM replicas stay up; no queue wait between rounds) ---
LAUNCHERS="$PIPELINE_DIR"
ROUNDS_PER_JOB="${SAH9B_ROUNDS_PER_JOB:-1}"; JOB_SECONDS="${SAH9B_JOB_SECONDS:-28800}"; MIN_ROUND_SECONDS="${SAH9B_MIN_ROUND_SECONDS:-9000}"; JOB_T0=$(date +%s); ROUNDS_DONE=0
while :; do
[ "$POOL_MODEL" = "$PROPOSER_MODEL" ] || { log "v12 worker runs one round per job when the proposer is trained (pool is $POOL_MODEL)"; break; }
# --- v11: never start a round against dead replicas ---
if ! ensure_replicas; then infra_fail "replicas dead before round $ROUND" || break; fi
# --- propose: M_phi(=M0) + H1 -> K candidate H2 per task ---
cd "$SAH_ROOT/src"
( while sleep 120; do [ -n "$(find "$ROUND_DIR" -newer "$HEARTBEAT" -type f 2>/dev/null | head -1)" ] && date -Is > "$HEARTBEAT"; done ) & PROP_BEAT_PID=$!
timeout --foreground --kill-after=60s "${SAH9B_PROPOSE_TIMEOUT:-10800}s" \
  "$SAH_PYTHON" -m outer.rounds.outer_round propose \
    --round-dir "$ROUND_DIR" --round "$ROUND" --k "$K" --tasks $TASKS \
    ${SAH9B_BASES_FILE:+--bases-file "$SAH9B_BASES_FILE"} ${SAH9B_SEED_PROGRAMS_FILE:+--seed-programs-file "$SAH9B_SEED_PROGRAMS_FILE"} \
    --n-replicas "$N_REPLICAS" --base-url "http://127.0.0.1:$PORT_BASE/v1" --model "$SERVED_MODEL" \
    --max-evals "$MAX_EVALS" --seed "$(( ${SAH9B_H1_SEED:-7311} + ROUND - 1 ))" --parallel "$PROPOSE_PAR" \
    2>&1 | tee "$ROUND_DIR/propose.log"
kill "$PROP_BEAT_PID" 2>/dev/null || true; beat "propose finished"
# --- v11: a propose that lost its replicas must not become an empty round ---
if grep -qaE "$INFRA_RE" "$ROUND_DIR/propose.log" 2>/dev/null || ! ensure_replicas; then infra_fail "propose of round $ROUND hit dead replicas (see propose.log)" && continue || break; fi
# v12: rollouts always run on the frozen executor -> swap the pool when the proposer is a trained checkpoint
if [ "$POOL_MODEL" != "$SAH_MODEL_PATH" ]; then switch_pool "$SAH_MODEL_PATH" || { log "could not bring up the executor pool"; exit 1; }; fi
"$SAH_PYTHON" -m outer.reward.trajectory_budget --round-dir "$ROUND_DIR" $FIXED_FLAG 2>&1 | tee "$ROUND_DIR/trajectory_budget.log"

# --- rollouts: frozen M0 + each H2 slot, bounded parallelism, round-robin over replicas ---
mapfile -t SLOTS < <("$SAH_PYTHON" - "$ROUND_DIR/h2_slot_plan.json" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["slots"]:
    print("\t".join((row["task_id"], str(row["k"]), row["h2_harness_dir"], row["h2_slot_mode"])))
PY
)
beat "rollouts: ${#SLOTS[@]} H2 slots, <= $ROLLOUT_PAR concurrent, watchdog ${ROLLOUT_WALL_TIMEOUT}s each"
mkdir -p "$ROUND_DIR/rollouts" "$ROUND_DIR/rollout_logs"
( while sleep 120; do [ -n "$(find "$ROUND_DIR/rollouts" -newer "$HEARTBEAT" -type f 2>/dev/null | head -1)" ] && date -Is > "$HEARTBEAT"; done ) & ROLL_BEAT_PID=$!
declare -a RPIDS=(); i=0
for slot in "${SLOTS[@]}"; do
  IFS=$'\t' read -r tid k hdir mode <<< "$slot"
  port=$((PORT_BASE + i % N_REPLICAS)); i=$((i + 1)); seed=$((H2_SEED_BASE + (ROUND - 1) * 1000 + k)); cand="cand$(printf '%02d' "$k")"
  ( timeout --foreground --kill-after=60s "${ROLLOUT_WALL_TIMEOUT}s" \
      "$SAH_PYTHON" -m inner.cli.run_baseline --harness-dir "$hdir" --ids "$tid" \
        --base-url "http://127.0.0.1:$port/v1" --model "$SERVED_MODEL" \
        --max-evals "$MAX_EVALS" --eval-timeout "$EVAL_TIMEOUT" --llm-timeout 600 --seed "$seed" \
        --seed-programs-file "$ROUND_DIR/seed_programs_in.json" --eval-python "$SAH_PYTHON" \
        --require-trajectory --out "$ROUND_DIR/rollouts/$tid/$cand" \
        > "$ROUND_DIR/rollout_logs/${tid}-${cand}.log" 2>&1; date -Is > "$HEARTBEAT" ) &
  RPIDS+=("$!")
  while [ "$(jobs -rp | wc -l)" -gt $((ROLLOUT_PAR + N_REPLICAS + 2)) ]; do sleep 10; done   # +2 = watchdog + rollout heartbeat loop
done
fails=0; for p in "${RPIDS[@]}"; do wait "$p" || fails=$((fails + 1)); done
kill "$ROLL_BEAT_PID" 2>/dev/null || true
beat "rollouts finished: $((${#RPIDS[@]} - fails)) ok, $fails failed (failed slots are kept as evidence)"
# --- v11: rollouts that lost their replicas are not evidence about the harnesses: redo the round instead of collecting ---
if grep -qaE "$INFRA_RE" "$ROUND_DIR"/rollout_logs/*.log 2>/dev/null || ! ensure_replicas; then infra_fail "rollouts of round $ROUND hit dead replicas (see rollout_logs)" && continue || break; fi
"$SAH_PYTHON" "$PIPELINE_DIR/audit_trajectories.py" "$ROUND_DIR/rollouts" 2>&1 | tee "$ROUND_DIR/trajectory_audit.log" || true

# --- collect: rewards + GRPO advantages -> grpo_batch.jsonl, round_summary.json, next_bases.json ---
"$SAH_PYTHON" -m outer.rounds.outer_round collect --round-dir "$ROUND_DIR" 2>&1 | tee "$ROUND_DIR/collect.log"
"$SAH_PYTHON" - "$ROUND_DIR" <<'PY'
import json, sys
from pathlib import Path
rd = Path(sys.argv[1]); s = json.load(open(rd / "round_summary.json"))
print("round_summary keys:", sorted(s.keys())[:20])
rows = json.load(open(rd / "grpo_batch.jsonl".replace(".jsonl", ".jsonl"))) if False else [json.loads(l) for l in open(rd / "grpo_batch.jsonl")]
by = {}
for r in rows:
    by.setdefault(r.get("task_id") or r.get("task"), []).append(r)
for t, rs in sorted(by.items()):
    rw = [r.get("reward") for r in rs]; print(f"{t:36s} slots={len(rs)} rewards={rw}")
PY
date -Is > "$RUN_DIR/ROUND1_DONE"
beat "done: $ROUND_DIR"
# --- v8 transition: next round inside this job, or leave it to the sbatch self-chain ---
ROUNDS_DONE=$((ROUNDS_DONE + 1))
[ "$ROUNDS_DONE" -lt "$ROUNDS_PER_JOB" ] || { log "rounds_per_job=$ROUNDS_PER_JOB reached"; break; }
remaining=$(( JOB_SECONDS - ($(date +%s) - JOB_T0) ))
[ "$remaining" -ge "$MIN_ROUND_SECONDS" ] || { log "only ${remaining}s left of the job (< ${MIN_ROUND_SECONDS}s): leaving round $((ROUND + 1)) to the sbatch self-chain"; break; }
NEXT=$((ROUND + 1)); [ "$NEXT" -le "${SAH9B_MAX_ROUND:-24}" ] || { log "round $NEXT exceeds SAH9B_MAX_ROUND=${SAH9B_MAX_ROUND:-24}"; break; }
if ! "$SAH_PYTHON" "$LAUNCHERS/build_bases.py" "$RUN_DIR" "$ROUND" "${CURVE_INHERIT_BEAM:-2}" > "$ROUND_DIR/build_bases.log" 2>&1; then
  log "build_bases failed (see $ROUND_DIR/build_bases.log): leaving round $NEXT to the sbatch self-chain"; break
fi
cat "$ROUND_DIR/build_bases.log"
export SAH9B_BASES_FILE="$ROUND_DIR/next_bases_beam2.json" SAH9B_SEED_PROGRAMS_FILE="$RUN_DIR/rounds/best_programs.json"
ROUND=$NEXT; ROUND_DIR="$RUN_DIR/rounds/round$(printf %03d "$ROUND")"; export LOGICAL_ROUND_INDEX=$((ROUND - 1))
[ -e "$ROUND_DIR/round.json" ] && { log "round $ROUND already exists: stopping"; break; }
mkdir -p "$ROUND_DIR"; beat "=== in-job transition to round $ROUND (bases=$SAH9B_BASES_FILE, ${remaining}s left) ==="
done
