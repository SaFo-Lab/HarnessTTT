#!/usr/bin/env bash
# HarnessTTT baseline worker (runs inside the SAH container, one job = one task):
#   frozen executor (SAH_MODEL_PATH) + the built-in H2 harness (src/inner/harness, "h2_initial") x SAH9B_BASELINE_SEEDS runs
#   of SAH9B_MAX_EVALS evaluator calls, plus a seed-only evaluation of the seed program, written as a baseline_h2_20ev-format
#   table to SAH9B_BASELINE_OUT.  With SAH9B_SEED_PROGRAMS_FILE the rollouts start from that registry's program instead of the
#   task's initial program (seed-program swap experiments); the pipeline then points SAH_BASELINE_JSON at the output.
# env: SAH9B_RUN_DIR SAH9B_SAH_ROOT SAH9B_TASKS (one task id) SAH9B_MAX_EVALS SAH9B_EVAL_TIMEOUT SAH9B_BASELINE_OUT
#      SAH9B_BASELINE_SEEDS [SAH9B_SEED_PROGRAMS_FILE] [SAH9B_EVAL_GPUS]
set -euo pipefail
RUN_DIR="$(cd "${SAH9B_RUN_DIR:?}" && pwd -P)"; SAH_ROOT="${SAH9B_SAH_ROOT:?}"; PIPELINE_DIR="$SAH_ROOT/scripts/pipeline"
TID="${SAH9B_TASKS:?}"; OUT_JSON="${SAH9B_BASELINE_OUT:?}"; SEEDS="${SAH9B_BASELINE_SEEDS:-2}"; MAX_EVALS="${SAH9B_MAX_EVALS:-20}"
SEED_FILE="${SAH9B_SEED_PROGRAMS_FILE:-}"
N_GPUS="${SLURM_GPUS_ON_NODE:?no GPU allocation}"; EVAL_GPUS="${SAH9B_EVAL_GPUS:-0}"; TP="${SAH9B_VLLM_TP:-1}"; MAX_MODEL_LEN="${SAH9B_MAX_MODEL_LEN:-131072}"
N_REPLICAS=$(( (N_GPUS - EVAL_GPUS) / TP ))   # TENSORPARALLEL-001: the single executor replica spans TP GPUs
[ "$N_REPLICAS" -ge 1 ] && [ $(( N_REPLICAS * TP + EVAL_GPUS )) -eq "$N_GPUS" ] || { echo "GPU layout mismatch (gpus=$N_GPUS eval_gpus=$EVAL_GPUS tp=$TP)" >&2; exit 65; }
if [ "$TP" -gt 1 ]; then VLLM_EXTRA=(--tensor-parallel-size "$TP" --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'); else VLLM_EXTRA=(--enforce-eager); fi
B="$RUN_DIR/baseline"; mkdir -p "$B"; LOG="$B/baseline.log"; PROG="$B/progress.txt"
log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }; beat(){ echo "$*" > "$PROG"; log "$*"; }

# --- environment (mirrors worker_round.sh) ---
export PATH="$SAH_AHC_NATIVE_BIN:$PATH"
export AHC_CASE_WORKERS=8 AHC_PIN_CORES=0 AHC_DOCKER_TIMEOUT=900 SIMPLETES_WORKER_ID=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=3"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$NEXAU_ROOT:$SAH_ROOT/src:$SAH_ROOT"
export SAH_DATASET_ROOT="$SAH_DATA_ROOT"
export SAH_TASK_TEXT_REGISTRY="${SAH_TASK_TEXT_REGISTRY:-$SAH_ROOT/config/task_text_registry.json}" SAH_TASK_TEXT_ENFORCE=1
export EVAL_TIMEOUT="${SAH9B_EVAL_TIMEOUT:-600}" OPENAI_API_KEY=EMPTY SAH_MAX_CONTEXT_TOKENS=131072
if [ -n "${SAH9B_BASE_MODEL:-}" ] && [ "${SAH_MODEL_PATH:-}" != "$SAH9B_BASE_MODEL" ]; then   # MODELCLOBBER-001 contract
  echo "MODELCLOBBER-001: SAH_MODEL_PATH=${SAH_MODEL_PATH:-unset} but this job was submitted for base model $SAH9B_BASE_MODEL" >&2; exit 66
fi
export MODEL_PATH="$SAH_MODEL_PATH" SERVED_MODEL="$SAH_SERVED_MODEL" SAH_MAX_CONTEXT_TOKENS="$MAX_MODEL_LEN"
export TMPDIR="/tmp/harnessttt-baseline-$SLURM_JOB_ID"; mkdir -p "$TMPDIR"
unset ANTHROPIC_API_KEY AZURE_API_KEY AZURE_OPENAI_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN GEMINI_API_KEY GOOGLE_API_KEY WANDB_API_KEY 2>/dev/null || true
if [ "$EVAL_GPUS" -gt 0 ]; then export SAH_EVAL_GPU_POOL="$(seq -s, $((N_REPLICAS * TP)) $((N_GPUS - 1)))"; fi
test -x "$SAH_PYTHON"; test -x "$VLLM_ENV/bin/vllm"; test -d "$SAH_MODEL_PATH"
[ -z "$SEED_FILE" ] || test -r "$SEED_FILE"
nvidia-smi -L | tee "$B/nvidia-smi-L.txt"
"$SAH_PYTHON" -c "from nexau import Agent; import numpy, scipy, yaml; print('runtime imports ok')"
log "baseline for $TID: max_evals=$MAX_EVALS seeds=$SEEDS eval_timeout=$EVAL_TIMEOUT seed_file=${SEED_FILE:-task_initial} gpus=$N_GPUS tp=$TP max_model_len=$MAX_MODEL_LEN model=$SAH_MODEL_PATH eval_gpus=$EVAL_GPUS commit $(git -C "$SAH_ROOT" rev-parse --short HEAD)"

# --- table writer: baseline_h2_20ev-format JSON from whatever runs have completed (called after every run; atomic) ---
write_table(){
"$SAH_PYTHON" - "$B" "$TID" "$OUT_JSON" "$MAX_EVALS" "$SEEDS" "${SEED_FILE:-}" <<'PY'
import glob, hashlib, json, os, sys, time
from pathlib import Path
B, tid, out, max_evals, seeds, seed_file = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
def rows(pattern):
    r = []
    for p in sorted(glob.glob(str(B / pattern / "*" / "summary.json"))):
        for row in json.load(open(p)):
            if row.get("task_id") == tid:
                row["_summary"] = p; r.append(row)
    return r
seed_rows = rows("seed_only")
if not seed_rows or seed_rows[-1].get("seed_score") is None:
    sys.exit(f"seed-only evaluation produced no score for {tid} (see {B}/seed_only.log)")
seed_score = float(seed_rows[-1]["seed_score"])
runs = []
for s in range(1, seeds + 1):
    for row in rows(f"agent_seed{s}"):
        runs.append({"run": s, "best_score": row.get("best_score"), "seed_score": row.get("seed_score"),
                     "evaluations": row.get("evaluations"), "error": row.get("error"), "summary": row["_summary"]})
scored = [float(r["best_score"]) for r in runs if r.get("best_score") is not None]
if not scored:
    sys.exit(f"no frozen-harness run produced a best_score for {tid} (see {B}/agent_seed*.log)")
seed_meta = {"mode": "task_initial", "registry": None}
if seed_file:
    ent = json.load(open(seed_file)).get(tid) or {}
    prog = ent.get("program", "")
    seed_meta = {"mode": "inherited_registry", "registry": seed_file, "registry_sha256": hashlib.sha256(Path(seed_file).read_bytes()).hexdigest(),
                 "program_sha256": hashlib.sha256(prog.encode()).hexdigest(), "registry_score": ent.get("score"), "source": ent.get("source")}
table = {
    "what": f"Frozen-harness baseline for {tid} with a {'swapped' if seed_file else 'task-initial'} seed program: seed = the seed program's "
            f"own evaluator score, h2_best = best combined_score over {len(scored)} run(s) of the built-in H2 harness with "
            f"{max_evals} evaluator calls each (same executor, container and evaluator as the lineage's rollouts).",
    "harness": "src/inner/harness (h2_initial)", "budget": {"max_evaluator_calls": max_evals, "seeds": seeds},
    "captured": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": os.environ.get("SLURM_JOB_ID"), "seed_program": seed_meta,
    "runs": runs, "runs_planned": seeds, "runs_completed": len(scored),
    "baseline": {tid: {"seed": seed_score, "h2_best": max(scored), "h2_runs": scored}},
}
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix(out.suffix + ".tmp"); tmp.write_text(json.dumps(table, indent=1) + "\n"); os.replace(tmp, out)
print(f"baseline table -> {out}: seed {seed_score:.6f} h2_best {max(scored):.6f} over {scored} ({len(scored)}/{seeds} runs)")
PY
}

# --- one frozen-executor replica on GPU 0 (per-job free port, see worker_round.sh) ---
PORT=$("$SAH_PYTHON" - "${SLURM_JOB_ID:-0}" <<'PY'
import socket, sys
job = int(sys.argv[1])
def free(p):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", p)); return True
    except OSError:
        return False
    finally:
        s.close()
start = 20000 + (job * 37) % 30000
for p in range(start, 60000, 16):
    if free(p):
        print(p); break
else:
    sys.exit("no free port")
PY
)
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TP - 1)))" setsid env -u VLLM_ENV "$VLLM_ENV/bin/vllm" serve "$SAH_MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_MODEL" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 "${VLLM_EXTRA[@]}" --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml >> "$B/vllm.log" 2>&1 &
VPID=$!
cleanup(){ kill -TERM -- "-$VPID" 2>/dev/null || kill -TERM "$VPID" 2>/dev/null || true; }
trap cleanup EXIT TERM INT
beat "vLLM executor replica starting on :$PORT (pid $VPID)"
ready=0
for _i in $(seq 1 240); do
  kill -0 "$VPID" 2>/dev/null || { tail -n 40 "$B/vllm.log" >&2; log "vLLM died during start"; exit 1; }
  if "$SAH_PYTHON" - "$PORT" "$SAH_MODEL_PATH" "$SERVED_MODEL" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
port, root, name = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10).read())
m = [x for x in d.get("data", []) if x.get("id") == name]
assert m and (m[0].get("root") or root) == root
body = json.dumps({"model": name, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}).encode()
urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}), timeout=180).read()
PY
  then ready=1; break; fi
  sleep 5
done
[ "$ready" = 1 ] || { log "vLLM not ready after 20 min"; exit 1; }
timeout --foreground --kill-after=30s 900s "$SAH_PYTHON" "$PIPELINE_DIR/vllm_protocol.py" \
  --base-url "http://127.0.0.1:$PORT/v1" --model "$SERVED_MODEL" --timeout 600 --out "$B/protocol" > "$B/protocol.log" 2>&1
test "$("$SAH_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["passed"])' "$B/protocol/validation.json")" = "True"
beat "executor replica ready and protocol-probed on :$PORT"

# --- seed-only score of the seed program (the table's "seed" column); a previous attempt's result is reused ---
cd "$SAH_ROOT/src"
DEADLINE="${SAH9B_BASELINE_DEADLINE_EPOCH:-0}"
have_summary(){ compgen -G "$B/$1/*/summary.json" > /dev/null && grep -q -E "\"$2\": [0-9]" "$B"/$1/*/summary.json; }
if have_summary seed_only seed_score; then
  beat "seed program score reused from a previous attempt: $(grep -h -o '"seed_score": [^,]*' "$B"/seed_only/*/summary.json | head -1)"
else
  "$SAH_PYTHON" -m inner.cli.run_baseline --seed-only --ids "$TID" ${SEED_FILE:+--seed-programs-file "$SEED_FILE"} \
    --eval-timeout "$EVAL_TIMEOUT" --eval-python "$SAH_PYTHON" --out "$B/seed_only" > "$B/seed_only.log" 2>&1
  beat "seed program evaluated: $(grep -h -o '"seed_score": [^,]*' "$B"/seed_only/*/summary.json | head -1)"
fi

# --- frozen-harness agent runs (built-in src/inner/harness, MAX_EVALS evaluator calls each) ---
#     Completed runs of a previous attempt are reused; a run is skipped when it cannot finish before the job's deadline
#     (estimate = 1.2 x the longest completed run, at least 30 min); the table is written after every completed run.
LAST_RUN_SECONDS=0
for s in $(seq 1 "$SEEDS"); do
  if have_summary agent_seed$s best_score; then
    beat "frozen-harness run $s/$SEEDS reused from a previous attempt: $(grep -h -o '"best_score": [^,]*' "$B"/agent_seed$s/*/summary.json | head -1)"
    [ -f "$B/agent_seed$s.seconds" ] && LAST_RUN_SECONDS=$(( $(cat "$B/agent_seed$s.seconds") > LAST_RUN_SECONDS ? $(cat "$B/agent_seed$s.seconds") : LAST_RUN_SECONDS ))
    continue
  fi
  if [ "$DEADLINE" -gt 0 ]; then
    est=$(( LAST_RUN_SECONDS * 12 / 10 )); [ "$est" -lt 1800 ] && est=1800
    if [ $(( $(date +%s) + est )) -gt "$DEADLINE" ]; then
      beat "frozen-harness run $s/$SEEDS SKIPPED: $(( (DEADLINE - $(date +%s)) / 60 )) min left before the job deadline, a run takes ~$(( est / 60 )) min"
      break
    fi
  fi
  beat "frozen-harness run $s/$SEEDS ($MAX_EVALS evals) starting"
  T_RUN0=$(date +%s)
  timeout --foreground --kill-after=60s "${SAH9B_ROLLOUT_WALL_TIMEOUT:-10800}s" \
    "$SAH_PYTHON" -m inner.cli.run_baseline --ids "$TID" \
      --base-url "http://127.0.0.1:$PORT/v1" --model "$SERVED_MODEL" \
      --max-evals "$MAX_EVALS" --eval-timeout "$EVAL_TIMEOUT" --llm-timeout 600 --seed $((300000 + s)) \
      ${SEED_FILE:+--seed-programs-file "$SEED_FILE"} --eval-python "$SAH_PYTHON" \
      --require-trajectory --out "$B/agent_seed$s" > "$B/agent_seed$s.log" 2>&1 || log "run $s exited non-zero (kept as evidence)"
  LAST_RUN_SECONDS=$(( $(date +%s) - T_RUN0 )); echo "$LAST_RUN_SECONDS" > "$B/agent_seed$s.seconds"
  beat "frozen-harness run $s/$SEEDS finished in $(( LAST_RUN_SECONDS / 60 )) min: $(grep -h -o '"best_score": [^,]*' "$B"/agent_seed$s/*/summary.json 2>/dev/null | head -1)"
  write_table && beat "baseline table updated after run $s ($OUT_JSON)"
done

write_table || { [ -f "$OUT_JSON" ] || exit 1; }
beat "baseline written to $OUT_JSON"
log "done"
