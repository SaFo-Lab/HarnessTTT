#!/usr/bin/env bash
# HarnessTTT proposer LoRA update for one round (inside the SAH container, one node, 8 GPUs):
#   replay.jsonl (src/trainer/grpo_to_replay.py; advantages from the collect step)
#     -> src/trainer/bound_replay.py : terminal-preserving, loss-mask-aware bound to SAH9B_TOKEN_BUDGET tokens
#     -> src/trainer/pad_replay.py   : pad to a multiple of the GPU count with zero-advantage duplicates (KL-only rows)
#     -> slime offline GRPO (FSDP; LoRA r64/alpha128 on all proj modules; KL 0.1 low_var_kl vs the frozen base; lr 1e-5 constant;
#        3 epochs; batch = all rows; cumulative adapter resumed with --load-weights-only)
#     -> src/trainer/merge_lora.sh   : merged HF checkpoint (served by vLLM as the next round's proposer)
#     -> model_state.json (model_path, adapter_path, weights_updated, skip_reason) + TRAIN_COMPLETE | FAILED
# Env in: SAH9B_TRAIN_DIR (out dir), SAH9B_TRAIN_REPLAY (replay.jsonl), SAH9B_TRAIN_PREV_STATE (model_state.json of the previous
#   round; empty for the first update), SAH9B_TRAIN_TAG, SAH9B_TRAIN_ROUND, SAH9B_TOKEN_BUDGET (18000), SAH9B_TRAIN_LR (1e-5),
#   SAH9B_TRAIN_KL (0.1), SAH9B_TRAIN_DRY_RUN (1 = validate the replay and write train_command.txt without touching a GPU).
# Site env (config/env.sh + config/local.env): SAH_MODEL_PATH, SAH_PYTHON, TRAIN_ENV, SLIME_ROOT, MEGATRON_ROOT.
set -euo pipefail
umask 027
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
SAH_ROOT="${SAH9B_SAH_ROOT:-$HARNESSTTT_ROOT}"
TRAIN_DIR="${SAH9B_TRAIN_DIR:?}"; REPLAY="${SAH9B_TRAIN_REPLAY:?}"; TAG="${SAH9B_TRAIN_TAG:?}"; ROUND="${SAH9B_TRAIN_ROUND:?}"
TOKEN_BUDGET="${SAH9B_TOKEN_BUDGET:-18000}"; DRY="${SAH9B_TRAIN_DRY_RUN:-0}"
TRAIN_ENV="${TRAIN_ENV:?config/local.env must set TRAIN_ENV (torch/transformers/flash-linear-attention env for slime)}"
: "${SLIME_ROOT:?}" "${MEGATRON_ROOT:?config/local.env must set MEGATRON_ROOT}" "${SAH_MODEL_PATH:?config/local.env must set SAH_MODEL_PATH}"
TRAINER="$SAH_ROOT/src/trainer"; TFPATCH="$TRAINER/tfpatch"
NGPU="${SLURM_GPUS_ON_NODE:?no GPU allocation (SLURM_GPUS_ON_NODE)}"
PY="${SAH_PYTHON:-}"; [ -n "$PY" ] && [ -x "$PY" ] || PY="${HARNESSTTT_HOST_PYTHON:-python3}"   # stdlib json helpers only
log(){ echo "[$(date -Is)] $*"; }
mkdir -p "$TRAIN_DIR"; cd "$TRAIN_DIR"
# any failure from here on leaves a FAILED marker (the runner retries once, then stops); stop_ray is a no-op until Ray starts
stop_ray(){ :; }
trap 'rc=$?; stop_ray; [ "$rc" -ne 0 ] && [ ! -s "$TRAIN_DIR/TRAIN_COMPLETE" ] && printf "failed_utc=%s rc=%s\n" "$(date -Is -u)" "$rc" > "$TRAIN_DIR/FAILED"; exit $rc' EXIT
# a previous attempt that FAILED leaves ckpt/merged/debug behind -> archive them so this attempt starts clean
if [ -e "$TRAIN_DIR/FAILED" ] || [ -e "$TRAIN_DIR/ckpt" ] || [ -e "$TRAIN_DIR/merged" ]; then
  arch="$TRAIN_DIR/attempt_failed_$(date +%Y%m%d-%H%M%S)"; mkdir -p "$arch"
  for x in ckpt merged debug train.log merge.log ray_head.log ray_status.log train_command.txt TRAINED FAILED bound_replay.log replay_bounded.jsonl; do [ -e "$TRAIN_DIR/$x" ] && mv "$TRAIN_DIR/$x" "$arch/"; done
  log "archived previous attempt to $arch"
fi
mkdir -p "$TRAIN_DIR/debug"
need(){ [ -e "$1" ] || { log "missing: $2 ($1)"; exit 64; }; }
need "$SLIME_ROOT" "slime checkout SLIME_ROOT"; need "$MEGATRON_ROOT" "Megatron-LM MEGATRON_ROOT"; need "$TFPATCH" "trainer tfpatch dir"
need "$REPLAY" "training replay"; [ -s "$REPLAY" ] || { log "empty training replay $REPLAY"; exit 64; }; need "$TRAINER/bound_replay.py" "bound_replay.py"
if [ "$DRY" != 1 ]; then need "$TRAIN_ENV/bin/python" "training env python TRAIN_ENV/bin/python"; nvidia-smi -L | tee "$TRAIN_DIR/nvidia-smi-L.txt"; fi
log "train worker start tag=$TAG round=$ROUND gpus=$NGPU replay=$REPLAY budget=$TOKEN_BUDGET dry_run=$DRY"

# previous policy (cumulative LoRA): resume from the previous adapter if any
previous_model="$SAH_MODEL_PATH"; previous_adapter=""
if [ -n "${SAH9B_TRAIN_PREV_STATE:-}" ] && [ -s "$SAH9B_TRAIN_PREV_STATE" ]; then
  previous_model=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' "$SAH9B_TRAIN_PREV_STATE")
  previous_adapter=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("adapter_path") or "")' "$SAH9B_TRAIN_PREV_STATE")
fi
STATE_OUT="$TRAIN_DIR/model_state.json"
publish_state(){ "$PY" - "$STATE_OUT" "$1" "$2" "$3" "$4" "$TAG" "$ROUND" "${SLURM_JOB_ID:-0}" <<'PY'
import json, os, sys, time
path, model, adapter, updated, reason, tag, rnd, job = sys.argv[1:]
payload = {"schema": 2, "tag": tag, "round": int(rnd), "model_path": model, "adapter_path": adapter or None,
           "weights_updated": updated == "1", "skip_reason": reason or None, "slurm_job_id": job,
           "published_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
tmp = path + ".tmp"; open(tmp, "w").write(json.dumps(payload, indent=2) + "\n"); os.replace(tmp, path)
PY
}
publish_skip(){
  if [ "$DRY" = 1 ]; then log "dry run: training would be SKIPPED ($1); proposer unchanged"; exit 0; fi
  publish_state "$previous_model" "$previous_adapter" 0 "$1"
  printf 'completed_utc=%s\nupdated=false\nreason=%s\n' "$(date -Is -u)" "$1" > "$TRAIN_DIR/TRAIN_COMPLETE"; log "training skipped: $1"; exit 0; }
count_rows(){ "$PY" - "$1" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
adv=[abs(float((r.get("metadata") or {}).get("advantage", 0.0))) for r in rows]
print(len(rows), max(adv, default=0.0))
PY
}
read -r train_rows max_abs_adv <<< "$(count_rows "$REPLAY")"
log "replay rows=$train_rows max|adv|=$max_abs_adv"
[ "$train_rows" -ge 4 ] || publish_skip "insufficient_rows_${train_rows}"
"$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 1e-6 else 1)" "$max_abs_adv" || publish_skip "zero_signal_maxadv_${max_abs_adv}"

# --- bound trajectories to TOKEN_BUDGET tokens (terminal-preserving) with the same qwen3_5 loss mask slime applies at train time ---
BOUNDED="$TRAIN_DIR/replay_bounded.jsonl"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="$TFPATCH:$SLIME_ROOT:$MEGATRON_ROOT:$SAH_ROOT/src:$SAH_ROOT"
if [ "$DRY" = 1 ]; then
  bounded_rows=$train_rows; log "dry run: skipping bound_replay.py (--budget $TOKEN_BUDGET --truncate-mode terminal) and pad_replay.py ($NGPU GPUs)"
else
  "$TRAIN_ENV/bin/python" "$TRAINER/bound_replay.py" --src "$REPLAY" --out "$BOUNDED" --budget "$TOKEN_BUDGET" \
    --truncate-mode terminal --model "$SAH_MODEL_PATH" 2>&1 | tee "$TRAIN_DIR/bound_replay.log"
  read -r bounded_rows bounded_max_abs <<< "$(count_rows "$BOUNDED")"
  log "bounded rows=$bounded_rows (dropped $((train_rows - bounded_rows))) max|adv|=$bounded_max_abs"
  [ "$bounded_rows" -ge 4 ] || publish_skip "post_bound_insufficient_rows_${bounded_rows}"
  # slime splits the batch over dp_size = NGPU and injects metadata-less dummy samples when rows < dp_size (KeyError 'raw_reward'):
  # pad to a multiple of NGPU with duplicates of real rows carrying advantage 0 (no policy-gradient contribution; KL term only).
  read -r padded_total n_pad <<< "$("$PY" "$TRAINER/pad_replay.py" "$BOUNDED" "$NGPU")"
  log "padded replay: $padded_total rows ($n_pad zero-advantage padding rows) for $NGPU GPUs"; bounded_rows=$padded_total
fi

# --- ray head (single node) ---
RAY_PORT=$((20000 + ${SLURM_JOB_ID:-1} % 19000)); PB=$((13000 + (${SLURM_JOB_ID:-1} % 7) * 256))
RAY_TMP="/tmp/htt-ray-${SLURM_JOB_ID:-x}"
if [ "$DRY" != 1 ]; then
  mkdir -p "$RAY_TMP"
  setsid "$TRAIN_ENV/bin/ray" start --head --block --node-ip-address=127.0.0.1 --port "$RAY_PORT" \
    --node-manager-port "$PB" --object-manager-port "$((PB + 1))" --min-worker-port "$((PB + 2))" --max-worker-port "$((PB + 253))" \
    --num-cpus "${SLURM_CPUS_PER_TASK:-16}" --num-gpus "$NGPU" --disable-usage-stats --include-dashboard=false --temp-dir "$RAY_TMP" \
    > "$TRAIN_DIR/ray_head.log" 2>&1 &
  RAY_PID=$!
  stop_ray(){ set +e; kill -TERM -- "-$RAY_PID" 2>/dev/null; for _ in $(seq 1 30); do kill -0 "$RAY_PID" 2>/dev/null || break; sleep 1; done; kill -KILL -- "-$RAY_PID" 2>/dev/null; set -e; }
  ready=0; for _ in $(seq 1 90); do "$TRAIN_ENV/bin/ray" status --address "127.0.0.1:$RAY_PORT" > "$TRAIN_DIR/ray_status.log" 2>&1 && grep -Fq 'Active:' "$TRAIN_DIR/ray_status.log" && { ready=1; break; }; kill -0 "$RAY_PID" 2>/dev/null || break; sleep 2; done
  [ "$ready" = 1 ] || { log "ray head did not come up"; exit 1; }
  export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
fi

# --- slime offline GRPO LoRA (advantages come from metadata.advantage; slime's reward normalisation disabled) ---
CKPT="$TRAIN_DIR/ckpt"; MERGED="$TRAIN_DIR/merged"; test ! -e "$CKPT"; test ! -e "$MERGED"
load_path="$SAH_MODEL_PATH"; resume=0; if [ -n "$previous_adapter" ]; then load_path="$previous_adapter"; resume=1; fi
runtime_env_json="{\"env_vars\":{\"PYTHONPATH\":\"${PYTHONPATH}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"HF_HUB_OFFLINE\":\"1\",\"TRANSFORMERS_OFFLINE\":\"1\",\"TOKENIZERS_PARALLELISM\":\"false\",\"WANDB_MODE\":\"disabled\"}}"
cmd=( "$TRAIN_ENV/bin/python" "$TRAINER/direct_ray_driver.py" "--runtime-env-json=$runtime_env_json" "--train-script=$SLIME_ROOT/train.py" --
  --hf-checkpoint "$SAH_MODEL_PATH" --load "$load_path" --ref-load "$SAH_MODEL_PATH" --save "$CKPT" --save-interval 1 --no-save-optim
  --rollout-function-path trainer.slime_replay_rollout.generate_rollout
  --prompt-data "$BOUNDED" --input-key messages --tool-key tools --metadata-key metadata --loss-mask-type qwen3_5 --rollout-shuffle
  --sglang-router-port "$((PB + 254))"
  --num-epoch 3 --rollout-batch-size "$bounded_rows" --n-samples-per-prompt 1 --global-batch-size "$bounded_rows" --micro-batch-size 1
  --loss-type policy_loss --advantage-estimator grpo --disable-rewards-normalization --debug-train-only --dump-details "$TRAIN_DIR/debug"
  --use-kl-loss --kl-loss-coef "${SAH9B_TRAIN_KL:-0.1}" --kl-loss-type low_var_kl
  --optimizer adam --lr "${SAH9B_TRAIN_LR:-1e-5}" --lr-decay-style constant --min-lr 0.0 --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.98
  --use-lora --lora-rank 64 --lora-alpha 128 --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
  --train-backend fsdp --gradient-checkpointing --attn-implementation flash_attention_2 --log-probs-chunk-size 128
  --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
  --actor-num-nodes 1 --actor-num-gpus-per-node "$NGPU" --num-gpus-per-node "$NGPU" --seed "$((91000 + ROUND))" )
[ "$resume" = 1 ] && cmd+=( --load-weights-only --start-rollout-id 0 )
printf '%s\n' "${cmd[@]}" > "$TRAIN_DIR/train_command.txt"
if [ "$DRY" = 1 ]; then log "dry run: slime command written to $TRAIN_DIR/train_command.txt (resume=$resume load=$load_path rows=$bounded_rows); not training"; exit 0; fi
mkdir -p "$CKPT"
cd "$SLIME_ROOT"; "${cmd[@]}" 2>&1 | tee "$TRAIN_DIR/train.log"; cd "$TRAIN_DIR"
printf 'trained_utc=%s\n' "$(date -Is -u)" > "$TRAIN_DIR/TRAINED"
stop_ray
# --- merge the LoRA into a full HF checkpoint for vLLM ---
PATH="$TRAIN_ENV/bin:$PATH" SLIME_ROOT="$SLIME_ROOT" HF_CKPT="$SAH_MODEL_PATH" CKPT_DIR="$CKPT" OUT="$MERGED" \
  /bin/bash "$TRAINER/merge_lora.sh" 2>&1 | tee "$TRAIN_DIR/merge.log"
test -s "$MERGED/config.json"
"$PY" - "$SAH_MODEL_PATH" "$MERGED" <<'PY'
import shutil, sys
from pathlib import Path
base, merged = Path(sys.argv[1]), Path(sys.argv[2])
for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json", "chat_template.jinja", "special_tokens_map.json"):
    if (base / name).is_file() and not (merged / name).exists(): shutil.copy(base / name, merged / name)
print("merged checkpoint files:", sorted(p.name for p in merged.iterdir())[:12])
PY
adapter_dir="$CKPT/iter_$(printf '%07d' "$(cat "$CKPT/latest_checkpointed_iteration.txt")")/model"
publish_state "$MERGED" "$adapter_dir" 1 ""
printf 'completed_utc=%s\nupdated=true\nrows=%s\nbounded_rows=%s\n' "$(date -Is -u)" "$train_rows" "$bounded_rows" > "$TRAIN_DIR/TRAIN_COMPLETE"
log "done: merged=$MERGED adapter=$adapter_dir"
