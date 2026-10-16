#!/usr/bin/env bash
# HarnessTTT: proposer LoRA update for one round (replay -> bounded replay -> slime FSDP LoRA -> merged HF checkpoint -> model_state.json).
#SBATCH --job-name=@@JOBNAME@@
@@SBATCH_ACCOUNT@@
@@SBATCH_PARTITION@@
@@SBATCH_QOS@@
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:@@TRAIN_GPUS@@
#SBATCH --cpus-per-task=@@TRAIN_CPUS@@
#SBATCH --mem=@@TRAIN_MEM@@   # pipeline config train_mem (default 0 = the node's memory: slime gathers the full state dict on every rank when it saves)
#SBATCH --time=@@TIME@@
#SBATCH --no-requeue
#SBATCH --output=@@RUN@@/out/train-r@@ROUND3@@-%j.out
#SBATCH --error=@@RUN@@/out/train-r@@ROUND3@@-%j.err
set -euo pipefail
export RUN_ROOT=@@RUN_ROOT@@   # the runner's run root, not the frozen worktree's default
source @@ENV_FILE@@
# per-lineage base model (pipeline.py config: model_path / served_model / vllm_tp / max_model_len; defaults = env.sh)
export SAH_MODEL_PATH=@@MODEL_PATH@@ SAH_SERVED_MODEL=@@SERVED_MODEL@@ SAH9B_VLLM_TP=@@VLLM_TP@@ SAH9B_MAX_MODEL_LEN=@@MAX_MODEL_LEN@@ SAH9B_BASE_MODEL=@@MODEL_PATH@@   # SAH9B_BASE_MODEL: the workers refuse to run if a sourced site file changes SAH_MODEL_PATH
export SAH9B_SAH_ROOT=@@SAH_ROOT@@
export SAH9B_TRAIN_TAG=@@TAG@@ SAH9B_TRAIN_ROUND=@@ROUND@@ SAH9B_TOKEN_BUDGET=@@TOKEN_BUDGET@@ SAH9B_TRAIN_LR=@@LR@@ SAH9B_TRAIN_KL=@@KL@@
export SAH9B_TRAIN_REPLAY=@@REPLAY@@ SAH9B_TRAIN_DIR=@@TRAIN_DIR@@ SAH9B_TRAIN_PREV_STATE=@@PREV_STATE@@
need(){ [ -e "$1" ] || { echo "[sbatch] missing: $2 ($1)" >&2; exit 64; }; }
clean_tree(){ git -C "$1" diff --quiet --no-ext-diff HEAD -- || { echo "[sbatch] $1 has uncommitted changes" >&2; exit 64; }
  [ -z "$(git -C "$1" ls-files --others --exclude-standard)" ] || { echo "[sbatch] $1 has untracked files" >&2; exit 64; }; }
need "$SAH_IMAGE" "container image SAH_IMAGE"; need "$SAH9B_TRAIN_REPLAY" "training replay SAH9B_TRAIN_REPLAY"
[ -s "$SAH9B_TRAIN_REPLAY" ] || { echo "[sbatch] empty training replay $SAH9B_TRAIN_REPLAY" >&2; exit 64; }
clean_tree "$SAH9B_SAH_ROOT"
srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --container-image="$SAH_IMAGE" --container-mounts="@@CONTAINER_MOUNTS@@" --no-container-mount-home \
  --container-workdir="$SAH9B_SAH_ROOT" \
  /bin/bash @@PIPELINE_DIR@@/train_worker.sh || echo "[sbatch] srun exit $? $(date -Is)"
@@EPILOGUE@@
