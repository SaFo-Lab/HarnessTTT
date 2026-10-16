#!/usr/bin/env bash
# HarnessTTT: proposer LoRA update for one round (replay -> bounded replay -> slime FSDP LoRA -> merged HF checkpoint -> model_state.json).
#SBATCH --job-name=@@JOBNAME@@
#SBATCH --account=@@ACCOUNT@@
#SBATCH --partition=@@PARTITION@@
#SBATCH --qos=@@QOS@@
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:@@TRAIN_GPUS@@
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000G   # TRAINMEM-001 (2026-10-02): 256G killed the 27B save (slime gathers the full state dict on every rank: 8 x 55 GB); the training nodes have 1.8 TB
#SBATCH --time=@@TIME@@
#SBATCH --no-requeue
#SBATCH --output=@@RUN@@/out/train-r@@ROUND3@@-%j.out
#SBATCH --error=@@RUN@@/out/train-r@@ROUND3@@-%j.err
set -euo pipefail
source @@ENV_FILE@@
# per-lineage base model (pipeline.py config: model_path / served_model / vllm_tp / max_model_len; defaults = env.sh)
export SAH_MODEL_PATH=@@MODEL_PATH@@ SAH_SERVED_MODEL=@@SERVED_MODEL@@ SAH9B_VLLM_TP=@@VLLM_TP@@ SAH9B_MAX_MODEL_LEN=@@MAX_MODEL_LEN@@ SAH9B_BASE_MODEL=@@MODEL_PATH@@   # SAH9B_BASE_MODEL: the workers refuse to run if a sourced site file changes SAH_MODEL_PATH (MODELCLOBBER-001)
export SAH9B_SAH_ROOT=@@SAH_ROOT@@
export SAH9B_TRAIN_TAG=@@TAG@@ SAH9B_TRAIN_ROUND=@@ROUND@@ SAH9B_TOKEN_BUDGET=@@TOKEN_BUDGET@@ SAH9B_TRAIN_LR=@@LR@@ SAH9B_TRAIN_KL=@@KL@@
export SAH9B_TRAIN_REPLAY=@@REPLAY@@ SAH9B_TRAIN_DIR=@@TRAIN_DIR@@ SAH9B_TRAIN_PREV_STATE=@@PREV_STATE@@
test -r "$SAH_IMAGE"; test -s "$SAH9B_TRAIN_REPLAY"
git -C "$SAH9B_SAH_ROOT" diff --quiet --no-ext-diff HEAD --
srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --container-image="$SAH_IMAGE" --container-mounts="@@CONTAINER_MOUNTS@@" --no-container-mount-home \
  --container-workdir="$SAH9B_SAH_ROOT" \
  /bin/bash @@PIPELINE_DIR@@/train_worker.sh || echo "[sbatch] srun exit $? $(date -Is)"
@@EPILOGUE@@
