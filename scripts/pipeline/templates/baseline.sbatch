#!/usr/bin/env bash
# HarnessTTT: frozen-harness baseline for a (swapped) seed program -> baseline_h2_20ev-format JSON (SAH_BASELINE_JSON of the lineage).
#SBATCH --job-name=@@JOBNAME@@
@@SBATCH_ACCOUNT@@
@@SBATCH_PARTITION@@
@@SBATCH_QOS@@
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:@@GPUS@@
#SBATCH --cpus-per-task=@@CPUS@@
#SBATCH --mem=@@MEM@@
#SBATCH --time=@@TIME@@
#SBATCH --no-requeue
#SBATCH --signal=TERM@600
#SBATCH --output=@@RUN@@/out/slurm-%j.out
#SBATCH --error=@@RUN@@/out/slurm-%j.err
set -euo pipefail
export RUN_ROOT=@@RUN_ROOT@@   # the runner's run root, not the frozen worktree's default
source @@ENV_FILE@@
# per-lineage base model (pipeline.py config: model_path / served_model / vllm_tp / max_model_len; defaults = env.sh)
export SAH_MODEL_PATH=@@MODEL_PATH@@ SAH_SERVED_MODEL=@@SERVED_MODEL@@ SAH9B_VLLM_TP=@@VLLM_TP@@ SAH9B_MAX_MODEL_LEN=@@MAX_MODEL_LEN@@ SAH9B_BASE_MODEL=@@MODEL_PATH@@   # SAH9B_BASE_MODEL: the workers refuse to run if a sourced site file changes SAH_MODEL_PATH
# one evaluation per reserved GPU at a time: kernel timings need an idle GPU and must stay comparable to the baseline
export SAH_EVAL_LOCK_PHASE=whole
export SAH9B_SAH_ROOT=@@SAH_ROOT@@
export SAH9B_RUN_DIR=@@RUN@@
export SAH9B_TASKS="@@TASK_ID@@"
export SAH9B_MAX_EVALS=@@MAX_EVALS@@ SAH9B_EVAL_TIMEOUT=@@EVAL_TIMEOUT@@ SAH9B_EVAL_GPUS=@@EVAL_GPUS@@
export SAH9B_SEED_PROGRAMS_FILE=@@SEED_PROGRAMS@@ SAH9B_BASELINE_OUT=@@BASELINE_OUT@@ SAH9B_BASELINE_SEEDS=@@BASELINE_SEEDS@@
export SAH_RATCHET_EVAL_LIMITS='{"eft__math__hadamard_maximal_det":350}'
# deadline for the worker: job start + wall limit - 15 min (a run that cannot finish before it is skipped; the table is
# written after every completed run, so a job that runs out of time still leaves a usable baseline)
export SAH9B_BASELINE_DEADLINE_EPOCH=$(( $(date +%s) + @@BASELINE_SECONDS@@ - 900 ))   # wall limit @@TIME@@ parsed by the runner
need(){ [ -e "$1" ] || { echo "[sbatch] missing: $2 ($1)" >&2; exit 64; }; }
clean_tree(){ git -C "$1" diff --quiet --no-ext-diff HEAD -- || { echo "[sbatch] $1 has uncommitted changes" >&2; exit 64; }
  [ -z "$(git -C "$1" ls-files --others --exclude-standard)" ] || { echo "[sbatch] $1 has untracked files" >&2; exit 64; }; }
need "$SAH_IMAGE" "container image SAH_IMAGE"; need "$VLLM_ENV/bin/vllm" "vLLM entry point VLLM_ENV/bin/vllm"
case "$SAH9B_TASKS" in *ahc*) need "$SAH_AHC_NATIVE_BIN/docker" "AHC docker shim SAH_AHC_NATIVE_BIN/docker";; esac
clean_tree "$SAH9B_SAH_ROOT"
srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --container-image="$SAH_IMAGE" \
  --container-mounts="@@CONTAINER_MOUNTS@@" \
  --no-container-mount-home \
  --container-workdir="$SAH9B_SAH_ROOT" \
  /bin/bash @@PIPELINE_DIR@@/baseline_worker.sh || echo "[sbatch] srun exit $? $(date -Is)"
