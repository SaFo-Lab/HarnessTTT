#!/usr/bin/env bash
# HarnessTTT: frozen-harness baseline for a (swapped) seed program -> baseline_h2_20ev-format JSON (SAH_BASELINE_JSON of the lineage).
#SBATCH --job-name=@@JOBNAME@@
#SBATCH --account=@@ACCOUNT@@
#SBATCH --partition=@@PARTITION@@
#SBATCH --qos=@@QOS@@
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
source @@ENV_FILE@@
# per-lineage base model (pipeline.py config: model_path / served_model / vllm_tp / max_model_len; defaults = env.sh)
export SAH_MODEL_PATH=@@MODEL_PATH@@ SAH_SERVED_MODEL=@@SERVED_MODEL@@ SAH9B_VLLM_TP=@@VLLM_TP@@ SAH9B_MAX_MODEL_LEN=@@MAX_MODEL_LEN@@ SAH9B_BASE_MODEL=@@MODEL_PATH@@   # SAH9B_BASE_MODEL: the workers refuse to run if a sourced site file changes SAH_MODEL_PATH (MODELCLOBBER-001)
# BENCHLOCK-001 reverted 2026-10-01 19:05 (EVALLOCK-002): with the lock held only around the timed benchmarks, the other
# trajectories' correctness tests and autotuning shared the evaluator GPU - 16 of 32 trajectories lost every evaluation to
# CUDA OOM, and the UNCHANGED SimpleTES seed re-timed 8% faster (1090 vs 1178 us) than in every single-process measurement,
# so scores were no longer comparable to the quiet baseline. One evaluation per GPU at a time (the default 'whole' phase)
# keeps timings consistent; the pruned seed compiles in ~30 s, so the autotune stall that motivated BENCHLOCK-001 is gone.
export SAH_EVAL_LOCK_PHASE=whole
export SAH9B_SAH_ROOT=@@SAH_ROOT@@
export SAH9B_RUN_DIR=@@RUN@@
export SAH9B_TASKS="@@TASK_ID@@"
export SAH9B_MAX_EVALS=@@MAX_EVALS@@ SAH9B_EVAL_TIMEOUT=@@EVAL_TIMEOUT@@ SAH9B_EVAL_GPUS=@@EVAL_GPUS@@
export SAH9B_SEED_PROGRAMS_FILE=@@SEED_PROGRAMS@@ SAH9B_BASELINE_OUT=@@BASELINE_OUT@@ SAH9B_BASELINE_SEEDS=@@BASELINE_SEEDS@@
export SAH_RATCHET_EVAL_LIMITS='{"eft__math__hadamard_maximal_det":350}'
# deadline for the worker: job start + wall limit - 15 min (a run that cannot finish before it is skipped; the table is
# written after every completed run, so a job that runs out of time still leaves a usable baseline)
IFS=: read -r _h _m _s <<< "@@TIME@@"; export SAH9B_BASELINE_DEADLINE_EPOCH=$(( $(date +%s) + 10#$_h * 3600 + 10#$_m * 60 + 10#$_s - 900 ))
test -r "$SAH_IMAGE"; test -x "$VLLM_ENV/bin/vllm"
git -C "$SAH9B_SAH_ROOT" diff --quiet --no-ext-diff HEAD --
test -z "$(git -C "$SAH9B_SAH_ROOT" ls-files --others --exclude-standard)"
srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --container-image="$SAH_IMAGE" \
  --container-mounts="@@CONTAINER_MOUNTS@@" \
  --no-container-mount-home \
  --container-workdir="$SAH9B_SAH_ROOT" \
  /bin/bash @@PIPELINE_DIR@@/baseline_worker.sh || echo "[sbatch] srun exit $? $(date -Is)"
