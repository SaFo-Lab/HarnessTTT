#!/usr/bin/env bash
# HarnessTTT environment.  Source this from every launcher, worker and interactive shell:
#     source /path/to/HarnessTTT/config/env.sh
# Resolution order: (1) variables already exported by the caller win; (2) config/local.env supplies the site payloads that are not
# part of this tree (frozen model, container image, Python environments, NexAU, Megatron-LM, Slurm account/partition/QOS);
# (3) the in-tree defaults below.  Nothing here may contain credentials: Slurm launchers source and log it.
_HTT_CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export HARNESSTTT_ROOT="${HARNESSTTT_ROOT:-$(cd "$_HTT_CFG/.." && pwd -P)}"
export SAH_ROOT="${SAH_ROOT:-$HARNESSTTT_ROOT}"                 # the SAH code reads SAH_ROOT; src/ lives directly below it

# --- in-tree components ---------------------------------------------------------------------------------------------
export SLIME_ROOT="${SLIME_ROOT:-$HARNESSTTT_ROOT/slime}"      # official THUDM/slime @ pinned commit + scripts/patches/slime
# the task snapshot: SAH_DATASET_ROOT (the loader's variable) and SAH_DATA_ROOT (the scripts') are one setting; either one set
# by the caller or config/local.env relocates the snapshot
export SAH_DATA_ROOT="${SAH_DATA_ROOT:-${SAH_DATASET_ROOT:-$HARNESSTTT_ROOT/dataset/self_adapt_harness}}"
export SAH_DATASET_ROOT="${SAH_DATASET_ROOT:-$SAH_DATA_ROOT}"   # read by src/inner/tasks/eft_task.py
export SAH_TASK_TEXT_REGISTRY="${SAH_TASK_TEXT_REGISTRY:-$HARNESSTTT_ROOT/config/task_text_registry.json}"
export SAH_FINCH_TARGETS="${SAH_FINCH_TARGETS:-$HARNESSTTT_ROOT/config/finch_targets.json}"
export SAH_BASELINE_JSON="${SAH_BASELINE_JSON:-$HARNESSTTT_ROOT/config/baseline_h2_20ev.json}"   # 20-eval baseline H2 scores (static round-1 bases)
export SAH_AHC_NATIVE_BIN="${SAH_AHC_NATIVE_BIN:-$SAH_DATA_ROOT/runtime/ahc_native/bin}"
export AHC_CACHE_DIR="${AHC_CACHE_DIR:-$SAH_DATA_ROOT/raw/simpletes-b7e0367/datasets/ahc/cache}"
export AHC_RUNNER_SCRIPT="${AHC_RUNNER_SCRIPT:-$SAH_DATA_ROOT/raw/simpletes-b7e0367/datasets/ahc/docker_runner.py}"

# --- site payloads (config/local.env; see config/local.env.example) ------------------------------------------------
if [ -r "$HARNESSTTT_ROOT/config/local.env" ]; then
  # shellcheck disable=SC1091
  source "$HARNESSTTT_ROOT/config/local.env"
fi
export HARNESSTTT_DEPS="${HARNESSTTT_DEPS:-$HARNESSTTT_ROOT/.deps}"       # where scripts/setup/setup_deps.sh clones external code
export NEXAU_ROOT="${NEXAU_ROOT:-$HARNESSTTT_DEPS/NexAU}"                  # H1/H2 agent runtime (pinned; see config/dependencies.lock.json)
export MEGATRON_ROOT="${MEGATRON_ROOT:-$HARNESSTTT_DEPS/Megatron-LM}"      # slime imports megatron.core even on the FSDP backend
export RUN_ROOT="${RUN_ROOT:-$HARNESSTTT_ROOT/runs}"                       # campaign run dirs, GPU_CAP file, generated sbatch files
export LOG_ROOT="${LOG_ROOT:-$RUN_ROOT}"
export SAH_SERVED_MODEL="${SAH_SERVED_MODEL:-qwen3.5-9b}"
export SAH_PYTHON="${SAH_PYTHON:-${SAH_RUNTIME_ENV:+$SAH_RUNTIME_ENV/bin/python}}"   # resolves inside the SAH container
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$HARNESSTTT_ROOT:$HARNESSTTT_ROOT,/tmp:/tmp}"
export HTT_SLURM_ACCOUNT="${HTT_SLURM_ACCOUNT:-}"                        # set in config/local.env (required to submit jobs)
export HTT_SLURM_PARTITION="${HTT_SLURM_PARTITION:-}"
export HTT_SLURM_QOS="${HTT_SLURM_QOS:-}"                                # canonical jobs use the low-priority QOS only
export HARNESSTTT_GPU_CAP="${HARNESSTTT_GPU_CAP:-48}"                      # chain.sh: total GPUs of the user's jobs incl. pending
export HARNESSTTT_MAX_JOBS="${HARNESSTTT_MAX_JOBS:-15}"                    # chain.sh: never submit above this queue depth

# --- keep library state out of $HOME ---------------------------------------------------------------------------------
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RUN_ROOT/cache/xdg}"
export HF_HOME="${HF_HOME:-$RUN_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$RUN_ROOT/cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_ROOT/cache/matplotlib}"
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export TMPDIR="${TMPDIR:-/tmp}"                                            # node-local; persistent artefacts stay on shared storage
unset _HTT_CFG
