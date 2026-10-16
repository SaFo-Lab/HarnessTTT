#!/usr/bin/env bash
# Read-only preflight for a HarnessTTT checkout: tree, pinned dependencies, site payloads, Slurm settings, Python imports.
#   bash scripts/setup/preflight.sh                  (exit 1 with a list of [missing] items when anything is off)
#   bash scripts/setup/preflight.sh --executor-only  (skip the training stack: slime, Megatron-LM, TRAIN_ENV)
# HARNESSTTT_HOST_PYTHON=<python with PyYAML/numpy> enables the import probe on the submitting host (SAH_PYTHON may resolve only
# inside the container).  HARNESSTTT_REQUIRE_LOW_QOS=1 makes the QOS check insist on a *_low QOS (site policy).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P); source "${HARNESSTTT_ENV:-$HERE/../../config/env.sh}"
LOCK="$HARNESSTTT_ROOT/config/dependencies.lock.json"; errors=0; TRAIN=1
for a in "$@"; do case "$a" in --executor-only) TRAIN=0;; *) echo "unknown argument: $a" >&2; exit 2;; esac; done
ok(){ printf '[ok] %s\n' "$*"; }; bad(){ printf '[missing] %s\n' "$*" >&2; errors=$((errors + 1)); }
req_dir(){ [ -d "$1" ] && ok "$2: $1" || bad "$2: $1"; }; req_file(){ [ -f "$1" ] && ok "$2: $1" || bad "$2: $1"; }
req_entry(){ { [ -e "$1" ] || [ -L "$1" ]; } && ok "$2: $1" || bad "$2: $1"; }
field(){ python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$LOCK" "$1"; }
echo "HarnessTTT root: $HARNESSTTT_ROOT"
req_file "$LOCK" "dependency lock"
for f in config/env.sh config/finch_targets.json config/task_text_registry.json config/baseline_h2_20ev.json src/inner/tasks/eft_task.py src/outer/rounds/outer_round.py src/trainer/grpo_to_replay.py \
         scripts/pipeline/worker_round.sh scripts/pipeline/train_worker.sh scripts/pipeline/chain.sh scripts/pipeline/templates/round.sbatch scripts/pipeline/templates/train.sbatch; do
  req_file "$HARNESSTTT_ROOT/$f" "tree"; done
[ -r "$HARNESSTTT_ROOT/config/local.env" ] && ok "config/local.env" || bad "config/local.env (copy config/local.env.example)"
# --- slime (official pin + patch) ---
if [ "$TRAIN" = 0 ]; then ok "training stack skipped (--executor-only)"
elif [ -d "$SLIME_ROOT/.git" ]; then
  exp=$(field slime.commit); obs=$(git -C "$SLIME_ROOT" rev-parse HEAD); [ "$obs" = "$exp" ] && ok "slime pinned at $obs" || bad "slime revision $obs != $exp"
  git -C "$SLIME_ROOT" remote get-url origin | grep -q 'THUDM/slime' && ok "slime remote is the official THUDM/slime" || bad "slime remote is not THUDM/slime"
  while IFS=$'\t' read -r rel needle; do grep -Fq -- "$needle" "$SLIME_ROOT/$rel" && ok "slime patch: $needle" || bad "slime patch: $needle ($rel)"; done \
    < <(python3 -c 'import json,sys
for rel, needle in json.load(open(sys.argv[1]))["slime"]["required_needles"]: print(f"{rel}\t{needle}")' "$LOCK")
else bad "slime checkout: $SLIME_ROOT (run scripts/setup/setup_slime.sh)"; fi
# --- dataset ---
req_dir "$SAH_DATA_ROOT" "evaluator snapshot"
if [ -f "$SAH_DATA_ROOT/MANIFEST_eft.json" ]; then
  [ "$(sha256sum "$SAH_DATA_ROOT/MANIFEST_eft.json" | awk '{print $1}')" = "$(field dataset.manifest_sha256)" ] && ok "dataset manifest hash" || bad "dataset manifest hash mismatch"; fi
req_entry "$SAH_AHC_NATIVE_BIN/docker" "AHC docker shim"
# --- external code ---
if [ -d "$NEXAU_ROOT" ]; then
  [ -d "$NEXAU_ROOT/.git" ] && { obs=$(git -C "$NEXAU_ROOT" rev-parse HEAD); [ "$obs" = "$(field nexau.commit)" ] && ok "NexAU pinned at $obs" || bad "NexAU revision $obs != $(field nexau.commit)"; }
  while IFS=$'\t' read -r rel needle; do grep -Fq -- "$needle" "$NEXAU_ROOT/$rel" && ok "NexAU patch: $needle" || bad "NexAU patch: $needle"; done \
    < <(python3 -c 'import json,sys
for rel, needle in json.load(open(sys.argv[1]))["nexau"]["needles"]: print(f"{rel}\t{needle}")' "$LOCK")
else bad "NexAU: $NEXAU_ROOT (run scripts/setup/setup_deps.sh --nexau)"; fi
if [ "$TRAIN" = 1 ]; then
  [ -f "$MEGATRON_ROOT/megatron/core/package_info.py" ] && ok "Megatron-LM: $MEGATRON_ROOT ($(grep -E '^(MAJOR|MINOR|PATCH|PRE_RELEASE)' "$MEGATRON_ROOT/megatron/core/package_info.py" | tr -d ' ' | tr '\n' ' '))" || bad "Megatron-LM: $MEGATRON_ROOT"
fi
# --- site payloads ---
req_dir "${SAH_MODEL_PATH:-<unset>}" "Qwen3.5-9B (SAH_MODEL_PATH)"
req_file "${SAH_IMAGE:-<unset>}" "container image (SAH_IMAGE)"
req_entry "${VLLM_ENV:-<unset>}/bin/vllm" "vLLM entry (VLLM_ENV)"
req_entry "${SAH_PYTHON:-<unset>}" "SAH runtime Python (SAH_PYTHON)"
[ "$TRAIN" = 1 ] && { req_entry "${TRAIN_ENV:-<unset>}/bin/ray" "training env ray (TRAIN_ENV)"; }
# --- Slurm settings ---
for v in HTT_SLURM_ACCOUNT HTT_SLURM_PARTITION HTT_SLURM_QOS; do
  val="${!v:-}"; case "$val" in ""|*"<"*) bad "$v is unset or still a placeholder (config/local.env)";; *) ok "$v=$val";; esac; done
if [ "${HARNESSTTT_REQUIRE_LOW_QOS:-0}" = 1 ]; then [[ "$HTT_SLURM_QOS" == *_low ]] && ok "low QOS: $HTT_SLURM_QOS" || bad "HARNESSTTT_REQUIRE_LOW_QOS=1 but HTT_SLURM_QOS=$HTT_SLURM_QOS is not a *_low QOS"; fi
grep -q -- '--qos=@@QOS@@' "$HARNESSTTT_ROOT/scripts/pipeline/templates/round.sbatch" "$HARNESSTTT_ROOT/scripts/pipeline/templates/train.sbatch" && ok "templates take QOS from env" || bad "templates hard-code a QOS"
# --- Python import probe ---
PROBE="${HARNESSTTT_HOST_PYTHON:-}"; [ -n "$PROBE" ] || { [ -n "${SAH_PYTHON:-}" ] && [ -x "$SAH_PYTHON" ] && PROBE="$SAH_PYTHON"; } || true
if [ -n "$PROBE" ] && [ -d "$NEXAU_ROOT" ]; then
  if PYTHONPATH="$HARNESSTTT_ROOT/src:$NEXAU_ROOT${PYTHONPATH:+:$PYTHONPATH}" SAH_DATASET_ROOT="$SAH_DATA_ROOT" "$PROBE" - <<'PY'
from nexau import Agent
import yaml, numpy
import inner.tasks.eft_task as e, outer.rounds.outer_round, outer.reward.rewards, trainer.grpo_to_replay, trainer.qwen35_format, trainer.qwen35_mask
tasks = e.load_tasks(); assert len(tasks) >= 11, len(tasks)
print(f"imports ok; {len(tasks)} tasks from {e.DATASET_ROOT}")
PY
  then ok "Python imports + task registry ($PROBE)"; else bad "Python imports / task registry ($PROBE)"; fi
else ok "import probe skipped (set HARNESSTTT_HOST_PYTHON on the submitting host; SAH_PYTHON may resolve only inside the container)"; fi
[ "$errors" -eq 0 ] || { echo "preflight failed: $errors issue(s)" >&2; exit 1; }
echo "HarnessTTT preflight passed"
