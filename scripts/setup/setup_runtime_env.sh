#!/usr/bin/env bash
# Build the Python 3.12 environment the executor runtime needs (NexAU agent runtime + the evaluator dependencies of the
# math / ADRS / Hadamard tasks) at $SAH_RUNTIME_ENV (config/local.env), and install this repository into it.
#   bash scripts/setup/setup_runtime_env.sh                 # python3.12 from PATH (or HARNESSTTT_PYTHON312=/path/to/python3.12)
#   bash scripts/setup/setup_runtime_env.sh --dest DIR      # build somewhere else than $SAH_RUNTIME_ENV
#   bash scripts/setup/setup_runtime_env.sh --no-eval-deps  # NexAU + repo only
# NexAU is installed from the pinned, patched checkout at $NEXAU_ROOT (scripts/setup/setup_deps.sh) as a regular package, so
# the checkout stays untouched; re-run after changing it.  vLLM (VLLM_ENV) and the training stack (TRAIN_ENV) are separate
# environments: see config/dependencies.lock.json "environments".
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P); source "${HARNESSTTT_ENV:-$HERE/../../config/env.sh}"
EVAL_DEPS=1; DEST=""
while [ $# -gt 0 ]; do case "$1" in --no-eval-deps) EVAL_DEPS=0;; --dest) DEST="${2:?--dest needs a directory}"; shift;; *) echo "unknown argument: $1" >&2; exit 2;; esac; shift; done
PY="${HARNESSTTT_PYTHON312:-$(command -v python3.12 || true)}"
[ -n "$PY" ] && [ -x "$PY" ] || { echo "python3.12 not found: install Python 3.12 or set HARNESSTTT_PYTHON312=/path/to/python3.12" >&2; exit 2; }
"$PY" -c 'import sys; assert sys.version_info[:2] >= (3, 12), sys.version' || { echo "$PY is not Python >= 3.12" >&2; exit 2; }
[ -d "$NEXAU_ROOT/.git" ] || { echo "NexAU checkout missing at $NEXAU_ROOT: run scripts/setup/setup_deps.sh --nexau first" >&2; exit 2; }
ENV_DIR="${DEST:-${SAH_RUNTIME_ENV:?set SAH_RUNTIME_ENV in config/local.env or pass --dest DIR}}"
if [ ! -x "$ENV_DIR/bin/python" ]; then mkdir -p "$(dirname "$ENV_DIR")"; "$PY" -m venv "$ENV_DIR"; echo "created venv $ENV_DIR"; fi
PIP=("$ENV_DIR/bin/python" -m pip)
"${PIP[@]}" install -q --upgrade pip
"${PIP[@]}" install -q "$NEXAU_ROOT"                         # the pinned, patched agent runtime (built from the checkout; the checkout stays clean)
"${PIP[@]}" install -q -e "$HARNESSTTT_ROOT"                 # harnessttt, inner, outer, trainer
if [ "$EVAL_DEPS" = 1 ]; then
  # evaluator dependencies of the core tasks (MANIFEST_eft.json "deps"): math tasks need numpy/scipy/jax/optax/tqdm, ADRS
  # tasks pandas and torch (CPU build is enough for EPLB), Hadamard/AHC the SimpleTES shim (vendored in the snapshot)
  "${PIP[@]}" install -q numpy scipy pandas psutil tqdm "jax[cpu]" optax
  "${PIP[@]}" install -q torch --index-url https://download.pytorch.org/whl/cpu || "${PIP[@]}" install -q torch
fi
"${PIP[@]}" check || echo "pip check reported conflicts (see above); continuing with the import probe" >&2
"$ENV_DIR/bin/python" - <<'PY'
import importlib.util, sys
mods = ["nexau", "yaml", "numpy", "scipy", "pandas", "psutil", "jsonschema", "openai", "inner", "outer", "trainer", "harnessttt"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("python", sys.version.split()[0], "| importable:", ", ".join(m for m in mods if m not in missing) or "-", "| missing:", ", ".join(missing) or "-")
sys.exit(1 if missing else 0)
PY
echo "runtime env ready: $ENV_DIR (SAH_PYTHON=$ENV_DIR/bin/python)"
