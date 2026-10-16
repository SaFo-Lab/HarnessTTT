#!/usr/bin/env bash
# Fetch the external code HarnessTTT needs but does not vendor: NexAU (agent runtime, pinned + transport patch) and Megatron-LM
# (slime import dependency, official tag).  Targets are NEXAU_ROOT / MEGATRON_ROOT from config/env.sh (+ config/local.env);
# with neither set they default to $HARNESSTTT_DEPS (= HarnessTTT/.deps, ignored by git).
#   usage: bash scripts/setup/setup_deps.sh [--nexau] [--megatron]     (default: both)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P); source "${HARNESSTTT_ENV:-$HERE/../../config/env.sh}"
LOCK="$HARNESSTTT_ROOT/config/dependencies.lock.json"
field(){ python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$LOCK" "$1"; }
DO_NEXAU=0; DO_MEG=0
for a in "$@"; do case "$a" in --nexau) DO_NEXAU=1;; --megatron) DO_MEG=1;; *) echo "unknown argument: $a" >&2; exit 2;; esac; done
[ "$DO_NEXAU$DO_MEG" != 00 ] || { DO_NEXAU=1; DO_MEG=1; }
clone_pin(){ local url=$1 commit=$2 target=$3
  if [ ! -d "$target/.git" ]; then [ ! -e "$target" ] || { echo "non-git path blocks the clone: $target" >&2; exit 2; }; mkdir -p "$(dirname "$target")"; git clone "$url" "$target"; fi
  if [ "$(git -C "$target" rev-parse HEAD)" != "$commit" ]; then
    [ -z "$(git -C "$target" status --porcelain)" ] || { echo "refusing to move a dirty checkout: $target" >&2; exit 2; }
    git -C "$target" fetch -q origin "$commit" || git -C "$target" fetch -q --tags origin
    git -C "$target" checkout -q --detach "$commit"
  fi; }
if [ "$DO_NEXAU" = 1 ]; then
  clone_pin "$(field nexau.url)" "$(field nexau.commit)" "$NEXAU_ROOT"
  PATCH="$HARNESSTTT_ROOT/$(field nexau.patch.path)"
  [ "$(sha256sum "$PATCH" | awk '{print $1}')" = "$(field nexau.patch.sha256)" ] || { echo "patch file $PATCH does not match config/dependencies.lock.json" >&2; exit 2; }
  if git -C "$NEXAU_ROOT" apply --check --reverse "$PATCH" 2>/dev/null; then echo "NexAU patch already applied"
  elif git -C "$NEXAU_ROOT" apply --check "$PATCH" 2>/dev/null; then git -C "$NEXAU_ROOT" apply "$PATCH"; echo "NexAU patch applied"
  else echo "NexAU checkout $NEXAU_ROOT neither has the patch applied nor accepts it cleanly (local modifications?)" >&2; exit 2; fi
  while IFS=$'\t' read -r rel needle; do grep -Fq -- "$needle" "$NEXAU_ROOT/$rel" || { echo "NexAU patch needle missing: $needle ($rel)" >&2; exit 2; }; done \
    < <(python3 -c 'import json,sys
for rel, needle in json.load(open(sys.argv[1]))["nexau"]["needles"]: print(f"{rel}\t{needle}")' "$LOCK")
  echo "NexAU ready: $NEXAU_ROOT @ $(git -C "$NEXAU_ROOT" rev-parse --short HEAD) + nexau_transport.patch"
fi
if [ "$DO_MEG" = 1 ]; then
  if [ -d "$MEGATRON_ROOT" ] && [ -f "$MEGATRON_ROOT/megatron/core/package_info.py" ]; then
    echo "Megatron-LM present: $MEGATRON_ROOT ($(grep -E '^(MAJOR|MINOR|PATCH|PRE_RELEASE)' "$MEGATRON_ROOT/megatron/core/package_info.py" | tr '\n' ' '))"
  else
    tag=$(field megatron_lm.tag); url=$(field megatron_lm.url)
    [ ! -e "$MEGATRON_ROOT" ] || { echo "non-git path blocks the clone: $MEGATRON_ROOT" >&2; exit 2; }
    mkdir -p "$(dirname "$MEGATRON_ROOT")"; git clone --depth 1 --branch "$tag" "$url" "$MEGATRON_ROOT"
    echo "Megatron-LM ready: $MEGATRON_ROOT @ $tag"
  fi
fi
