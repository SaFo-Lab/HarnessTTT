#!/usr/bin/env bash
# Clone the OFFICIAL slime (THUDM/slime) at the pinned commit into HarnessTTT/slime (branch harnessttt-pin) and apply the HarnessTTT
# patch set: OpenClaw-RL's Qwen3.5 / FSDP-LoRA modifications + SAH's seven training patches, regenerated as one patch against the pin.
# Idempotent: re-running verifies the pin, the patch checksum and the needles the training launcher depends on.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P); source "${HARNESSTTT_ENV:-$HERE/../../config/env.sh}"
LOCK="$HARNESSTTT_ROOT/config/dependencies.lock.json"
field(){ python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$LOCK" "$1"; }
URL=$(field slime.url); COMMIT=$(field slime.commit); PATCH="$HARNESSTTT_ROOT/$(field slime.patch.path)"; PATCH_SHA=$(field slime.patch.sha256)
if [ ! -d "$SLIME_ROOT/.git" ]; then
  [ ! -e "$SLIME_ROOT" ] || { echo "non-git path blocks the clone: $SLIME_ROOT" >&2; exit 2; }
  git clone "$URL" "$SLIME_ROOT"
fi
if [ "$(git -C "$SLIME_ROOT" rev-parse HEAD)" != "$COMMIT" ]; then
  [ -z "$(git -C "$SLIME_ROOT" status --porcelain)" ] || { echo "refusing to move a dirty slime checkout: $SLIME_ROOT" >&2; exit 2; }
  git -C "$SLIME_ROOT" fetch -q origin "$COMMIT" || git -C "$SLIME_ROOT" fetch -q origin
  git -C "$SLIME_ROOT" checkout -q -B harnessttt-pin "$COMMIT"
fi
echo "$PATCH_SHA  $PATCH" | sha256sum -c - >/dev/null || { echo "patch checksum mismatch: $PATCH" >&2; exit 2; }
if git -C "$SLIME_ROOT" apply --check "$PATCH" 2>/dev/null; then
  git -C "$SLIME_ROOT" apply "$PATCH"; echo "applied $(basename "$PATCH")"
elif git -C "$SLIME_ROOT" apply --check --reverse "$PATCH" 2>/dev/null; then
  echo "patch already applied"
else
  echo "patch neither applies nor is applied cleanly; inspect $SLIME_ROOT (git status / git diff)" >&2; exit 2
fi
fail=0
needles=$(python3 -c 'import json,sys
for rel, needle in json.load(open(sys.argv[1]))["slime"]["required_needles"]: print(f"{rel}\t{needle}")' "$LOCK")   # a failing python3 aborts here (set -e)
while IFS=$'\t' read -r rel needle; do
  grep -Fq -- "$needle" "$SLIME_ROOT/$rel" && echo "[ok] $rel: $needle" || { echo "[missing] $rel: $needle" >&2; fail=1; }
done <<< "$needles"
[ "$fail" = 0 ] || exit 2
echo "slime ready: $SLIME_ROOT @ $(git -C "$SLIME_ROOT" rev-parse --short HEAD) (official $URL) + HarnessTTT patch"
