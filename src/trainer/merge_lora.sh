#!/usr/bin/env bash
# Merge the latest slime LoRA adapter into the frozen HF base -> a full checkpoint vLLM serves as the next round's proposer.
# Runs inside the training env (transformers + peft with Qwen3.5 linear-attention support; PATH must start with $TRAIN_ENV/bin).
# slime provides the merge tool (tools/merge_lora_adapter.py, part of the HarnessTTT slime patch); this script owns checkpoint
# selection and the non-zero adapter gate: an all-zero lora_B means training was a no-op, so the merge is refused.
#   env: SLIME_ROOT, HF_CKPT (frozen base), CKPT_DIR (slime --save dir), OUT (merged output dir)
set -euo pipefail
: "${SLIME_ROOT:?}" "${HF_CKPT:?}" "${CKPT_DIR:?}" "${OUT:?}"
export PYTHONPATH="$SLIME_ROOT:${PYTHONPATH:-}"
ITER=$(<"$CKPT_DIR/latest_checkpointed_iteration.txt"); printf -v ITER_PAD 'iter_%07d' "$ITER"
ADAPTER="$CKPT_DIR/$ITER_PAD/model"
[ -f "$ADAPTER/adapter_weights.pt" ] || { echo "adapter weights not found: $ADAPTER/adapter_weights.pt" >&2; exit 2; }
echo "merging adapter $ADAPTER (iteration $ITER) into $HF_CKPT"
python3 - "$ADAPTER/adapter_weights.pt" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
values = [value.float().abs().sum().item() for name, value in state.items() if "lora_B" in name]
total = sum(values)
print(f"[sanity] {len(values)} lora_B tensors, sum|B|={total:.6g}")
if not values or total <= 0:
    raise SystemExit("LoRA adapter is empty or all-zero; refusing to merge")
print("[sanity] OK: adapter is trained")
PY
python3 "$SLIME_ROOT/tools/merge_lora_adapter.py" --base-model "$HF_CKPT" --adapter "$ADAPTER" --output "$OUT" --force
echo "MERGE_DONE $OUT"
