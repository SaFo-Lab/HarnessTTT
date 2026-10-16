#!/usr/bin/env python3
"""Mask-aware trajectory bounding for the proposer update (replay.jsonl -> replay_bounded.jsonl).

Uses the SAME qwen3_5 loss-mask generator slime applies at train time (``trainer.qwen35_mask``), so every truncation point
(a) fits the real token budget and (b) keeps a non-zero loss mask (response_length > 0).  This avoids both the
flash-linear-attention OOM on long H1 trajectories (14k-32k tokens) and slime's ``response_length == 0`` assertion.

Stage 1: clip verbose tool observations (the loss is masked to assistant tokens, so no learning signal is lost).
Stage 2: if the tokenized row still exceeds the budget, choose a sub-trajectory that fits and has a non-zero mask:
  --truncate-mode terminal (default; what the pipeline uses): keep the system + first user turn plus the longest fitting
      final suffix that starts at an assistant action, so the rewarded terminal action (submit_harness) is never discarded;
  --truncate-mode prefix: keep the longest fitting prefix (any endpoint whose mask is non-zero).
Rows that cannot be made to fit are dropped and reported.  Prints one RESULT line with kept/dropped/truncated counts and
the real-token length percentiles of the output, then re-verifies every kept row with the mask generator.

Ported from the earlier SFT replay-bounding script (the one the 2026-09 runs used).  Behaviour is unchanged apart from
the defaults (--budget 18000, --truncate-mode terminal, --model from SAH_MODEL_PATH), which are the pipeline's arguments.

usage: bound_replay.py --src replay.jsonl --out replay_bounded.jsonl [--budget 18000] [--truncate-mode terminal] [--model <hf dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # HarnessTTT/src: `trainer` importable when run as a script
from trainer.qwen35_mask import Qwen35MultiTurnLossMaskGenerator  # noqa: E402


def clip(s, head, tail):
    if not s or len(s) <= head + tail + 40:
        return s
    return s[:head] + f"\n...[truncated {len(s) - head - tail} chars]...\n" + s[-tail:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=18000, help="real token budget per row (default 18000)")
    ap.add_argument("--tool-head", type=int, default=600)
    ap.add_argument("--tool-tail", type=int, default=300)
    ap.add_argument("--truncate-mode", choices=("prefix", "terminal"), default="terminal",
                    help="terminal keeps system/user plus the longest fitting final action suffix (default)")
    ap.add_argument("--model", default=os.environ.get("SAH_MODEL_PATH"),
                    help="HF checkpoint dir whose tokenizer/chat template slime trains with (default: $SAH_MODEL_PATH)")
    a = ap.parse_args()
    if not a.model:
        ap.error("--model (or SAH_MODEL_PATH) is required: the tokenizer of the frozen base")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
    gen = Qwen35MultiTurnLossMaskGenerator(tok)

    def eval_msgs(msgs, tools):
        # (real_token_len, response_length) from the true train-time generator
        tids, mask = gen.get_loss_mask(msgs, tools=tools or None)
        return len(tids), gen.get_response_lengths([mask])[0]

    rows = [json.loads(line) for line in open(a.src) if line.strip()]
    kept, dropped, truncated = [], 0, 0
    for r in rows:
        tools = r.get("tools")
        msgs = []
        for m in r["messages"]:
            if m.get("role") == "tool":
                m = {**m, "content": clip(m.get("content"), a.tool_head, a.tool_tail)}
            msgs.append(m)

        tlen, rl = eval_msgs(msgs, tools)
        if tlen > a.budget or rl == 0:
            chosen = None
            if a.truncate_mode == "terminal":
                # The GRPO reward belongs to the terminal outcome: preserve the original system/user contract and the longest
                # final suffix beginning at an assistant action, so the rewarded submit action is never discarded merely to fit.
                for i in range(2, len(msgs)):
                    if msgs[i].get("role") != "assistant":
                        continue
                    cand = msgs[:2] + msgs[i:]
                    cl, crl = eval_msgs(cand, tools)
                    if cl <= a.budget and crl > 0:
                        chosen = cand
                        if i > 2:
                            truncated += 1
                        break
            else:
                # Consider all prefix endpoints (not just assistant): the qwen3_5 generator marks assistant turns closed by a
                # following tool/user message, so a tool endpoint retains a non-zero mask.
                for i in reversed(range(2, len(msgs))):
                    cand = msgs[: i + 1]
                    cl, crl = eval_msgs(cand, tools)
                    if cl <= a.budget and crl > 0:
                        chosen = cand
                        if i + 1 < len(msgs):
                            truncated += 1
                        break
            if chosen is None:
                dropped += 1
                continue
            msgs = chosen
        kept.append({**r, "messages": msgs})

    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # final verification pass with the same generator
    lens, zero = [], 0
    for r in kept:
        tl, rl = eval_msgs(r["messages"], r.get("tools"))
        lens.append(tl)
        if rl == 0:
            zero += 1
    lens.sort()
    n = len(lens)
    stats = (f"real_tok p50={lens[n // 2]} p90={lens[int(n * 0.9)]} max={lens[-1]}" if n else "real_tok n/a")
    print(f"RESULT mode={a.truncate_mode} in={len(rows)} kept={n} dropped={dropped} truncated={truncated} "
          f"| {stats} | over_budget={sum(x > a.budget for x in lens)} ZERO_MASK={zero}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
