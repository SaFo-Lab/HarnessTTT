#!/usr/bin/env python3
"""Pad a bounded replay to a multiple of the training GPU count with zero-advantage duplicates (KL-only rows).

slime splits the training batch over dp_size = number of training GPUs and injects metadata-less dummy samples when there
are fewer rows than GPUs (``KeyError: 'raw_reward'`` on 7 rows / 8 GPUs, observed 2026-09-23).  Duplicates of real rows with
``metadata.advantage = 0.0`` contribute no policy-gradient term (only the KL term) and are tagged
``metadata.padding = true`` / ``metadata.padding_of = <sample_id>`` so analyses can drop them.

usage: pad_replay.py <replay.jsonl> <n_gpus>      (rewrites the file in place; prints "<total_rows> <padding_rows>")
"""
from __future__ import annotations

import copy
import json
import sys


def main() -> int:
    path, n = sys.argv[1], int(sys.argv[2])
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        print("0 0")
        return 0
    target = ((len(rows) + n - 1) // n) * n
    pad = []
    for i in range(target - len(rows)):
        r = copy.deepcopy(rows[i % len(rows)])
        m = r.setdefault("metadata", {})
        m["advantage"] = 0.0
        m["padding"] = True
        m["padding_of"] = m.get("sample_id")
        pad.append(r)
    if pad:
        with open(path, "w") as fh:
            for r in rows + pad:
                fh.write(json.dumps(r) + "\n")
    print(len(rows) + len(pad), len(pad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
