#!/usr/bin/env python3
"""Pin the canonical task texts so tampering with a task message is detectable.

Writes config/task_text_registry.json: {task_id: {spec_sha256, initial_program_sha256}} computed from the frozen dataset via
the same loader the pipeline uses.  outer_round verifies every round against this registry (fail-closed under
SAH_TASK_TEXT_ENFORCE=1) -- the mechanical answer to the curated-note incident, where an edited task message carried a
known program.  Re-run only when the dataset snapshot changes; commit the result.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("HARNESSTTT_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "src"))

from inner.tasks.eft_task import load_tasks  # noqa: E402


def main() -> None:
    check = "--check" in sys.argv[1:]      # compare the snapshot with the committed registry instead of rewriting it
    registry = {}
    for t in load_tasks():
        registry[t.task_id] = {
            "spec_sha256": hashlib.sha256(t.spec.encode()).hexdigest(),
            "initial_program_sha256": hashlib.sha256(t.initial_program.encode()).hexdigest(),
        }
    out = Path(os.environ.get("SAH_TASK_TEXT_REGISTRY", str(ROOT / "config" / "task_text_registry.json")))
    if check:
        pinned = json.loads(out.read_text()).get("tasks", {}) if out.is_file() else {}
        bad = [tid for tid, row in registry.items() if pinned.get(tid) not in (None, row)]
        unpinned = [tid for tid in registry if tid not in pinned]
        print(f"{len(registry)} tasks loaded; {len(registry) - len(bad) - len(unpinned)} match the registry, "
              f"{len(unpinned)} unpinned, {len(bad)} MISMATCH" + (": " + ", ".join(bad) if bad else ""))
        raise SystemExit(1 if bad else 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "task-text-registry/1.0", "tasks": registry}, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(registry)} tasks)")


if __name__ == "__main__":
    main()
