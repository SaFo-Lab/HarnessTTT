"""Verify live task texts against the pinned registry (anti-tampering).

The curated-note incident entered through an edited task message.  The
registry (scripts/tools/build_task_text_registry.py) pins sha256 of every task's
spec text and initial program from the frozen dataset; each round verifies the
texts it is about to serve.  Mismatches are recorded always and fatal under
``SAH_TASK_TEXT_ENFORCE=1``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

REGISTRY_PATH = Path(os.environ.get(
    "SAH_TASK_TEXT_REGISTRY",
    str(Path(__file__).resolve().parents[3] / "config" / "task_text_registry.json"),
))


def verify_task_texts(tasks: Dict[str, Any]) -> Dict[str, Any]:
    """tasks: {task_id: EFTTask}.  Returns the provenance record."""
    enforce = os.environ.get("SAH_TASK_TEXT_ENFORCE", "0") == "1"
    record: Dict[str, Any] = {
        "schema": "task-text-provenance/1.0",
        "registry": str(REGISTRY_PATH),
        "registry_present": REGISTRY_PATH.is_file(),
        "enforced": enforce,
        "tasks": {},
    }
    if not record["registry_present"]:
        if enforce:
            raise RuntimeError(
                "SAH_TASK_TEXT_ENFORCE=1 but the task-text registry is missing; "
                "run scripts/tools/build_task_text_registry.py first"
            )
        return record
    registry = json.loads(REGISTRY_PATH.read_text()).get("tasks", {})
    mismatches = []
    for tid, task in tasks.items():
        pinned = registry.get(tid)
        row = {
            "spec_sha256": hashlib.sha256(task.spec.encode()).hexdigest(),
            "initial_program_sha256": hashlib.sha256(
                task.initial_program.encode()
            ).hexdigest(),
        }
        if pinned is None:
            row["status"] = "unpinned"
        elif (row["spec_sha256"] == pinned["spec_sha256"]
                and row["initial_program_sha256"] == pinned["initial_program_sha256"]):
            row["status"] = "match"
        else:
            row["status"] = "MISMATCH"
            mismatches.append(tid)
        record["tasks"][tid] = row
    record["mismatches"] = mismatches
    if mismatches and enforce:
        raise RuntimeError(
            f"task text mismatch vs pinned registry for {mismatches}; "
            "a task message or seed program was modified"
        )
    return record
