"""Build an auditable H2 rollout-slot plan for one outer round.

Legacy campaigns launch H2 only for valid H1 proposals.  The inference-16
comparison instead reserves exactly one H2 slot for every one of its eight H1
slots.  If H1 produces an invalid or duplicate proposal, that H2 slot runs the
task-local incumbent harness as an explicit no-op fallback.  The proposal
remains invalid for reward attribution; the fallback exists only to spend the
pre-declared inference trajectory slot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_h2_slot_plan(
    meta: dict[str, Any], *, fixed_slots: bool, forbid_fallback: bool = False,
    allow_missing_invalid: bool = False,
) -> list[dict[str, Any]]:
    """Return ordered H2 rollout slots described by ``round.json``.

    In fixed-slot mode there must be exactly ``K`` candidate records for every
    task, and the returned plan always contains exactly ``K`` H2 slots per
    task.  Invalid proposals use the incoming task-local base package.  In
    legacy mode only valid, materialized candidates are returned.

    ``forbid_fallback`` hard-disables the incumbent-fallback pathway: causal
    curve protocols must never charge an intervention slot that would run the
    incumbent package, so an invalid proposal raises instead of silently
    becoming a fallback rollout (the fair16-v5 failure mode).
    """
    k_target = int(meta["k"])
    plan: list[dict[str, Any]] = []
    for task_id in meta["tasks_order"]:
        task = meta["per_task"][task_id]
        candidates = list(task.get("candidates") or [])
        if fixed_slots and len(candidates) != k_target:
            raise ValueError(
                f"{task_id}: fixed-slot round has {len(candidates)} H1 records, "
                f"expected K={k_target}"
            )
        seen: set[int] = set()
        for candidate in sorted(candidates, key=lambda row: int(row["k"])):
            k = int(candidate["k"])
            if k in seen:
                raise ValueError(f"{task_id}: duplicate candidate index k={k}")
            seen.add(k)
            valid = bool(candidate.get("valid"))
            if not fixed_slots and not valid:
                continue
            if allow_missing_invalid and not valid:
                # REPAIR-001 min-valid gate: an invalid (unrepaired) proposal
                # is charged as a proposer trajectory and trained at minimum
                # reward, but never rolls out — no candidate, no fallback.
                continue
            if forbid_fallback and not valid:
                raise ValueError(
                    f"{task_id}/k{k}: invalid proposal cannot occupy a charged "
                    "intervention slot (stop_reason="
                    f"{candidate.get('stop_reason')!r}); the incumbent-fallback "
                    "pathway is disabled for causal curve runs"
                )
            fallback = not valid
            harness_dir = (
                str(task["base_package"])
                if fallback
                else str(candidate.get("dir") or "")
            )
            if not harness_dir:
                raise ValueError(
                    f"{task_id}/k{k}: valid proposal has no materialized harness"
                )
            # BEAM-001: under beam inheritance a slot may descend from a
            # non-incumbent beam member; its paired control MUST roll out
            # THAT parent, so the parent travels with the slot.
            parent_package = str(
                candidate.get("parent_package") or task["base_package"]
            )
            plan.append({
                "task_id": task_id,
                "k": k,
                "h1_proposal_valid": valid,
                "h2_harness_dir": harness_dir,
                "h2_parent_package": parent_package,
                "h2_slot_mode": (
                    "incumbent_fallback" if fallback else "candidate_harness"
                ),
                "eligible_for_h1_reward": valid,
            })
        if fixed_slots and seen != set(range(k_target)):
            raise ValueError(
                f"{task_id}: fixed-slot candidate indices are {sorted(seen)}, "
                f"expected 0..{k_target - 1}"
            )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", required=True, type=Path)
    parser.add_argument("--fixed-slots", action="store_true")
    parser.add_argument("--forbid-fallback", action="store_true",
                        help="fail instead of planning incumbent-fallback slots "
                             "(required for causal curve protocols)")
    parser.add_argument("--allow-invalid-slots", action="store_true",
                        help="min-valid gate mode: omit invalid proposals from "
                             "the rollout plan instead of failing (they stay "
                             "charged and are trained at minimum reward)")
    args = parser.parse_args()

    meta = json.loads((args.round_dir / "round.json").read_text())
    plan = build_h2_slot_plan(meta, fixed_slots=args.fixed_slots,
                              forbid_fallback=args.forbid_fallback,
                              allow_missing_invalid=args.allow_invalid_slots)
    payload = {
        "schema": 1,
        "axis_unit": "generated_agent_trajectory",
        "fixed_h2_slots": bool(args.fixed_slots),
        "h1_slots_per_task": int(meta["k"]),
        "h2_slots_per_task": (
            (meta.get("inference_trajectory_budget") or {}).get(
                "h2_slots_per_task", int(meta["k"])
            ) if args.fixed_slots else None
        ),
        "h2_slot_plan_rows_per_task": (
            int(meta["k"]) if args.fixed_slots else None
        ),
        "paired_control_slots_per_task": (
            (meta.get("inference_trajectory_budget") or {}).get(
                "paired_control_slots_per_task", 0
            )
        ),
        "causal_attribution": meta.get("causal_attribution"),
        "logical_round_index": (
            meta.get("inference_trajectory_budget") or {}
        ).get("logical_round_index"),
        "axis_x_after_round": (
            meta.get("inference_trajectory_budget") or {}
        ).get("axis_x_after_round"),
        "analyzer_model_calls_excluded_from_axis": True,
        "post_submit_reviewer_model_calls": 0,
        "fallback_slots_forbidden": bool(args.forbid_fallback),
        "invalid_slots_omitted": sorted(
            int(c["k"])
            for task_id in meta["tasks_order"]
            for c in (meta["per_task"][task_id].get("candidates") or [])
            if not c.get("valid")
        ) if args.allow_invalid_slots else [],
        "emitted_h2_slots": len(plan),
        "slots": plan,
    }
    out = args.round_dir / "h2_slot_plan.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
