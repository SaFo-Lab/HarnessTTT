"""Outer-round orchestrator CLI (instance-wise).

Per round, for EACH task instance tau in the task list:
K H1-agent runs conditioned on tau -> K candidate H2 packages for tau ->
each rolled out once on tau by the inner loop -> per-task GRPO group.

Stages (run inside the serving job; see scripts/outer_round.sbatch):

  propose  for every (task, k): run H1 against M_phi -> validate -> materialize
           <round_dir>/tasks/<task_id>/candNN/ + round.json/prompts/trajectories.
  collect  after rollouts: per-task rewards + group advantages ->
           grpo_batch.jsonl (K x n_tasks rows) + round_summary.json +
           next_bases.json (per-task best harness -> next round's bases).

Bases registry: {task_id: {"package": dir, "score": s}} — which H2 package is
the current best for each task and what it scored. Round 1 defaults to the
initial hand-written H2 + results/baseline_h2_20ev.json scores; later rounds
pass --bases-file <prev_round>/next_bases.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from outer.proposing import proposer_io as pio, propose as pp  # noqa: E402
from outer.proposing.component_portfolio import (  # noqa: E402
    ALLOWED_SURFACES,
    PORTFOLIO_VERSION,
    h2_compute_contract,
    implementation_change_inventory,
    portfolio_enabled,
    role_for_slot,
)
from outer.genome import harness_spec as hs  # noqa: E402
from outer.reward import rewards as rw  # noqa: E402
from outer.reward.program_ratchet import (  # noqa: E402
    mark_over_cap_candidates, suppress_task_training, update_program_ratchet,
)
from outer.compiling.materialize import materialize, INNER_HARNESS  # noqa: E402
from inner.runtime.package_hash import h2_sha256  # noqa: E402

REPO = _SRC.parent
def _baseline_json() -> Path:
    """The 20-eval baseline table, wherever this tree keeps it.

    The research tree ships it under results/ and the public tree under data/.
    That difference used to live as two divergent copies of this line, guarded
    by a whitelist entry in scripts/check_tree_sync.py -- which meant a routine
    file sync between the trees silently clobbered it.  It did: a round job
    then died five times on a missing data/baseline_h2_20ev.json before the
    driver gave up.  Resolving the location at runtime removes the divergence
    instead of documenting it.  SAH_BASELINE_JSON still wins when set.
    """
    override = os.environ.get("SAH_BASELINE_JSON")
    if override:
        return Path(override)
    # HarnessTTT keeps every repo-level reference table under config/; the two legacy locations are still probed.
    for candidate in (REPO / "config" / "baseline_h2_20ev.json",
                      REPO / "results" / "baseline_h2_20ev.json",
                      REPO / "data" / "baseline_h2_20ev.json"):
        if candidate.is_file():
            return candidate
    return REPO / "config" / "baseline_h2_20ev.json"


BASELINE_JSON = _baseline_json()

# HISTORICAL ONLY: nothing is seeded into a candidate draft any more, so no
# new round can produce these names.  They stay recognised because rounds
# recorded before the forced-tool contract was removed still carry a seeded
# `quick_probe*` / `slot_tool_k*` tool in changed_fields, and those must not be
# counted as components the proposer chose to write.
_SCAFFOLD_TOOL_FIELDS = ("new_tools.slot_tool_k", "new_tools.quick_probe")


def _is_scaffold_field(field: Any) -> bool:
    return str(field).startswith(_SCAFFOLD_TOOL_FIELDS)


DEFAULT_TASKS = [
    "eft__math__circle_packing",
    "eft__math__hadamard_maximal_det",
    "eft__math__erdos_min_overlap",
    "adrs__prism",
    "adrs__txn_scheduling",
    "adrs__eplb",
    "eft__algotune__convolve2d_full_fill",
    "eft__algotune__psd_cone_projection",
]


def _load_bases(bases_file: str | None, tasks: list) -> Dict[str, Dict[str, Any]]:
    """task_id -> {package, score, seed_score}.

    COLDSTART-001: a task absent from the static baseline file (any newly
    onboarded task — the next phase runs CODE tasks) starts from the fixed
    H2 with score 0.0 and a ``cold_start`` marker instead of KeyError-ing
    the round.  The number is honest within one round: round 0's paired
    controls are rollouts of this very base, and BASEREFRESH-001 replaces
    the placeholder with their mean before round 1 inherits it.
    """
    static = json.loads(BASELINE_JSON.read_text())["baseline"]

    def _static_entry(t: str) -> Dict[str, Any]:
        if t in static:
            return {"package": str(INNER_HARNESS),
                    "score": static[t]["h2_best"]}
        print(f"[bases] {t}: not in {BASELINE_JSON.name} — cold start "
              "(score 0.0 until round-0 controls refresh it)")
        return {"package": str(INNER_HARNESS), "score": 0.0,
                "cold_start": True}

    if bases_file:
        bases = json.loads(Path(bases_file).read_text())
    else:
        bases = {t: _static_entry(t) for t in tasks}
    for t in tasks:
        if t not in bases:
            bases[t] = _static_entry(t)
        bases[t].setdefault("seed_score", static.get(t, {}).get("seed", 0.0))
    return bases


def _input_file_provenance(path_value: str | None) -> Dict[str, Any]:
    """Hash a mutable cross-round input at the moment this round consumes it."""
    if not path_value:
        return {"path": None, "exists": False, "sha256": None, "bytes": 0}
    path = Path(path_value)
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None, "bytes": 0}
    raw = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _replica_base_urls(base_url: str, n_replicas: int) -> list[str]:
    """Expand a concrete base endpoint across consecutive local ports."""
    if n_replicas <= 0:
        return [base_url]
    parsed = urlsplit(base_url)
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(
            "--base-url must include a hostname and port when --n-replicas > 0"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("userinfo is not supported in replicated --base-url")
    last_port = parsed.port + n_replicas - 1
    if last_port > 65535:
        raise ValueError("replica port range exceeds 65535")
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    return [
        urlunsplit((
            parsed.scheme,
            f"{host}:{parsed.port + index}",
            parsed.path,
            parsed.query,
            parsed.fragment,
        ))
        for index in range(n_replicas)
    ]


def _load_slot_prompt_map(round_dir: Path):
    """Load exact per-slot proposer inputs, or ``None`` for legacy rounds.

    v1.1 records every exact model input because preload appends the private H2
    filesystem. Condition-identical slots must contain byte-identical messages,
    but training must still bind each response to the exact input it saw.
    Falling back to ``prompts.json`` for a new round would lose that proof, so
    malformed new artifacts fail closed. Only older rounds retain the fallback.
    """

    path = Path(round_dir) / "slot_prompts.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"cannot read exact proposer prompts: {path}") from exc
    if payload.get("schema") != "h1-slot-prompts/1.0":
        raise RuntimeError(f"unsupported slot prompt schema in {path}")
    rows = payload.get("slots")
    if not isinstance(rows, list):
        raise RuntimeError(f"slot prompt artifact has no slots list: {path}")
    prompts = {}
    for row in rows:
        try:
            key = (str(row["task_id"]), int(row["k"]))
            message = row["user_message"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"malformed slot prompt row in {path}") from exc
        if key in prompts:
            raise RuntimeError(f"duplicate exact proposer prompt for {key}")
        if row.get("sent_to_model") is not True or not isinstance(message, str) \
                or not message:
            raise RuntimeError(f"slot {key} has no exact model input in {path}")
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
        if row.get("sha256") != digest:
            raise RuntimeError(f"slot {key} proposer prompt hash mismatch")
        prompts[key] = message
    return prompts


def _canonical_sha256(payload: Dict[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _resolved_h2_compute_contract(
    spec: Dict[str, Any], *, max_evals: int,
) -> Dict[str, Any]:
    """Bind proposal metadata to the orchestrator's actual H2 ceilings."""

    raw_context = os.environ.get("SAH_MAX_CONTEXT_TOKENS", "")
    max_context = (
        min(131072, int(raw_context))
        if raw_context.isdigit() and int(raw_context) > 0 else 131072
    )
    raw_min_iterations = os.environ.get("SAH_MIN_ITERS", "0")
    return h2_compute_contract(
        spec,
        max_evaluator_calls=int(max_evals),
        # run_baseline/H2Config fixes this at 30; do not sign an ambient env
        # value that the runtime does not consume.
        max_probe_calls=30,
        eval_timeout_s=float(os.environ.get("EVAL_TIMEOUT", "360") or 360),
        trajectory_wall_timeout_s=float(
            os.environ.get("ROLLOUT_WALL_TIMEOUT", "0") or 0
        ),
        max_context_tokens=max_context,
        min_executor_iterations=(
            int(raw_min_iterations) if raw_min_iterations.isdigit() else 0
        ),
    )


def _proposal_condition(
    *, task_id: str, round_index: int, role: str,
    parent_h2_sha256: str | None, input_prompt_sha256: str | None,
    system_prompt_sha256: str, behavior_policy: str,
    behavior_policy_identity: str | None, resolved_h1_config: Dict[str, Any],
    parent_compute_contract: Dict[str, Any], sampling_mode: str,
) -> tuple[str, Dict[str, Any]]:
    """Bind a GRPO group to every condition that changes its distribution."""

    config_hash = _canonical_sha256(resolved_h1_config)
    budget = {
        "h1_max_iterations": resolved_h1_config.get("max_iterations"),
        "h1_max_context_tokens": resolved_h1_config.get("max_context_tokens"),
        "h1_model_max_output_tokens": (
            resolved_h1_config.get("sampling") or {}
        ).get("max_tokens"),
        "h2": parent_compute_contract,
    }
    condition = {
        "schema": "h1-proposal-condition/1.0",
        # Separate repeated batches even if their prompt bytes and parent happen
        # to recur; their policy checkpoint can have changed between rounds.
        "round": int(round_index),
        "task_id": str(task_id),
        "role": str(role),
        "parent_h2_sha256": parent_h2_sha256,
        "input_prompt_sha256": input_prompt_sha256,
        "system_prompt_sha256": system_prompt_sha256,
        "behavior_policy": behavior_policy,
        "behavior_policy_identity": behavior_policy_identity,
        "resolved_h1_config_sha256": config_hash,
        "resolved_h1_config": resolved_h1_config,
        "budget_signature": _canonical_sha256(budget),
        "budget": budget,
        "sampling_mode": sampling_mode,
    }
    return "h1g_" + _canonical_sha256(condition)[:24], condition


def _condition_candidate_groups(
    candidates: list[Dict[str, Any]],
) -> list[tuple[str, list[Dict[str, Any]]]]:
    """Partition candidates by exact sampling condition, failing closed."""

    has_ids = [bool(c.get("proposal_group_id")) for c in candidates]
    if any(has_ids) and not all(has_ids):
        raise RuntimeError("proposal group identity is missing for some candidates")
    if not any(has_ids):
        # Read compatibility for artifacts that predate condition-aware groups.
        return [("legacy_task_group", list(candidates))]
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    conditions: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        group_id = str(candidate["proposal_group_id"])
        condition = candidate.get("proposal_condition")
        if not isinstance(condition, dict):
            raise RuntimeError(f"{group_id}: proposal_condition is missing")
        expected = "h1g_" + _canonical_sha256(condition)[:24]
        if expected != group_id:
            raise RuntimeError(
                f"proposal condition hash mismatch: recorded={group_id} "
                f"expected={expected}"
            )
        if group_id in conditions and conditions[group_id] != condition:
            raise RuntimeError(f"proposal group hash collision for {group_id}")
        conditions[group_id] = condition
        grouped.setdefault(group_id, []).append(candidate)
    return [
        (group_id, sorted(rows, key=lambda row: int(row["k"])))
        for group_id, rows in sorted(grouped.items())
    ]


def _trainable_condition_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return an exchangeable on-policy subset, or none for a singleton."""

    selected = [
        candidate for candidate in candidates
        if candidate.get("training_eligible") is True
        and candidate.get("infrastructure_failure", False) is not True
        and not candidate.get("training_block_reason")
    ]
    return selected if len(selected) >= 2 else []


def _bind_training_advantages(
    evaluation_group: Dict[str, Any],
    training_group: Optional[Dict[str, Any]],
) -> None:
    """Keep all evaluation rows but source advantages only from trainables."""

    training_rows = {
        int(row["k"]): row
        for row in ((training_group or {}).get("rows") or [])
    }
    for row in evaluation_group["rows"]:
        training_row = training_rows.get(int(row["k"]))
        row["advantage_eligible"] = training_row is not None
        row["advantage"] = (
            float(training_row["advantage"])
            if training_row is not None else 0.0
        )
    evaluation_group.update({
        "training_slots": sorted(training_rows),
        "training_adv_mode": (
            training_group.get("adv_mode", "legacy")
            if training_group is not None else "insufficient_trainable"
        ),
        "training_reward_mean": (
            training_group.get("reward_mean")
            if training_group is not None else None
        ),
        "training_reward_std": (
            training_group.get("reward_std")
            if training_group is not None else None
        ),
    })


def _aggregate_condition_reward_groups(
    *, task_id: str, base_score: float,
    condition_groups: list[tuple[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Keep advantages local while selecting a task-level promotion winner."""

    rows: list[Dict[str, Any]] = []
    summaries: list[Dict[str, Any]] = []
    for group_id, group in condition_groups:
        for row in group["rows"]:
            row["proposal_group_id"] = group_id
            rows.append(row)
        summaries.append({
            "proposal_group_id": group_id,
            "slots": [int(row["k"]) for row in group["rows"]],
            "adv_mode": group.get("adv_mode", "legacy"),
            "reward_mean": group.get("reward_mean"),
            "reward_std": group.get("reward_std"),
            "best_k": group.get("best_k"),
            "best_score": group.get("best_score"),
            "best_causal_delta": group.get("best_causal_delta"),
            "best_component_credit_delta": group.get(
                "best_component_credit_delta"
            ),
            "promotion_best_k": group.get("promotion_best_k"),
            "training_slots": group.get("training_slots", []),
            "training_adv_mode": group.get("training_adv_mode"),
            "training_reward_mean": group.get("training_reward_mean"),
            "training_reward_std": group.get("training_reward_std"),
        })
    rows.sort(key=lambda row: int(row["k"]))
    rewards = [float(row["reward"]) for row in rows]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    std = (
        sum((value - mean) ** 2 for value in rewards) / len(rewards)
    ) ** 0.5 if rewards else 0.0
    modes = sorted({str(group.get("adv_mode", "legacy"))
                    for _, group in condition_groups})
    out: Dict[str, Any] = {
        "task_id": task_id,
        "base_score": float(base_score),
        "reward_mean": mean,
        "reward_std": std,
        "rows": rows,
        "adv_mode": "condition_partitioned:" + "+".join(modes),
        "condition_groups": summaries,
    }

    paired = any("paired_same_seed" in mode for mode in modes)
    if paired:
        credited = [row for row in rows if row.get("attribution_valid")]
        causal_best = max(
            credited,
            key=lambda row: (
                float(row["reward"]),
                float(row.get("component_credit_delta", row["causal_delta"])),
                float(row["score"]),
            ),
        ) if credited else None
        promotion_rows = [
            row for row in credited if row.get("promotion_eligible") is True
        ]
        promotion_best = max(
            promotion_rows,
            key=lambda row: (
                float(row.get("component_credit_delta", row["causal_delta"])),
                float(row["score"]),
            ),
        ) if promotion_rows else None
        selected = promotion_best or causal_best
        absolute_best = max(
            credited, key=lambda row: float(row["score"])
        ) if credited else None
        program_samples = [
            (row["k"], sample)
            for row in credited
            for sample in (row.get("pair_contract") or {}).get("samples", [])
        ]
        program_best = max(
            program_samples,
            key=lambda item: float(item[1]["candidate_score"]),
        ) if program_samples else None
        min_effect = max(
            (float(group.get("min_effect", 0.0))
             for _, group in condition_groups), default=0.0,
        )
        out.update({
            "credit_basis": "candidate_minus_parent_same_decode_seed",
            "paired_repeats": max(
                (int(group.get("paired_repeats", 1))
                 for _, group in condition_groups), default=1,
            ),
            "min_promotion_repeats": max(
                (int(group.get("min_promotion_repeats", 3))
                 for _, group in condition_groups), default=3,
            ),
            "min_effect": min_effect,
            "best_k": selected["k"] if selected else None,
            "best_score": selected["score"] if selected else None,
            "best_control_score": (
                selected["control_score"] if selected else None
            ),
            "best_causal_delta": (
                selected["causal_delta"] if selected else None
            ),
            "best_component_credit_delta": (
                selected.get("component_credit_delta", selected["causal_delta"])
                if selected else None
            ),
            "diagnostic_causal_best_k": (
                causal_best["k"] if causal_best else None
            ),
            "promotion_best_k": (
                promotion_best["k"] if promotion_best else None
            ),
            "absolute_best_k": absolute_best["k"] if absolute_best else None,
            "absolute_best_score": (
                absolute_best["score"] if absolute_best else None
            ),
            "program_best_k": program_best[0] if program_best else None,
            "program_best_score": (
                program_best[1]["candidate_score"] if program_best else None
            ),
            "program_improved": bool(
                program_best and float(program_best[1]["candidate_score"])
                > float(base_score) + rw.EPS
            ),
            "improved": promotion_best is not None,
        })
    else:
        valid = [
            row for row in rows
            if row.get("valid") and row.get("score") is not None
        ]
        best = max(valid, key=lambda row: float(row["score"])) if valid else None
        out.update({
            "best_k": best["k"] if best else None,
            "best_score": best["score"] if best else None,
            "improved": bool(
                best and float(best["score"]) > float(base_score)
            ),
        })
    ceilings = [
        group.get("ceiling") for _, group in condition_groups
        if group.get("ceiling") is not None
    ]
    if ceilings:
        out["ceiling"] = ceilings[0]
    return out


def _aggregate_feedback_repeats(
    repeat_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate telemetry without last-result-wins or fake token zeros."""

    if not repeat_rows:
        return {}
    numeric = (
        "evals", "probes", "llm_calls", "input_tokens", "output_tokens",
        "sandbox_seconds", "component_invocations", "middleware_fires",
        "component_usable_outputs", "component_semantic_successes",
        "component_semantic_rejections",
        "candidate_geometry_calls", "valid_edit_rate", "new_best_edits",
        "structure_gate_satisfied", "structure_postcondition_failed",
        "evaluated_edits", "distinct_program_families",
    )
    means: Dict[str, Any] = {}
    totals: Dict[str, Any] = {}
    for key in numeric:
        values = [item.get(key) for item in repeat_rows]
        if any(value is None for value in values):
            means[key] = None
            totals[key] = None
        else:
            totals[key] = sum(float(value) for value in values)
            means[key] = totals[key] / len(values)
    stop_counts: Dict[str, int] = {}
    reject_reason_counts: Dict[str, int] = {}
    for item in repeat_rows:
        stop = str(item.get("stop") or "unknown")
        stop_counts[stop] = stop_counts.get(stop, 0) + 1
        for reason, count in (item.get("component_reject_reasons") or {}).items():
            reject_reason_counts[str(reason)] = (
                reject_reason_counts.get(str(reason), 0) + int(count or 0)
            )
    return {
        "per_repeat": repeat_rows,
        "repeat_count": len(repeat_rows),
        "usage_mean": means,
        "usage_sum": totals,
        "evals": means["evals"],
        "probes": means["probes"],
        "llm_calls": means["llm_calls"],
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "sandbox_seconds": totals["sandbox_seconds"],
        "component_invocations": totals["component_invocations"],
        "component_usable_outputs": totals["component_usable_outputs"],
        "component_semantic_successes": totals[
            "component_semantic_successes"
        ],
        "component_semantic_rejections": totals[
            "component_semantic_rejections"
        ],
        "middleware_fires": totals["middleware_fires"],
        "candidate_geometry_calls": totals["candidate_geometry_calls"],
        "valid_edit_rate": means["valid_edit_rate"],
        "new_best_edits": totals["new_best_edits"],
        "structure_gate_satisfied": totals["structure_gate_satisfied"],
        "structure_postcondition_failed": totals[
            "structure_postcondition_failed"
        ],
        "evaluated_edits": totals["evaluated_edits"],
        "distinct_program_families": means["distinct_program_families"],
        "score_progression": [
            item.get("score_progression") or [] for item in repeat_rows
        ],
        "stop_reasons": stop_counts,
        "component_reject_reasons": dict(sorted(reject_reason_counts.items())),
        "stop": next(iter(stop_counts)) if len(stop_counts) == 1 else "mixed",
        "err": next(
            (item["err"] for item in repeat_rows if item.get("err")), None
        ),
    }


def cmd_propose(args) -> None:
    from inner.tasks.eft_task import get_task  # tasks registry (spec + seed program)

    round_source_revision = pio.source_revision()
    if round_source_revision.get("proposer_sources_dirty") is not False:
        raise RuntimeError(
            "proposer/runtime source tree is dirty; refusing to spend H1 "
            "rollout budget without immutable provenance"
        )
    round_dir = Path(args.round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    bases = _load_bases(args.bases_file, args.tasks)
    feedback_input = _input_file_provenance(
        getattr(args, "feedback_file", None)
    )
    program_input = _input_file_provenance(
        getattr(args, "seed_programs_file", None)
    )
    bases_input = _input_file_provenance(getattr(args, "bases_file", None))
    base_urls = _replica_base_urls(args.base_url, args.n_replicas)

    # per-task context: base spec (that task's current best harness) + user message
    inherited = {}
    logical_index = int(os.environ.get("LOGICAL_ROUND_INDEX", "0") or 0)
    if os.environ.get("SAH_FIXED_INFERENCE_SLOTS", "0") == "1" \
            and logical_index > 0 \
            and not getattr(args, "seed_programs_file", None):
        raise RuntimeError(
            "canonical post-cold round requires --seed-programs-file"
        )
    if getattr(args, "seed_programs_file", None):
        try:
            inherited = json.loads(Path(args.seed_programs_file).read_text())
            if not isinstance(inherited, dict):
                raise TypeError("seed-program registry must be a JSON object")
        except Exception as e:
            if os.environ.get("SAH_FIXED_INFERENCE_SLOTS", "0") == "1" \
                    and logical_index > 0:
                raise RuntimeError(
                    "canonical post-cold round requires its task-local "
                    "seed-program registry"
                ) from e
            print(f"[propose] WARNING: seed-programs-file unreadable ({e}); using task seeds")
            inherited = {}
    # Rollouts consume this immutable round-local copy, never the mutable
    # workspace registry that the collector will update after the batch.
    seed_snapshot = round_dir / "seed_programs_in.json"
    seed_snapshot.write_text(
        json.dumps(inherited, indent=2, ensure_ascii=False) + "\n"
    )
    snapshot_raw = seed_snapshot.read_bytes()
    program_input["snapshot_path"] = str(seed_snapshot.resolve())
    program_input["snapshot_sha256"] = hashlib.sha256(snapshot_raw).hexdigest()
    program_input["snapshot_bytes"] = len(snapshot_raw)

    # Anti-tampering: verify the task texts about to be served against the
    # pinned registry (fail-closed under SAH_TASK_TEXT_ENFORCE=1).
    from outer.rounds.task_text_registry import verify_task_texts
    _tasks_by_id = {tid: get_task(tid) for tid in args.tasks}
    task_text_provenance = verify_task_texts(_tasks_by_id)
    (round_dir / "task_text_provenance.json").write_text(
        json.dumps(task_text_provenance, indent=1)
    )
    for _tid, _row in task_text_provenance.get("tasks", {}).items():
        if _row.get("status") == "MISMATCH":
            print(f"[task-text] WARNING: {_tid} differs from the pinned registry")

    ctx: Dict[str, Dict[str, Any]] = {}
    for tid in args.tasks:
        task = _tasks_by_id[tid]
        # BEAM-001: with CURVE_INHERIT_BEAM=B>1 the task carries B parent
        # packages; slots are dealt round-robin across them so a runner-up
        # lineage (e.g. the only component-carrying candidate) survives a
        # round instead of being deleted by strict single inheritance.
        beam_entries = list(bases[tid].get("beam") or [])
        if not beam_entries:
            beam_entries = [{"package": bases[tid]["package"],
                             "score": bases[tid]["score"], "from": "incumbent"}]
        base_spec = hs.read_base_spec(Path(bases[tid]["package"]))
        beam_specs = [hs.read_base_spec(Path(e["package"])) for e in beam_entries]
        ent = inherited.get(tid)
        if ent:  # inheritance: rollouts start from the current best program
            seed_prog = ent["program"] if isinstance(ent, dict) else ent
            seed_sc = float(ent.get("score", bases[tid]["score"])) if isinstance(ent, dict) \
                else float(bases[tid]["score"])
        else:
            seed_prog, seed_sc = task.initial_program, float(bases[tid]["seed_score"])
        # GRPO-CONTEXT-001: prior rollout feedback, failure taxonomies, campaign
        # summaries, and analyzer output are intentionally excluded from the H1
        # model input. They drift across rounds and would make relative rewards
        # reflect different context rather than proposer-policy quality.
        analysis_requested = any(
            os.environ.get(name, "0") == "1"
            for name in ("SAH_ANALYSIS", "SAH_ANALYSIS_REQUIRED")
        )
        analysis_meta = {
            "enabled": False,
            "requested_but_suppressed": analysis_requested,
            "stateless_proposer_context": True,
            "feedback_available": False,
            "brief_attached": False,
            "model_calls": 0,
            "specialists": [],
            "feedback_input": feedback_input,
        }
        # Optional novelty rejection remains validator-side and is never
        # rendered into the proposer context. Canonical GRPO runs disable it.
        futile = []
        try:
            import collections
            rounds_root = round_dir.parent
            sig_deltas: dict = collections.defaultdict(list)
            for prev in sorted(rounds_root.glob("round0*")):
                if prev == round_dir:
                    continue
                gb = prev / "grpo_batch.jsonl"
                if not gb.is_file():
                    continue
                deltas_by_k = {}
                for line in gb.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("task_id") != tid:
                        continue
                    if row.get("causal_delta") is not None:
                        deltas_by_k[int(row["k"])] = abs(
                            float(row["causal_delta"]))
                for md in sorted((prev / "tasks" / tid).glob("cand*/meta.json")):
                    try:
                        meta_c = json.loads(md.read_text())
                    except Exception:
                        continue
                    k_c = meta_c.get("k")
                    if k_c is None or int(k_c) not in deltas_by_k:
                        continue
                    # component-level signature only: prompt edits are a
                    # class, not a repeatable proposal; named components are.
                    sig = ";".join(sorted(
                        str(f) for f in meta_c.get("changed_fields") or []
                        if not str(f).startswith("system_prompt")
                        and not _is_scaffold_field(f)
                    ))
                    if sig:
                        sig_deltas[sig].append(deltas_by_k[int(k_c)])
            futile = sorted(
                sig for sig, ds in sig_deltas.items()
                if len(ds) >= 2 and max(ds) < 1e-4
            )
        except Exception as e:
            print(f"[propose] WARNING: novelty history skipped ({e})")
        futile_sigs = set(futile)
        ctx[tid] = {
            "futile_sigs": futile_sigs,
            "base_spec": base_spec,
            "beam_entries": beam_entries,
            "beam_specs": beam_specs,
            "analysis": analysis_meta,
            "user_message": pio.build_user_message(
                task_id=tid, task_spec=task.spec,
                seed_program=seed_prog,
                seed_score=seed_sc,
                base_score=float(bases[tid]["score"]),
                base_spec=base_spec, max_evals=args.max_evals),
        }

    jobs = [(tid, k) for tid in args.tasks for k in range(args.k)]
    print(f"[propose] {len(args.tasks)} tasks x K={args.k} = {len(jobs)} H1 runs "
          f"across {len(base_urls)} replica(s)")

    # Every slot sees the same fixed task/contract message and safe editing
    # surface. Independent sampling supplies diversity; no slot-specific brief
    # or prior-rollout context is appended. Keep the legacy CLI value inert.
    force_frac = 0.0
    # Frozen-policy proposals are evaluation-only and therefore cannot occupy
    # one of the canonical trainable slots. Legacy experiments may opt in, but
    # the default K=8 portfolio remains one complete phi group.
    frozen_epsilon = os.environ.get("SAH_FROZEN_EPSILON", "0") == "1"

    def _msg_for(tid, k):
        _ = k
        return ctx[tid]["user_message"]

    # PROPOSALARTIFACT-001: every attempt gets its own immutable workspace
    # root.  A requeue after a failed gate therefore preserves the first
    # attempt instead of silently overwriting the files the proposer authored.
    workspace_attempts = round_dir / "proposal_workspace_attempts"
    workspace_attempts.mkdir(exist_ok=True)
    prior_attempts = [
        int(path.name.split("_", 1)[1])
        for path in workspace_attempts.iterdir()
        if path.is_dir() and path.name.startswith("attempt_")
        and path.name.split("_", 1)[1].isdigit()
    ]
    proposal_attempt = max(prior_attempts, default=0) + 1
    proposal_workspace_root = (
        workspace_attempts / f"attempt_{proposal_attempt:02d}"
    )
    proposal_workspace_root.mkdir()

    # REPAIR-001: the takeover agent must run on the FROZEN executor model,
    # never the trained proposer. Under split serving base_urls[0] is M_phi
    # and the last replica is frozen M0; without split serving every replica
    # is frozen, so the fallback is also safe.
    repair_enabled = os.environ.get("SAH_REPAIR", "0") == "1"
    repair_url = (
        os.environ.get("SAH_REPAIR_BASE_URL") or base_urls[-1]
    ) if repair_enabled else None
    repair_model = os.environ.get("SAH_REPAIR_MODEL") or args.model
    repair_policy_identity = (
        os.environ.get("SAH_REPAIR_MODEL_PATH")
        or os.environ.get("MODEL_PATH")
        or repair_model
    )
    repair_max_iters = int(os.environ.get("SAH_REPAIR_MAX_ITERS", "16"))

    def run(idx_job):
        idx, (tid, k) = idx_job
        seed = (args.seed * 1000 + idx) if args.seed is not None else None
        novelty_guard = os.environ.get("SAH_NOVELTY_GUARD", "0") == "1"
        k_total = int(args.k)
        # POLICY ARM (not a role): one candidate per round samples from the
        # FROZEN BASE replica so component-authoring behaviour cannot vanish
        # from the training distribution as phi drifts.  It changes which
        # policy proposes, never what the proposal is allowed to contain.
        eps = bool(
            frozen_epsilon and repair_url
            and k == max(int(k_total) - 1, 0)
        )
        # BEAM-001 dealing rule: the frozen-base arm inherits the component
        # lineage when one exists, so a component can compound across rounds;
        # every other candidate keeps improving the promoted package.
        # Width 1 => everyone on the head.
        beam = ctx[tid]["beam_entries"]
        b_idx = 1 if (len(beam) > 1 and eps) else 0
        # SAH_BEAM_DEAL=roundrobin: every beam parent receives slots (k mod width),
        # so each parent forms its own exact-condition group and the frozen
        # proposer's structural collapse onto one parent no longer wastes the
        # whole batch.  Default keeps the head-only dealing above.
        if (len(beam) > 1 and not eps
                and os.environ.get("SAH_BEAM_DEAL", "head") == "roundrobin"):
            b_idx = int(k) % len(beam)
        slot_base_spec = ctx[tid]["beam_specs"][b_idx]
        slot_url = (repair_url if (eps and repair_url)
                    else base_urls[idx % len(base_urls)])
        rec = pp.run_once(
            k, base_spec=slot_base_spec, user_message=_msg_for(tid, k),
            task_id=tid,
            futile_signatures=(
                ctx[tid].get("futile_sigs") if novelty_guard else None),
            base_url=slot_url, model=args.model,
            api_key="EMPTY", seed=seed, timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            repair_base_url=repair_url,
            repair_model=repair_model,
            repair_policy_identity=repair_policy_identity,
            repair_endpoint_role="frozen_base_executor",
            repair_max_iters=repair_max_iters,
            workspace_archive=(
                proposal_workspace_root / tid / f"cand{k:02d}" / "h2"
            ),
            round_index=args.round,
        )
        rec.behavior_policy = "frozen_base" if eps else "trained_phi"
        rec.parent_package = str(beam[b_idx]["package"])
        rec.parent_score = float(beam[b_idx]["score"])
        rec.parent_spec_hash = hs.spec_hash(slot_base_spec)
        rec.parent_compute_contract = _resolved_h2_compute_contract(
            slot_base_spec, max_evals=args.max_evals
        )
        return tid, rec

    # -- optional legacy sequential sampling (non-portfolio campaigns only) --- #
    # Proposer-policy GRPO relies on independent condition-identical samples:
    # sharing earlier actions makes later proposals conditional and destroys
    # exchangeability. Reject sequential portfolio runs fail-closed.
    seq = os.environ.get("SAH_SEQUENTIAL", "0") == "1"
    if seq and portfolio_enabled():
        raise RuntimeError(
            "portfolio discovery requires independent parallel proposals; "
            "set SAH_SEQUENTIAL=0"
        )
    max_shared = int(os.environ.get("SAH_SEQ_MAX_SHARED", "6") or "6")

    def run_task_sequential(tid):
        recs, shared = [], []
        for k in range(args.k):
            seed = (args.seed * 1000 + k) if args.seed is not None else None
            msg = _msg_for(tid, k)
            if shared:
                msg = msg + pio.render_prior_actions(shared[-max_shared:])
            rec = pp.run_once(
                k, base_spec=ctx[tid]["base_spec"], user_message=msg,
                task_id=tid,
                futile_signatures=(
                    ctx[tid].get("futile_sigs")
                    if os.environ.get("SAH_NOVELTY_GUARD", "0") == "1"
                    else None),
                base_url=base_urls[k % len(base_urls)], model=args.model,
                api_key="EMPTY", seed=seed, timeout=args.llm_timeout,
                max_retries=args.llm_max_retries,
                repair_base_url=repair_url,
                repair_model=repair_model,
                repair_policy_identity=repair_policy_identity,
                repair_endpoint_role="frozen_base_executor",
                repair_max_iters=repair_max_iters,
                workspace_archive=(
                    proposal_workspace_root / tid / f"cand{k:02d}" / "h2"
                ),
                round_index=args.round,
            )
            rec.behavior_policy = "trained_phi"
            rec.parent_package = str(ctx[tid]["beam_entries"][0]["package"])
            rec.parent_score = float(ctx[tid]["beam_entries"][0]["score"])
            rec.parent_spec_hash = hs.spec_hash(ctx[tid]["base_spec"])
            rec.parent_compute_contract = _resolved_h2_compute_contract(
                ctx[tid]["base_spec"], max_evals=args.max_evals
            )
            recs.append((tid, rec))
            if rec.valid and rec.effective is not None:
                shared.append({"k": k, "changed_fields": rec.changed_fields,
                               "spec": rec.spec})
        return recs

    if seq:
        print(f"[propose] sequential sampling ON (share <= {max_shared} prior valid actions)")
        with ThreadPoolExecutor(max_workers=min(args.parallel, len(args.tasks))) as ex:
            results = [r for task_recs in ex.map(run_task_sequential, args.tasks)
                       for r in task_recs]
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            results = list(ex.map(run, enumerate(jobs)))

    per_task: Dict[str, Any] = {}
    trajectories = []
    repair_trajectories = []
    slot_prompt_rows = []
    h1_system_raw = (pio.H1_PACKAGE / "system.md").read_bytes()
    h1_system_sha256 = hashlib.sha256(h1_system_raw).hexdigest()
    for tid in args.tasks:
        records = sorted((r for t, r in results if t == tid), key=lambda r: r.k)
        # Compute group identity from the conditions that actually reached the
        # model.  This is deliberately done after behavior-policy/parent
        # assignment and before any reward is observed.
        for rec in records:
            role_key = (
                role_for_slot(rec.k, args.round).key
                if portfolio_enabled() else "unconditioned"
            )
            exact_prompt = rec.exact_user_message
            prompt_hash = (
                hashlib.sha256(exact_prompt.encode("utf-8")).hexdigest()
                if exact_prompt else None
            )
            parent_hash = (
                h2_sha256(Path(rec.parent_package))
                if rec.parent_package else None
            )
            if rec.behavior_policy == "frozen_base":
                policy_identity = os.environ.get("MODEL_PATH") or args.model
            else:
                policy_identity = (
                    os.environ.get("MPHI_PATH")
                    or os.environ.get("MODEL_PATH")
                    or args.model
                )
            group_id, condition = _proposal_condition(
                task_id=tid,
                round_index=args.round,
                role=role_key,
                parent_h2_sha256=parent_hash,
                input_prompt_sha256=prompt_hash,
                system_prompt_sha256=h1_system_sha256,
                behavior_policy=rec.behavior_policy,
                behavior_policy_identity=policy_identity,
                resolved_h1_config=rec.h1_config,
                parent_compute_contract=rec.parent_compute_contract,
                sampling_mode="sequential" if seq else "parallel",
            )
            rec.proposal_group_id = group_id
            rec.proposal_condition = condition
            rec.training_eligible = bool(
                rec.behavior_policy == "trained_phi"
                and not seq
                and not rec.infrastructure_failure
            )
        # Duplicate invalidation is itself part of reward assignment and must
        # therefore remain inside an exchangeable condition group. Cross-group
        # repeats are analysis signals only; they never alter training reward.
        records_by_group: Dict[str, List[Any]] = {}
        for rec in records:
            records_by_group.setdefault(rec.proposal_group_id, []).append(rec)
        for grouped_records in records_by_group.values():
            pp.dedup_group(grouped_records, ctx[tid]["base_spec"])
        first_by_spec: Dict[str, Any] = {}
        for rec in records:
            if not rec.spec_hash:
                continue
            prior = first_by_spec.get(rec.spec_hash)
            if (prior is not None
                    and prior.proposal_group_id != rec.proposal_group_id):
                rec.cross_group_duplicates.append({
                    "k": prior.k,
                    "proposal_group_id": prior.proposal_group_id,
                    "analysis_only": True,
                })
                prior.cross_group_duplicates.append({
                    "k": rec.k,
                    "proposal_group_id": rec.proposal_group_id,
                    "analysis_only": True,
                })
            else:
                first_by_spec[rec.spec_hash] = rec
        group_sizes: Dict[str, int] = {}
        trainable_group_sizes: Dict[str, int] = {}
        for rec in records:
            group_sizes[rec.proposal_group_id] = (
                group_sizes.get(rec.proposal_group_id, 0) + 1
            )
            if rec.training_eligible:
                trainable_group_sizes[rec.proposal_group_id] = (
                    trainable_group_sizes.get(rec.proposal_group_id, 0) + 1
                )
        for rec in records:
            # A singleton has no within-condition relative advantage. Keep it
            # evaluable/promotable, but fail closed for policy optimization.
            if trainable_group_sizes.get(rec.proposal_group_id, 0) < 2:
                rec.training_eligible = False
        cands = []
        for rec in records:
            portfolio_role = (
                role_for_slot(rec.k, args.round)
                if portfolio_enabled() else None
            )
            portfolio_meta = {
                "version": PORTFOLIO_VERSION,
                "enabled": bool(portfolio_role is not None),
                "role": (
                    portfolio_role.key
                    if portfolio_role is not None else "unconditioned"
                ),
                "title": (
                    portfolio_role.title
                    if portfolio_role is not None else "portfolio ablation"
                ),
                "required_kind": (
                    portfolio_role.required_kind
                    if portfolio_role is not None else "none"
                ),
                "selection_policy": (
                    "model_selected_portfolio"
                    if portfolio_role is not None else "none"
                ),
                "allowed_surfaces": (
                    list(ALLOWED_SURFACES) if portfolio_role is not None else []
                ),
                # Compatibility alias for older artifact consumers.
                "allowed_kinds": (
                    list(ALLOWED_SURFACES) if portfolio_role is not None else []
                ),
            }
            exact_prompt = rec.exact_user_message
            prompt_sha256 = (
                hashlib.sha256(exact_prompt.encode("utf-8")).hexdigest()
                if exact_prompt else None
            )
            workspace_dir = (
                proposal_workspace_root / tid / f"cand{rec.k:02d}" / "h2"
            )
            workspace_rel = workspace_dir.relative_to(round_dir).as_posix()
            primary_workspace_rel = (
                Path(rec.primary_workspace_archive).relative_to(
                    round_dir
                ).as_posix()
                if rec.primary_workspace_archive else None
            )
            repair_workspace_rel = (
                Path(rec.repair_workspace_archive).relative_to(
                    round_dir
                ).as_posix()
                if rec.repair_workspace_archive else None
            )
            record_path = workspace_dir.parent / "record.json"
            record_rel = record_path.relative_to(round_dir).as_posix()
            slot_prompt_rows.append({
                "task_id": tid,
                "k": rec.k,
                "sent_to_model": bool(exact_prompt),
                "user_message": exact_prompt,
                "sha256": prompt_sha256,
                "system_sha256": h1_system_sha256,
                "portfolio": dict(portfolio_meta),
                "workspace": workspace_rel,
                "proposal_group_id": rec.proposal_group_id,
                "proposal_condition": rec.proposal_condition,
                "training_eligible": rec.training_eligible,
                "infrastructure_failure": rec.infrastructure_failure,
                "training_block_reason": rec.training_block_reason,
            })
            entry = {"k": rec.k, "valid": rec.valid, "errors": rec.errors,
                     "behavior_policy": getattr(rec, "behavior_policy", None),
                     "training_eligible": rec.training_eligible,
                     "infrastructure_failure": rec.infrastructure_failure,
                     "training_block_reason": rec.training_block_reason,
                     "h1_runtime_events": rec.h1_runtime_events,
                     "proposal_group_id": rec.proposal_group_id,
                     "proposal_condition": rec.proposal_condition,
                     "parent_package": getattr(rec, "parent_package", None),
                     "parent_score": getattr(rec, "parent_score", None),
                     "parent_compute_contract": rec.parent_compute_contract,
                     "candidate_compute_contract": (
                         _resolved_h2_compute_contract(
                             rec.effective, max_evals=args.max_evals
                         )
                         if rec.effective else None
                     ),
                     # BEAM-001: the paired contract must be checked against
                     # THIS slot's parent, not the task's head package.
                     "parent_package_sha256": (
                         h2_sha256(Path(rec.parent_package))
                         if getattr(rec, "parent_package", None) else None),
                     "spec_hash": rec.spec_hash, "changed_fields": rec.changed_fields,
                     "stop_reason": rec.stop_reason, "llm_calls": rec.llm_calls,
                     "tool_schema_guard": getattr(rec, "schema_guard", {}),
                     "validation_profile": rec.validation_profile,
                     "proposal_prompt_sha256": prompt_sha256,
                     "proposal_workspace": workspace_rel,
                     "proposal_primary_workspace": primary_workspace_rel,
                     "proposal_repair_workspace": repair_workspace_rel,
                     "proposal_record": record_rel,
                     "proposal_workspace_archived": bool(
                         rec.workspace_archive
                     ),
                     "proposal_artifact_errors": rec.artifact_errors,
                     "validations": rec.validations,
                     "failed_validations": rec.failed_validations,
                     "repair": getattr(rec, "repair", {}),
                     "review_log": getattr(rec, "review_log", []),
                     "component_transactions": getattr(
                         rec, "component_transactions", []
                     ),
                     "cross_group_duplicates": rec.cross_group_duplicates,
                     "component_semantic_fingerprint": (
                         rec.component_semantic_fingerprint
                     ),
                     "component_semantic_fingerprint_record": (
                         rec.component_semantic_fingerprint_record
                     ),
                     "within_group_semantic_duplicates": (
                         rec.within_group_semantic_duplicates
                     ),
                     "component_behavior_fingerprint": (
                         rec.component_behavior_fingerprint
                     ),
                     "component_behavior_fingerprint_record": (
                         rec.component_behavior_fingerprint_record
                     ),
                     "within_group_behavior_duplicates": (
                         rec.within_group_behavior_duplicates
                     ),
                     "portfolio": {
                         **portfolio_meta,
                         "implementation_changes":
                             implementation_change_inventory(
                                 ctx[tid]["base_spec"], rec.effective or {}
                             ),
                     },
                     "added_generated_tools": sorted(
                         hs.added_generated_component_names(
                             ctx[tid]["base_spec"], rec.effective or {},
                             "new_tools",
                         )
                     )}
            if rec.valid:
                cdir = round_dir / "tasks" / tid / f"cand{rec.k:02d}"
                component_lineage = hs.generated_component_lineage(
                    ctx[tid]["base_spec"], rec.effective
                )
                materialize(rec.effective, cdir, raw_spec_text=rec.raw_submission,
                            meta={"round": args.round, "task_id": tid, "k": rec.k,
                                  "spec_hash": rec.spec_hash,
                                  "component_semantic_fingerprint":
                                      rec.component_semantic_fingerprint,
                                  "component_semantic_fingerprint_record":
                                      rec.component_semantic_fingerprint_record,
                                  "component_behavior_fingerprint":
                                      rec.component_behavior_fingerprint,
                                  "component_behavior_fingerprint_record":
                                      rec.component_behavior_fingerprint_record,
                                  "changed_fields": rec.changed_fields,
                                  "base_package": bases[tid]["package"],
                                  "base_component_inventory":
                                      hs.generated_component_inventory(
                                          ctx[tid]["base_spec"]),
                                  "component_lineage": component_lineage,
                                  "effective": rec.effective})
                entry["dir"] = str(cdir)
                entry["package_sha256"] = h2_sha256(cdir)
                entry["component_lineage"] = component_lineage
            cands.append(entry)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps({
                "schema": "h1-candidate-artifact/1.0",
                "task_id": tid,
                "k": rec.k,
                "valid": rec.valid,
                "errors": rec.errors,
                "stop_reason": rec.stop_reason,
                "changed_fields": rec.changed_fields,
                "spec_hash": rec.spec_hash,
                "component_semantic_fingerprint": (
                    rec.component_semantic_fingerprint
                ),
                "component_semantic_fingerprint_record": (
                    rec.component_semantic_fingerprint_record
                ),
                "within_group_semantic_duplicates": (
                    rec.within_group_semantic_duplicates
                ),
                "component_behavior_fingerprint": (
                    rec.component_behavior_fingerprint
                ),
                "component_behavior_fingerprint_record": (
                    rec.component_behavior_fingerprint_record
                ),
                "within_group_behavior_duplicates": (
                    rec.within_group_behavior_duplicates
                ),
                "prompt_sha256": prompt_sha256,
                "system_sha256": h1_system_sha256,
                "workspace": workspace_rel,
                "workspace_archived": bool(rec.workspace_archive),
                "workspace_sha256": (
                    h2_sha256(Path(rec.workspace_archive))
                    if rec.workspace_archive else None
                ),
                "primary_workspace": primary_workspace_rel,
                "primary_workspace_archived": bool(
                    rec.primary_workspace_archive
                ),
                "primary_workspace_sha256": (
                    h2_sha256(Path(rec.primary_workspace_archive))
                    if rec.primary_workspace_archive else None
                ),
                "repair_workspace": repair_workspace_rel,
                "repair_workspace_archived": bool(
                    rec.repair_workspace_archive
                ),
                "repair_workspace_sha256": (
                    h2_sha256(Path(rec.repair_workspace_archive))
                    if rec.repair_workspace_archive else None
                ),
                "artifact_errors": rec.artifact_errors,
                "validations": rec.validations,
                "failed_validations": rec.failed_validations,
                "component_transactions": rec.component_transactions,
                "primary_component_transactions": (
                    rec.primary_component_transactions
                ),
                "repair_component_transactions": (
                    rec.repair_component_transactions
                ),
                "cross_group_duplicates": rec.cross_group_duplicates,
                "behavior_policy": rec.behavior_policy,
                "training_eligible": rec.training_eligible,
                "infrastructure_failure": rec.infrastructure_failure,
                "training_block_reason": rec.training_block_reason,
                "h1_runtime_events": rec.h1_runtime_events,
                "proposal_group_id": rec.proposal_group_id,
                "proposal_condition": rec.proposal_condition,
                "validation_profile": rec.validation_profile,
                "primary_file_events": rec.primary_file_events,
                "repair_file_events": rec.repair_file_events,
                "raw_submission_sha256": (
                    hashlib.sha256(
                        rec.raw_submission.encode("utf-8")
                    ).hexdigest()
                    if rec.raw_submission else None
                ),
            }, indent=2, ensure_ascii=False) + "\n")
            trajectories.append({
                "task_id": tid,
                "k": rec.k,
                "prompt_sha256": prompt_sha256,
                "candidate_record": record_rel,
                "workspace": workspace_rel,
                "behavior_policy": rec.behavior_policy,
                "training_eligible": rec.training_eligible,
                "infrastructure_failure": rec.infrastructure_failure,
                "training_block_reason": rec.training_block_reason,
                "h1_runtime_events": rec.h1_runtime_events,
                "proposal_group_id": rec.proposal_group_id,
                "proposal_condition": rec.proposal_condition,
                "validation_profile": rec.validation_profile,
                "component_semantic_fingerprint": (
                    rec.component_semantic_fingerprint
                ),
                "component_semantic_fingerprint_record": (
                    rec.component_semantic_fingerprint_record
                ),
                "within_group_semantic_duplicates": (
                    rec.within_group_semantic_duplicates
                ),
                "component_behavior_fingerprint": (
                    rec.component_behavior_fingerprint
                ),
                "component_behavior_fingerprint_record": (
                    rec.component_behavior_fingerprint_record
                ),
                "within_group_behavior_duplicates": (
                    rec.within_group_behavior_duplicates
                ),
                "raw_submission": rec.raw_submission,
                "trajectory": rec.trajectory,
            })
            if getattr(rec, "repair_trajectory", None):
                repair_trajectories.append({
                    "task_id": tid, "k": rec.k,
                    "repair": getattr(rec, "repair", {}),
                    "user_message": rec.repair_user_message,
                    "prompt_sha256": hashlib.sha256(
                        rec.repair_user_message.encode("utf-8")
                    ).hexdigest(),
                    "trajectory": rec.repair_trajectory,
                })
        n_ok = sum(c["valid"] for c in cands)
        valid_behavior_fingerprints = sorted({
            str(c.get("component_behavior_fingerprint"))
            for c in cands
            if c.get("valid") and c.get("component_behavior_fingerprint")
        })
        valid_semantic_fingerprints = sorted({
            str(c.get("component_semantic_fingerprint"))
            for c in cands
            if c.get("valid") and c.get("component_semantic_fingerprint")
        })
        print(f"  {tid}: {n_ok}/{len(cands)} valid | " + " ".join(
            f"c{c['k']:02d}{'+' if c['valid'] else '-'}" for c in cands))
        per_task[tid] = {"base_package": bases[tid]["package"],
                         "base_package_sha256": h2_sha256(
                             Path(bases[tid]["package"])
                         ),
                         "base_score": bases[tid]["score"],
                         "seed_score": bases[tid]["seed_score"],
                         "base_spec_hash": hs.spec_hash(ctx[tid]["base_spec"]),
                         "analysis": ctx[tid]["analysis"],
                         "proposal_metrics": {
                             "valid_candidates": n_ok,
                             "distinct_executable_structures": len(
                                 valid_semantic_fingerprints
                             ),
                             "distinct_middleware_behaviors": len(
                                 valid_behavior_fingerprints
                             ),
                             "semantic_fingerprints": (
                                 valid_semantic_fingerprints
                             ),
                             "middleware_behavior_fingerprints": (
                                 valid_behavior_fingerprints
                             ),
                         },
                         "candidates": cands}

    # Write proposer evidence before applying the proposal gate.  Both failed
    # and successful attempts therefore retain the exact H1 system bytes,
    # exact per-slot user input (including preload), trajectory, raw submitted
    # spec, and final private candidate filesystem.
    (round_dir / "h1_system_prompt.txt").write_bytes(h1_system_raw)
    (round_dir / "prompts.json").write_text(json.dumps(
        {tid: ctx[tid]["user_message"] for tid in args.tasks},
        indent=2, ensure_ascii=False,
    ) + "\n")
    (round_dir / "slot_prompts.json").write_text(json.dumps({
        "schema": "h1-slot-prompts/1.0",
        "h1_version": pio.H1_VERSION,
        "portfolio_version": PORTFOLIO_VERSION,
        "system_path": "h1_system_prompt.txt",
        "system_sha256": h1_system_sha256,
        "slots": slot_prompt_rows,
    }, indent=2, ensure_ascii=False) + "\n")
    (round_dir / "trajectories.json").write_text(
        json.dumps(trajectories, indent=2, ensure_ascii=False) + "\n"
    )
    if repair_trajectories:
        # Separate file: trajectories.json stays exactly one primary (phi)
        # trajectory per slot for the training/audit contracts.
        (round_dir / "repair_trajectories.json").write_text(
            json.dumps(repair_trajectories, indent=2, ensure_ascii=False)
            + "\n"
        )

    # Fail-closed proposal gate (SAH_PROPOSAL_GATE=all): a causal comparison
    # round must not proceed to rollouts unless every charged slot holds a
    # valid materialized candidate. The audit file is written on both outcomes
    # so a passing round proves the gate actually ran; on violation the
    # proposer evidence (prompts + trajectories + audit) is preserved and the
    # round fails before round.json exists, so a retry re-proposes cleanly.
    gate_mode = os.environ.get("SAH_PROPOSAL_GATE", "off")
    try:
        gate_kind, gate_min_valid = pp.parse_gate_mode(gate_mode, args.k)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    gate_tasks: Dict[str, Any] = {}
    for tid in args.tasks:
        cands = per_task[tid]["candidates"]
        slot_issues = (
            pp.proposal_gate_issues(cands, k=args.k)
            if gate_kind in ("all", "min") else []
        )
        valid_slots = sum(bool(c.get("valid")) for c in cands)
        if gate_kind == "all":
            issues = slot_issues
        elif gate_kind == "min" and valid_slots < gate_min_valid:
            issues = slot_issues + [
                f"only {valid_slots}/{len(cands)} valid slots "
                f"(minimum {gate_min_valid} required)"
            ]
        else:
            issues = []
        gate_tasks[tid] = {
            "charged_slots": len(cands),
            "valid_slots": valid_slots,
            "repaired_slots": sum(
                1 for c in cands
                if (c.get("repair") or {}).get("succeeded")
            ),
            "min_valid_required": gate_min_valid,
            "slot_issues": slot_issues,
            "issues": issues,
        }
    gate_passed = all(not row["issues"] for row in gate_tasks.values())
    (round_dir / "proposal_gate_audit.json").write_text(json.dumps({
        "schema": "proposal-gate/1.0",
        "mode": gate_mode,
        "enforced": gate_kind in ("all", "min"),
        "k": int(args.k),
        "passed": gate_passed,
        "tasks": gate_tasks,
    }, indent=2))
    if not gate_passed:
        # A requeued attempt re-runs propose and overwrites the round-local
        # files — archive each failed attempt's evidence immutably (failures
        # must be retained and remain diagnosable).
        archive_root = round_dir / "proposal_gate_failures"
        archive_root.mkdir(exist_ok=True)
        attempt = 1 + sum(1 for p in archive_root.iterdir() if p.is_dir())
        archive = archive_root / f"attempt_{attempt:02d}"
        archive.mkdir()
        for name in (
            "proposal_gate_audit.json", "h1_system_prompt.txt",
            "prompts.json", "slot_prompts.json", "trajectories.json",
            "repair_trajectories.json",
        ):
            source = round_dir / name
            if source.is_file():
                (archive / name).write_bytes(source.read_bytes())
        (archive / "proposal_workspace_ref.json").write_text(json.dumps({
            "schema": "h1-workspace-attempt-ref/1.0",
            "proposal_attempt": proposal_attempt,
            "workspace_root": proposal_workspace_root.relative_to(
                round_dir
            ).as_posix(),
        }, indent=2) + "\n")
        detail = "; ".join(
            f"{tid} {row['valid_slots']}/{row['charged_slots']} valid"
            for tid, row in gate_tasks.items() if row["issues"]
        )
        raise RuntimeError(
            f"proposal gate failed (SAH_PROPOSAL_GATE={gate_mode}): refusing to charge "
            f"intervention slots for invalid H2 proposals — {detail}; "
            f"evidence archived in {archive}"
        )

    fixed_inference_slots = (
        os.environ.get("SAH_FIXED_INFERENCE_SLOTS", "0") == "1"
    )
    paired_controls = (
        os.environ.get("SAH_PAIRED_CONTROLS", "0") == "1"
        or os.environ.get("SAH_ADV", "") == "paired"
    )
    paired_repeats = (
        int(os.environ.get("SAH_PAIRED_REPEATS", "1"))
        if paired_controls else 1
    )
    if paired_repeats < 1:
        raise RuntimeError("SAH_PAIRED_REPEATS must be >= 1")
    logical_round_raw = os.environ.get("LOGICAL_ROUND_INDEX", "")
    logical_round_index = (
        int(logical_round_raw) if logical_round_raw.strip() else None
    )
    if fixed_inference_slots and logical_round_index is None:
        raise RuntimeError(
            "fixed inference slots require LOGICAL_ROUND_INDEX for x-axis provenance"
        )
    decode_round_index = logical_round_index or 0
    h2_slots = int(args.k) * (paired_repeats if paired_controls else 1)
    control_slots = int(args.k) * paired_repeats if paired_controls else 0
    nominal_total = (
        int(args.k) + h2_slots + control_slots
        if fixed_inference_slots else None
    )
    h2_seed_stride = max(16, int(args.k) * paired_repeats)
    shared_anchor_trajectories = int(
        os.environ.get("SAH_SHARED_ANCHOR_TRAJECTORIES", "1")
    )
    if shared_anchor_trajectories < 0:
        raise RuntimeError("SAH_SHARED_ANCHOR_TRAJECTORIES must be >= 0")
    ending_source_revision = pio.source_revision()
    if ending_source_revision != round_source_revision:
        raise RuntimeError(
            "proposer/runtime source revision changed during H1 rollout; "
            "refusing unstable provenance"
        )
    round_source_revision = {
        **round_source_revision,
        "stable_during_proposal": True,
    }
    (round_dir / "round.json").write_text(json.dumps({
        "round": args.round, "created": time.strftime("%Y%m%d-%H%M%S"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "mode": "instance_wise",
        "h1_version": pio.H1_VERSION, "h1_package_hash": pio.h1_hash(),
        "proposer_contract_hash": pio.proposer_contract_hash(),
        "source_revision": round_source_revision,
        "proposal_artifacts": {
            "schema": "h1-proposal-artifacts/1.0",
            "proposal_attempt": proposal_attempt,
            "system_prompt": "h1_system_prompt.txt",
            "system_prompt_sha256": h1_system_sha256,
            "base_prompts": "prompts.json",
            "exact_slot_prompts": "slot_prompts.json",
            "trajectories": "trajectories.json",
            "workspace_root": proposal_workspace_root.relative_to(
                round_dir
            ).as_posix(),
        },
        "proposer": {
            "base_urls": base_urls,
            "model": args.model,
            "seed": args.seed,
            "checkpoint": (
                os.environ.get("MPHI_PATH")
                or os.environ.get("MODEL_PATH")
                or None
            ),
            "executor_checkpoint": os.environ.get("MODEL_PATH") or None,
        },
        "tasks_order": args.tasks, "max_evals": args.max_evals, "k": args.k,
        "proposal_contract": {
            "force_tool_fraction": force_frac,
            "enforcement": (
                "v1.1-model-selected-evidence-portfolio"
                if portfolio_enabled() else "portfolio-ablation"
            ),
            "portfolio_version": PORTFOLIO_VERSION,
            "intervention_selection": "model_selected_portfolio",
            "allowed_intervention_surfaces": list(ALLOWED_SURFACES),
            "whole_harness_attribution": True,
            "parallel_independent": not seq,
            "grpo_grouping": "exact-condition-hash/1.0",
            "sampling_mode": "sequential" if seq else "parallel",
            "proposer_context_policy": "stateless-condition-identical/1.0",
            "dynamic_context_excluded": [
                "campaign_evidence", "previous_telemetry",
                "failure_taxonomy", "analysis_brief", "slot_advisory_brief",
            ],
            "h2_compute_policy": (
                "freeze-parent-agent.max_iterations-and-sampling.max_tokens"
            ),
        },
        "cross_round_inputs": {
            "bases": bases_input,
            "feedback": feedback_input,
            "seed_programs": program_input,
        },
        "inference_trajectory_budget": {
            "schema": 1,
            "axis_unit": "generated_agent_trajectory",
            "fixed_h1_plus_h2_slots": fixed_inference_slots,
            "logical_round_index": logical_round_index,
            "h1_slots_per_task": int(args.k),
            "h2_slots_per_task": (
                h2_slots if fixed_inference_slots else None
            ),
            "paired_control_slots_per_task": (
                control_slots if fixed_inference_slots and paired_controls else 0
            ),
            "nominal_total_slots_per_task": nominal_total,
            "shared_anchor_x": (
                shared_anchor_trajectories if fixed_inference_slots else None
            ),
            "axis_x_after_round": (
                shared_anchor_trajectories
                + (logical_round_index + 1) * nominal_total
                if fixed_inference_slots else None
            ),
            "h2_decode_seed_base": int(os.environ.get("H2_SEED_BASE", "200000")),
            "h2_decode_seed_stride": h2_seed_stride,
            "h2_decode_seeds": [
                int(os.environ.get("H2_SEED_BASE", "200000"))
                + decode_round_index * h2_seed_stride + k
                for k in range(int(args.k))
            ],
            "h2_decode_seed_matrix": [
                [
                    int(os.environ.get("H2_SEED_BASE", "200000"))
                    + decode_round_index * h2_seed_stride
                    + repeat * int(args.k) + k
                    for repeat in range(paired_repeats)
                ]
                for k in range(int(args.k))
            ],
            "analyzer_calls_excluded_from_axis": True,
            "post_submit_reviewer_model_calls": 0,
        },
        "causal_attribution": {
            "schema": "paired-parent-control/1.0",
            "enabled": paired_controls,
            "candidate_root": "rollouts",
            "control_root": "paired_controls" if paired_controls else None,
            "pairing": "same_task_program_budget_model_decode_seed",
            "repeats_per_candidate": paired_repeats if paired_controls else 0,
            "minimum_promotion_repeats": int(
                os.environ.get("SAH_MIN_PROMOTION_REPEATS", "3")
            ),
            "minimum_effect": float(
                os.environ.get("SAH_MIN_PAIRED_EFFECT", "1e-4")
            ),
            "activation_contract": (
                "generated_tool_top_level_ok_true_or_middleware_fire"
            ),
        },
        "program_ratchet_mode": os.environ.get(
            "SAH_PROGRAM_RATCHET_MODE", "legacy_qd"
        ),
        "eval_timeout_s": int(float(os.environ.get("EVAL_TIMEOUT", "0") or 0)),
        "trajectory_wall_timeout_s": int(float(
            os.environ.get("ROLLOUT_WALL_TIMEOUT", "0") or 0
        )),
        "proposal_parallelism": {
            "requested_workers": int(args.parallel),
            "serving_replicas": len(base_urls),
        },
        "bases_in": bases,
        "per_task": per_task,
    }, indent=2))
    total_ok = sum(c["valid"] for t in per_task.values() for c in t["candidates"])
    n_rep = sum(
        1 for t in per_task.values() for c in t["candidates"]
        if (c.get("repair") or {}).get("succeeded")
    )
    print(f"[propose] {total_ok}/{len(jobs)} valid candidates "
          f"({n_rep} repaired by frozen M0) -> {round_dir}")


def cmd_collect(args) -> None:
    round_dir = Path(args.round_dir)
    meta = json.loads((round_dir / "round.json").read_text())
    prompts = json.loads((round_dir / "prompts.json").read_text())
    exact_slot_prompts = _load_slot_prompt_map(round_dir)
    trajs = {(t["task_id"], t["k"]): t
             for t in json.loads((round_dir / "trajectories.json").read_text())}
    recorded_system = round_dir / "h1_system_prompt.txt"
    system_text = (
        recorded_system.read_text()
        if recorded_system.is_file()
        else (pio.H1_PACKAGE / "system.md").read_text()
    )

    adv_mode = os.environ.get("SAH_ADV", "v2")
    if adv_mode not in ("v3", "v2", "grpo", "legacy", "anchored", "paired"):
        raise ValueError(f"unknown SAH_ADV={adv_mode!r}; expected one of v3, v2, grpo, legacy, anchored, paired")
    # v3 == v2 except on no_signal groups, so results alone can't tell you which
    # impl ran — log it so collect.log is proof the env survived sbatch/srun.
    print(f"[collect] advantage impl: SAH_ADV={adv_mode} "
          f"(hist_lambda={os.environ.get('SAH_HIST_LAMBDA', '-')})")
    ceilings = {}
    try:
        ft_path = Path(os.environ.get("SAH_FINCH_TARGETS") or (
            Path(__file__).resolve().parents[3] / "config" / "finch_targets.json"))
        ft = json.loads(ft_path.read_text())
        ceilings = {t: (v.get("sota_combined") or v.get("finch_combined"))
                    for t, v in ft.get("targets", {}).items()}
    except Exception:
        pass

    groups, batch_rows = {}, []
    next_bases = dict(meta.get("bases_in", {}))  # carry forward ALL tasks' bases
    for tid in meta["tasks_order"]:
        pt = meta["per_task"][tid]

        def compute_condition_group(
            candidates: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
            """Score candidates; callers decide which rows normalize GRPO."""

            if adv_mode == "legacy":
                return rw.compute_task_group(
                    task_id=tid, candidates=candidates,
                    rollout_root=round_dir / "rollouts",
                    base_score=float(pt["base_score"]),
                )
            if adv_mode == "paired":
                attribution = meta.get("causal_attribution") or {}
                if attribution.get("enabled") is not True:
                    raise RuntimeError(
                        "SAH_ADV=paired requires paired controls declared at "
                        "proposal time (SAH_PAIRED_CONTROLS=1)"
                    )
                return rw.compute_task_group_paired(
                    task_id=tid, candidates=candidates,
                    rollout_root=round_dir / "rollouts",
                    control_root=round_dir / "paired_controls",
                    base_score=float(pt["base_score"]),
                    ceiling=ceilings.get(tid),
                    sharpen_alpha=float(os.environ.get("SAH_ALPHA", "0.3")),
                    min_effect=float(
                        attribution.get("minimum_effect", 1e-4)
                    ),
                    expected_control_h2_sha256=pt.get("base_package_sha256"),
                    expected_repeats=int(
                        attribution.get("repeats_per_candidate", 1)
                    ),
                    min_promotion_repeats=int(
                        attribution.get("minimum_promotion_repeats", 3)
                    ),
                )
            if adv_mode == "anchored":
                return rw.compute_task_group_anchored(
                    task_id=tid, candidates=candidates,
                    rollout_root=round_dir / "rollouts",
                    base_score=float(pt["base_score"]),
                    ceiling=ceilings.get(tid),
                )
            if adv_mode == "grpo":
                return rw.compute_task_group_grpo(
                    task_id=tid, candidates=candidates,
                    rollout_root=round_dir / "rollouts",
                    base_score=float(pt["base_score"]),
                    ceiling=ceilings.get(tid),
                )
            if adv_mode == "v3":
                return rw.compute_task_group_v3(
                    task_id=tid, candidates=candidates,
                    rollout_root=round_dir / "rollouts",
                    base_score=float(pt["base_score"]),
                    ceiling=ceilings.get(tid),
                    sharpen_alpha=float(os.environ.get("SAH_ALPHA", "0.3")),
                    hist_lambda=float(
                        os.environ.get("SAH_HIST_LAMBDA", "0.3")
                    ),
                )
            return rw.compute_task_group_v2(
                task_id=tid, candidates=candidates,
                rollout_root=round_dir / "rollouts",
                base_score=float(pt["base_score"]),
                ceiling=ceilings.get(tid),
                sharpen_alpha=float(os.environ.get("SAH_ALPHA", "0.3")),
            )

        # OVERCAP-001 (opt-in, SAH_OVER_CAP_AS_INVALID=1): a proposal whose best
        # program exceeds the ratchet eval-time cap is an unacceptable submission,
        # exactly like a schema failure: it takes INVALID_REWARD, cannot become
        # best_k or a beam parent, and no longer blocks the promotion of a
        # slower-but-under-cap improvement or suppresses the whole task's
        # training (hadamard lineage, rounds 3/4: 0.809 @ 333 s and 0.686 @
        # 334 s hid a 0.548 @ 142 s improvement and zeroed two rounds).
        over_cap_marks = mark_over_cap_candidates(round_dir, tid, pt["candidates"])
        for mark in over_cap_marks:
            print(f"  [{tid}] cand{mark['k']:02d} eval_time {mark['eval_time']:.1f}s "
                  f"> cap {mark['cap']:.1f}s -> treated as INVALID "
                  f"(program score {mark['score']:.6f} discarded)")
        computed_groups: list[tuple[str, Dict[str, Any]]] = []
        for proposal_group_id, conditioned_candidates in \
                _condition_candidate_groups(pt["candidates"]):
            subgroup = compute_condition_group(conditioned_candidates)
            trainable_candidates = _trainable_condition_candidates(
                conditioned_candidates
            )
            training_group: Optional[Dict[str, Any]] = None
            if trainable_candidates:
                training_group = (
                    subgroup
                    if len(trainable_candidates) == len(conditioned_candidates)
                    else compute_condition_group(trainable_candidates)
                )
            _bind_training_advantages(subgroup, training_group)
            computed_groups.append((proposal_group_id, subgroup))
            print(
                f"  [{tid}/{proposal_group_id}] advantages: "
                f"{subgroup.get('training_adv_mode')} "
                f"training_slots={subgroup.get('training_slots')} "
                f"evaluation_slots={[c['k'] for c in conditioned_candidates]}"
            )
        g = _aggregate_condition_reward_groups(
            task_id=tid,
            base_score=float(pt["base_score"]),
            condition_groups=computed_groups,
        )
        groups[tid] = g
        if over_cap_marks:
            g["over_cap_policy"] = "invalid"
            g["over_cap_candidates"] = over_cap_marks
            _marks_by_k = {int(m["k"]): m for m in over_cap_marks}
            for row in g["rows"]:
                if int(row["k"]) in _marks_by_k:
                    row["over_cap"] = _marks_by_k[int(row["k"])]
        candidate_by_k = {
            int(candidate["k"]): candidate for candidate in pt["candidates"]
        }
        for row in g["rows"]:
            candidate = candidate_by_k[int(row["k"])]
            t = trajs.get((tid, row["k"]), {})
            if exact_slot_prompts is None:
                # Backwards compatibility for artifacts produced before exact
                # per-slot proposer inputs were retained.
                proposer_user = prompts[tid]
            else:
                key = (tid, int(row["k"]))
                if key not in exact_slot_prompts:
                    raise RuntimeError(
                        f"missing exact proposer input for {tid}/k{row['k']}"
                    )
                proposer_user = exact_slot_prompts[key]
            batch_rows.append({
                "round": meta["round"], "task_id": tid, "k": row["k"],
                "system": system_text, "user": proposer_user,
                "response": t.get("raw_submission", ""),
                "trajectory": t.get("trajectory", []),
                "reward": row["reward"], "advantage": row["advantage"],
                "valid": row["valid"], "score": row["score"],
                "spec_hash": row["spec_hash"],
                "control_score": row.get("control_score"),
                "causal_delta": row.get("causal_delta"),
                "attribution_valid": row.get("attribution_valid"),
                "attribution_status": row.get("attribution_status"),
                "pair_contract": row.get("pair_contract"),
                "proposal_group_id": row.get("proposal_group_id"),
                "proposal_condition": candidate.get("proposal_condition"),
                "behavior_policy": candidate.get("behavior_policy"),
                "training_eligible": bool(
                    candidate.get("training_eligible", True)
                    and row.get("advantage_eligible") is True
                ),
                "infrastructure_failure": bool(
                    candidate.get("infrastructure_failure", False)
                ),
                "training_block_reason": candidate.get(
                    "training_block_reason"
                ),
            })
        # next round's base for this task = best candidate iff it beat the base
        if g["improved"]:
            next_bases[tid] = {
                "package": str(round_dir / "tasks" / tid / f"cand{g['best_k']:02d}"),
                "score": g["best_score"], "seed_score": pt["seed_score"],
                "from": f"round{meta['round']:03d}/cand{g['best_k']:02d}"}
        else:
            # BASEREFRESH-001: the paired controls ARE rollouts of the
            # retained incumbent; once enough of them agree, they are a
            # better estimate of its score than the 2-seed measurement from
            # its promotion round (observed: base recorded 0.9393 while 112
            # subsequent same-harness rollouts scored 0.9645).  Refresh with
            # full provenance; the ratchet inherits the honest number.
            ctrl_scores = [
                float(row["control_score"]) for row in g.get("rows", [])
                if row.get("control_score") is not None
            ]
            refreshed = pt["base_score"]
            refresh_note = None
            if len(ctrl_scores) >= 4:
                est = sum(ctrl_scores) / len(ctrl_scores)
                if abs(est - pt["base_score"]) > 1e-3:
                    refreshed = est
                    refresh_note = {
                        "refreshed_from": "paired_control_rollouts",
                        "samples": len(ctrl_scores),
                        "previous_score": pt["base_score"],
                    }
            next_bases[tid] = {"package": pt["base_package"],
                               "score": refreshed,
                               "seed_score": pt["seed_score"], "from": "unchanged"}
            if refresh_note:
                next_bases[tid]["score_refresh"] = refresh_note
        # BEAM-001: publish the inheritance beam alongside the incumbent.
        # beam[0] is always the promoted/retained package (the curve score);
        # beam[1..] are exploration parents, chosen to PRESERVE DIVERSITY:
        # the best component-carrying candidate within BEAM_TOL of the best
        # wins the slot, else the runner-up by score.  Width 1 reproduces
        # strict single inheritance byte-for-byte.
        beam_width = int(os.environ.get("CURVE_INHERIT_BEAM", "1") or 1)
        if beam_width > 1:
            tol = float(os.environ.get("CURVE_BEAM_TOL", "0.02") or 0.02)
            head = {"package": next_bases[tid]["package"],
                    "score": next_bases[tid]["score"], "from": "incumbent"}
            pool = []
            for row in g.get("rows", []):
                if row.get("score") is None or not row.get("valid", True):
                    continue
                cand_dir = str(round_dir / "tasks" / tid
                               / f"cand{int(row['k']):02d}")
                if cand_dir == head["package"]:
                    continue
                has_component = bool([
                    f for f in (row.get("changed_fields") or [])
                    if not str(f).startswith("system_prompt")
                    and not _is_scaffold_field(f)
                ])
                pool.append({"package": cand_dir, "score": float(row["score"]),
                             "from": f"round{meta['round']:03d}/cand{int(row['k']):02d}",
                             "has_component": has_component})
            pool.sort(key=lambda e: -e["score"])
            beam = [head]
            near = [e for e in pool
                    if e["has_component"] and head["score"] - e["score"] <= tol]
            if near:
                beam.append(near[0])
            for entry in pool:
                if len(beam) >= beam_width:
                    break
                if entry["package"] not in {b["package"] for b in beam}:
                    beam.append(entry)
            next_bases[tid]["beam"] = beam[:beam_width]
            next_bases[tid]["beam_policy"] = {
                "width": beam_width, "tolerance": tol,
                "rule": "head=promoted/retained; then best component-carrying "
                        "candidate within tolerance; then by score",
            }
        bs = f"{g['best_score']:.6g}" if g["best_score"] is not None else "n/a"
        print(f"  {tid}: base={g['base_score']:.6g} best={bs} "
              f"(cand{g['best_k']:02d})" if g["best_k"] is not None else
              f"  {tid}: base={g['base_score']:.6g} best=n/a",
              "IMPROVED" if g["improved"] else "")

    # Persist telemetry for offline analysis and causal reward attribution. It
    # is deliberately not injected into the next H1 user message.
    fb_path = round_dir.parent / "task_feedback.json"
    try:
        feedback = json.loads(fb_path.read_text()) if fb_path.exists() else {}
    except Exception:
        feedback = {}
    for tid, g in groups.items():
        cands = []
        for row in g["rows"]:
            ent = {"k": row["k"], "score": row["score"],
                   "control_score": row.get("control_score"),
                   "causal_delta": row.get("causal_delta"),
                   "component_credit_delta": row.get(
                       "component_credit_delta"
                   ),
                   "attribution_status": row.get("attribution_status"),
                   "changed": [c.split(".")[-1] for c in row.get("changed_fields", [])][:6]}
            rl = row.get("review_log") or []
            if rl:
                ent["tools"] = [f"{x['name']}:{'ok' if x['ok'] else 'dropped('+str(x.get('error',''))[:40]+')'}" for x in rl]
            repeat_rows = []
            for res in sorted((round_dir / "rollouts" / tid / f"cand{row['k']:02d}").glob("*/results/*.json")):
                try:
                    d = json.loads(res.read_text())
                    led = d.get("ledger") or {}
                    tool_audit = d.get("tool_audit") or {}
                    middleware_audit = d.get("middleware_audit") or {}
                    evaluated_edits = [
                        step for step in (d.get("steps") or [])
                        if step.get("kind") in {"diff", "full_rewrite"}
                    ]
                    gate_rows = [
                        audit.get("gate") or {}
                        for audit in middleware_audit.values()
                        if isinstance(audit, dict)
                    ]
                    component_reject_reasons: Dict[str, int] = {}
                    for audit in tool_audit.values():
                        if not isinstance(audit, dict):
                            continue
                        reasons = audit.get("reject_reasons") or {}
                        if isinstance(reasons, dict):
                            for reason, count in reasons.items():
                                label = str(reason)
                                component_reject_reasons[label] = (
                                    component_reject_reasons.get(label, 0)
                                    + int(count or 0)
                                )
                    repeat_rows.append({
                        "artifact": res.relative_to(round_dir).as_posix(),
                        "decode_seed": d.get("decode_seed"),
                        "evals": led.get("evaluator_calls"),
                        "probes": led.get("probe_calls"),
                        "llm_calls": led.get("llm_calls"),
                        "input_tokens": led.get("llm_input_tokens"),
                        "output_tokens": led.get("llm_output_tokens"),
                        "input_tokens_source": led.get(
                            "llm_input_tokens_source", "unknown"
                        ),
                        "output_tokens_source": led.get(
                            "llm_output_tokens_source", "unknown"
                        ),
                        "sandbox_seconds": led.get("sandbox_seconds"),
                        "candidate_geometry_calls": led.get(
                            "candidate_geometry_calls", 0
                        ),
                        "valid_edit_rate": (
                            sum(
                                float(step.get("validity", 0.0) or 0.0) >= 1.0
                                for step in evaluated_edits
                            ) / len(evaluated_edits)
                            if evaluated_edits else 0.0
                        ),
                        "new_best_edits": sum(
                            bool(step.get("is_new_best"))
                            for step in evaluated_edits
                        ),
                        "evaluated_edits": len(evaluated_edits),
                        "distinct_program_families": led.get(
                            "distinct_program_families", 0
                        ),
                        "score_progression": [
                            step.get("combined_score")
                            for step in (d.get("steps") or [])
                            if step.get("kind") != "note"
                        ],
                        "component_invocations": sum(
                            int(audit.get("invocations", 0) or 0)
                            for audit in tool_audit.values()
                            if isinstance(audit, dict)
                        ),
                        "component_usable_outputs": sum(
                            int(audit.get("usable_outputs", 0) or 0)
                            for audit in tool_audit.values()
                            if isinstance(audit, dict)
                        ),
                        "component_semantic_successes": sum(
                            int(audit.get(
                                "semantic_successful_invocations", 0
                            ) or 0)
                            for audit in tool_audit.values()
                            if isinstance(audit, dict)
                        ),
                        "component_semantic_rejections": sum(
                            int(audit.get("semantic_rejections", 0) or 0)
                            for audit in tool_audit.values()
                            if isinstance(audit, dict)
                        ),
                        "component_reject_reasons": component_reject_reasons,
                        "middleware_fires": sum(
                            int(audit.get("fires", 0) or 0)
                            for audit in middleware_audit.values()
                            if isinstance(audit, dict)
                        ),
                        "structure_gate_satisfied": sum(
                            int(gate.get("satisfied", 0) or 0)
                            for gate in gate_rows
                        ),
                        "structure_postcondition_failed": sum(
                            int(gate.get("postcondition_failed", 0) or 0)
                            for gate in gate_rows
                        ),
                        "stop": d.get("stop_reason"),
                        "err": (
                            str(d.get("error"))[:120]
                            if d.get("error") else None
                        ),
                    })
                except Exception:
                    pass
            if repeat_rows:
                ent.update(_aggregate_feedback_repeats(repeat_rows))
            activation = row.get("component_evidence") or {}
            if activation:
                ent["component_activation_status"] = activation.get("status")
                ent["component_activated"] = activation.get("activated")
                ent["component_activation_consistent"] = activation.get(
                    "activation_consistent"
                )
                ent["inactive_components"] = activation.get(
                    "inactive_components", []
                )
                ent["component_reject_reasons"] = activation.get(
                    "reject_reasons", ent.get("component_reject_reasons", {})
                )
            if not row["valid"]:
                ent["invalid"] = True
                ent["failure_taxonomy"] = "invalid_component"
            elif ent.get("component_activation_status") == "not_delivered":
                ent["failure_taxonomy"] = "component_not_delivered"
            elif ent.get("component_activation_status") \
                    == "delivered_but_inactive":
                reasons = ent.get("component_reject_reasons") or {}
                ent["failure_taxonomy"] = (
                    "component_rejected:bridge_stale"
                    if "bridge_stale" in reasons
                    else "delivered_but_inactive"
                )
            elif ent.get("component_activation_status") \
                    == "activation_inconsistent_across_repeats":
                ent["failure_taxonomy"] = (
                    "activation_inconsistent_across_repeats"
                )
            elif ent.get("component_invocations", 0) == 0 \
                    and any("new_tools" in str(field)
                            for field in row.get("changed_fields", [])):
                ent["failure_taxonomy"] = "component_never_invoked"
            elif ((row.get("proposal_condition") or {}).get("role")
                  == "constraint_checker") \
                    and ent.get("candidate_geometry_calls", 0) == 0:
                ent["failure_taxonomy"] = "current_candidate_bridge_never_executed"
            elif ent.get("middleware_fires", 0) == 0 \
                    and any("new_middlewares" in str(field)
                            for field in row.get("changed_fields", [])):
                ent["failure_taxonomy"] = "middleware_causally_inert"
            elif ent.get("evaluated_edits", 0) > 1 \
                    and ent.get("distinct_program_families", 0) <= 1:
                ent["failure_taxonomy"] = "repeated_same_family_edits"
            elif row.get("causal_delta") is not None \
                    and float(row["causal_delta"]) <= 0:
                ent["failure_taxonomy"] = "causally_neutral_or_negative"
            elif ent.get("stop") in {"iteration_limit", "budget_exhausted"}:
                ent["failure_taxonomy"] = "search_budget_exhausted"
            else:
                ent["failure_taxonomy"] = "positive_or_unresolved"
            cands.append(ent)
        n_stuck = sum(1 for c in cands if c.get("score") is not None
                      and abs((c["score"] or 0) - g["base_score"]) < 1e-9)
        prev_note = (feedback.get(tid) or {}).get("analyst_note")
        feedback[tid] = {"round": meta["round"], "base_score": g["base_score"],
                         "best_score": g["best_score"], "best_k": g["best_k"],
                         "n_stuck_at_base": n_stuck, "candidates": cands}
        if prev_note:  # analyst notes are curated externally — never wipe them
            # anti-leak (leakage guard): the analyst_note is the
            # only free-form text reaching the proposer; drop it if it carries a
            # leak marker, else neutralize unproven result-claim words so they
            # can't survive as facts. Default-on; a no-op on clean structured notes.
            if os.environ.get("SAH_LEAK_NEUTRALIZE", "1") == "1":
                from outer.safety.leak_guard import code_signals, sanitize_note
                dropped_for = code_signals(prev_note)
                prev_note = sanitize_note(prev_note)
                if dropped_for:
                    print(f"[leak-guard] {tid}: analyst_note DROPPED "
                          f"(code-injection signals: {dropped_for})")
            if prev_note:
                feedback[tid]["analyst_note"] = prev_note
    fb_path.write_text(json.dumps(feedback, indent=1))
    (round_dir / "task_feedback_after.json").write_text(
        json.dumps(feedback, indent=1)
    )

    # Route-independent program inheritance.  Canonical comparison runs set
    # SAH_PROGRAM_RATCHET_MODE=strict_single; legacy exploratory campaigns keep
    # their historical QD/crossover memory behind an explicit mode.
    bp_path = round_dir.parent / "best_programs.json"
    ratchet_mode = os.environ.get("SAH_PROGRAM_RATCHET_MODE", "legacy_qd")
    ratchet_mode_source = (
        "env" if "SAH_PROGRAM_RATCHET_MODE" in os.environ else "default"
    )
    if (os.environ.get("SAH_REQUIRE_STRICT_RATCHET", "0") == "1"
            and ratchet_mode != "strict_single"):
        raise RuntimeError(
            "SAH_REQUIRE_STRICT_RATCHET=1 but program ratchet mode is "
            f"{ratchet_mode!r} ({ratchet_mode_source}); canonical campaigns "
            "must run strict_single"
        )
    logical_index = (
        meta.get("inference_trajectory_budget") or {}
    ).get("logical_round_index")
    try:
        best_programs = json.loads(bp_path.read_text()) if bp_path.exists() else {}
        if not isinstance(best_programs, dict):
            raise TypeError("program incumbent registry must be a JSON object")
    except Exception as exc:
        if ratchet_mode == "strict_single":
            raise RuntimeError(
                f"canonical program incumbent registry is unreadable: {bp_path}"
            ) from exc
        best_programs = {}
    if ratchet_mode == "strict_single" and logical_index not in (None, 0) \
            and not bp_path.is_file():
        raise RuntimeError(
            f"canonical post-cold round lacks program incumbent registry: {bp_path}"
        )
    best_programs, ratchet_audit = update_program_ratchet(
        mode=ratchet_mode,
        round_dir=round_dir,
        groups=groups,
        bases_in=meta.get("bases_in", {}),
        previous=best_programs,
        round_id=int(meta["round"]),
    )
    for tid, row in ratchet_audit["tasks"].items():
        groups[tid]["program_ratchet"] = row
        groups[tid]["accepted_improvement"] = bool(row.get("promoted"))
        if ratchet_mode == "strict_single" and groups[tid].get("improved") \
                and not row.get("promoted"):
            # A raw score cannot advance H2 when it cannot be bound to a newly
            # improved program from the same rollout.  This closes the legacy
            # step-0/inherited-program attribution hole.
            pt = meta["per_task"][tid]
            next_bases[tid] = {
                "package": pt["base_package"],
                "score": pt["base_score"],
                "seed_score": pt["seed_score"],
                "from": "unchanged_program_attribution_failed",
            }
            # Do not let an unbound step-0/noisy score update proposer weights
            # indirectly.  Keeping the diagnostic rewards while zeroing every
            # task-row advantage makes replay conversion skip this update.
            suppress_task_training(batch_rows, tid, row.get("reason"))
            groups[tid]["training_suppressed"] = True
            groups[tid]["training_suppression_reason"] = row.get("reason")
        # COMPRATCHET-001 runs HERE, not in the per-task loop above: the
        # strict_single revert replaces next_bases[tid] wholesale, which
        # silently discarded a graft the ledger had already recorded as
        # admitted (and left the graft dir orphaned on disk).  Grafting after
        # the base is final also means components land on the package that
        # actually gets inherited.
        try:
            from outer.rounds.component_ratchet import apply_component_ratchet
            ratchet = apply_component_ratchet(
                round_dir=round_dir, tid=tid, round_index=int(meta["round"]),
                candidates=(meta["per_task"][tid].get("candidates") or []),
                rows=groups[tid].get("rows") or [],
                improved=bool(groups[tid].get("improved")),
                best_k=groups[tid].get("best_k"),
                next_base_entry=next_bases[tid],
            )
            if ratchet and ratchet.get("admitted"):
                names = ", ".join(a["name"] for a in ratchet["admitted"])
                print(f"  [{tid}] component ratchet admitted: {names} "
                      f"-> {next_bases[tid]['package']}")
        except Exception as exc:
            print(f"  [{tid}] component ratchet skipped on error: {exc}")
        if tid in feedback:
            feedback[tid]["accepted_improvement"] = bool(row.get("promoted"))
            feedback[tid]["program_ratchet_reason"] = row.get("reason")
            feedback[tid]["outgoing_base_score"] = next_bases[tid]["score"]
        print(
            f"  [program-ratchet:{ratchet_mode}] {tid}: "
            f"promoted={row.get('promoted')} reason={row.get('reason', 'legacy')}"
        )

    bp_path.write_text(json.dumps(best_programs, indent=1))
    (round_dir / "best_programs_after.json").write_text(
        json.dumps(best_programs, indent=1)
    )
    ratchet_audit["mode_source"] = ratchet_mode_source
    ratchet_audit["strict_required"] = (
        os.environ.get("SAH_REQUIRE_STRICT_RATCHET", "0") == "1"
    )
    (round_dir / "program_ratchet_audit.json").write_text(
        json.dumps(ratchet_audit, indent=2)
    )
    # Rewrite after the joint H2/program transition so the next context sees
    # the accepted state, not merely a noisy/raw candidate maximum.
    fb_path.write_text(json.dumps(feedback, indent=1))
    (round_dir / "task_feedback_after.json").write_text(
        json.dumps(feedback, indent=1)
    )

    with open(round_dir / "grpo_batch.jsonl", "w") as f:
        for row in batch_rows:
            f.write(json.dumps(row) + "\n")
    # Publish every transition dependency before the summary marker.  Resumable
    # campaign workers treat round_summary.json as the commit record and must
    # never observe it without the corresponding next_bases registry.
    (round_dir / "next_bases.json").write_text(json.dumps(next_bases, indent=2))
    (round_dir / "round_summary.json").write_text(json.dumps({
        "round": meta["round"], "groups": groups,
        "inference_trajectory_budget": meta.get("inference_trajectory_budget"),
        "improved_tasks": [t for t, g in groups.items() if g["improved"]],
        "accepted_improved_tasks": [
            t for t, g in groups.items() if g.get("accepted_improvement")
        ],
    }, indent=2))
    n_imp = sum(bool(g.get("accepted_improvement")) for g in groups.values())
    print(f"[collect] {len(batch_rows)} GRPO rows | {n_imp}/{len(groups)} tasks accepted-improved "
          f"-> grpo_batch.jsonl, round_summary.json, next_bases.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("propose", help="K H1 runs per task; validate; materialize")
    p.add_argument("--round-dir", required=True)
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--bases-file", default=None,
                   help="per-task bases registry (prev round's next_bases.json)")
    p.add_argument("--base-url", default="http://127.0.0.1:8800/v1")
    p.add_argument("--n-replicas", type=int, default=0,
                   help="if >0, fan H1 runs from --base-url across consecutive ports")
    p.add_argument("--model", default="qwen3.5-9b")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--max-evals", type=int, default=20)
    p.add_argument("--llm-timeout", type=float, default=600.0,
                   help="per-request timeout for each proposer trajectory")
    p.add_argument("--llm-max-retries", type=int, default=2,
                   help="transport retries for each proposer request")
    p.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    p.add_argument("--force-tool-frac", type=float, default=0.0,
                   help="DEPRECATED and ignored: v1.1 uses a deterministic "
                        "condition-identical discovery context; every slot "
                        "receives the same safe intervention surfaces. Accepted "
                        "so existing campaign scripts still run.")
    p.add_argument("--seed-programs-file", default=None,
                   help="global best_programs.json; H1 sees the inherited program as the rollout starting point")
    p.add_argument("--feedback-file", default=None,
                   help="global task_feedback.json provenance input; retained "
                        "for offline analysis but never injected into H1")
    p.set_defaults(fn=cmd_propose)

    c = sub.add_parser("collect", help="per-task rewards + GRPO batch + next bases")
    c.add_argument("--round-dir", required=True)
    c.set_defaults(fn=cmd_collect)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
