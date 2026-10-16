"""Mechanistic activation evidence for proposer-authored H2 components.

Transport activity is not behavior. A generated tool can be invoked and still
return a structured rejection; a middleware hook is called every turn but only
intervenes when it fires. This module gives reward collection and component
inheritance one fail-closed interpretation of the runtime audit.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping


COMPONENT_EVIDENCE_SCHEMA = "sah-component-activation-evidence/2.0"
_FIELDS = {
    "new_tools": "tools",
    "new_skills": "skills",
    "new_middlewares": "middlewares",
}


def changed_component_names(candidate: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Return executable component names encoded in ``changed_fields``."""

    changed = {kind: [] for kind in _FIELDS.values()}
    for raw in candidate.get("changed_fields") or []:
        field, separator, name = str(raw).partition(".")
        kind = _FIELDS.get(field)
        if kind and separator and name:
            changed[kind].append(name)
    return {kind: sorted(set(names)) for kind, names in changed.items()}


def _int(row: Mapping[str, Any], key: str) -> int:
    try:
        return max(0, int(row.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _merge_reasons(target: Counter, value: Any) -> None:
    if isinstance(value, Mapping):
        for reason, count in value.items():
            try:
                amount = max(0, int(count or 0))
            except (TypeError, ValueError):
                amount = 0
            if amount:
                target[str(reason)[:160]] += amount
    elif value:
        target[str(value)[:160]] += 1


def _aggregate_named_audit(
    results: List[Mapping[str, Any]], audit_key: str, names: Iterable[str],
    counters: Iterable[str], activation_counter: str,
) -> Dict[str, Dict[str, Any]]:
    repeats = len(results)
    aggregated: Dict[str, Dict[str, Any]] = {}
    for name in names:
        totals = {counter: 0 for counter in counters}
        delivered_repeats = 0
        active_repeats = 0
        reasons: Counter = Counter()
        for result in results:
            audit = result.get(audit_key) or {}
            row = audit.get(name) if isinstance(audit, Mapping) else None
            if not isinstance(row, Mapping):
                continue
            delivered = _int(row, "mounts") > 0
            active = _int(row, activation_counter) > 0
            delivered_repeats += int(delivered)
            active_repeats += int(active)
            for counter in totals:
                if counter == "gate_refusals":
                    gate = row.get("gate") or {}
                    totals[counter] += (
                        _int(gate, "refused")
                        if isinstance(gate, Mapping) else 0
                    )
                else:
                    totals[counter] += _int(row, counter)
            _merge_reasons(reasons, row.get("reject_reasons"))
            if not row.get("reject_reasons"):
                _merge_reasons(reasons, row.get("last_reject_reason"))
        aggregated[name] = {
            **totals,
            "delivered_repeats": delivered_repeats,
            "active_repeats": active_repeats,
            "repeat_count": repeats,
            "delivered": repeats > 0 and delivered_repeats == repeats,
            "activated": active_repeats > 0,
            "activation_consistent": (
                repeats > 0 and active_repeats == repeats
            ),
            "reject_reasons": dict(sorted(reasons.items())),
        }
    return aggregated


def component_evidence(
    candidate: Mapping[str, Any], results: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate activation evidence across candidate rollout repeats.

    Tool activation is intentionally strict: only the runtime counter produced
    by a top-level ``{"ok": true, ...}`` result counts. ``completed`` and
    ``usable_outputs`` remain diagnostics, not attribution evidence.
    Middleware activation requires ``fires > 0``. Generated skills count when
    their full playbook was loaded or injected before the first edit.
    """

    terminal_results = [row for row in results if isinstance(row, Mapping)]
    changed = changed_component_names(candidate)
    tools = _aggregate_named_audit(
        terminal_results, "tool_audit", changed["tools"],
        (
            "mounts", "invocations", "completed", "errors",
            "usable_outputs", "semantic_successful_invocations",
            "semantic_rejections",
        ),
        "semantic_successful_invocations",
    )
    middlewares = _aggregate_named_audit(
        terminal_results, "middleware_audit", changed["middlewares"],
        ("mounts", "invocations", "fires", "errors", "gate_refusals"),
        "fires",
    )
    skills: Dict[str, Dict[str, Any]] = {}
    repeats = len(terminal_results)
    for name in changed["skills"]:
        delivered_repeats = 0
        active_repeats = 0
        totals = {
            "mounts": 0, "loads_before_first_edit": 0,
            "runtime_injections": 0, "tool_loads": 0,
        }
        for result in terminal_results:
            audit = result.get("skill_audit") or {}
            row = audit.get(name) if isinstance(audit, Mapping) else None
            if not isinstance(row, Mapping):
                continue
            delivered_repeats += int(_int(row, "mounts") > 0)
            active = (
                _int(row, "loads_before_first_edit") > 0
                or _int(row, "runtime_injections") > 0
            )
            active_repeats += int(active)
            for counter in totals:
                totals[counter] += _int(row, counter)
        skills[name] = {
            **totals,
            "delivered_repeats": delivered_repeats,
            "active_repeats": active_repeats,
            "repeat_count": repeats,
            "delivered": repeats > 0 and delivered_repeats == repeats,
            "activated": active_repeats > 0,
            "activation_consistent": (
                repeats > 0 and active_repeats == repeats
            ),
            "reject_reasons": {},
        }

    named = [*tools.values(), *middlewares.values(), *skills.values()]
    activation_required = bool(named)
    delivered = activation_required and all(row["delivered"] for row in named)
    activated = activation_required and all(row["activated"] for row in named)
    activation_consistent = activation_required and all(
        row["activation_consistent"] for row in named
    )
    inactive = sorted(
        name
        for collection in (tools, middlewares, skills)
        for name, row in collection.items()
        if not row["activated"]
    )
    inconsistent = sorted(
        name
        for collection in (tools, middlewares, skills)
        for name, row in collection.items()
        if row["activated"] and not row["activation_consistent"]
    )
    rejection_reasons: Counter = Counter()
    for row in tools.values():
        rejection_reasons.update(row["reject_reasons"])
    if not activation_required:
        status = "no_component_activation_required"
    elif not delivered:
        status = "not_delivered"
    elif not activated:
        status = "delivered_but_inactive"
    elif not activation_consistent:
        status = "activation_inconsistent_across_repeats"
    else:
        status = "activated"
    return {
        "schema": COMPONENT_EVIDENCE_SCHEMA,
        "repeat_count": repeats,
        "changed": changed,
        "tools": tools,
        "middlewares": middlewares,
        "skills": skills,
        "activation_required": activation_required,
        "delivered": delivered,
        "activated": activated,
        "activation_consistent": activation_consistent,
        "positive_credit_eligible": bool(
            activation_required and activation_consistent
        ),
        "status": status,
        "inactive_components": inactive,
        "inconsistent_components": inconsistent,
        "tool_invocations": sum(row["invocations"] for row in tools.values()),
        "tool_usable_outputs": sum(
            row["usable_outputs"] for row in tools.values()
        ),
        "tool_semantic_successful_invocations": sum(
            row["semantic_successful_invocations"] for row in tools.values()
        ),
        "tool_semantic_rejections": sum(
            row["semantic_rejections"] for row in tools.values()
        ),
        "middleware_invocations": sum(
            row["invocations"] for row in middlewares.values()
        ),
        "middleware_fires": sum(row["fires"] for row in middlewares.values()),
        "middleware_gate_refusals": sum(
            row["gate_refusals"] for row in middlewares.values()
        ),
        "reject_reasons": dict(sorted(rejection_reasons.items())),
    }


def promotion_gate(
    *, evidence: Mapping[str, Any], pair_samples: Iterable[Mapping[str, Any]],
    score: float | None, base_score: float, causal_delta: float | None,
    min_effect: float, min_repeats: int,
) -> Dict[str, Any]:
    """Return an auditable conservative paired-promotion decision."""

    samples = [sample for sample in pair_samples if isinstance(sample, Mapping)]
    deltas = [float(sample["causal_delta"]) for sample in samples]
    blockers: List[str] = []
    if evidence.get("activation_required"):
        if not evidence.get("activated"):
            blockers.append(str(evidence.get("status") or "component_inactive"))
        elif not evidence.get("activation_consistent"):
            blockers.append("activation_inconsistent_across_repeats")
    if len(samples) < int(min_repeats):
        blockers.append(f"paired_repeats_below_{int(min_repeats)}")
    if deltas and not all(delta > 0.0 for delta in deltas):
        blockers.append("paired_delta_sign_inconsistent")
    if causal_delta is None or float(causal_delta) <= float(min_effect):
        blockers.append(f"mean_delta_not_above_{float(min_effect):g}")
    if score is None or float(score) <= float(base_score):
        blockers.append("candidate_not_above_historical_base")
    return {
        "eligible": not blockers,
        "blockers": blockers,
        "observed_repeats": len(samples),
        "minimum_repeats": int(min_repeats),
        "positive_sign_consistent": bool(deltas) and all(
            delta > 0.0 for delta in deltas
        ),
        "minimum_effect": float(min_effect),
    }
