"""Open proposer portfolio metadata and fixed-compute submission contract.

All samples in one task/parent/policy group receive the same prompt and safe
editing surface. Diversity comes from independent model sampling under the
evidence-driven H1 system prompt, not from slot-specific briefs. The
submit-time cross-candidate contract preserves fixed compute while normal
workspace validation enforces structure and safety.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from outer.genome import harness_spec as hs


PORTFOLIO_VERSION = "evidence-portfolio/v1.1"
ALLOWED_SURFACES = (
    "prompt", "skill", "sampling", "tool", "middleware", "plumbing",
)


@dataclass(frozen=True)
class ComponentRole:
    key: str
    title: str
    required_kind: str  # compatibility metadata; always "open_portfolio"


# One fixed condition label is shared by every parallel sample. It is artifact
# metadata only and never adds text to the model input.
PORTFOLIO: Tuple[ComponentRole, ...] = (
    ComponentRole(
        "evidence_driven_discovery",
        "evidence-driven discovery",
        "open_portfolio",
    ),
)


def portfolio_enabled() -> bool:
    """Allow an explicit ablation while keeping v1.1 fail-closed by default."""

    import os
    return os.environ.get("SAH_H1_COMPONENT_PORTFOLIO", "1") != "0"


def role_for_slot(
    slot_index: int, round_index: int = 0, group_size: int | None = None,
) -> ComponentRole:
    """Return the fixed metadata role shared by all exchangeable samples."""

    _ = slot_index, round_index, group_size
    return PORTFOLIO[0]


def h2_compute_contract(
    spec: Dict[str, Any], *,
    max_evaluator_calls: int = 20,
    max_probe_calls: int = 30,
    eval_timeout_s: float = 360.0,
    trajectory_wall_timeout_s: float = 0.0,
    max_context_tokens: int = 131072,
    min_executor_iterations: int = 0,
) -> Dict[str, Any]:
    """Immutable phase-one H2 compute ceilings and stable signature.

    The two per-agent values are inherited from the parent and frozen by the
    proposal contract. Remaining ceilings are orchestrator-owned; all are
    signed so candidate/control results prove one complete compute contract.
    """

    iterations = max(
        int((spec.get("agent") or {}).get("max_iterations", 36)),
        int(min_executor_iterations or 0),
    )
    per_call_output = int((spec.get("sampling") or {}).get("max_tokens", 8192))
    max_context = max(1, min(131072, int(max_context_tokens or 131072)))
    eval_timeout = max(1.0, float(eval_timeout_s or 360.0))
    derived_sandbox = (
        (int(max_evaluator_calls) + 1) * (eval_timeout + 30.0)
        + int(max_probe_calls) * (min(eval_timeout, 120.0) + 30.0)
    )
    wall_cap = float(trajectory_wall_timeout_s or 0.0)
    payload = {
        "agent_max_iterations": iterations,
        "model_max_context_tokens": max_context,
        "model_max_output_tokens": per_call_output,
        "max_executor_model_calls": iterations,
        "max_executor_input_tokens": iterations * max_context,
        "max_executor_output_tokens": iterations * per_call_output,
        "max_full_evaluations": int(max_evaluator_calls),
        "max_probe_cost": int(max_probe_calls),
        "max_sandbox_seconds": (
            min(derived_sandbox, wall_cap) if wall_cap > 0.0
            else derived_sandbox
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _rows_by_name(spec: Dict[str, Any], field: str) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("name")): row
        for row in (spec.get(field) or [])
        if isinstance(row, dict) and row.get("name")
    }


def implementation_changes(
    base_spec: Dict[str, Any], effective: Dict[str, Any], field: str,
) -> List[str]:
    """Names whose executable implementation was added or changed.

    Description/schema-only edits do not count: the v1.1 postcondition is
    deliberately about new executable behavior, not metadata that merely
    looks like a component in ``changed_fields``.
    """

    base = _rows_by_name(base_spec, field)
    changed: List[str] = []
    for name, row in _rows_by_name(effective, field).items():
        prior = base.get(name)
        if prior is None or row.get("implementation_py") != prior.get("implementation_py"):
            changed.append(name)
    return sorted(changed)


def implementation_change_inventory(
    base_spec: Dict[str, Any], effective: Dict[str, Any],
) -> Dict[str, List[str]]:
    return {
        "tools": implementation_changes(base_spec, effective, "new_tools"),
        "middlewares": implementation_changes(
            base_spec, effective, "new_middlewares"
        ),
    }


def contract_errors(
    base_spec: Dict[str, Any], effective: Dict[str, Any], slot_index: int,
    round_index: int = 0,
) -> List[str]:
    """Return only externally fixed compute-contract errors.

    The evidence-portfolio prompt deliberately allows prompt, skill, sampling,
    tool, middleware, plumbing, and coherent multi-component changes. General
    genome/workspace validation owns their structural and safety invariants;
    this cross-candidate contract retains only compute fairness.
    """

    errors: List[str] = []
    # Compute fairness is a causal-evaluation invariant, not a portfolio
    # feature. Keep it active in portfolio ablations as well.
    base_budget = h2_compute_contract(base_spec)
    candidate_budget = h2_compute_contract(effective)
    if candidate_budget["sha256"] != base_budget["sha256"]:
        errors.append(
            "H2 compute budget is externally fixed for causal comparison: "
            f"parent={base_budget}, candidate={candidate_budget}. Restore "
            "agent.max_iterations and sampling.max_tokens to the parent values; "
            "a component must win by behavior, not by buying more executor compute."
        )
    _ = slot_index, round_index  # retained for the stable validation API
    return errors
