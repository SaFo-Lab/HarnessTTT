"""Trusted semantic validation profiles for proposer-authored components.

Profiles are selected from orchestrator-owned task and portfolio identities.
They are never inferred from H1-authored descriptions, schemas, code, or
prompt text.  That distinction is important: executor-visible prose is an
honesty surface to validate, not an authority that can enable or disable a
task contract.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional


PROFILE_SCHEMA = "sah-component-validation-profile/1.0"
CP26_TASK_ID = "eft__math__circle_packing"


def trusted_component_profile(
    task_id: Optional[str], portfolio_role: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Return the immutable validation contract for one scheduled role.

    Every scheduled portfolio role is recorded so role-specific validators
    (for example the stagnation-gate truth table) do not depend on wording H1
    can edit.  CP26 adds an exact structured-input contract only for the
    constraint-checker role; other CP26 computational tools remain free to use
    task-appropriate schemas.
    """

    role = str(portfolio_role or "").strip()
    if not role:
        return None
    profile: Dict[str, Any] = {
        "schema": PROFILE_SCHEMA,
        "task_id": str(task_id) if task_id is not None else None,
        "portfolio_role": role,
    }
    if str(task_id or "") == CP26_TASK_ID and role == "constraint_checker":
        profile["tool_contract"] = {
            "kind": "fixed_circle_packing_constraint_checker",
            "cardinality": 26,
            "dimension": 2,
            "centers_field": "centers",
            "radii_field": "radii",
        }
    return profile


def fixed_circle_checker_contract(
    profile: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a validated fixed-circle contract, or ``None`` when inactive."""

    if not isinstance(profile, dict) or profile.get("schema") != PROFILE_SCHEMA:
        return None
    contract = profile.get("tool_contract")
    if not isinstance(contract, dict) or contract.get("kind") != (
        "fixed_circle_packing_constraint_checker"
    ):
        return None
    return contract


def normalize_tool_schema(
    profile: Optional[Dict[str, Any]], schema: Dict[str, Any],
) -> tuple[Dict[str, Any], list[str]]:
    """Apply trusted task-role schema plumbing for an atomic transaction.

    CP26 checkers may be called with no geometry arguments, in which case
    their implementation must use ``ctx.get_candidate_geometry()``.  Explicit
    centers/radii remain a paired fallback for counterfactual diagnostics and
    semantic validation.  The proposer owns the rest of the schema.
    """

    contract = fixed_circle_checker_contract(profile)
    out = copy.deepcopy(schema)
    if contract is None:
        return out, []
    cardinality = int(contract["cardinality"])
    dimension = int(contract["dimension"])
    centers_field = str(contract["centers_field"])
    radii_field = str(contract["radii_field"])
    out["type"] = "object"
    properties = out.setdefault("properties", {})
    properties[centers_field] = {
        "type": "array",
        "minItems": cardinality,
        "maxItems": cardinality,
        "items": {
            "type": "array",
            "minItems": dimension,
            "maxItems": dimension,
            "items": {"type": "number"},
        },
        "description": (
            "Optional explicit CP26 centers; provide together with radii. "
            "Omit both to inspect the current candidate through the trusted bridge."
        ),
    }
    properties[radii_field] = {
        "type": "array",
        "minItems": cardinality,
        "maxItems": cardinality,
        "items": {"type": "number", "minimum": 0},
        "description": (
            "Optional explicit CP26 radii; provide together with centers. "
            "Zero radii are valid under the public evaluator contract."
        ),
    }
    # The trusted current-candidate path is deliberately argument-free.  Any
    # proposer-authored tuning fields must therefore have implementation
    # defaults; the semantic self-test exercises ``{}`` and rejects code that
    # still assumes a required knob.
    out["required"] = []
    marker = "sah/trusted-cp26-explicit-or-current-candidate"
    all_of = [
        row for row in (out.get("allOf") or [])
        if not isinstance(row, dict) or row.get("$comment") != marker
    ]
    all_of.append({
        "$comment": marker,
        "oneOf": [
            {"required": [centers_field, radii_field]},
            {"not": {"anyOf": [
                {"required": [centers_field]},
                {"required": [radii_field]},
            ]}},
        ],
    })
    out["allOf"] = all_of
    out["additionalProperties"] = False
    return out, [
        "CP26 centers/radii normalized to exact 26x2/26 shapes",
        "centers/radii made paired optional with current-candidate bridge fallback",
        "all proposer-authored arguments made optional for an empty current-candidate call",
        "radii minimum normalized to inclusive zero",
    ]
