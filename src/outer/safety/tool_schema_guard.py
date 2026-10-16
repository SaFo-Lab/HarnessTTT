"""Mechanical argument guard for H1's fixed tool surface (TOOLSCHEMA-001).

NexAU validates tool arguments only inside ``Tool.execute`` (after the
middleware chain), and the raw failure surfaces as a Python/validation error
string.  In the fair16-v5 campaign a drifted proposer burned whole
trajectories on malformed calls without recovering: CP round 4 issued eight
``harness_shell`` calls with no ``command`` at all, and an LLM-SQL context
round called ``edit_harness_file`` with hallucinated ``avg_len`` /
``unique_count`` properties.  Every such call cost an iteration and returned
a message that never taught the model the exact expected shape.

This module is the pure, nexau-free half of the fix.  The
``SchemaGuardMiddleware`` (outer/harness/middlewares/schema_guard.py) wraps
every tool call:

- unknown properties with all required keys present are DROPPED and the call
  proceeds (repair — hallucinated extras carry no intent);
- missing required keys REFUSE the call before execution with a corrective
  message that states the exact argument list and a literal example call;
- repeated refusals escalate to the full six-tool usage catalog.

Every decision is recorded on the ProposeSession so round.json can attribute
a failed slot to malformed tool calls instead of a bare "no_submission".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Literal example calls beat generated placeholders for corrective quality.
# Keyed by tool name; the generic fallback covers any future tool.
TOOL_EXAMPLES: Dict[str, str] = {
    "harness_shell": 'harness_shell(command="cat agent.yaml")',
    "write_harness_file": (
        'write_harness_file(path="skills/my-skill/SKILL.md", '
        'content="# Skill\\n...")'
    ),
    "edit_harness_file": (
        'edit_harness_file(path="prompt.md", '
        'append_text="## Task strategy\\n- probe before full eval")'
    ),
    "delete_harness_file": 'delete_harness_file(path="custom_tools/old.py")',
    "upsert_skill": (
        'upsert_skill(name="restart-search", body="# Restart\n...", '
        'description="Switch construction family.", usage_guidance="On stall", '
        'failure_mode="Repeated family", hypothesis="A playbook redirects search", '
        'expected_behavior_change="Try a new family", novelty_basis="Not inherited", '
        'risk="May redirect too early")'
    ),
    "repair_component": (
        'repair_component(kind="tool", name="shape_checker", '
        'surface="input_schema", replacement_schema={"type":"object",'
        '"properties":{}})'
    ),
    "validate_harness": "validate_harness()",
    "submit_harness": "submit_harness()",
}


def load_input_schemas(package_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Read ``{tool name: input_schema}`` for every tool mounted in agent.yaml.

    Fail-closed: a duplicate name, a schema path escaping the package, or a
    missing/mismatched schema file raises rather than silently guarding a
    different surface than the one being served.
    """

    package_dir = Path(package_dir).resolve()
    agent = yaml.safe_load((package_dir / "agent.yaml").read_text())
    schemas: Dict[str, Dict[str, Any]] = {}
    for mount in agent.get("tools") or []:
        name = str(mount.get("name") or "")
        rel = str(mount.get("yaml_path") or "")
        if not name or not rel:
            raise ValueError(f"agent.yaml tool mount missing name/yaml_path: {mount}")
        if name in schemas:
            raise ValueError(f"duplicate tool mount: {name}")
        path = (package_dir / rel).resolve()
        if package_dir not in path.parents:
            raise ValueError(f"{name}: schema path escapes the package: {rel}")
        doc = yaml.safe_load(path.read_text())
        if doc.get("name") != name:
            raise ValueError(
                f"{name}: schema file declares name {doc.get('name')!r}"
            )
        schemas[name] = dict(doc.get("input_schema") or {})
    if not schemas:
        raise ValueError(f"no tools mounted in {package_dir / 'agent.yaml'}")
    return schemas


def usage_hint(name: str, schema: Dict[str, Any]) -> str:
    """One-line argument contract plus a literal example call."""

    properties: Dict[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    parts = []
    for prop, spec in properties.items():
        kind = spec.get("type", "any")
        parts.append(
            f"{prop} ({kind}{', REQUIRED' if prop in required else ''})"
        )
    example = TOOL_EXAMPLES.get(name) or (
        name + "(" + ", ".join(
            f'{prop}="<{(properties[prop].get("type", "value"))}>"'
            for prop in sorted(required)
        ) + ")"
    )
    return (
        f"{name} accepts exactly: {'; '.join(parts) if parts else '(no arguments)'}. "
        f"Example: {example}"
    )


def full_catalog(schemas: Dict[str, Dict[str, Any]]) -> str:
    """The escalation payload: every mounted tool's usage hint."""

    lines = ["Complete tool catalog — use these exact argument shapes:"]
    for name in schemas:
        lines.append(f"- {usage_hint(name, schemas[name])}")
    lines.append(
        "Procedure: inspect with harness_shell; prefer atomic upserts for "
        "components; use raw file editors only as fallback; then validate. "
        "A valid final workspace is auto-finalized."
    )
    return "\n".join(lines)


@dataclass
class GuardDecision:
    action: str  # "pass" | "repair" | "refuse"
    tool: str
    repaired_args: Dict[str, Any] = field(default_factory=dict)
    dropped: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    message: str = ""


def check_tool_args(
    name: str, schema: Dict[str, Any], args: Dict[str, Any],
) -> GuardDecision:
    """Decide pass/repair/refuse for one tool call's arguments.

    Only presence is judged here (unknown / missing-required keys — the two
    observed failure classes).  Value types still go through NexAU's own
    schema validation after the guard.
    """

    properties = schema.get("properties") or {}
    required = [key for key in (schema.get("required") or []) if key not in args]
    unknown = sorted(key for key in args if key not in properties)
    if required:
        detail = (
            f"missing required argument(s) {', '.join(required)}"
            + (f"; unknown argument(s) {', '.join(unknown)} ignored" if unknown else "")
        )
        return GuardDecision(
            action="refuse", tool=name, missing=required, dropped=unknown,
            message=(
                f"REFUSED (malformed arguments for {name}): {detail}. "
                f"{usage_hint(name, schema)}"
            ),
        )
    if unknown:
        repaired = {key: value for key, value in args.items() if key in properties}
        return GuardDecision(
            action="repair", tool=name, repaired_args=repaired, dropped=unknown,
            message=(
                f"{name}: dropped unknown argument(s) {', '.join(unknown)}; "
                f"{usage_hint(name, schema)}"
            ),
        )
    return GuardDecision(action="pass", tool=name, repaired_args=dict(args))


def refusal_text(
    decision: GuardDecision,
    schemas: Dict[str, Dict[str, Any]],
    consecutive_refusals: int,
    *, escalate_after: int = 2,
) -> str:
    """The refusal message, escalating to the full catalog when the model
    keeps emitting malformed calls for the same tool."""

    if consecutive_refusals >= escalate_after:
        return decision.message + "\n\n" + full_catalog(schemas)
    return decision.message
