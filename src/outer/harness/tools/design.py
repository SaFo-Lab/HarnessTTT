"""NexAU tool bindings for H1's fixed action space.

Bound by ``outer.harness.tools.design:<fn>`` in agent.yaml. Plain functions
(str in, str out) reaching the active ProposeSession through the runtime
bridge.
"""
from __future__ import annotations

from typing import Any, Dict

from outer.harness.tools.runtime import get_session


def validate_spec(spec_yaml: str) -> str:
    """Validate a draft H2 spec; report changed fields or exact errors."""
    return get_session().validate(spec_yaml)


def submit_spec(spec_yaml: str) -> str:
    """Submit the final candidate H2 spec (stop tool)."""
    return get_session().submit(spec_yaml)


def harness_shell(command: str) -> str:
    """Inspect the private H2 with a read-only shell-shaped command."""
    return get_session().inspect_harness(command)


def write_harness_file(path: str, content: str) -> str:
    """Create or replace one mutable file in the private H2 package."""
    return get_session().write_harness_file(path, content)


def edit_harness_file(
    path: str,
    old_text: str = "",
    new_text: str = "",
    append_text: str = "",
) -> str:
    """Replace one exact occurrence or append guidance to a mutable H2 file."""
    return get_session().edit_harness_file(
        path, old_text, new_text, append_text
    )


def delete_harness_file(path: str) -> str:
    """Delete one mutable H2 file; agent.yaml must also stop mounting it."""
    return get_session().delete_harness_file(path)


def upsert_skill(
    name: str,
    body: str,
    description: str,
    usage_guidance: str,
    failure_mode: str,
    hypothesis: str,
    expected_behavior_change: str,
    novelty_basis: str,
    risk: str,
) -> str:
    """Atomically add/update a skill and its framework-owned plumbing."""

    return get_session().upsert_skill(
        name=name, body=body, description=description,
        usage_guidance=usage_guidance, failure_mode=failure_mode,
        hypothesis=hypothesis,
        expected_behavior_change=expected_behavior_change,
        novelty_basis=novelty_basis, risk=risk,
    )


def upsert_tool(
    name: str,
    implementation: str,
    description: str,
    input_schema: Dict[str, Any],
    usage_guidance: str,
    failure_mode: str,
    hypothesis: str,
    expected_behavior_change: str,
    novelty_basis: str,
    risk: str,
) -> str:
    """Atomically add/update and validate a generated computational tool."""

    return get_session().upsert_tool(
        name=name, implementation=implementation, description=description,
        input_schema=input_schema, usage_guidance=usage_guidance,
        failure_mode=failure_mode, hypothesis=hypothesis,
        expected_behavior_change=expected_behavior_change,
        novelty_basis=novelty_basis, risk=risk,
    )


def upsert_middleware(
    name: str,
    implementation: str,
    description: str,
    usage_guidance: str,
    failure_mode: str,
    hypothesis: str,
    expected_behavior_change: str,
    novelty_basis: str,
    risk: str,
) -> str:
    """Atomically add/update and validate a generated before_model hook."""

    return get_session().upsert_middleware(
        name=name, implementation=implementation, description=description,
        usage_guidance=usage_guidance, failure_mode=failure_mode,
        hypothesis=hypothesis,
        expected_behavior_change=expected_behavior_change,
        novelty_basis=novelty_basis, risk=risk,
    )


def repair_component(
    kind: str, name: str, old_text: str = "", new_text: str = "",
    surface: str = "implementation", replacement_text: str = "",
    replacement_schema: Dict[str, Any] | None = None,
) -> str:
    """Replace or patch one retained transaction surface and revalidate."""

    return get_session().repair_component(
        kind=kind, name=name, old_text=old_text, new_text=new_text,
        surface=surface, replacement_text=replacement_text,
        replacement_schema=replacement_schema,
    )


def validate_harness() -> str:
    """Parse, safety-check, and canonically compile the current H2 workspace."""
    return get_session().validate_harness()


def submit_harness() -> str:
    """Submit the current validated H2 workspace (stop tool)."""
    return get_session().submit_harness()
