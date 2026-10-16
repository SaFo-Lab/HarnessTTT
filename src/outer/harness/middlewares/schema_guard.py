"""Schema-guard middleware for H1's fixed tool surface (TOOLSCHEMA-001).

Wraps every tool call before NexAU's own late validation: hallucinated extra
properties are dropped (the call proceeds), calls missing required arguments
are refused with the exact argument contract and a literal example, and
repeated refusals escalate to the full tool catalog.  Decisions are recorded
on the ProposeSession so round.json can attribute a failed slot to malformed
tool calls.  Fails open: any guard error falls through to the unguarded call.
"""
from __future__ import annotations

from pathlib import Path

from nexau.archs.main_sub.execution.hooks import Middleware

from outer.harness.tools.runtime import get_session
from outer.safety.tool_schema_guard import (
    check_tool_args,
    load_input_schemas,
    refusal_text,
)

_H1_PACKAGE = Path(__file__).resolve().parents[1]


class SchemaGuardMiddleware(Middleware):
    def __init__(self, *, escalate_after: int = 2) -> None:
        self.escalate_after = int(escalate_after)
        self._schemas = load_input_schemas(_H1_PACKAGE)
        self._consecutive: dict[str, int] = {}

    def _record(self, event: dict) -> None:
        try:
            get_session().record_schema_guard(event)
        except Exception:
            pass

    def wrap_tool_call(self, params, call_next):
        try:
            tool_name = str(params.tool_name)
            schema = self._schemas.get(tool_name)
            if schema is None:
                return call_next(params)
            decision = check_tool_args(
                tool_name, schema, dict(params.parameters)
            )
        except Exception:
            return call_next(params)

        if decision.action == "pass":
            self._consecutive[tool_name] = 0
            return call_next(params)

        if decision.action == "repair":
            self._consecutive[tool_name] = 0
            for key in decision.dropped:
                # execution_params was built from the raw arguments BEFORE the
                # wrap chain runs, so both dicts must be repaired.
                params.parameters.pop(key, None)
                params.execution_params.pop(key, None)
            self._record({
                "action": "repaired", "tool": tool_name,
                "dropped": decision.dropped,
            })
            return call_next(params)

        count = self._consecutive.get(tool_name, 0) + 1
        self._consecutive[tool_name] = count
        self._record({
            "action": "refused", "tool": tool_name,
            "missing": decision.missing, "dropped": decision.dropped,
            "consecutive": count,
            "escalated": count >= self.escalate_after,
        })
        return refusal_text(
            decision, self._schemas, count, escalate_after=self.escalate_after,
        )
