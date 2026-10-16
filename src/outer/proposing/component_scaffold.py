"""Placeholder component bodies — used for DETECTION, never for seeding.

Component slots used to start from pre-seeded, already-mounted scaffolding
(a tool file plus schema and mount, or a hook wrapper) so the proposer only
had to write the body.  That is gone: nothing is written into a candidate
draft on the proposer's behalf, because a component the pipeline authored
measures nothing and still lands in the record as if the proposer had
invented it.

The bodies below survive so an INHERITED placeholder — one that reached a
base package from an older campaign — is still recognised as scaffolding
rather than credited as a component.
"""
from __future__ import annotations

from pathlib import Path

import yaml

TOOL_PLACEHOLDER = (
    "def run(ctx, args):\n"
    "    # PLACEHOLDER — replace this body with a real capability for THIS\n"
    "    # task.  Shipping it unchanged is refused.\n"
    "    return {\"replace_me\": True}\n"
)

MIDDLEWARE_PLACEHOLDER = (
    "def before_model(hook_input):\n"
    "    # PLACEHOLDER — replace this body.  Runs automatically before every\n"
    "    # model call and costs the executor no turn; return a string to\n"
    "    # inject a framework note, or None for no message.\n"
    "    return None\n"
)

_MW_WRAPPER = '''"""Generated middleware (h2spec/1.0), lifecycle-audited."""
from nexau.archs.main_sub.execution.hooks import (
    BeforeModelHookInput, HookResult, Middleware)
from nexau.core.messages import Message, Role, TextBlock
from inner.harness.middleware.generated_context import GeneratedHookTracker
from inner.harness.tools.runtime import get_session

# --USER-HOOK-START--
{user_code}
# --USER-HOOK-END--

class GeneratedMiddleware(Middleware):
    def __init__(self):
        self._name = {name_repr}
        self._hook = "before_model"
        self._tracker = GeneratedHookTracker()
        self._registered = False
        self._ensure_registered()

    def _ensure_registered(self):
        if self._registered:
            return
        try:
            get_session().register_middleware(self._name, self._hook)
            self._registered = True
        except Exception:
            pass

    def before_model(self, hook_input: BeforeModelHookInput) -> HookResult:
        self._ensure_registered()
        try:
            session = get_session()
            state = self._tracker.snapshot(hook_input, session)
            note = before_model({{"state": state}})
        except Exception:
            return HookResult.no_changes()
        if not note:
            return HookResult.no_changes()
        return HookResult.with_modifications(messages=[
            *hook_input.messages,
            Message(role=Role.FRAMEWORK, content=[TextBlock(text=str(note))]),
        ])
'''


def placeholder_bodies() -> set:
    """Bodies that count as untouched scaffolding, whatever the file name."""
    return {TOOL_PLACEHOLDER, MIDDLEWARE_PLACEHOLDER}
