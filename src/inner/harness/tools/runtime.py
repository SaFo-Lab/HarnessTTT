"""Run-scoped bridge between the synchronous NexAU tools and the active task.

Mirrors the earlier runtime's ``harness/nexau/tools/runtime.py`` pattern: the harness sets the
active :class:`~inner.session.InnerSession` in a contextvar for the duration of
one agent run; the tool bindings and middleware resolve it via
:func:`get_session`. The contextvar itself lives in ``inner.session`` (the
single owner of session state); this module re-exports it so components import
it from the conventional ``inner.harness.tools.runtime`` location.
"""
from __future__ import annotations

from inner.runtime.session import get_session, session_scope

__all__ = ["get_session", "session_scope"]
