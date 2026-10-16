"""Run-scoped bridge between H1's synchronous tools and the active propose run.

Mirrors inner/harness/tools/runtime.py: the session contextvar lives in
outer/propose_session.py (single owner); this module re-exports it so H1
components import from the conventional package-local location.
"""
from __future__ import annotations

from outer.workspace.propose_session import get_session, propose_scope

__all__ = ["get_session", "propose_scope"]
