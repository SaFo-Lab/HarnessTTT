"""Stall-restart middleware for the inner discovery harness.

When the best score has not improved for ``stall_after`` consecutive
evaluations, injects one framework message telling the agent to abandon
incremental mutation and restart from a structurally different approach
(perturb the best program, switch construction family, or hybridize with an
alternative solution shown in the task message). Fires at most
``max_restarts`` times per run.

Attacks the plateau pathology observed on erdos/AC1: within-basin mutation
saturates long before the evaluation budget is spent.

Fails open: any error resolves to "no changes".
"""
from __future__ import annotations

from nexau.archs.main_sub.execution.hooks import (
    BeforeModelHookInput,
    HookResult,
    Middleware,
)
from nexau.core.messages import Message, Role, TextBlock

from inner.harness.tools.runtime import get_session


class StallRestartMiddleware(Middleware):
    """Nudge the agent to restart the search when progress has stalled."""

    def __init__(self, *, stall_after: int = 8, max_restarts: int = 2) -> None:
        self.stall_after = int(stall_after)
        self.max_restarts = int(max_restarts)
        self._fired = 0
        self._last_best: float | None = None
        self._stalled_evals = 0
        self._seen_evals = 0

    def before_model(self, hook_input: BeforeModelHookInput) -> HookResult:
        try:
            session = get_session()
            evals = session.ledger.evaluator_calls
            best = session.best_score
        except Exception:
            return HookResult.no_changes()
        if best == float("-inf"):
            return HookResult.no_changes()

        if evals != self._seen_evals:
            if self._last_best is not None and best is not None \
                    and best <= self._last_best:
                self._stalled_evals += evals - self._seen_evals
            else:
                self._stalled_evals = 0
            self._seen_evals = evals
            self._last_best = best

        if self._fired >= self.max_restarts or self._stalled_evals < self.stall_after:
            return HookResult.no_changes()

        self._fired += 1
        self._stalled_evals = 0
        text = (
            f"PLATEAU DETECTED: the best score ({best}) has not improved for "
            f"{self.stall_after}+ evaluations. Incremental tweaks of the current "
            "approach are saturating. For the next edit, make a STRUCTURAL move "
            "instead: (1) restart from a significantly perturbed copy of the best "
            "program, (2) switch to a different construction/algorithm family, or "
            "(3) hybridize with one of the alternative approaches shown in the "
            "task message, if any. Do not resubmit a small variation of the last "
            "edit."
        )
        messages = [
            *hook_input.messages,
            Message(role=Role.FRAMEWORK, content=[TextBlock(text=text)]),
        ]
        return HookResult.with_modifications(messages=messages)
