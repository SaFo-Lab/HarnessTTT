"""Budget-reminder middleware for the inner discovery harness.

Appends a bounded framework message before a model call once the evaluation
budget is running low, so the agent consolidates and submits in time. Follows
the NexAU middleware hook API (same interface the earlier runtime's middleware uses).

Fails open: any error resolves to "no changes" so a reminder never breaks a run.
"""
from __future__ import annotations

from typing import Any

from nexau.archs.main_sub.execution.hooks import (
    BeforeModelHookInput,
    HookResult,
    Middleware,
)
from nexau.core.messages import Message, Role, TextBlock

from inner.harness.tools.runtime import get_session


class BudgetReminderMiddleware(Middleware):
    """Inject an evaluation-budget reminder when few evaluations remain."""

    def __init__(self, *, remind_from_left: int = 3) -> None:
        self.remind_from_left = int(remind_from_left)

    def before_model(self, hook_input: BeforeModelHookInput) -> HookResult:
        try:
            session = get_session()
        except Exception:
            return HookResult.no_changes()
        left = session.ledger.evaluator_budget_left()
        if left > self.remind_from_left:
            return HookResult.no_changes()

        if left <= 0:
            text = (
                f"Evaluation budget exhausted. Best combined_score so far is "
                f"{session.best_score:.6g}. Call finish now."
            )
        else:
            text = (
                f"Only {left} evaluation(s) left. Best combined_score so far is "
                f"{session.best_score:.6g}. Make your remaining edit(s) count on the "
                f"most promising approach, then finish."
            )
        messages = [
            *hook_input.messages,
            Message(role=Role.FRAMEWORK, content=[TextBlock(text=text)]),
        ]
        return HookResult.with_modifications(messages=messages)
