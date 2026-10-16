"""Generated-tool gate/self-test utility.

Canonical reward-route runs call :func:`review_tool_code` with
``repair_fn=None, max_rounds=0`` *inside* ``validate_harness``.  That path is a
deterministic static gate plus local mock-context self-test and never changes
the bytes submitted by H1.

The optional ``repair_fn`` loop remains only for explicitly noncanonical legacy
experiments.  It returns a proposed repaired string to its caller; it never
mutates a candidate package and is not called by ``outer.propose`` or
``submit_harness``.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from outer.safety.static_gates import check_tool_code, check_tool_utility
from outer.safety.reviewer.selftest import selftest_tool_code

# Legacy repair-path anti-leak invariants (canonical runs never enter it):
#   1. repair_fn MUST be the SAME frozen served M0 endpoint the inner loop uses
#      — never a stronger external model. Capability parity is what keeps the
#      reviewer from injecting solution knowledge the executor doesn't have.
#   2. the repair prompt carries ONLY the code + gate errors — never the task
#      spec, evaluator, sample data, or scores.
#   3. a repair must be a LOCALIZED fix, not a rewrite: if the "fix" diverges
#      too far from the original it is rejected (fail-closed), so the reviewer
#      cannot smuggle in a new algorithm under the guise of a bug fix.
MIN_KEEP_RATIO = 0.55          # repaired vs original similarity floor
_LEAK_MARKERS = ("combined_score", "best_score", "evaluator", "answer",
                 "ground truth", "expected output", "test case", "finch",
                 "sota", "target score")


def _prompt_leak_markers(text: str) -> List[str]:
    """Markers that would signal task/eval CONTENT leaked into the repair prompt.
    Non-fatal: a hit means we refuse to send this to the reviewer and drop the
    tool (fail-closed), rather than crashing the whole propose round."""
    low = text.lower()
    return [m for m in ("evaluator", "ground truth", "expected output",
                        "finch", "sota", "target score") if m in low]


REPAIR_SYSTEM = (
    "You are a Python code reviewer. You are given a tool function that must "
    "define exactly `def run(ctx, args):` and use ONLY these imports: math, re, "
    "json, itertools, functools, collections, heapq, bisect, random, statistics, "
    "string, typing, dataclasses, numpy, pandas. It may NOT import os/sys/"
    "subprocess/io/pathlib or call open/exec/eval. All side effects go through "
    "`ctx` (ctx.get_program(), ctx.stage_edit(code), ctx.probe(), ctx.evaluate(), "
    "ctx.read_input_sample(name, nrows), ctx.list_task_inputs(), ctx.budget_left(), "
    "ctx.scratch_write/read, ctx.log). Return str/dict/list/number. "
    "Fix ONLY the reported errors with the SMALLEST possible change. Do NOT "
    "add new algorithmic logic, do NOT improve the strategy, do NOT rewrite "
    "working parts — you are a mechanic making the code runnable, not a "
    "problem solver. Reply with ONLY the corrected code in a single "
    "```python block."
)


@dataclass
class ReviewOutcome:
    ok: bool
    code: str
    rounds: int
    history: List[str] = field(default_factory=list)
    final_error: Optional[str] = None


def _extract_code(text: str) -> str:
    if "```" in text:
        blk = text.split("```", 2)
        inner = blk[1] if len(blk) > 1 else text
        if inner.startswith("python"):
            inner = inner[len("python"):]
        return inner.strip()
    return text.strip()


def review_tool_code(
    code: str,
    *,
    input_schema: Optional[Dict[str, Any]] = None,
    description: str = "",
    repair_fn: Optional[Callable[[str, str], str]] = None,
    max_rounds: int = 2,
    selftest_timeout: float = 30.0,
    require_utility: bool = False,
    validation_profile: Optional[Dict[str, Any]] = None,
    contract_text: str = "",
) -> ReviewOutcome:
    """Gate + self-test, repairing via ``repair_fn(system, user)`` if provided."""
    history: List[str] = []
    current = code
    for r in range(max_rounds + 1):
        ok_static, static_errs = check_tool_code(current)
        if ok_static and require_utility:
            ok_utility, utility_errs = check_tool_utility(
                current, input_schema=input_schema, description=description,
                validation_profile=validation_profile,
                contract_text=contract_text,
            )
            if not ok_utility:
                ok_static = False
                static_errs = [f"utility gate: {err}" for err in utility_errs]
        ok_test, test_detail = (False, "skipped (static gate failed)")
        if ok_static:
            ok_test, test_detail = selftest_tool_code(
                current,
                timeout_s=selftest_timeout,
                input_schema=input_schema,
                description=description,
                validation_profile=(
                    validation_profile if require_utility else None
                ),
            )
        if ok_static and ok_test:
            return ReviewOutcome(True, current, r, history)

        problems = "; ".join(static_errs) if not ok_static else f"self-test: {test_detail}"
        history.append(f"round {r}: {problems}")
        if repair_fn is None or r == max_rounds:
            return ReviewOutcome(False, current, r, history, final_error=problems)

        user = (f"The tool code below failed review.\n\nERRORS:\n{problems}\n\n"
                f"CODE:\n```python\n{current}\n```\n\nReturn only the fixed code.")
        leak = _prompt_leak_markers(user)
        if leak:  # never ship flagged content to the reviewer — drop, don't crash
            history.append(f"round {r}: repair skipped (leak-marker {leak} in prompt)")
            return ReviewOutcome(False, current, r, history,
                                 final_error=f"leak markers in code, not repaired: {leak}")
        try:
            fixed = _extract_code(repair_fn(REPAIR_SYSTEM, user))
        except Exception as e:  # repairer unavailable -> give up cleanly
            return ReviewOutcome(False, current, r, history, final_error=f"repair_fn error: {e}")
        if not fixed or fixed == current:
            return ReviewOutcome(False, current, r, history, final_error=problems)
        # localization guard: a bug fix stays close to the original; a rewrite
        # (possible knowledge injection) is rejected, keeping fail-closed.
        keep = difflib.SequenceMatcher(None, current, fixed).ratio()
        if keep < MIN_KEEP_RATIO:
            history.append(f"round {r}: repair rejected (similarity {keep:.2f} "
                           f"< {MIN_KEEP_RATIO}; looks like a rewrite, not a fix)")
            return ReviewOutcome(False, current, r, history,
                                 final_error="repair diverged too far (leak guard)")
        current = fixed
    return ReviewOutcome(False, current, max_rounds, history, final_error="max rounds")
