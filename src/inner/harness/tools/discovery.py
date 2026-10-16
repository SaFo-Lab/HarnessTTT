"""NexAU tool bindings for H2's fixed action space.

Three tools bound by ``inner.harness.tools.discovery:<fn>`` in ``agent.yaml``.
They are plain functions (str in, str out) and reach the active task state
through the :mod:`inner.harness.tools.runtime` contextvar bridge.
"""
from __future__ import annotations

from inner.harness.tools.runtime import get_session


def edit_solution(code: str) -> str:
    """Replace the EVOLVE-BLOCK body with ``code`` (full rewrite of the region)."""
    session = get_session()
    if session.ledger.evaluator_exhausted():
        return (
            "REFUSED: evaluation budget is exhausted, so an edited program "
            "cannot be validated or become best. No edit was staged. Call "
            "finish now."
        )
    refusal = session.check_tool_gate("edit_solution")
    if refusal:
        return refusal
    return session.apply_edit(code)


def evaluate_solution() -> str:
    """Score the current program; report combined_score, validity, best, budget."""
    session = get_session()
    refusal = session.check_tool_gate("evaluate_solution")
    if refusal:
        return refusal
    ledger = session.ledger
    if ledger.evaluator_exhausted():
        return (
            f"Evaluation budget exhausted ({ledger.evaluator_calls}/"
            f"{ledger.max_evaluator_calls}). Best combined_score so far: "
            f"{session.best_score:.6g}. Call finish to end."
        )
    out = session.evaluate()
    left = ledger.evaluator_budget_left()
    parts = [f"combined_score = {out.combined_score:.6g}", f"validity = {out.validity:g}"]
    if out.error:
        parts.append(f"error = {out.error}")
    if out.metrics:
        extra = {k: round(v, 6) for k, v in out.metrics.items()
                 if k not in ("combined_score", "validity")}
        if extra:
            parts.append(f"metrics = {extra}")
    tag = "  <-- NEW BEST" if (session.history and session.history[-1].is_new_best) else ""
    parts.append(f"best_so_far = {session.best_score:.6g}{tag}")
    parts.append(f"evaluations_left = {left}")
    if left == 0:
        parts.append("No evaluations left — call finish.")
    return "\n".join(parts)


def probe_solution() -> str:
    """Cheap approximate score of the CURRENT program on subsampled data."""
    session = get_session()
    if session.ledger.evaluator_exhausted():
        return (
            "REFUSED: evaluation budget is exhausted, so a probe can no "
            "longer lead to a validated best program. No probe was run. Call "
            "finish now."
        )
    refusal = session.check_tool_gate("probe_solution")
    if refusal:
        return refusal
    if session.ledger.probe_calls >= session.ledger.max_probe_calls:
        return "probe budget exhausted; use evaluate_solution for the real score."
    out = session.probe()
    left = session.ledger.max_probe_calls - session.ledger.probe_calls
    if out.error:
        return (f"PROBE (subsampled, approximate): FAILED — {out.error}\n"
                f"{left} probes left. Fix the program before a full evaluation.")
    return (f"PROBE (subsampled, approximate) combined_score = {out.combined_score:.6g}. "
            f"NOT comparable to full scores and NOT counted as an evaluation; use it "
            f"only to rank your own variants cheaply. {left} probes left. "
            f"Run evaluate_solution when confident.")


def finish(summary: str) -> str:
    """End the session (stop tool)."""
    session = get_session()
    return (
        f"Session finished. Best combined_score = {session.best_score:.6g} after "
        f"{session.ledger.evaluator_calls} evaluations. Summary: {summary}"
    )
