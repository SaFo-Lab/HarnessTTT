"""Submit-reminder middleware for H1's file-native H2 editing loop.

From ``remind_from_iteration`` on, if the proposer has not yet submitted a
spec, append a bounded framework message urging it to validate + submit before
the iteration cap — an un-submitted run is an invalid candidate (minimum
reward). Fails open: any error resolves to no-changes.
"""
from __future__ import annotations

from nexau.archs.main_sub.execution.hooks import (
    BeforeModelHookInput,
    HookResult,
    Middleware,
)
from nexau.core.messages import Message, Role, TextBlock

from outer.harness.tools.runtime import get_session


class SubmitReminderMiddleware(Middleware):
    """Portfolio-aware submit reminder and bounded repair escalation.

    The first two-round smoke of the gated pipeline showed the dominant
    no-submission mode: a slot writes an ambitious generated tool, gets a
    real validator error, fixes ONE issue per turn, and repair-loops until
    the iteration cap without ever abandoning the doomed component.  From
    ``bailout_from_left`` remaining turns (after >= ``bailout_after_failures``
    failed validations) the reminder switches to an exact patch recipe over
    the retained rolled-back attempt; from
    ``freeze_from_left`` remaining turns further generated-component edits
    are refused mechanically while the bounded repair transaction remains
    available, so the recipe can still complete.
    """

    def __init__(self, *, remind_from_iteration: int = 6, max_iterations: int = 8,
                 bailout_from_left: int = 10, bailout_after_failures: int = 2,
                 freeze_from_left: int = 6) -> None:
        import os
        # PRELOAD-001: the runtime may shrink the agent's iteration budget
        # (SAH_H1_MAX_ITERS) below the fixed agent.yaml params; the reminder's
        # turns-left arithmetic must follow the ACTUAL cap, and thresholds
        # scale down proportionally for short runs.
        override = os.environ.get("SAH_H1_MAX_ITERS")
        if override and override.isdigit() and int(override) > 0:
            actual = int(override)
            if actual < int(max_iterations):
                scale = actual / float(max_iterations)
                remind_from_iteration = max(
                    2, round(remind_from_iteration * scale)
                )
                bailout_from_left = max(3, round(bailout_from_left * scale))
                freeze_from_left = max(2, round(freeze_from_left * scale))
            max_iterations = actual
        self.remind_from_iteration = int(remind_from_iteration)
        self.max_iterations = int(max_iterations)
        self.bailout_from_left = int(bailout_from_left)
        self.bailout_after_failures = int(bailout_after_failures)
        self.freeze_from_left = int(freeze_from_left)
        self._current_iteration = 0

    # Passes the static gate and the mock self-test verbatim; small enough to
    # fit the late-turn freeze.  Kept as data so recipes and tests share it.
    # A shrink target for the late-turn freeze, NOT a shippable tool: it must
    # pass the static gate and the self-test so a cornered slot can always
    # produce something valid, while the submit guard still refuses it. An
    # earlier revision used a best-score/budget probe here; because that reads
    # like a real capability, slots shipped it and the artifacts recorded a
    # tool the proposer never actually wrote.
    MINIMAL_TOOL_TEMPLATE = (
        "def run(ctx, args):\n"
        "    # PLACEHOLDER — replace this body with a real capability for "
        "THIS\n"
        "    # task.  Shipping it unchanged is refused.\n"
        "    return {\"replace_me\": True}\n"
    )
    # A small generated-component write stays allowed during the freeze so a
    # slot can still shrink a doomed component instead of only deleting it;
    # ambitious rewrites stay blocked.
    FREEZE_MAX_WRITE_CHARS = 6500

    def _component_start_recipe(self, session) -> str:
        _ = session
        return (
            "First state the task/search failure mode "
            "and compare behaviorally distinct alternatives. Then begin the "
            "smallest evidence-backed portfolio using any exposed surface: "
            "atomic upsert_skill/upsert_tool/upsert_middleware for generated "
            "components, or safe workspace editors for prompt, budget-neutral "
            "sampling, and supported inherited registration. No slot-specific brief "
            "assigns a surface, family, or portfolio size."
        )

    def _bailout_recipe(self, session) -> str:
        targets = (
            session.failed_component_targets()
            or session.generated_component_targets()
        )
        files = ", ".join(targets) or "the current component file"
        return (
            "STOP growing the failing generated component — it has failed "
            f"validation {session.failed_validations} times and there is not "
            "enough budget left for speculative features. "
            f"Repair {files} with repair_component on the named failing surface "
            "using replacement_text or replacement_schema, or remove the failed "
            "component if the remaining portfolio still addresses the diagnosis. "
            "Use a full upsert only if its algorithm or schema must materially "
            "change. Validate the strongest coherent subset before the "
            "deadline; never replace it with a probe/score/budget relay."
        )

    def _stuck_on_component(self, session) -> bool:
        return (
            session.validated_revision != session.workspace_revision
            and session.failed_validations >= self.bailout_after_failures
            and bool(
                session.failed_component_targets()
                or session.generated_component_targets()
            )
        )

    def before_model(self, hook_input: BeforeModelHookInput) -> HookResult:
        try:
            session = get_session()
            iteration = int(hook_input.current_iteration)
            self._current_iteration = iteration
        except Exception:
            return HookResult.no_changes()
        if session.submitted or iteration < self.remind_from_iteration:
            return HookResult.no_changes()
        left = max(0, self.max_iterations - iteration)
        if left <= self.bailout_from_left and self._stuck_on_component(session):
            action = self._bailout_recipe(session)
        elif session.workspace_revision == 0:
            action = (
                "No H2 file has changed. Do not call edit_solution, "
                "evaluate_solution, probe_solution, finish, or LoadSkill; "
                "those are future-executor tools and are unavailable to "
                "H1. " + self._component_start_recipe(session)
            )
        elif left <= 3:
            # DEADLINE-001: reserve the final calls for a valid workspace;
            # deterministic auto-finalize banks it after the agent returns.
            action = (
                f"DEADLINE: only {left} tool calls remain. STOP expanding the portfolio. "
                "Call validate_harness NOW; if it reports one small error, "
                "fix only that or drop the invalid low-value change. Once VALID, "
                "stop; the runtime auto-finalizes it."
            )
        elif session.validated_revision != session.workspace_revision:
            action = (
                "The H2 has unvalidated edits. Call validate_harness now and "
                "repair only the errors it reports."
            )
        else:
            action = (
                "The current H2 is validated. If the selected portfolio is "
                "complete, stop now; the runtime auto-finalizes it. Otherwise implement only a "
                "remaining evidence-backed component with distinct marginal "
                "value, then validate the complete portfolio again."
            )
        text = (
            f"Only ~{left} turn(s) remain and H2 is not finalized. "
            f"{action} An invalid final workspace scores the minimum reward."
        )
        return HookResult.with_modifications(messages=[
            *hook_input.messages,
            Message(role=Role.FRAMEWORK, content=[TextBlock(text=text)]),
        ])

    def wrap_tool_call(self, params, call_next):
        """Mechanically preserve enough turns for edit -> validate."""

        try:
            session = get_session()
            tool_name = str(params.tool_name)
            tool_input = dict(params.parameters)
        except Exception:
            return call_next(params)

        left = max(0, self.max_iterations - self._current_iteration)

        if tool_name in {"write_harness_file", "edit_harness_file"}:
            path = str(tool_input.get("path", ""))
            supplied_text = "\n".join(str(tool_input.get(key, "")) for key in (
                "content", "old_text", "new_text", "append_text",
            ))
            if path.endswith("solution.py") or "EVOLVE-BLOCK" in supplied_text:
                return (
                    "REFUSED: H1 must not copy or implement the seed solution or "
                    "EVOLVE-BLOCK/solution.py in H2. Author only reusable harness "
                    "behavior selected from task analysis, then validate."
                )

        if (
            self._current_iteration >= self.remind_from_iteration
            and session.workspace_revision == 0
            and tool_name == "harness_shell"
        ):
            return (
                "REFUSED: the inspection deadline has passed and no H2 file has "
                "changed. " + self._component_start_recipe(session)
            )

        # BAILOUT-001 freeze: with the budget nearly gone and a generated
        # component still failing validation, refuse to keep growing it.
        # Deletes, agent.yaml/prompt.md edits, validate, and submit remain
        # available so the recovery recipe can always complete.  A small
        # write still gets through first, so a slot can shrink a doomed
        # component rather than being forced to delete it outright.
        if (
            left <= self.freeze_from_left
            and self._stuck_on_component(session)
            and tool_name in {"write_harness_file", "edit_harness_file"}
        ):
            path = str(tool_input.get("path", ""))
            if (path.startswith("custom_tools/")
                    or (path.startswith("tools/")
                        and path.endswith(".tool.yaml"))
                    or path.startswith("middlewares/")
                    or path.startswith("skills/")):
                supplied = sum(
                    len(str(tool_input.get(key, "")))
                    for key in ("content", "new_text", "append_text")
                )
                if supplied > self.FREEZE_MAX_WRITE_CHARS:
                    return "REFUSED (write too large): " + \
                        self._bailout_recipe(session)
                # A compact replacement is the only valid v1.1 recovery.
                return call_next(params)
        return call_next(params)
