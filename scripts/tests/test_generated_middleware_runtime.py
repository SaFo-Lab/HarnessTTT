from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inner.harness.middleware.generated_context import GeneratedHookTracker  # noqa: E402
from inner.harness.tools.runtime import session_scope  # noqa: E402
from inner.runtime.harness_runner import _expected_generated_middlewares  # noqa: E402
from inner.runtime.session import BudgetLedger, InnerSession  # noqa: E402
from inner.runtime.program_structure import (  # noqa: E402
    STRUCTURAL_CANONICAL_SCHEMA,
    normalized_ast_digest,
    normalized_ast_payload,
)
from outer.genome.harness_spec import parse_and_validate  # noqa: E402
from outer.compiling.materialize import _build_custom_middlewares  # noqa: E402
from outer.safety.static_gates import check_middleware_code  # noqa: E402


def _session(program: str = "x = 1"):
    session = InnerSession(
        task=SimpleNamespace(initial_program=program),
        ledger=BudgetLedger(max_evaluator_calls=20, max_probe_calls=30),
    )
    session.best_score = 1.0
    return session


class GeneratedMiddlewareRuntimeTest(unittest.TestCase):
    def test_structural_digest_rejects_cosmetic_family_switches(self) -> None:
        baseline = """
def solve(values):
    total = 0
    for value in values:
        total += value + 1
    return total
"""
        renamed_literals_and_noops = """
\"\"\"cosmetic docstring\"\"\"
def compute(items):
    accumulator = 99
    pass
    accumulator = accumulator
    for item in items:
        accumulator += item + 200
    return accumulator
"""
        different_operation = baseline.replace("total += value + 1", "total *= value + 1")
        self.assertEqual(
            normalized_ast_digest(baseline),
            normalized_ast_digest(renamed_literals_and_noops),
        )
        self.assertNotEqual(
            normalized_ast_digest(baseline),
            normalized_ast_digest(different_operation),
        )

    def test_structural_digest_preserves_free_operation_names(self) -> None:
        self.assertNotEqual(
            normalized_ast_digest("result = sum(values)"),
            normalized_ast_digest("result = max(values)"),
        )

    def test_structural_digest_rejects_i9_semantic_noop_bypasses(self) -> None:
        baseline = """
def solve(value):
    result = value + 1
    return result
"""
        noop_variant = """
def solve(value):
    unused = 99
    result = 1 + value
    result += 0
    result = result + 0
    if False:
        return value * 1000
    return result
"""
        self.assertEqual(
            normalized_ast_digest(baseline),
            normalized_ast_digest(noop_variant),
        )

    def test_structural_digest_rejects_dead_loop_ternary_and_literal_expr(self):
        baseline = """
def solve(value):
    result = value + 1
    return result
"""
        noop_variant = """
def solve(value):
    unused = 1 + 2
    40 + 2
    while False:
        return value * 1000
    result = value * 1000 if False else value + 1
    return result
"""
        self.assertEqual(
            normalized_ast_digest(baseline),
            normalized_ast_digest(noop_variant),
        )

    def test_structural_payload_is_explicitly_versioned(self) -> None:
        payload = normalized_ast_payload("def run(x):\n    return x + 1\n")
        self.assertIsNotNone(payload)
        self.assertIn(STRUCTURAL_CANONICAL_SCHEMA, payload)
        self.assertNotIn("type_params", payload)
        self.assertNotIn("type_comment", payload)

    def test_tracks_structural_family_not_literal_values(self) -> None:
        tracker = GeneratedHookTracker()
        session = _session("x = 1")

        first = tracker.snapshot(SimpleNamespace(current_iteration=0), session)
        session.ledger.edit_calls += 1
        session.current_program = "x = 2"
        second = tracker.snapshot(SimpleNamespace(current_iteration=1), session)
        session.ledger.edit_calls += 1
        session.current_program = "x = [2]"
        third = tracker.snapshot(SimpleNamespace(current_iteration=2), session)

        self.assertEqual(first["family_streak"], 1)
        self.assertEqual(second["family_streak"], 2)
        self.assertEqual(len(second["families_explored"]), 1)
        self.assertEqual(third["family_streak"], 1)
        self.assertEqual(len(third["families_explored"]), 2)

    def test_family_streak_does_not_increment_on_model_turns_without_edits(self):
        tracker = GeneratedHookTracker()
        session = _session("x = 1")
        first = tracker.snapshot(SimpleNamespace(current_iteration=0), session)
        second = tracker.snapshot(SimpleNamespace(current_iteration=1), session)
        third = tracker.snapshot(SimpleNamespace(current_iteration=2), session)
        self.assertEqual(
            [first["family_streak"], second["family_streak"],
             third["family_streak"]],
            [1, 1, 1],
        )

    def test_pending_evaluation_distinguishes_staged_edit_from_last_outcome(self):
        tracker = GeneratedHookTracker()
        session = _session("x = 1")
        before = tracker.snapshot(SimpleNamespace(current_iteration=0), session)
        session.apply_edit("x = 2")
        staged = tracker.snapshot(SimpleNamespace(current_iteration=1), session)
        session._pending_edit_mode = "edit_eval"
        after = tracker.snapshot(SimpleNamespace(current_iteration=2), session)
        session.apply_edit(
            "<<<<<<< SEARCH\nmissing = 1\n=======\nmissing = 2\n"
            ">>>>>>> REPLACE"
        )
        no_match = tracker.snapshot(
            SimpleNamespace(current_iteration=3), session
        )
        self.assertFalse(before["pending_evaluation"])
        self.assertIsNone(before["last_edit_staged"])
        self.assertTrue(staged["pending_evaluation"])
        self.assertTrue(staged["last_edit_staged"])
        self.assertFalse(after["pending_evaluation"])
        self.assertFalse(no_match["pending_evaluation"])
        self.assertFalse(no_match["last_edit_staged"])

    def test_exposes_stall_and_last_error(self) -> None:
        tracker = GeneratedHookTracker()
        session = _session()
        tracker.snapshot(SimpleNamespace(current_iteration=0), session)

        session.ledger.evaluator_calls = 2
        session.history.append(SimpleNamespace(
            kind="edit_eval", error=None, validity=1.0, combined_score=1.0,
        ))
        session.history.extend([
            SimpleNamespace(kind="probe", error=None, validity=1.0,
                            combined_score=0.9),
            SimpleNamespace(kind="probe", error="SyntaxError: bad candidate",
                            validity=0.0, combined_score=0.0),
        ])
        view = tracker.snapshot(SimpleNamespace(current_iteration=3), session)

        self.assertEqual(view["stalled_evals"], 2)
        self.assertEqual(view["last_error"], "SyntaxError: bad candidate")
        self.assertEqual(view["evaluations_remaining"], 18)
        self.assertEqual(view["evals_done"], 2)
        self.assertEqual(view["probes_since_eval"], 2)
        self.assertEqual(view["last_family"], view["families_explored"][0])
        self.assertEqual(view["state"]["iteration"], 3)

    def test_gate_accepts_documented_diversity_state(self) -> None:
        code = '''
def before_model(hook_input):
    state = hook_input.get("state", {})
    if state.get("family_streak", 0) >= 5:
        return "switch structure"
    return None
'''
        ok, errors = check_middleware_code(code, "before_model")
        self.assertTrue(ok, errors)

    def test_gate_rejects_undefined_state_and_framework_attributes(self) -> None:
        undefined = '''
def before_model(hook_input):
    state = hook_input.get("state", {})
    return "switch" if state.get("magic_counter", 0) else None
'''
        direct_framework_access = '''
def before_model(hook_input):
    return str(hook_input.current_iteration)
'''

        ok, errors = check_middleware_code(undefined, "before_model")
        self.assertFalse(ok)
        self.assertTrue(any("magic_counter" in e for e in errors))
        ok, errors = check_middleware_code(direct_framework_access, "before_model")
        self.assertFalse(ok)
        self.assertTrue(any("unsupported hook_input attribute" in e for e in errors))

    def test_spec_rejects_unadapted_non_model_hooks(self) -> None:
        result = parse_and_validate('''
schema: h2spec/1.0
new_middlewares:
  - name: unsafe_tool_hook
    hook: after_tool
    description: unsupported framework contract
    implementation_py: |
      def after_tool(hook_input):
          return "note"
''')
        self.assertFalse(result.valid)
        self.assertTrue(any("hook" in error for error in result.errors))

    def test_materialized_diversity_hook_fires_and_is_audited(self) -> None:
        class HookResult:
            @staticmethod
            def no_changes():
                return {"changed": False}

            @staticmethod
            def with_modifications(**kwargs):
                return {"changed": True, **kwargs}

        class Middleware:
            pass

        class Message:
            def __init__(self, role, content):
                self.role, self.content = role, content

        class Role:
            FRAMEWORK = "framework"

        class TextBlock:
            def __init__(self, text):
                self.text = text

        modules = {
            "nexau": types.ModuleType("nexau"),
            "nexau.archs": types.ModuleType("nexau.archs"),
            "nexau.archs.main_sub": types.ModuleType("nexau.archs.main_sub"),
            "nexau.archs.main_sub.execution": types.ModuleType(
                "nexau.archs.main_sub.execution"
            ),
            "nexau.archs.main_sub.execution.hooks": types.ModuleType(
                "nexau.archs.main_sub.execution.hooks"
            ),
            "nexau.core": types.ModuleType("nexau.core"),
            "nexau.core.messages": types.ModuleType("nexau.core.messages"),
        }
        hooks = modules["nexau.archs.main_sub.execution.hooks"]
        hooks.BeforeModelHookInput = object
        hooks.HookResult = HookResult
        hooks.Middleware = Middleware
        messages = modules["nexau.core.messages"]
        messages.Message = Message
        messages.Role = Role
        messages.TextBlock = TextBlock
        previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)

        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                code = '''
def before_model(hook_input):
    state = hook_input.get("state", {})
    if state.get("iteration", 0) >= 5 and state.get("family_streak", 0) >= 5:
        return "DIVERSIFICATION_STALLED"
    return None
'''
                _build_custom_middlewares({"new_middlewares": [{
                    "name": "enforce_diversification",
                    "hook": "before_model",
                    "implementation_py": code,
                }]}, root)
                path = root / "middlewares" / "enforce_diversification.py"
                spec = importlib.util.spec_from_file_location("generated_mw_test", path)
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)

                session = _session()
                with session_scope(session):
                    middleware = module.GeneratedMiddleware()
                    outcomes = []
                    for i in range(6):
                        if i:
                            session.ledger.edit_calls += 1
                            session.current_program = f"x = {i + 1}"
                        outcomes.append(middleware.before_model(SimpleNamespace(
                            current_iteration=i, messages=[]
                        )))

                self.assertEqual(
                    [row["changed"] for row in outcomes],
                    [False, False, False, False, False, True],
                )
                notes = [row.edit_note for row in session.history if row.kind == "note"]
                self.assertTrue(any("FIRED iteration=5" in note for note in notes))
                audit = session.middleware_audit["enforce_diversification"]
                self.assertEqual(audit["mounts"], 1)
                self.assertEqual(audit["invocations"], 6)
                self.assertEqual(audit["fires"], 1)
                self.assertEqual(audit["errors"], 0)
                self.assertEqual(
                    session.middleware_participation_issues(
                        ["enforce_diversification"]
                    ),
                    [],
                )
        finally:
            for name, old in previous.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old

    def test_middleware_may_be_ineffective_but_must_be_invoked(self) -> None:
        session = _session()
        session.register_middleware("quiet_hook", "before_model")
        for iteration in range(3):
            session.record_middleware_event(
                "quiet_hook", "invoked", iteration=iteration
            )

        self.assertEqual(session.middleware_audit["quiet_hook"]["fires"], 0)
        self.assertEqual(
            session.middleware_participation_issues(["quiet_hook"]), []
        )

    def test_missing_or_crashing_middleware_is_not_participating(self) -> None:
        session = _session()
        self.assertIn(
            "not mounted",
            session.middleware_participation_issues(["missing_hook"])[0],
        )

        session.register_middleware("broken_hook", "before_model")
        session.record_middleware_event("broken_hook", "invoked", iteration=0)
        session.record_middleware_event(
            "broken_hook", "error", iteration=0, error="boom"
        )
        issue = session.middleware_participation_issues(["broken_hook"])[0]
        self.assertIn("execution error", issue)

    def test_expected_middleware_names_come_from_materialized_agent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "agent.yaml"
            path.write_text('''
middlewares:
  - import: middlewares.enforce_diversification:GeneratedMiddleware
    params: {}
  - import: middlewares.budget_reminder:BudgetReminderMiddleware
    params: {}
''')
            self.assertEqual(
                _expected_generated_middlewares(path),
                ["enforce_diversification"],
            )




class GeneratedMiddlewareGateTest(unittest.TestCase):
    """Dict-return hooks: require_tools enforcement and malformed shapes."""

    def _stub_modules(self):
        class HookResult:
            @staticmethod
            def no_changes():
                return {"changed": False}

            @staticmethod
            def with_modifications(**kwargs):
                return {"changed": True, **kwargs}

        class Middleware:
            pass

        class Message:
            def __init__(self, role, content):
                self.role, self.content = role, content

        class Role:
            FRAMEWORK = "framework"

        class TextBlock:
            def __init__(self, text):
                self.text = text

        modules = {
            "nexau": types.ModuleType("nexau"),
            "nexau.archs": types.ModuleType("nexau.archs"),
            "nexau.archs.main_sub": types.ModuleType("nexau.archs.main_sub"),
            "nexau.archs.main_sub.execution": types.ModuleType(
                "nexau.archs.main_sub.execution"
            ),
            "nexau.archs.main_sub.execution.hooks": types.ModuleType(
                "nexau.archs.main_sub.execution.hooks"
            ),
            "nexau.core": types.ModuleType("nexau.core"),
            "nexau.core.messages": types.ModuleType("nexau.core.messages"),
        }
        hooks = modules["nexau.archs.main_sub.execution.hooks"]
        hooks.BeforeModelHookInput = object
        hooks.HookResult = HookResult
        hooks.Middleware = Middleware
        messages = modules["nexau.core.messages"]
        messages.Message = Message
        messages.Role = Role
        messages.TextBlock = TextBlock
        return modules

    def _materialize(self, root: Path, code: str):
        _build_custom_middlewares({"new_middlewares": [{
            "name": "probe_gate",
            "hook": "before_model",
            "implementation_py": code,
        }]}, root)
        path = root / "middlewares" / "probe_gate.py"
        spec = importlib.util.spec_from_file_location("generated_mw_gate", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_require_tools_return_sets_the_session_gate(self) -> None:
        modules = self._stub_modules()
        previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        try:
            with tempfile.TemporaryDirectory() as td:
                code = (
                    "def before_model(hook_input):\n"
                    "    state = hook_input.get(\"state\", {})\n"
                    "    if state.get(\"iteration\", 0) >= 1:\n"
                    "        return {\"note\": \"stage a fresh candidate then probe\","
                    " \"require_tools\": [\"probe_solution\"]}\n"
                    "    return None\n"
                )
                module = self._materialize(Path(td), code)
                session = _session()
                with session_scope(session):
                    middleware = module.GeneratedMiddleware()
                    middleware.before_model(SimpleNamespace(
                        current_iteration=0, messages=[]))
                    out = middleware.before_model(SimpleNamespace(
                        current_iteration=1, messages=[]))
                self.assertTrue(out["changed"])
                self.assertIsNotNone(session.tool_gate)
                self.assertEqual(session.tool_gate["require"], ("probe_solution",))
                delivered = out["messages"][-1].content[0].text
                self.assertIn("NEXT CALL: probe_solution", delivered)
                self.assertNotIn("stage a fresh candidate", delivered)
                audit = session.middleware_audit["probe_gate"]
                self.assertEqual(audit["fires"], 1)
                self.assertEqual(audit["errors"], 0)
                self.assertEqual(audit["gate"]["enforced"], 1)
        finally:
            for name, mod in previous.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

    def test_unknown_dict_keys_are_hook_errors_without_effects(self) -> None:
        modules = self._stub_modules()
        previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        try:
            with tempfile.TemporaryDirectory() as td:
                code = (
                    "def before_model(hook_input):\n"
                    "    return {\"force_tool\": \"probe_solution\"}\n"
                )
                module = self._materialize(Path(td), code)
                session = _session()
                with session_scope(session):
                    middleware = module.GeneratedMiddleware()
                    out = middleware.before_model(SimpleNamespace(
                        current_iteration=0, messages=[]))
                self.assertFalse(out["changed"])
                self.assertIsNone(session.tool_gate)
                audit = session.middleware_audit["probe_gate"]
                self.assertEqual(audit["errors"], 1)
                self.assertIn("unknown hook-result keys", audit["last_error"])
        finally:
            for name, mod in previous.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

    def test_structure_change_effect_sets_trusted_postcondition(self) -> None:
        modules = self._stub_modules()
        previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        try:
            with tempfile.TemporaryDirectory() as td:
                code = (
                    "def before_model(hook_input):\n"
                    "    state = hook_input.get('state', {})\n"
                    "    if state.get('family_streak', 0) >= 1:\n"
                    "        return {'require_tools': ['edit_solution'], "
                    "'require_structure_change': True}\n"
                    "    return None\n"
                )
                module = self._materialize(Path(td), code)
                session = _session("x = 1")
                with session_scope(session):
                    out = module.GeneratedMiddleware().before_model(
                        SimpleNamespace(current_iteration=1, messages=[])
                    )
                self.assertTrue(out["changed"])
                self.assertIsNotNone(session.tool_gate)
                baseline = session.tool_gate["require_structure_change_from"]
                self.assertEqual(baseline, GeneratedHookTracker().snapshot(
                    SimpleNamespace(current_iteration=1), session
                )["last_family"])
                delivered = out["messages"][-1].content[0].text
                self.assertIn("normalized Python AST", delivered)
                self.assertIn("identifier renames", delivered)
                self.assertIn("self-assignment", delivered)
        finally:
            for name, mod in previous.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

    def test_losing_gate_note_is_suppressed_with_the_gate(self) -> None:
        modules = self._stub_modules()
        previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        try:
            with tempfile.TemporaryDirectory() as td:
                code = (
                    "def before_model(hook_input):\n"
                    "    return {'note': 'edit_solution is now mandatory', "
                    "'require_tools': ['edit_solution']}\n"
                )
                module = self._materialize(Path(td), code)
                session = _session()
                session.request_tool_gate("first_gate", ["probe_solution"])
                with session_scope(session):
                    out = module.GeneratedMiddleware().before_model(
                        SimpleNamespace(current_iteration=1, messages=[])
                    )
                self.assertFalse(out["changed"])
                self.assertEqual(session.tool_gate["set_by"], "first_gate")
                audit = session.middleware_audit["probe_gate"]
                self.assertEqual(audit["fires"], 0)
                self.assertEqual(audit["gate"]["suppressed"], 1)
                notes = [row.edit_note for row in session.history]
                self.assertTrue(any("SUPPRESSED" in note for note in notes))
                self.assertFalse(any(
                    "edit_solution is now mandatory" in note for note in notes
                ))
        finally:
            for name, mod in previous.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod


if __name__ == "__main__":
    unittest.main()


class RegistrationTimingTest(unittest.TestCase):
    """MWREG-001: NexAU builds middlewares BEFORE session_scope opens.

    The first-ever valid generated-middleware rollout died with
    'MiddlewareParticipationError: not mounted' because __init__-time
    registration ran outside the session and the exception was swallowed.
    The wrapper must therefore re-register lazily at hook entry.
    """

    def _materialized_wrapper(self, tmp):
        from outer.compiling.materialize import _MW_WRAPPER
        code = _MW_WRAPPER.format(
            user_code=("def before_model(hook_input):\n"
                       "    return 'NOTE'\n"),
            hook="before_model", name_repr=repr("t_hook"),
            hook_repr=repr("before_model"))
        path = Path(tmp) / "t_hook.py"
        path.write_text(code)
        return path

    def test_hook_entry_registers_when_init_could_not(self):
        import importlib.util, sys, tempfile, types
        from unittest import mock
        from inner.runtime.session import (
            InnerSession, BudgetLedger, session_scope)
        from inner.tasks.eft_task import EFTTask

        fake = {}
        hooks_mod = types.ModuleType("nexau.archs.main_sub.execution.hooks")
        hooks_mod.BeforeModelHookInput = object

        class HookResult:
            @staticmethod
            def no_changes():
                return {"changed": False}

            @staticmethod
            def with_modifications(**kw):
                return {"changed": True, **kw}

        hooks_mod.HookResult = HookResult
        hooks_mod.Middleware = object
        msgs = types.ModuleType("nexau.core.messages")
        msgs.Message = lambda **kw: kw
        msgs.Role = types.SimpleNamespace(FRAMEWORK="framework")
        msgs.TextBlock = lambda text: {"text": text}
        for name, mod in {
            "nexau": types.ModuleType("nexau"),
            "nexau.archs": types.ModuleType("nexau.archs"),
            "nexau.archs.main_sub": types.ModuleType("nexau.archs.main_sub"),
            "nexau.archs.main_sub.execution":
                types.ModuleType("nexau.archs.main_sub.execution"),
            "nexau.archs.main_sub.execution.hooks": hooks_mod,
            "nexau.core": types.ModuleType("nexau.core"),
            "nexau.core.messages": msgs,
        }.items():
            fake[name] = sys.modules.get(name)
            sys.modules[name] = mod
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = self._materialized_wrapper(tmp)
                spec = importlib.util.spec_from_file_location("t_hook", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # 1) NexAU timing: instantiation happens with NO session
                mw = module.GeneratedMiddleware()
                task = mock.Mock(spec=EFTTask)
                task.task_id = "t"
                session = InnerSession(
                    task=task, ledger=BudgetLedger(max_evaluator_calls=1))
                self.assertNotIn("t_hook", session.middleware_audit)
                # 2) the hook fires inside the scope -> must register + invoke
                with session_scope(session):
                    out = mw.before_model(
                        types.SimpleNamespace(current_iteration=1, messages=[]))
                row = session.middleware_audit.get("t_hook")
                self.assertIsNotNone(row, "hook entry did not register")
                self.assertGreaterEqual(row["mounts"], 1)
                self.assertGreaterEqual(row["invocations"], 1)
                self.assertEqual(
                    session.middleware_participation_issues(["t_hook"]), [])
        finally:
            for name, prev in fake.items():
                if prev is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prev
