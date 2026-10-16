from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from outer.workspace.propose_session import ProposeSession, propose_scope


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "src/outer/harness/middlewares/context_budget.py"


def _load_module():
    modules = {
        name: types.ModuleType(name)
        for name in (
            "nexau", "nexau.archs", "nexau.archs.main_sub",
            "nexau.archs.main_sub.execution",
            "nexau.archs.main_sub.execution.hooks",
            "nexau.archs.main_sub.utils",
            "nexau.archs.main_sub.utils.token_counter",
        )
    }
    modules["nexau.archs.main_sub.execution.hooks"].Middleware = object

    class PlaceholderCounter:
        def count_tokens(self, messages, tools=None):
            del messages, tools
            return 0

    modules["nexau.archs.main_sub.utils.token_counter"].TokenCounter = (
        PlaceholderCounter
    )
    spec = importlib.util.spec_from_file_location(
        "h1_context_budget_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class H1ContextBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    @staticmethod
    def _params():
        return SimpleNamespace(
            messages=[{"role": "user", "content": "x"}],
            api_params={
                "max_tokens": 4096, "tools": [],
                "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            },
            max_tokens=4096,
            llm_config=SimpleNamespace(max_tokens=4096),
        )

    def test_qwen_parity_cushion_prevents_i10_one_token_overflow(self):
        middleware = self.module.H1ContextBudgetMiddleware(
            max_context_tokens=49152, safety_tokens=1536,
            estimate_multiplier=1.05, minimum_response_tokens=256,
        )
        middleware.token_counter = SimpleNamespace(
            count_tokens=lambda messages, tools=None: 44400
        )
        params = self._params()
        calls = []
        session = ProposeSession(base_spec={})
        with propose_scope(session):
            result = middleware.wrap_model_call(
                params, lambda value: calls.append(value) or "OK"
            )
        expected = 49152 - math.ceil(44400 * 1.05) - 1536
        self.assertEqual(result, "OK")
        self.assertEqual(len(calls), 1)
        self.assertEqual(params.api_params["max_tokens"], expected)
        self.assertLessEqual(45425 + expected, 49152)
        self.assertEqual(session.h1_runtime_events[-1]["kind"], "context_budget")
        self.assertEqual(session.h1_runtime_events[-1]["sampling"], {
            "max_tokens": expected,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "seed": None, "enable_thinking": None,
        })
        self.assertEqual(session.h1_runtime_events[-1]["call_index"], 1)
        self.assertFalse(session.h1_runtime_events[-1]["training_blocking"])

    def test_nested_extra_body_sampling_is_recorded_as_sent(self):
        middleware = self.module.H1ContextBudgetMiddleware()
        middleware.token_counter = SimpleNamespace(
            count_tokens=lambda messages, tools=None: 1000
        )
        params = self._params()
        params.api_params.pop("top_k")
        params.api_params["extra_body"] = {
            "top_k": 20,
            "seed": 1102001,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        session = ProposeSession(base_spec={})
        with propose_scope(session):
            result = middleware.wrap_model_call(params, lambda value: "OK")
        self.assertEqual(result, "OK")
        self.assertEqual(session.h1_runtime_events[-1]["sampling"], {
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "seed": 1102001,
            "enable_thinking": False,
        })

    def test_hard_stop_is_structured_and_training_blocking(self):
        middleware = self.module.H1ContextBudgetMiddleware()
        middleware.token_counter = SimpleNamespace(
            count_tokens=lambda messages, tools=None: 46000
        )
        session = ProposeSession(base_spec={})
        calls = []
        with propose_scope(session):
            result = middleware.wrap_model_call(
                self._params(), lambda value: calls.append(value)
            )
        self.assertIsNone(result)
        self.assertEqual(calls, [])
        self.assertEqual(session.h1_training_block_reason(), "context_exhausted")
        self.assertEqual(session.h1_runtime_events[-1]["kind"], "context_exhausted")

    def test_provider_context_error_is_structured_and_not_retried(self):
        middleware = self.module.H1ContextBudgetMiddleware()
        middleware.token_counter = SimpleNamespace(
            count_tokens=lambda messages, tools=None: 1000
        )
        session = ProposeSession(base_spec={})
        calls = []

        def fail_once(value):
            calls.append(value)
            raise RuntimeError("maximum context length is 49152 tokens")

        with propose_scope(session), self.assertRaises(RuntimeError):
            middleware.wrap_model_call(self._params(), fail_once)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            session.h1_training_block_reason(), "provider_context_overflow"
        )


if __name__ == "__main__":
    unittest.main()
