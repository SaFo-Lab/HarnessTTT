from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from inner.runtime.harness_runner import _finalize_llm_accounting
from inner.runtime.session import BudgetLedger


ROOT = Path(__file__).resolve().parents[2]


class TokenLedgerAccountingTest(unittest.TestCase):
    def test_actual_and_estimated_input_usage_is_provenanced(self):
        ledger = BudgetLedger()
        ledger.record_llm_call(
            input_tokens=100, output_tokens=20,
            input_source="provider_usage", output_source="provider_usage",
        )
        ledger.record_llm_call(
            input_tokens=80, output_tokens=10,
            input_source="token_counter_estimate", output_source="provider_usage",
        )
        self.assertEqual(ledger.llm_calls, 2)
        self.assertEqual(ledger.llm_input_tokens, 180)
        self.assertEqual(ledger.llm_output_tokens, 30)
        self.assertEqual(
            ledger.llm_input_tokens_source, "mixed_actual_and_estimated"
        )
        self.assertEqual(ledger.llm_output_tokens_source, "provider_usage")

    def test_unknown_usage_stays_null_instead_of_becoming_zero(self):
        ledger = BudgetLedger()
        ledger.record_llm_call(
            input_tokens=100, output_tokens=None,
            input_source="token_counter_estimate", output_source="unknown",
        )
        self.assertEqual(ledger.llm_input_tokens, 100)
        self.assertIsNone(ledger.llm_output_tokens)
        self.assertEqual(ledger.llm_output_tokens_source, "unknown")

    def test_immutable_usage_middleware_is_in_base_and_materializer(self):
        mount = "inner.runtime.token_ledger:LLMUsageLedgerMiddleware"
        self.assertIn(mount, (ROOT / "src/inner/harness/agent.yaml").read_text())
        self.assertIn(
            mount,
            (ROOT / "src/outer/compiling/materialize.py").read_text(),
        )

    def test_provider_attempts_are_not_overwritten_by_assistant_history(self):
        ledger = BudgetLedger()
        ledger.record_llm_call(
            input_tokens=10, output_tokens=None,
            input_source="token_counter_estimate", output_source="unknown",
            error="ProviderError: rejected",
        )
        ledger.record_llm_call(
            input_tokens=12, output_tokens=4,
            input_source="provider_usage", output_source="provider_usage",
        )
        agent = SimpleNamespace(history=[
            SimpleNamespace(role="assistant"),
            SimpleNamespace(role="tool"),
        ])

        _finalize_llm_accounting(ledger, agent)

        self.assertEqual(ledger.llm_calls, 2)
        self.assertEqual(len(ledger.llm_usage_records), 2)
        self.assertEqual(ledger.llm_calls_source, "provider_middleware")
        self.assertEqual(ledger.assistant_turns, 1)

    def test_history_count_is_explicit_legacy_fallback(self):
        ledger = BudgetLedger()
        agent = SimpleNamespace(history=[
            SimpleNamespace(role="assistant"),
            SimpleNamespace(role="assistant"),
        ])

        _finalize_llm_accounting(ledger, agent)

        self.assertEqual(ledger.llm_calls, 2)
        self.assertEqual(ledger.llm_calls_source, "assistant_history_fallback")
        self.assertEqual(ledger.assistant_turns, 2)


if __name__ == "__main__":
    unittest.main()
