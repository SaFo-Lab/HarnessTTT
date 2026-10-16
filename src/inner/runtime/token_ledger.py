"""Immutable-runtime accounting for H2 provider tokens."""
from __future__ import annotations

from typing import Any, Optional

from nexau.archs.main_sub.execution.hooks import Middleware
from nexau.archs.main_sub.utils.token_counter import TokenCounter


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and value.get(name) is not None:
            return value.get(name)
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _nonnegative_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


class LLMUsageLedgerMiddleware(Middleware):
    """Count every visible provider call and retain actual/estimated provenance."""

    def __init__(self) -> None:
        self.token_counter = TokenCounter()

    @staticmethod
    def _ledger() -> Any:
        from inner.runtime.session import get_session
        return get_session().ledger

    def _estimate_input(self, params: Any) -> Optional[int]:
        tools = params.api_params.get("tools")
        try:
            return int(self.token_counter.count_tokens(
                params.messages,
                tools=tools if isinstance(tools, list) else None,
            ))
        except Exception:
            return None

    @staticmethod
    def _provider_usage(response: Any) -> tuple[Optional[int], Optional[int]]:
        usage = _field(response, "usage")
        if usage is None:
            return None, None
        return (
            _nonnegative_int(_field(usage, "prompt_tokens", "input_tokens")),
            _nonnegative_int(_field(
                usage, "completion_tokens", "output_tokens"
            )),
        )

    def wrap_model_call(self, params: Any, call_next: Any) -> Any:
        estimated_input = self._estimate_input(params)
        try:
            response = call_next(params)
        except Exception as exc:
            self._ledger().record_llm_call(
                input_tokens=estimated_input, output_tokens=None,
                input_source=(
                    "token_counter_estimate"
                    if estimated_input is not None else "unknown"
                ),
                output_source="unknown",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        # A context guard may stop before the provider. None is not a call.
        if response is None:
            return None
        actual_input, actual_output = self._provider_usage(response)
        self._ledger().record_llm_call(
            input_tokens=(
                actual_input if actual_input is not None else estimated_input
            ),
            output_tokens=actual_output,
            input_source=(
                "provider_usage" if actual_input is not None
                else "token_counter_estimate"
                if estimated_input is not None else "unknown"
            ),
            output_source=(
                "provider_usage" if actual_output is not None else "unknown"
            ),
        )
        return response
