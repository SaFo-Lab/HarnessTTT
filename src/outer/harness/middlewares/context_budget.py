"""Proactive one-call context guard for the H1 proposer."""
from __future__ import annotations

import logging
import math
from typing import Any

from nexau.archs.main_sub.execution.hooks import Middleware
from nexau.archs.main_sub.utils.token_counter import TokenCounter


logger = logging.getLogger(__name__)


class H1ContextBudgetMiddleware(Middleware):
    """Narrow response tokens before the sole provider call; never retry."""

    def __init__(
        self, *, max_context_tokens: int = 49152,
        safety_tokens: int = 1536, minimum_response_tokens: int = 256,
        estimate_multiplier: float = 1.05,
    ) -> None:
        self.max_context_tokens = int(max_context_tokens)
        self.safety_tokens = int(safety_tokens)
        self.minimum_response_tokens = int(minimum_response_tokens)
        self.estimate_multiplier = max(1.0, float(estimate_multiplier))
        self.token_counter = TokenCounter()
        self._call_index = 0

    @staticmethod
    def _record(event: dict, *, training_blocking: bool = False) -> None:
        try:
            from outer.harness.tools.runtime import get_session
            get_session().record_h1_runtime_event(
                event, training_blocking=training_blocking,
            )
        except Exception:
            pass

    @staticmethod
    def _provider_error_kind(exc: Exception) -> str:
        text = f"{type(exc).__name__}: {exc}".lower()
        context_markers = (
            "maximum context length", "context length", "context_length",
            "maximum context", "too many tokens", "max context",
        )
        if any(marker in text for marker in context_markers):
            return "provider_context_overflow"
        return "provider_call_error"

    def set_llm_runtime(
        self, llm_config: Any, openai_client: Any, **kwargs: Any,
    ) -> None:
        del llm_config, openai_client
        runtime_cap = kwargs.get("max_context_tokens")
        if runtime_cap is not None:
            self.max_context_tokens = min(
                self.max_context_tokens, int(runtime_cap)
            )

    @staticmethod
    def _requested(params: Any) -> int:
        value = params.api_params.get("max_tokens")
        if value is None:
            value = getattr(params, "max_tokens", None)
        if value is None and getattr(params, "llm_config", None) is not None:
            value = getattr(params.llm_config, "max_tokens", None)
        return max(1, int(value or 4096))

    @staticmethod
    def _apply(params: Any, value: int) -> None:
        params.api_params["max_tokens"] = int(value)
        params.max_tokens = int(value)

    @staticmethod
    def _sampling(params: Any, *, max_tokens: int) -> dict:
        """Resolve the SDK-shaped values that reach the provider request."""

        api = params.api_params if isinstance(params.api_params, dict) else {}
        extra_body = api.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        llm = getattr(params, "llm_config", None)
        llm_extra = getattr(llm, "extra_params", None)
        if not isinstance(llm_extra, dict):
            llm_extra = {}
        llm_body = llm_extra.get("extra_body")
        if not isinstance(llm_body, dict):
            llm_body = {}

        def first(*values):
            return next((value for value in values if value is not None), None)

        chat_kwargs = first(
            extra_body.get("chat_template_kwargs"),
            llm_body.get("chat_template_kwargs"),
        )
        if not isinstance(chat_kwargs, dict):
            chat_kwargs = {}
        return {
            "max_tokens": int(max_tokens),
            "temperature": first(
                api.get("temperature"), getattr(params, "temperature", None),
                getattr(llm, "temperature", None),
            ),
            "top_p": first(
                api.get("top_p"), getattr(params, "top_p", None),
                getattr(llm, "top_p", None),
            ),
            "top_k": first(
                api.get("top_k"), extra_body.get("top_k"),
                llm_extra.get("top_k"), llm_body.get("top_k"),
            ),
            "seed": first(
                api.get("seed"), extra_body.get("seed"),
                llm_extra.get("seed"), llm_body.get("seed"),
            ),
            "enable_thinking": first(
                chat_kwargs.get("enable_thinking"),
                extra_body.get("enable_thinking"),
                llm_body.get("enable_thinking"),
            ),
        }

    def wrap_model_call(self, params: Any, call_next: Any) -> Any:
        tools = params.api_params.get("tools")
        self._call_index += 1
        call_index = self._call_index
        requested = self._requested(params)
        sampling = self._sampling(params, max_tokens=requested)
        try:
            prompt_tokens = self.token_counter.count_tokens(
                params.messages,
                tools=tools if isinstance(tools, list) else None,
            )
        except Exception as exc:
            # Estimation failure must not create a replay; the normal single
            # provider call remains authoritative.
            logger.warning("H1 context estimate unavailable: %s", exc)
            estimate_event = {
                "kind": "context_estimate_unavailable",
                "call_index": call_index,
                "error": f"{type(exc).__name__}: {exc}",
                "sampling": sampling,
            }
            try:
                result = call_next(params)
            except Exception as provider_exc:
                kind = self._provider_error_kind(provider_exc)
                self._record({
                    "kind": kind,
                    "call_index": call_index,
                    "error": f"{type(provider_exc).__name__}: {provider_exc}",
                    "context_estimate_error": estimate_event["error"],
                    "max_context_tokens": self.max_context_tokens,
                    "sampling": sampling,
                }, training_blocking=True)
                raise
            self._record(estimate_event)
            return result

        conservative_prompt_tokens = int(math.ceil(
            int(prompt_tokens) * self.estimate_multiplier
        ))

        available = (
            self.max_context_tokens
            - conservative_prompt_tokens
            - self.safety_tokens
        )
        event = {
            "kind": "context_budget",
            "call_index": call_index,
            "estimated_prompt_tokens": int(prompt_tokens),
            "conservative_prompt_tokens": conservative_prompt_tokens,
            "estimate_multiplier": self.estimate_multiplier,
            "safety_tokens": self.safety_tokens,
            "max_context_tokens": self.max_context_tokens,
            "requested_response_tokens": requested,
            "available_response_tokens": max(0, int(available)),
            "sampling": sampling,
        }
        if available < self.minimum_response_tokens:
            logger.warning(
                "H1 context hard stop before provider call: prompt=%d cap=%d",
                conservative_prompt_tokens, self.max_context_tokens,
            )
            event.update({
                "kind": "context_exhausted",
                "applied_response_tokens": 0,
            })
            self._record(event, training_blocking=True)
            return None
        if available < requested:
            self._apply(params, available)
            logger.warning(
                "H1 context adjusted max_tokens %d -> %d (prompt=%d cap=%d)",
                requested, available, prompt_tokens, self.max_context_tokens,
            )
        applied = min(requested, int(available))
        event["applied_response_tokens"] = applied
        event["sampling"] = self._sampling(params, max_tokens=applied)
        self._record(event)
        try:
            return call_next(params)
        except Exception as exc:
            kind = self._provider_error_kind(exc)
            self._record({
                **event,
                "kind": kind,
                "error": f"{type(exc).__name__}: {exc}",
            }, training_blocking=True)
            raise
