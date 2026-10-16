"""Qwen3.5 chat-format conversion for training rows.

The trainer is slime (`OpenClaw-RL/slime`).  slime consumes replay rows whose
assistant turns are already in Qwen3.5's *output* form: parsed `tool_calls`
materialised as the `<tool_call><function=...>` XML the model actually emits
at eval time.  If a replay row instead carries structured `tool_calls`, the
tokenised training text differs from the tokenised eval text and the loss is
computed against a format the policy never produces.

This module owns that conversion so the SAH tree depends on the trainer only
through the on-disk replay format, not through another repository's Python
path.  The behaviour is deliberately identical to the converter this pipeline
was validated against; `SAH_QWEN35_FORMAT_MODULE` can point at an external
implementation if a future trainer changes the contract.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from typing import Any, Dict, List

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def _render_qwen35_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    lines = ["<tool_call>", f"<function={name}>"]
    for argument_name, value in arguments.items():
        key = str(argument_name).strip()
        if not key or any(char in key for char in "<>\r\n"):
            raise ValueError("tool call argument name is invalid")
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        elif value is None:
            rendered = "None"
        else:
            rendered = str(value)
        lines.extend([f"<parameter={key}>", rendered, "</parameter>"])
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def _extract_qwen35_tool_calls(content: str) -> List[str]:
    return [m.group(0).strip() for m in _TOOL_CALL_BLOCK_RE.finditer(content)]


def normalize_qwen35_messages(messages: Any) -> List[Dict[str, Any]]:
    """Materialise parsed tool calls as the exact Qwen3.5 assistant XML."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    output = copy.deepcopy(messages)
    for message in output:
        if not isinstance(message, dict):
            continue
        if message.get("content") is None:
            message["content"] = ""
        calls = message.get("tool_calls")
        if message.get("role") != "assistant":
            continue
        if calls is None:
            for stale in ("tool_calls", "function_call", "provider_specific_fields"):
                message.pop(stale, None)
            continue
        if not isinstance(calls, list):
            raise ValueError("assistant tool_calls must be a list")

        rendered_calls = []
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("tool_calls entries must be objects")
            function = call.get("function") if isinstance(call.get("function"), dict) else call
            name = str(function.get("name") or "").strip()
            if not name or any(char in name for char in "<>\r\n"):
                raise ValueError("tool call function name is invalid")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError("tool call arguments are not valid JSON") from exc
            if not isinstance(arguments, dict):
                raise ValueError("tool call arguments must decode to an object")
            rendered_calls.append(_render_qwen35_tool_call(name, arguments))

        content = str(message.get("content") or "").rstrip()
        if rendered_calls and _extract_qwen35_tool_calls(content):
            raise ValueError("assistant message contains both raw and parsed tool calls")
        if rendered_calls:
            separator = "\n\n" if content else ""
            message["content"] = content + separator + "\n".join(rendered_calls)
        else:
            message["content"] = content
        for stale in ("tool_calls", "function_call", "provider_specific_fields"):
            message.pop(stale, None)
    return output


def resolve_normalizer():
    """Return the converter, honouring an external override if one is set.

    ``SAH_QWEN35_FORMAT_MODULE`` names an importable module exposing
    ``normalize_qwen35_messages``; ``SAH_QWEN35_FORMAT_PATH`` optionally
    prepends a directory to ``sys.path`` first.  Unset (the default), the
    in-tree implementation above is used and nothing outside SAH is imported.
    """
    module_name = os.environ.get("SAH_QWEN35_FORMAT_MODULE")
    if not module_name:
        return normalize_qwen35_messages
    extra_path = os.environ.get("SAH_QWEN35_FORMAT_PATH")
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, "normalize_qwen35_messages")
