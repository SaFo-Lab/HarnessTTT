#!/usr/bin/env python3
"""Exercise the OpenAI-compatible protocol needed by both SAH agents.

This is deliberately stricter than a health check.  It retains every request
and response and requires a normal assistant turn, a parsed function call, and
a successful assistant turn after a tool result is returned.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _request(url: str, *, payload: dict[str, Any] | None = None,
             timeout: float = 300.0) -> tuple[dict[str, Any], float]:
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer EMPTY",
            "Content-Type": "application/json",
        },
        method="GET" if payload is None else "POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {raw.decode(errors='replace')[:4000]}"
        ) from exc
    elapsed = time.monotonic() - started
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"non-object JSON response from {url}")
    return parsed, elapsed


def _message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("chat response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("chat response has no assistant message")
    return message


def _nonempty_content(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, str) and bool(content.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8800/v1")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    endpoint = args.base_url.rstrip("/")
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model": args.model,
        "checks": {},
        "timings_s": {},
    }

    models, elapsed = _request(f"{endpoint}/models", timeout=args.timeout)
    _write(args.out / "models_response.json", models)
    model_ids = [row.get("id") for row in models.get("data", [])
                 if isinstance(row, dict)]
    report["timings_s"]["models"] = elapsed
    report["checks"]["served_model_listed"] = args.model in model_ids

    basic_request = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": "Reply with the single word READY."},
        ],
        "temperature": 0,
        "seed": 101,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    _write(args.out / "basic_request.json", basic_request)
    basic_response, elapsed = _request(
        f"{endpoint}/chat/completions", payload=basic_request,
        timeout=args.timeout,
    )
    _write(args.out / "basic_response.json", basic_response)
    basic_message = _message(basic_response)
    report["timings_s"]["basic_chat"] = elapsed
    report["checks"]["basic_assistant_content"] = _nonempty_content(basic_message)

    tools = [{
        "type": "function",
        "function": {
            "name": "lookup_temperature",
            "description": "Return the current temperature for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius"]},
                },
                "required": ["city", "unit"],
                "additionalProperties": False,
            },
        },
    }]
    tool_user = {
        "role": "user",
        "content": (
            "Use lookup_temperature exactly once for Paris in celsius. "
            "Do not answer from memory."
        ),
    }
    tool_request = {
        "model": args.model,
        "messages": [tool_user],
        "tools": tools,
        "tool_choice": "required",
        "temperature": 0,
        "seed": 102,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    _write(args.out / "tool_request.json", tool_request)
    tool_response, elapsed = _request(
        f"{endpoint}/chat/completions", payload=tool_request,
        timeout=args.timeout,
    )
    _write(args.out / "tool_response.json", tool_response)
    tool_message = _message(tool_response)
    report["timings_s"]["tool_call"] = elapsed

    tool_calls = tool_message.get("tool_calls")
    parsed_args: dict[str, Any] | None = None
    call: dict[str, Any] | None = None
    if isinstance(tool_calls, list) and len(tool_calls) == 1 \
            and isinstance(tool_calls[0], dict):
        call = tool_calls[0]
        function = call.get("function")
        if isinstance(function, dict):
            raw_args = function.get("arguments")
            try:
                parsed_args = (raw_args if isinstance(raw_args, dict)
                               else json.loads(raw_args))
            except Exception:
                parsed_args = None
    function = call.get("function", {}) if call else {}
    report["checks"]["one_parsed_tool_call"] = call is not None
    report["checks"]["tool_name"] = function.get("name") == "lookup_temperature"
    report["checks"]["tool_arguments_json"] = isinstance(parsed_args, dict)
    report["checks"]["tool_city"] = (
        isinstance(parsed_args, dict)
        and str(parsed_args.get("city", "")).strip().lower() == "paris"
    )
    report["checks"]["tool_unit"] = (
        isinstance(parsed_args, dict)
        and parsed_args.get("unit") == "celsius"
    )

    tool_call_id = call.get("id") if call else None
    followup_request = {
        "model": args.model,
        "messages": [
            tool_user,
            tool_message,
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"city": "Paris", "temperature": 21,
                                        "unit": "celsius"}),
            },
        ],
        "tools": tools,
        "temperature": 0,
        "seed": 103,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    _write(args.out / "tool_followup_request.json", followup_request)
    followup_response, elapsed = _request(
        f"{endpoint}/chat/completions", payload=followup_request,
        timeout=args.timeout,
    )
    _write(args.out / "tool_followup_response.json", followup_response)
    followup_message = _message(followup_response)
    report["timings_s"]["tool_followup"] = elapsed
    report["checks"]["tool_followup_assistant_content"] = _nonempty_content(
        followup_message
    )

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = all(report["checks"].values())
    _write(args.out / "validation.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit("vLLM protocol validation failed")


if __name__ == "__main__":
    main()
