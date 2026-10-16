"""Qwen3.5 multi-turn loss mask for slime training rows.

slime's offline-GRPO path needs, per replay row, the token ids and a mask
selecting exactly the assistant tokens that carry gradient.  Only assistant
turns CLOSED by a later user/tool message count: an unterminated final turn
has no stop token to learn, so training on it teaches the policy to run on.
``grpo_to_replay`` appends a terminal user turn for precisely this reason.

Owned here so the SAH tree depends on slime through its documented API
(``slime.utils.types.Sample``) rather than through another repository's
Python path."""

from __future__ import annotations

from typing import Any


def _find_sublist(values: list[int], needle: list[int], start: int, stop: int) -> int:
    if not needle:
        return -1
    last = stop - len(needle)
    for index in range(start, last + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def extract_input_ids(value: Any) -> list[int]:
    """Normalize list/tensor/BatchEncoding chat-template return values."""
    if isinstance(value, dict) or hasattr(value, "keys"):
        try:
            value = value["input_ids"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "chat template returned a mapping without input_ids"
            ) from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    while (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"chat template input_ids has unsupported type {type(value).__name__}"
        )
    try:
        token_ids = [int(token) for token in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("chat template input_ids is not a flat integer sequence") from exc
    if not token_ids:
        raise ValueError("chat template returned empty input_ids")
    return token_ids


class Qwen35MultiTurnLossMaskGenerator:
    """Generate full token IDs and a mask for closed assistant action turns."""

    def __init__(self, tokenizer: Any, tokenizer_type: str = "qwen3_5"):
        if tokenizer_type != "qwen3_5":
            raise ValueError(f"unsupported tokenizer type: {tokenizer_type}")
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self._assistant_header = self._encode("<|im_start|>assistant\n")
        self._im_end = self._encode("<|im_end|>")
        self._think_open = self._encode("<think>\n")
        self._think_close = self._encode("</think>\n\n")
        if not self._assistant_header:
            raise ValueError("Qwen3.5 assistant header tokenization is empty")
        if len(self._im_end) != 1:
            raise ValueError(
                f"Qwen3.5 <|im_end|> must be one token, got {self._im_end}"
            )

    def _encode(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        encoded = self.tokenizer(text, add_special_tokens=False)
        return list(encoded["input_ids"])

    @staticmethod
    def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
        return [
            len(mask) - mask.index(1) if 1 in mask else 0
            for mask in loss_masks
        ]

    def _assistant_spans(self, token_ids: list[int]) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        end_id = self._im_end[0]
        while cursor <= len(token_ids) - len(self._assistant_header):
            header = _find_sublist(
                token_ids,
                self._assistant_header,
                cursor,
                len(token_ids),
            )
            if header < 0:
                break
            content_start = header + len(self._assistant_header)
            try:
                message_end = token_ids.index(end_id, content_start)
            except ValueError as exc:
                raise ValueError(
                    f"assistant header at token {header} has no <|im_end|>"
                ) from exc

            # The template, not the demonstration, owns this wrapper.  Exclude
            # both empty and non-empty reasoning_content from Mode-2 action loss.
            if token_ids[
                content_start : content_start + len(self._think_open)
            ] == self._think_open:
                close = _find_sublist(
                    token_ids,
                    self._think_close,
                    content_start + len(self._think_open),
                    message_end,
                )
                if close < 0:
                    raise ValueError(
                        f"assistant turn at token {header} has an unclosed think wrapper"
                    )
                content_start = close + len(self._think_close)

            # Include im_end as the learned assistant stop token.
            spans.append((content_start, message_end + 1))
            cursor = message_end + 1
        return spans

    def get_loss_mask(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[list[int], list[int]]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            tools=tools,
            enable_thinking=False,
        )
        token_ids = extract_input_ids(rendered)
        spans = self._assistant_spans(token_ids)
        assistant_positions = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ]
        if len(spans) != len(assistant_positions):
            raise ValueError(
                "Qwen3.5 assistant boundary mismatch: "
                f"messages={len(assistant_positions)} token_spans={len(spans)}"
            )

        loss_mask = [0] * len(token_ids)
        for span, message_index in zip(spans, assistant_positions):
            message = messages[message_index]
            closed = any(
                later.get("role") in {"user", "tool"}
                for later in messages[message_index + 1 :]
            )
            if not closed or message.get("step_loss_mask", 1) != 1:
                continue
            start, end = span
            if start >= end:
                raise ValueError(
                    f"Qwen3.5 assistant turn {message_index} has no content tokens"
                )
            loss_mask[start:end] = [1] * (end - start)
        return token_ids, loss_mask

    def get_text_from_loss_mask(
        self, token_ids: list[int], loss_mask: list[int]
    ) -> list[str]:
        selected: list[str] = []
        current: list[int] = []
        for token, keep in zip(token_ids, loss_mask):
            if keep:
                current.append(token)
            elif current:
                selected.append(self.tokenizer.decode(current))
                current = []
        if current:
            selected.append(self.tokenizer.decode(current))
        return selected
