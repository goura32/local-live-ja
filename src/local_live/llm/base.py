from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from .events import LLMEvent


Message = dict[str, Any]


class LLMProvider(Protocol):
    requested_model: str

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        cancel_event: Any = None,
    ) -> Iterable[LLMEvent]: ...


def text_message(text: str, role: str = "user") -> Message:
    return {"role": role, "content": text}


def copy_messages(messages: Iterable[Mapping[str, Any]]) -> list[Message]:
    return [dict(message) for message in messages]
