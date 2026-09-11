from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class LLMEvent:
    kind: ClassVar[str] = "event"


@dataclass(frozen=True)
class TextDelta(LLMEvent):
    text: str
    requested_model: str | None = None
    actual_model: str | None = None
    kind: ClassVar[str] = "text_delta"


@dataclass(frozen=True)
class ToolCall(LLMEvent):
    call_id: str
    name: str
    arguments: dict[str, Any]
    requested_model: str | None = None
    actual_model: str | None = None
    kind: ClassVar[str] = "tool_call"


@dataclass(frozen=True)
class ToolResult(LLMEvent):
    call_id: str
    name: str
    content: str
    kind: ClassVar[str] = "tool_result"


@dataclass(frozen=True)
class Completion(LLMEvent):
    reason: str
    requested_model: str | None = None
    actual_model: str | None = None
    usage: dict[str, Any] | None = None
    kind: ClassVar[str] = "completion"


@dataclass(frozen=True)
class Cancelled(LLMEvent):
    reason: str = "cancel_requested"
    kind: ClassVar[str] = "cancelled"


@dataclass(frozen=True)
class LLMError(LLMEvent):
    message: str
    retryable: bool = False
    status_code: int | None = None
    kind: ClassVar[str] = "error"
