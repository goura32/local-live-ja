from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, ClassVar


def safe_error_details(status_code: int, payload: Any) -> dict[str, Any]:
    """Keep provider diagnostics small and never retain bearer-like values."""
    details: dict[str, Any] = {"status_code": status_code}
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            for key in ("code", "type", "message"):
                if key in error:
                    details[key] = _redact_error_text(error[key])
        elif error is not None:
            details["error"] = _redact_error_text(error)
        elif payload.get("message") is not None:
            details["message"] = _redact_error_text(payload["message"])
    elif payload:
        details["body_type"] = type(payload).__name__
    return details


def _redact_error_text(value: Any) -> str:
    text = str(value)
    text = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text[:512]


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
    details: dict[str, Any] | None = None
    kind: ClassVar[str] = "error"
