from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import httpx

from ..config import load_openrouter_key
from .events import Cancelled, Completion, LLMError, TextDelta, ToolCall, LLMEvent


def parse_sse_payload(payload: dict[str, Any], requested_model: str) -> LLMEvent | None:
    actual_model = payload.get("model") or requested_model
    choices = payload.get("choices") or []
    if not choices:
        return None
    choice = choices[0] or {}
    delta = choice.get("delta") or {}
    content = delta.get("content") or ""
    if content:
        return TextDelta(str(content), requested_model=requested_model, actual_model=actual_model)
    finish_reason = choice.get("finish_reason")
    if finish_reason:
        return Completion(
            reason=str(finish_reason),
            requested_model=requested_model,
            actual_model=actual_model,
            usage=payload.get("usage"),
        )
    return None


class OpenRouterLLM:
    name = "openrouter"

    def __init__(
        self,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openrouter/free",
        credential_path: str | None = None,
        num_ctx: int = 8192,
        max_tokens: int = 96,
        temperature: float = 0.2,
        timeout_s: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.requested_model = model
        self.credential_path = credential_path
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.last_actual_model: str | None = None

    def _credential(self):
        return load_openrouter_key(self.credential_path)

    def probe(self) -> dict[str, Any]:
        credential = self._credential()
        if credential.value is None:
            return {
                "credential_present": False,
                "credential_reason": credential.reason,
                "authenticated": False,
                "status_code": None,
            }
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {credential.value}"},
                timeout=20.0,
            )
            return {
                "credential_present": True,
                "credential_reason": credential.reason,
                "authenticated": response.status_code < 400,
                "status_code": response.status_code,
            }
        except httpx.TimeoutException:
            return {"credential_present": True, "credential_reason": credential.reason, "authenticated": False, "error_type": "timeout"}
        except httpx.HTTPError as exc:
            return {"credential_present": True, "credential_reason": credential.reason, "authenticated": False, "error_type": type(exc).__name__}

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        cancel_event: Any = None,
    ) -> Iterable[LLMEvent]:
        credential = self._credential()
        if credential.value is None:
            yield LLMError(f"OpenRouter credential unavailable: {credential.reason}", retryable=False)
            return
        body: dict[str, Any] = {
            "model": self.requested_model,
            "messages": messages,
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {credential.value}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/goura32/local-live-ja",
            "X-Title": "local-live-ja",
        }
        tool_buffers: dict[int, dict[str, Any]] = {}
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=self.timeout_s,
            ) as response:
                if response.status_code >= 400:
                    yield LLMError(
                        f"OpenRouter HTTP {response.status_code}",
                        retryable=response.status_code >= 500 or response.status_code == 429,
                        status_code=response.status_code,
                    )
                    return
                for line in response.iter_lines():
                    if cancel_event is not None and cancel_event.is_set():
                        yield Cancelled()
                        return
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        for event in self._finish_tools(tool_buffers):
                            yield event
                        return
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("model"):
                        self.last_actual_model = payload["model"]
                    choices = payload.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        for item in delta.get("tool_calls") or []:
                            index = int(item.get("index", 0))
                            buffer = tool_buffers.setdefault(index, {"id": None, "name": "", "arguments": ""})
                            buffer["id"] = item.get("id") or buffer["id"]
                            function = item.get("function") or {}
                            buffer["name"] += function.get("name") or ""
                            buffer["arguments"] += function.get("arguments") or ""
                    event = parse_sse_payload(payload, self.requested_model)
                    if event is not None:
                        if hasattr(event, "actual_model"):
                            self.last_actual_model = event.actual_model
                        yield event
                    if choices and choices[0].get("finish_reason") == "tool_calls":
                        for event in self._finish_tools(tool_buffers):
                            yield event
                        tool_buffers.clear()
        except httpx.TimeoutException:
            yield LLMError("OpenRouter request timed out", retryable=True)
        except httpx.HTTPError as exc:
            yield LLMError(f"OpenRouter transport error: {type(exc).__name__}", retryable=True)

    def _finish_tools(self, buffers: dict[int, dict[str, Any]]) -> list[LLMEvent]:
        events: list[LLMEvent] = []
        for index in sorted(buffers):
            item = buffers[index]
            raw = item["arguments"] or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = {"_raw": raw}
            if not isinstance(arguments, dict):
                arguments = {"_raw": str(arguments)}
            events.append(
                ToolCall(
                    call_id=str(item["id"] or f"openrouter-tool-{index}"),
                    name=str(item["name"] or "unknown"),
                    arguments=arguments,
                    requested_model=self.requested_model,
                    actual_model=self.last_actual_model,
                )
            )
        return events
