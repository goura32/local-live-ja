from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

import httpx

from .events import Completion, LLMError, TextDelta, ToolCall, LLMEvent, Cancelled


def choose_qwen35_9b_model(tags: list[Mapping[str, Any]]) -> str | None:
    """Select an observed local Qwen3.5 9B tag; never synthesize a tag name."""
    candidates: list[tuple[int, str]] = []
    for tag in tags:
        name = str(tag.get("name", ""))
        lowered = name.casefold()
        details = tag.get("details") or {}
        parameter_size = str(details.get("parameter_size", "")).casefold()
        if "qwen3.5" not in lowered:
            continue
        if "9b" not in lowered and not parameter_size.startswith("9"):
            continue
        # Prefer the exact observed qwen3.5 family over tags with a suffix.
        priority = 0 if lowered.startswith("qwen3.5:") else 1
        candidates.append((priority, name))
    return sorted(candidates)[0][1] if candidates else None


def parse_ollama_events(line: str, requested_model: str) -> list[LLMEvent]:
    payload = json.loads(line)
    actual_model = payload.get("model") or requested_model
    message = payload.get("message") or {}
    events: list[LLMEvent] = []
    content = message.get("content") or payload.get("response") or ""
    if content:
        events.append(TextDelta(str(content), requested_model=requested_model, actual_model=actual_model))
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"_raw": str(arguments)}
        call_id = str(call.get("id") or f"ollama-tool-{index}")
        events.append(
            ToolCall(
                call_id=call_id,
                name=str(function.get("name") or "unknown"),
                arguments=arguments,
                requested_model=requested_model,
                actual_model=actual_model,
            )
        )
    if payload.get("done"):
        usage = {
            key: payload[key]
            for key in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")
            if key in payload
        }
        events.append(
            Completion(
                reason=str(payload.get("done_reason") or "stop"),
                requested_model=requested_model,
                actual_model=actual_model,
                usage=usage or None,
            )
        )
    return events


def parse_ollama_line(line: str, requested_model: str) -> LLMEvent | None:
    events = parse_ollama_events(line, requested_model)
    return events[0] if events else None


class OllamaLLM:
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "auto",
        num_ctx: int = 8192,
        max_tokens: int = 96,
        temperature: float = 0.2,
        timeout_s: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.requested_model = model if model != "auto" else ""
        self._model_selector = model
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.last_actual_model: str | None = None

    def resolve_model(self) -> str:
        if self._model_selector != "auto":
            self.requested_model = self._model_selector
            return self.requested_model
        response = httpx.get(f"{self.base_url}/api/tags", timeout=15.0)
        response.raise_for_status()
        payload = response.json()
        model = choose_qwen35_9b_model(payload.get("models", []))
        if not model:
            raise RuntimeError("ollama API has no observed Qwen3.5 9B model")
        self.requested_model = model
        return model

    def list_models(self) -> list[dict[str, Any]]:
        response = httpx.get(f"{self.base_url}/api/tags", timeout=15.0)
        response.raise_for_status()
        return list(response.json().get("models", []))

    def probe(self) -> dict[str, Any]:
        try:
            models = self.list_models()
            selected = choose_qwen35_9b_model(models)
            return {"reachable": True, "model_count": len(models), "selected_qwen35_9b": selected}
        except httpx.HTTPStatusError as exc:
            return {"reachable": False, "status_code": exc.response.status_code, "error_type": "http"}
        except Exception as exc:
            return {"reachable": False, "error_type": type(exc).__name__}

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        cancel_event: Any = None,
    ) -> Iterable[LLMEvent]:
        try:
            model = self.resolve_model()
        except Exception as exc:
            yield LLMError(str(exc), retryable=False)
            return
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        if tools:
            body["tools"] = tools
        try:
            with httpx.stream("POST", f"{self.base_url}/api/chat", json=body, timeout=self.timeout_s) as response:
                if response.status_code >= 400:
                    yield LLMError(
                        f"Ollama HTTP {response.status_code}",
                        retryable=response.status_code >= 500 or response.status_code == 429,
                        status_code=response.status_code,
                    )
                    return
                for line in response.iter_lines():
                    if cancel_event is not None and cancel_event.is_set():
                        yield Cancelled()
                        return
                    if not line:
                        continue
                    try:
                        events = parse_ollama_events(line, model)
                    except (ValueError, TypeError, json.JSONDecodeError) as exc:
                        yield LLMError(f"invalid Ollama stream event: {type(exc).__name__}")
                        return
                    for event in events:
                        if hasattr(event, "actual_model"):
                            self.last_actual_model = event.actual_model
                        yield event
        except httpx.TimeoutException:
            yield LLMError("Ollama request timed out", retryable=True)
        except httpx.HTTPError as exc:
            yield LLMError(f"Ollama transport error: {type(exc).__name__}", retryable=True)
