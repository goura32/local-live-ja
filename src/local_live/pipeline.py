from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sentence_chunker import SentenceChunker
from .telemetry import EventLog
from .tools import MockToolRegistry
from .llm.events import Cancelled, Completion, LLMError, TextDelta, ToolCall, ToolResult


@dataclass(frozen=True)
class PipelineEvent:
    kind: str
    monotonic_ns: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    assistant_text: str = ""
    audio_paths: list[str] = field(default_factory=list)
    cancelled: bool = False
    error: str | None = None
    events: list[PipelineEvent] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)


class Cancellation:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.requested_ns: int | None = None
        self.completed_ns: int | None = None

    def request(self) -> None:
        if not self.event.is_set():
            self.requested_ns = time.monotonic_ns()
            self.event.set()


class LivePipeline:
    """VAD/ASR is supplied by the caller; this class owns LLM->TTS->playback."""

    def __init__(
        self,
        *,
        llm: Any,
        tts: Any,
        playback: Any,
        artifact_dir: str | Path = "results/artifacts",
        sentence_max_chars: int = 48,
        sentence_timeout_s: float = 0.8,
        tool_registry: MockToolRegistry | None = None,
        max_tool_rounds: int = 3,
    ) -> None:
        self.llm = llm
        self.tts = tts
        self.playback = playback
        self.artifact_dir = Path(artifact_dir)
        self.chunker_kwargs = {"max_chars": sentence_max_chars, "timeout_s": sentence_timeout_s}
        self.tool_registry = tool_registry
        self.max_tool_rounds = max_tool_rounds
        self._active_cancel: Cancellation | None = None

    def cancel(self) -> None:
        if self._active_cancel:
            self._active_cancel.request()

    def respond(self, user_text: str, cancel: Cancellation | None = None) -> PipelineResult:
        cancellation = cancel or Cancellation()
        self._active_cancel = cancellation
        log = EventLog()
        result = PipelineResult()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "日本語で短く自然に答えてください。音声合成向けに一文を短くします。"},
            {"role": "user", "content": user_text},
        ]
        try:
            for round_index in range(self.max_tool_rounds + 1):
                log.mark("llm_start", round=round_index)
                text_parts: list[str] = []
                tool_calls: list[ToolCall] = []
                completed = False
                for event in self.llm.stream(
                    messages,
                    tools=self.tool_registry.definitions() if self.tool_registry else None,
                    cancel_event=cancellation.event,
                ):
                    if isinstance(event, TextDelta):
                        if text_parts == []:
                            log.mark("llm_first_token", round=round_index)
                        text_parts.append(event.text)
                    elif isinstance(event, ToolCall):
                        tool_calls.append(event)
                    elif isinstance(event, Completion):
                        completed = True
                        log.mark("llm_end", reason=event.reason, actual_model=event.actual_model)
                    elif isinstance(event, Cancelled):
                        self._finish_cancel(log, result, cancellation)
                        return result
                    elif isinstance(event, LLMError):
                        result.error = event.message
                        result.events = self._events(log)
                        return result
                    if cancellation.event.is_set():
                        self._finish_cancel(log, result, cancellation)
                        return result

                llm_text = "".join(text_parts)
                if tool_calls and self.tool_registry:
                    assistant_tool_calls = []
                    for call in tool_calls:
                        assistant_tool_calls.append(
                            {
                                "id": call.call_id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": call.arguments},
                            }
                        )
                    messages.append({"role": "assistant", "content": llm_text or None, "tool_calls": assistant_tool_calls})
                    for call in tool_calls:
                        log.mark("tool_call", name=call.name, call_id=call.call_id)
                        content = self.tool_registry.call(call.name, call.arguments)
                        messages.append(
                            {"role": "tool", "tool_call_id": call.call_id, "name": call.name, "content": content}
                        )
                        log.mark("tool_result", name=call.name, call_id=call.call_id)
                    if round_index < self.max_tool_rounds:
                        continue
                result.assistant_text += llm_text
                if not completed and not llm_text and not tool_calls:
                    result.error = "LLM returned no content"
                break

            chunker = SentenceChunker(**self.chunker_kwargs)
            chunks = chunker.push(result.assistant_text)
            chunks.extend(chunker.flush())
            for index, chunk in enumerate(chunks):
                if cancellation.event.is_set():
                    self._finish_cancel(log, result, cancellation)
                    return result
                output_path = self.artifact_dir / f"assistant_{time.monotonic_ns()}_{index}.wav"
                log.mark("tts_request", text_chars=len(chunk))
                generated = self.tts.synthesize(chunk, output_path=output_path, cancel_event=cancellation.event)
                log.mark("tts_end", path=str(output_path))
                if cancellation.event.is_set():
                    self._finish_cancel(log, result, cancellation)
                    return result
                path = str(generated.get("path", output_path)) if isinstance(generated, dict) else str(output_path)
                result.audio_paths.append(path)
                log.mark("playback_start", path=path)
                playback_result = self.playback.play(path, cancel_event=cancellation.event)
                playback_cancelled = isinstance(playback_result, dict) and bool(playback_result.get("cancelled"))
                log.mark("playback_end", path=path, cancelled=playback_cancelled)
                if cancellation.event.is_set() or playback_cancelled:
                    self._finish_cancel(log, result, cancellation)
                    return result

            result.events = self._events(log)
            result.timing = self._timing(log)
            return result
        except Exception as exc:  # boundary for a live turn; do not leak credentials
            result.error = f"{type(exc).__name__}: {exc}"
            result.events = self._events(log)
            return result
        finally:
            self._active_cancel = None

    def _finish_cancel(self, log: EventLog, result: PipelineResult, cancellation: Cancellation) -> None:
        if cancellation.requested_ns is None:
            cancellation.requested_ns = time.monotonic_ns()
        log.mark("cancel_requested", requested_ns=cancellation.requested_ns)
        cancellation.completed_ns = time.monotonic_ns()
        log.mark("cancel_completed")
        result.cancelled = True
        result.events = self._events(log) + [
            PipelineEvent("cancelled", cancellation.completed_ns, {"reason": "cancel_requested"})
        ]
        result.timing = self._timing(log)

    @staticmethod
    def _events(log: EventLog) -> list[PipelineEvent]:
        return [PipelineEvent(item["event"], item["monotonic_ns"], {k: v for k, v in item.items() if k not in {"event", "monotonic_ns"}}) for item in log.events]

    @staticmethod
    def _timing(log: EventLog) -> dict[str, Any]:
        return {item["event"]: item["monotonic_ns"] for item in log.events}
