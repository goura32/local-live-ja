from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sentence_chunker import SentenceChunker
from .telemetry import EventLog
from .echo_rejection import EchoRejectionConfig, classify_echo_candidate
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
    generated_text: str = ""
    spoken_text: str = ""
    audio_paths: list[str] = field(default_factory=list)
    cancelled: bool = False
    error: str | None = None
    retryable: bool = False
    failure_phase: str | None = None
    events: list[PipelineEvent] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    state: str = "IDLE"


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
        echo_rejection_config: EchoRejectionConfig | None = None,
    ) -> None:
        self.llm = llm
        self.tts = tts
        self.playback = playback
        self.artifact_dir = Path(artifact_dir)
        self.chunker_kwargs = {"max_chars": sentence_max_chars, "timeout_s": sentence_timeout_s}
        self.tool_registry = tool_registry
        self.max_tool_rounds = max_tool_rounds
        self._active_cancel: Cancellation | None = None
        self.state = "IDLE"
        self.echo_rejection_config = echo_rejection_config or EchoRejectionConfig()

    def classify_vad_candidate(
        self,
        reference: Any,
        microphone: Any,
        sample_rate: int,
        *,
        playback_active: bool,
    ) -> dict[str, Any]:
        """Apply the post-VAD echo gate without disabling VAD during playback."""
        return classify_echo_candidate(
            reference,
            microphone,
            sample_rate,
            playback_active=playback_active,
            config=self.echo_rejection_config,
        )

    def cancel(self) -> None:
        """Stop playback first, then TTS, then signal the active LLM stream."""
        if self._active_cancel:
            for owner in (self.playback, self.tts):
                cancel = getattr(owner, "cancel", None)
                if callable(cancel):
                    try:
                        cancel()
                    except Exception:
                        # Cancellation is best effort; the turn still observes the event.
                        pass
            self._active_cancel.request()
            llm_cancel = getattr(self.llm, "cancel", None)
            if callable(llm_cancel):
                try:
                    llm_cancel()
                except Exception:
                    pass

    def respond(
        self,
        user_text: str,
        cancel: Cancellation | None = None,
        *,
        history: list[dict[str, Any]] | None = None,
    ) -> PipelineResult:
        """Run one turn, feeding non-tool LLM deltas into TTS incrementally."""
        cancellation = cancel or Cancellation()
        self._active_cancel = cancellation
        self.state = "RUNNING"
        log = EventLog()
        result = PipelineResult()
        messages = [dict(message) for message in history] if history else [
            {"role": "system", "content": "日本語で短く自然に答えてください。音声合成向けに一文を短くします。"},
        ]
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_text:
            messages.append({"role": "user", "content": user_text})
        incremental = self.tool_registry is None
        spoken_chunk_index = 0
        drain_after_playback_cancel = False
        current_phase = "llm"

        def speak_chunk(chunk: str) -> bool:
            nonlocal current_phase, drain_after_playback_cancel, spoken_chunk_index
            if cancellation.event.is_set():
                self._finish_cancel(log, result, cancellation)
                return False
            output_path = self.artifact_dir / f"assistant_{time.monotonic_ns()}_{spoken_chunk_index}.wav"
            spoken_chunk_index += 1
            log.mark("tts_request", text_chars=len(chunk), incremental=incremental)
            current_phase = "tts"
            if bool(getattr(self.tts, "streaming", False)):
                if not hasattr(self.tts, "synthesize_stream") or not hasattr(self.playback, "start"):
                    result.error = "streaming TTS requires a persistent PCM playback backend"
                    result.failure_phase = "tts"
                    return False
                generated = self.tts.synthesize_stream(
                    chunk,
                    output_path=output_path,
                    playback=self.playback,
                    cancel_event=cancellation.event,
                    event_log=log,
                )
                log.mark("tts_end", path=str(output_path), streaming=True)
                if generated.get("cancelled"):
                    self._finish_cancel(log, result, cancellation)
                    return False
                if generated.get("status") != "measured":
                    result.error = "streaming TTS failed"
                    result.retryable = True
                    result.failure_phase = "tts"
                    return False
                path = str(generated.get("path", output_path))
                result.audio_paths.append(path)
                self._commit_spoken(log, result, chunk, path, streaming=True)
                log.mark("playback_end", path=path, streaming=True, cancelled=False)
                return not cancellation.event.is_set()
            generated = self.tts.synthesize(chunk, output_path=output_path, cancel_event=cancellation.event)
            log.mark("tts_end", path=str(output_path))
            if cancellation.event.is_set():
                self._finish_cancel(log, result, cancellation)
                return False
            path = str(generated.get("path", output_path)) if isinstance(generated, dict) else str(output_path)
            log.mark("playback_start", path=path)
            current_phase = "playback"
            playback_result = self.playback.play(path, cancel_event=cancellation.event)
            playback_cancelled = isinstance(playback_result, dict) and bool(playback_result.get("cancelled"))
            log.mark("playback_end", path=path, cancelled=playback_cancelled)
            if playback_cancelled:
                self._finish_cancel(log, result, cancellation)
                return False
            result.audio_paths.append(path)
            self._commit_spoken(log, result, chunk, path)
            if cancellation.event.is_set():
                self._finish_cancel(log, result, cancellation)
                drain_after_playback_cancel = True
                return True
            return True

        try:
            for round_index in range(self.max_tool_rounds + 1):
                current_phase = "llm"
                log.mark("llm_start", round=round_index)
                text_parts: list[str] = []
                tool_calls: list[ToolCall] = []
                completed = False
                chunker = SentenceChunker(**self.chunker_kwargs) if incremental else None
                for event in self.llm.stream(
                    messages,
                    tools=self.tool_registry.definitions() if self.tool_registry else None,
                    cancel_event=cancellation.event,
                ):
                    if isinstance(event, TextDelta):
                        if not text_parts:
                            log.mark("llm_first_token", round=round_index)
                        text_parts.append(event.text)
                        result.generated_text = "".join(text_parts)
                        if chunker is not None:
                            for chunk in chunker.push(event.text):
                                if not speak_chunk(chunk):
                                    result.events = self._events(log)
                                    result.timing = self._timing(log)
                                    return result
                    elif isinstance(event, ToolCall):
                        tool_calls.append(event)
                    elif isinstance(event, Completion):
                        completed = True
                        log.mark("llm_end", reason=event.reason, actual_model=event.actual_model)
                    elif isinstance(event, Cancelled):
                        result.generated_text = "".join(text_parts)
                        self._finish_cancel(log, result, cancellation)
                        return result
                    elif isinstance(event, LLMError):
                        result.generated_text = "".join(text_parts)
                        result.error = "LLM stream error"
                        result.retryable = event.retryable
                        result.failure_phase = "llm"
                        result.state = "ERROR"
                        result.events = self._events(log)
                        return result
                    if cancellation.event.is_set() and not drain_after_playback_cancel:
                        result.generated_text = "".join(text_parts)
                        self._finish_cancel(log, result, cancellation)
                        return result
                    drain_after_playback_cancel = False

                llm_text = "".join(text_parts)
                result.generated_text = llm_text
                if tool_calls and self.tool_registry:
                    assistant_tool_calls = []
                    for call in tool_calls:
                        if getattr(self.llm, "tool_call_format", "openai") == "ollama":
                            assistant_tool_calls.append({"function": {"name": call.name, "arguments": call.arguments}})
                        else:
                            assistant_tool_calls.append(
                                {
                                    "id": call.call_id,
                                    "type": "function",
                                    "function": {"name": call.name, "arguments": call.arguments},
                                }
                            )
                    assistant_content = llm_text if llm_text else ("" if getattr(self.llm, "tool_call_format", "openai") == "ollama" else None)
                    messages.append({"role": "assistant", "content": assistant_content, "tool_calls": assistant_tool_calls})
                    for call in tool_calls:
                        log.mark("tool_call", name=call.name, call_id=call.call_id)
                        content = self.tool_registry.call(call.name, call.arguments)
                        if getattr(self.llm, "tool_call_format", "openai") == "ollama":
                            messages.append({"role": "tool", "content": content})
                        else:
                            messages.append({"role": "tool", "tool_call_id": call.call_id, "name": call.name, "content": content})
                        log.mark("tool_result", name=call.name, call_id=call.call_id)
                    if round_index < self.max_tool_rounds:
                        continue
                if not completed and not llm_text and not tool_calls:
                    result.error = "LLM returned no content"
                    result.retryable = True
                    result.failure_phase = "llm"
                if chunker is not None:
                    for chunk in chunker.flush():
                        if not speak_chunk(chunk):
                            result.events = self._events(log)
                            result.timing = self._timing(log)
                            return result
                break

            if result.error:
                result.state = "ERROR"
                result.events = self._events(log)
                result.timing = self._timing(log)
                return result
            if not incremental:
                chunker = SentenceChunker(**self.chunker_kwargs)
                chunks = chunker.push(result.generated_text)
                chunks.extend(chunker.flush())
                for chunk in chunks:
                    if not speak_chunk(chunk):
                        result.events = self._events(log)
                        result.timing = self._timing(log)
                        return result
            result.events = self._events(log)
            result.timing = self._timing(log)
            result.state = "IDLE"
            return result
        except Exception as exc:  # boundary for a live turn; do not leak credentials
            result.error = f"{type(exc).__name__}"
            result.retryable = bool(getattr(exc, "retryable", True))
            result.failure_phase = current_phase
            result.state = "ERROR"
            result.events = self._events(log)
            result.timing = self._timing(log)
            return result
        finally:
            self.state = result.state
            self._active_cancel = None

    @staticmethod
    def _commit_spoken(log: EventLog, result: PipelineResult, text: str, path: str, *, streaming: bool = False) -> None:
        result.spoken_text += text
        # assistant_text is the history-safe compatibility field. Generated-only
        # text remains available in generated_text and is never committed here.
        result.assistant_text = result.spoken_text
        log.mark("spoken_text_committed", text=text, path=path, streaming=streaming)

    def _finish_cancel(self, log: EventLog, result: PipelineResult, cancellation: Cancellation) -> None:
        if cancellation.requested_ns is None:
            cancellation.requested_ns = time.monotonic_ns()
        log.mark("cancel_requested", requested_ns=cancellation.requested_ns)
        cancellation.completed_ns = time.monotonic_ns()
        log.mark("cancel_completed")
        log.mark("cancelled", reason="cancel_requested")
        result.cancelled = True
        result.state = "IDLE"
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
