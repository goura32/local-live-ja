from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from .pipeline import Cancellation, LivePipeline, PipelineResult


class SessionState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    FINALIZING = "FINALIZING"
    TRANSCRIBING = "TRANSCRIBING"
    ASSISTANT_THINKING = "ASSISTANT_THINKING"
    ASSISTANT_SPEAKING = "ASSISTANT_SPEAKING"
    INTERRUPTING = "INTERRUPTING"
    ERROR_RECOVERY = "ERROR_RECOVERY"
    STOPPING = "STOPPING"


@dataclass(frozen=True)
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    min_speech_duration_s: float = 0.12
    end_silence_s: float = 0.12
    max_utterance_duration_s: float = 8.0
    threshold: float = 0.015
    noise_multiplier: float = 3.0
    initial_noise_floor: float = 0.003

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.frame_ms <= 0:
            raise ValueError("sample_rate and frame_ms must be positive")
        if self.min_speech_duration_s <= 0 or self.end_silence_s <= 0:
            raise ValueError("speech duration and end silence must be positive")
        if self.max_utterance_duration_s < self.min_speech_duration_s:
            raise ValueError("max utterance duration must cover minimum speech duration")
        if self.threshold < 0 or self.noise_multiplier < 1:
            raise ValueError("invalid VAD threshold or noise multiplier")


@dataclass(frozen=True)
class VADEvent:
    kind: str
    start_s: float
    end_s: float
    duration_s: float
    rms: float
    noise_floor_rms: float
    samples: np.ndarray | None = field(default=None, repr=False, compare=False)
    reason: str | None = None


class StreamingVAD:
    """Bounded CPU energy VAD for a persistent microphone stream."""

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self.frame_length = max(1, round(self.config.sample_rate * self.config.frame_ms / 1000))
        self.end_silence_frames = max(1, int(np.ceil(self.config.end_silence_s * 1000 / self.config.frame_ms)))
        self._pending = np.empty(0, dtype=np.float32)
        self._speech = np.empty(0, dtype=np.float32)
        self._speech_active = False
        self._silence_frames = 0
        self._cursor_samples = 0
        self._speech_start_sample = 0
        self._noise_floor = float(self.config.initial_noise_floor)
        self.noise_floor_history: list[float] = []

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    @property
    def noise_floor_rms(self) -> float:
        return self._noise_floor

    def process(self, samples: Any) -> list[VADEvent]:
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim > 1:
            array = array.mean(axis=1)
        if not len(array):
            return []
        self._pending = np.concatenate((self._pending, array))
        events: list[VADEvent] = []
        started_this_call = False
        ended_this_call = False
        while len(self._pending) >= self.frame_length:
            frame = self._pending[: self.frame_length]
            self._pending = self._pending[self.frame_length :]
            frame_start = self._cursor_samples
            self._cursor_samples += self.frame_length
            frame_rms = float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0
            active = frame_rms >= max(self.config.threshold, self._noise_floor * self.config.noise_multiplier)
            if not self._speech_active and not active:
                self._noise_floor = 0.95 * self._noise_floor + 0.05 * frame_rms
            self.noise_floor_history.append(self._noise_floor)
            if not self._speech_active:
                if not active:
                    continue
                self._speech_active = True
                self._speech_start_sample = frame_start
                self._speech = frame.copy()
                self._silence_frames = 0
                started_this_call = True
                events.append(self._event("speech_start", frame_start, self._cursor_samples, frame_rms, frame.copy()))
                continue

            self._speech = np.concatenate((self._speech, frame))
            if active:
                self._silence_frames = 0
            else:
                self._silence_frames += 1
            elapsed = len(self._speech) / self.config.sample_rate
            if elapsed >= self.config.max_utterance_duration_s:
                events.append(self._finish("speech_end", "max_utterance_duration"))
                ended_this_call = True
            elif self._silence_frames >= self.end_silence_frames:
                if elapsed >= self.config.min_speech_duration_s:
                    events.append(self._finish("speech_end", "end_silence"))
                else:
                    events.append(self._finish("too_short", "minimum_speech_duration"))
                ended_this_call = True
        if self._speech_active and not started_this_call and not ended_this_call:
            events.append(
                self._event(
                    "speech_continuation",
                    self._speech_start_sample,
                    self._cursor_samples,
                    self._rms(self._speech),
                    None,
                )
            )
        return events

    def flush(self) -> list[VADEvent]:
        events = self.process(np.zeros(0, dtype=np.float32))
        if self._speech_active:
            elapsed = len(self._speech) / self.config.sample_rate
            if elapsed >= self.config.min_speech_duration_s:
                events.append(self._finish("speech_end", "stream_end"))
            else:
                events.append(self._finish("too_short", "stream_end"))
        self._pending = np.empty(0, dtype=np.float32)
        return events

    def reset(self) -> None:
        self._pending = np.empty(0, dtype=np.float32)
        self._speech = np.empty(0, dtype=np.float32)
        self._speech_active = False
        self._silence_frames = 0
        self._cursor_samples = 0
        self._speech_start_sample = 0
        self._noise_floor = float(self.config.initial_noise_floor)
        self.noise_floor_history.clear()

    def _update_noise_floor(self, frame_rms: float) -> None:
        if not self._speech_active:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * frame_rms
        self.noise_floor_history.append(self._noise_floor)

    def _finish(self, kind: str, reason: str) -> VADEvent:
        samples = self._speech.copy()
        start = self._speech_start_sample
        end = start + len(samples)
        event = self._event(kind, start, end, self._rms(samples), samples, reason=reason)
        self._speech = np.empty(0, dtype=np.float32)
        self._speech_active = False
        self._silence_frames = 0
        return event

    def _event(
        self,
        kind: str,
        start_sample: int,
        end_sample: int,
        rms: float,
        samples: np.ndarray | None,
        *,
        reason: str | None = None,
    ) -> VADEvent:
        return VADEvent(
            kind=kind,
            start_s=start_sample / self.config.sample_rate,
            end_s=end_sample / self.config.sample_rate,
            duration_s=max(0.0, (end_sample - start_sample) / self.config.sample_rate),
            rms=float(rms),
            noise_floor_rms=float(self._noise_floor),
            samples=samples,
            reason=reason,
        )

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0


class AudioSource(Protocol):
    sample_rate: int
    eof: bool

    def start(self) -> None: ...
    def read(self, timeout_s: float = 0.1) -> np.ndarray | None: ...
    def stop(self) -> None: ...


class FixtureAudioSource:
    """Deterministic source for application-level tests; no audio device required."""

    def __init__(self, chunks: list[np.ndarray], *, sample_rate: int = 16000, delay_s: float = 0.0) -> None:
        self.chunks = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
        self.sample_rate = sample_rate
        self.delay_s = delay_s
        self.index = 0
        self.started = False
        self.stopped = False
        self.eof = False
        self.dropped_frames = 0

    def start(self) -> None:
        self.started = True
        self.stopped = False
        self.eof = False
        self.dropped_frames = 0

    def read(self, timeout_s: float = 0.1) -> np.ndarray | None:
        if not self.started or self.stopped:
            return None
        if self.index >= len(self.chunks):
            self.eof = True
            return None
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk.copy()

    def stop(self) -> None:
        self.stopped = True


class RealMicrophoneSource:
    """Persistent callback capture using the already-supported sounddevice dependency."""

    def __init__(
        self,
        *,
        target: str | None = None,
        sample_rate: int = 16000,
        block_ms: int = 20,
        queue_size: int = 128,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.target = target
        self.sample_rate = sample_rate
        self.block_size = max(1, round(sample_rate * block_ms / 1000))
        self._queue: deque[np.ndarray] = deque(maxlen=queue_size)
        self._stream: Any = None
        self._lock = threading.Lock()
        self._error: Exception | None = None
        self.dropped_frames = 0
        self.eof = False

    @property
    def callback_error(self) -> str | None:
        return type(self._error).__name__ if self._error else None

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is not installed; run uv sync --extra voice") from exc
        self._error = None
        self.dropped_frames = 0
        self.eof = False

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info
            if status:
                self._error = RuntimeError(str(status))
            with self._lock:
                if len(self._queue) == self._queue.maxlen:
                    self._queue.popleft()
                    self.dropped_frames += 1
                    self._error = RuntimeError("microphone queue overflow")
                self._queue.append(np.asarray(indata, dtype=np.float32).reshape(-1).copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="float32",
                device=self.target,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise RuntimeError(f"microphone unavailable: {type(exc).__name__}") from exc

    def read(self, timeout_s: float = 0.1) -> np.ndarray | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._queue:
                    return self._queue.popleft()
            time.sleep(0.005)
        return None

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        with self._lock:
            self._queue.clear()
        self.eof = True


class ConversationHistory:
    """Bounded system/user/assistant history; assistant text is spoken-only."""

    def __init__(self, system_prompt: str, *, max_turns: int = 12, max_chars: int = 8000) -> None:
        if max_turns < 1 or max_chars < 1:
            raise ValueError("history bounds must be positive")
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._messages: list[dict[str, str]] = []
        self.truncated_count = 0

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}] + [dict(item) for item in self._messages]

    def append_user(self, text: str) -> None:
        content = str(text).strip()
        if not content:
            raise ValueError("user history text must not be empty")
        if len(content) > self.max_chars:
            content = content[: self.max_chars]
            self.truncated_count += 1
        self._messages.append({"role": "user", "content": content})
        self._trim()

    def commit_assistant_spoken(self, spoken_text: str) -> None:
        content = str(spoken_text).strip()
        if not content:
            return
        if len(content) > self.max_chars:
            content = content[: self.max_chars]
            self.truncated_count += 1
        self._messages.append({"role": "assistant", "content": content})
        self._trim()

    def spoken_assistant_texts(self) -> list[str]:
        return [item["content"] for item in self._messages if item["role"] == "assistant"]

    def reset(self) -> None:
        self._messages.clear()
        self.truncated_count = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages(),
            "max_turns": self.max_turns,
            "max_chars": self.max_chars,
            "truncated_count": self.truncated_count,
        }

    def _trim(self) -> None:
        while self._turn_count() > self.max_turns or self._char_count() > self.max_chars:
            if not self._messages:
                return
            del self._messages[0]
            if self._messages and self._messages[0]["role"] == "assistant":
                del self._messages[0]

    def _turn_count(self) -> int:
        return sum(item["role"] == "user" for item in self._messages)

    def _char_count(self) -> int:
        return sum(len(item["content"]) for item in self._messages)


class SessionController:
    """Continuous capture/session state machine shared by chat and app tests."""

    def __init__(
        self,
        *,
        source: AudioSource,
        asr: Any,
        pipeline: LivePipeline,
        config: VADConfig | None = None,
        artifact_dir: str | Path = "results/artifacts",
        history: ConversationHistory | None = None,
        max_retries: int = 1,
        read_timeout_s: float = 0.1,
        max_pending_utterances: int = 2,
    ) -> None:
        self.source = source
        self.asr = asr
        self.pipeline = pipeline
        self.vad = StreamingVAD(config or VADConfig(sample_rate=source.sample_rate))
        self.artifact_dir = Path(artifact_dir)
        self.history = history or ConversationHistory(
            "日本語で短く自然に答えてください。音声合成向けに一文を短くします。"
        )
        self.max_retries = max(0, max_retries)
        self.read_timeout_s = read_timeout_s
        if max_pending_utterances < 1:
            raise ValueError("max_pending_utterances must be positive")
        self.max_pending_utterances = max_pending_utterances
        self.state = SessionState.IDLE
        self.events: list[dict[str, Any]] = []
        self.transitions: list[dict[str, Any]] = []
        self.application_success_count = 0
        self.application_failure_count = 0
        self.recovery_count = 0
        self.barge_in_count = 0
        self.cancel_count = 0
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._turn_thread: threading.Thread | None = None
        self._active_cancel: Cancellation | None = None
        self._pending_utterances: deque[np.ndarray] = deque()
        self.queue_drop_count = 0
        self._source_restart_attempted = False
        self._barge_in_pending = False
        self._turn_counter = 0

    @property
    def turn_active(self) -> bool:
        return self._turn_thread is not None and self._turn_thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.state not in {SessionState.IDLE, SessionState.ERROR_RECOVERY}:
                return
            self._stop_event.clear()
            self._source_restart_attempted = False
            self.source.start()
            self._transition(SessionState.LISTENING, "capture_started")

    def stop(self) -> None:
        with self._lock:
            if self.state == SessionState.STOPPING:
                return
            self._transition(SessionState.STOPPING, "shutdown_requested")
            self._stop_event.set()
            if self._active_cancel is not None:
                self._cancel_active_turn("shutdown")
            discarded = len(self._pending_utterances)
            self._pending_utterances.clear()
            if discarded:
                self._record("utterance_queue_discarded", count=discarded, reason="shutdown")
            try:
                self.source.stop()
            finally:
                self._transition(SessionState.IDLE, "shutdown_complete")
        self.wait_for_idle(timeout=5.0)

    def reset(self) -> None:
        self.stop()
        with self._lock:
            self.history.reset()
            self.vad.reset()
            self.events.clear()
            self.transitions.clear()
            self._pending_utterances.clear()
            self._barge_in_pending = False
            self.state = SessionState.IDLE

    def process_audio_chunk(self, samples: Any) -> list[VADEvent]:
        events = self.vad.process(samples)
        for event in events:
            self._handle_vad_event(event)
        return events

    def submit_utterance(self, samples: Any) -> bool:
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        minimum = self.vad.config.min_speech_duration_s * self.vad.config.sample_rate
        if len(array) < minimum:
            self._record("too_short_rejected", samples=len(array))
            return False
        with self._lock:
            if self.turn_active:
                return self._enqueue_pending(array, "submit_while_turn_active")
            self._launch_turn(array)
            return True

    def interrupt(self, reason: str = "possible_user_speech") -> None:
        with self._lock:
            if not self.turn_active or self._barge_in_pending:
                return
            self._barge_in_pending = True
            self.barge_in_count += 1
            self._transition(SessionState.INTERRUPTING, reason)
            self._cancel_active_turn(reason)

    def wait_for_idle(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                thread = self._turn_thread
                pending = bool(self._pending_utterances)
                active = thread is not None and thread.is_alive()
            if not active and not pending:
                return True
            if thread is not None:
                thread.join(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        return not self.turn_active and not self._pending_utterances

    def run(self, *, max_turns: int | None = None) -> dict[str, Any]:
        try:
            self.start()
            while not self._stop_event.is_set():
                if max_turns is not None and self.application_success_count >= max_turns:
                    break
                callback_error = getattr(self.source, "callback_error", None)
                if callback_error:
                    self._record("capture_error", error_type=callback_error)
                    if self._source_restart_attempted:
                        self._transition(SessionState.ERROR_RECOVERY, "capture_error_repeated")
                        break
                    self._source_restart_attempted = True
                    self.recovery_count += 1
                    self.source.stop()
                    self.source.start()
                    self._transition(SessionState.LISTENING, "capture_restarted")
                    continue
                chunk = self.source.read(self.read_timeout_s)
                if chunk is None:
                    if self.source.eof:
                        for event in self.vad.flush():
                            self._handle_vad_event(event)
                        self.wait_for_idle(timeout=10.0)
                        break
                    continue
                self.process_audio_chunk(chunk)
            self.wait_for_idle(timeout=10.0)
        finally:
            self.stop()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "application_success_count": self.application_success_count,
            "application_failure_count": self.application_failure_count,
            "recovery_count": self.recovery_count,
            "barge_in_count": self.barge_in_count,
            "cancel_count": self.cancel_count,
            "pending_utterance_count": len(self._pending_utterances),
            "queue_drop_count": self.queue_drop_count,
            "source_dropped_frames": int(getattr(self.source, "dropped_frames", 0)),
            "history": self.history.to_dict(),
            "transitions": list(self.transitions),
            "events": list(self.events),
        }

    def _handle_vad_event(self, event: VADEvent) -> None:
        self._record(
            event.kind,
            start_s=event.start_s,
            end_s=event.end_s,
            duration_s=event.duration_s,
            rms=event.rms,
            noise_floor_rms=event.noise_floor_rms,
            reason=event.reason,
        )
        if event.kind == "speech_start":
            if self.turn_active:
                if not self._playback_active():
                    self._record("speech_queued_while_thinking")
                    return
                decision = self._classify_candidate(event.samples)
                if decision.get("decision") == "possible_user_speech":
                    self.interrupt("possible_user_speech")
                return
            self._transition(SessionState.USER_SPEAKING, "speech_start")
        elif event.kind == "speech_continuation":
            if not self.turn_active and self.state == SessionState.LISTENING:
                self._transition(SessionState.USER_SPEAKING, "speech_continuation")
        elif event.kind == "speech_end":
            if self._barge_in_pending:
                if event.samples is not None:
                    self._enqueue_pending(event.samples, "barge_in")
                return
            if self.turn_active:
                if event.samples is not None:
                    self._enqueue_pending(event.samples, "assistant_busy")
                return
            self._transition(SessionState.FINALIZING, event.reason or "speech_end")
            if event.samples is not None:
                self.submit_utterance(event.samples)
        elif event.kind == "too_short":
            self._transition(SessionState.LISTENING, "too_short_rejected")

    def _enqueue_pending(self, samples: np.ndarray, reason: str) -> bool:
        with self._lock:
            if len(self._pending_utterances) >= self.max_pending_utterances:
                self.queue_drop_count += 1
                self._record("utterance_queue_full", reason=reason, capacity=self.max_pending_utterances)
                return False
            self._pending_utterances.append(samples.copy())
            self._record("utterance_queued", reason=reason, depth=len(self._pending_utterances))
            return True

    def _classify_candidate(self, samples: np.ndarray | None) -> dict[str, Any]:
        reference = getattr(self.pipeline.playback, "reference_samples", None)
        if samples is None:
            return {"decision": "possible_user_speech", "reason": "missing_candidate"}
        try:
            return self.pipeline.classify_vad_candidate(
                reference,
                samples,
                self.vad.config.sample_rate,
                playback_active=self.turn_active,
            )
        except Exception as exc:
            self._record("echo_rejection_error", error_type=type(exc).__name__)
            return {"decision": "possible_user_speech", "reason": "echo_gate_error"}

    def _playback_active(self) -> bool:
        active = getattr(self.pipeline.playback, "active", None)
        if active is None:
            return self.state == SessionState.ASSISTANT_SPEAKING
        return bool(active)

    def _launch_turn(self, samples: np.ndarray) -> None:
        self._turn_counter += 1
        turn_id = self._turn_counter
        self._active_cancel = Cancellation()
        self._transition(SessionState.TRANSCRIBING, f"turn_{turn_id}")
        self._turn_thread = threading.Thread(
            target=self._turn_worker,
            args=(turn_id, samples.copy(), self._active_cancel),
            name=f"local-live-turn-{turn_id}",
            daemon=True,
        )
        self._turn_thread.start()

    def _turn_worker(self, turn_id: int, samples: np.ndarray, cancellation: Cancellation) -> None:
        result: PipelineResult | None = None
        try:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            input_path = self.artifact_dir / f"chat_turn_{turn_id:04d}_input.wav"
            sf.write(str(input_path), samples, self.vad.config.sample_rate)
            asr_result = self._transcribe(samples, input_path)
            user_text = str(getattr(asr_result, "text", "") or (asr_result.get("text", "") if isinstance(asr_result, dict) else "")).strip()
            self._record("asr_result", turn_id=turn_id, text_chars=len(user_text))
            if not user_text:
                self.recovery_count += 1
                self._transition(SessionState.ERROR_RECOVERY, "empty_asr_result")
                return
            self.history.append_user(user_text)
            self._transition(SessionState.ASSISTANT_THINKING, "asr_complete")
            self._transition(SessionState.ASSISTANT_SPEAKING, "incremental_response_started")
            for attempt in range(self.max_retries + 1):
                result = self.pipeline.respond(user_text, cancellation, history=self.history.messages())
                if not result.error or result.cancelled:
                    break
                if getattr(result, "retryable", False) and attempt < self.max_retries:
                    self.recovery_count += 1
                    self._record(
                        "turn_retry",
                        turn_id=turn_id,
                        attempt=attempt + 1,
                        error_type=result.error,
                        failure_phase=getattr(result, "failure_phase", None),
                    )
                    continue
            if result is None:
                self.application_failure_count += 1
                return
            if result.spoken_text:
                self.history.commit_assistant_spoken(result.spoken_text)
            if result.cancelled:
                self._record("turn_cancelled", turn_id=turn_id, spoken_chars=len(result.spoken_text))
            elif result.error:
                self.application_failure_count += 1
                self._transition(SessionState.ERROR_RECOVERY, "turn_error")
                self._record(
                    "turn_error",
                    turn_id=turn_id,
                    error_type=result.error,
                    failure_phase=getattr(result, "failure_phase", None),
                    retryable=getattr(result, "retryable", False),
                )
                if not getattr(result, "retryable", False):
                    self._stop_event.set()
            else:
                self.application_success_count += 1
                self._record(
                    "turn_completed",
                    turn_id=turn_id,
                    generated_chars=len(result.generated_text),
                    spoken_chars=len(result.spoken_text),
                )
        except Exception as exc:
            self.application_failure_count += 1
            self.recovery_count += 1
            self._transition(SessionState.ERROR_RECOVERY, "turn_exception")
            self._record("turn_exception", turn_id=turn_id, error_type=type(exc).__name__)
        finally:
            with self._lock:
                self._active_cancel = None
                self._turn_thread = None
                self._barge_in_pending = False
                pending = self._pending_utterances.popleft() if self._pending_utterances else None
                if self._stop_event.is_set():
                    self._transition(SessionState.IDLE, "stopped")
                elif pending is not None and len(pending) >= self.vad.config.min_speech_duration_s * self.vad.config.sample_rate:
                    self._launch_turn(pending)
                else:
                    self._transition(SessionState.LISTENING, "turn_finished")

    def _transcribe(self, samples: np.ndarray, input_path: Path) -> Any:
        transcribe_samples = getattr(self.asr, "transcribe_samples", None)
        if callable(transcribe_samples):
            return transcribe_samples(samples, sample_rate=self.vad.config.sample_rate)
        return self.asr.transcribe(input_path)

    def _cancel_active_turn(self, reason: str) -> None:
        self.cancel_count += 1
        cancellation = self._active_cancel
        try:
            self.pipeline.cancel()
        finally:
            if cancellation is not None:
                cancellation.request()
            self._record("barge_in_cancel_requested", reason=reason)

    def _transition(self, state: SessionState, reason: str) -> None:
        with self._lock:
            previous = self.state
            self.state = state
            self.transitions.append(
                {
                    "from": previous.value,
                    "to": state.value,
                    "reason": reason,
                    "monotonic_ns": time.monotonic_ns(),
                }
            )

    def _record(self, kind: str, **payload: Any) -> None:
        self.events.append({"event": kind, "monotonic_ns": time.monotonic_ns(), **payload})


__all__ = [
    "AudioSource",
    "ConversationHistory",
    "FixtureAudioSource",
    "RealMicrophoneSource",
    "SessionController",
    "SessionState",
    "StreamingVAD",
    "VADEvent",
    "VADConfig",
]
