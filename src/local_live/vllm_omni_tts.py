from __future__ import annotations

import time
import wave
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf

from .audio_metrics import clipping_ratio, measure_generated_audio_leading_silence, signal_rms
from .telemetry import EventLog

VLLM_OMNI_SAMPLE_RATE = 24000
VLLM_OMNI_SAMPLE_WIDTH = 2


class PCMChunkParser:
    """Keep raw signed-16 PCM samples intact across network chunk boundaries."""

    def __init__(self, *, sample_width: int = VLLM_OMNI_SAMPLE_WIDTH) -> None:
        if sample_width < 1:
            raise ValueError("sample_width must be positive")
        self.sample_width = sample_width
        self._pending = b""

    def feed(self, payload: bytes | bytearray | memoryview) -> list[bytes]:
        data = bytes(payload)
        if not data:
            return []
        chunks: list[bytes] = []
        if self._pending:
            needed = self.sample_width - len(self._pending)
            if len(data) < needed:
                self._pending += data
                return chunks
            chunks.append(self._pending + data[:needed])
            self._pending = b""
            data = data[needed:]
        complete = len(data) - (len(data) % self.sample_width)
        if complete:
            chunks.append(data[:complete])
        self._pending = data[complete:]
        return chunks

    def finish(self) -> None:
        if self._pending:
            raise ValueError("incomplete PCM sample at end of stream")


def decode_pcm16(payload: bytes) -> np.ndarray:
    """Decode little-endian signed 16-bit mono PCM to float32."""
    if len(payload) % VLLM_OMNI_SAMPLE_WIDTH:
        raise ValueError("PCM payload must contain an even number of bytes")
    if not payload:
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


def write_pcm_wav(path: str | Path, payload: bytes, *, sample_rate: int = VLLM_OMNI_SAMPLE_RATE) -> None:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if len(payload) % VLLM_OMNI_SAMPLE_WIDTH:
        raise ValueError("PCM payload must contain an even number of bytes")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(VLLM_OMNI_SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)


def audio_metrics_from_pcm(payload: bytes, *, sample_rate: int = VLLM_OMNI_SAMPLE_RATE) -> dict[str, Any]:
    samples = decode_pcm16(payload)
    analysis = measure_generated_audio_leading_silence(samples, sample_rate) if len(samples) else {
        "detected": False,
        "stable_speech_onset_s": None,
        "leading_silence_duration_s": None,
    }
    return {
        "sample_rate": sample_rate,
        "frames": int(len(samples)),
        "audio_duration_s": len(samples) / sample_rate,
        "rms": signal_rms(samples),
        "peak": float(np.max(np.abs(samples))) if len(samples) else 0.0,
        "clipping_ratio": clipping_ratio(samples),
        "generated_audio_analysis": analysis,
    }


def _delta_s(timing: dict[str, int | None], end: str, start: str) -> float | None:
    end_ns = timing.get(end)
    start_ns = timing.get(start)
    if end_ns is None or start_ns is None:
        return None
    if end_ns < start_ns:
        raise ValueError(f"timing boundaries are not monotonic: {start} > {end}")
    return (end_ns - start_ns) / 1e9


def aggregate_stream_timing(timing: dict[str, int | None], *, audio_duration_s: float | None) -> dict[str, float | None]:
    """Derive stream timings from monotonic nanosecond boundaries."""
    request_start = timing.get("request_start")
    if request_start is not None:
        for name, value in timing.items():
            if name == "record_start":
                continue
            if value is not None and value < request_start:
                raise ValueError(f"timing boundaries are not monotonic: {name} before request_start")
    for end, start in (
        ("first_audio_chunk_received", "request_start"),
        ("first_audio_chunk_queued", "first_audio_chunk_received"),
        ("last_audio_chunk_received", "first_audio_chunk_received"),
        ("playback_completed", "playback_stream_started"),
    ):
        _delta_s(timing, end, start)
    generation_s = _delta_s(timing, "last_audio_chunk_received", "request_start")
    return {
        "request_to_first_audio_chunk_s": _delta_s(timing, "first_audio_chunk_received", "request_start"),
        "first_audio_chunk_receive_to_queue_s": _delta_s(timing, "first_audio_chunk_queued", "first_audio_chunk_received"),
        "request_to_playback_stream_started_s": _delta_s(timing, "playback_stream_started", "request_start"),
        "request_to_physical_audio_s": _delta_s(timing, "physical_audio_detected", "request_start"),
        "first_audio_to_playback_start_s": _delta_s(timing, "playback_stream_started", "first_audio_chunk_received"),
        "playback_stream_to_physical_audio_s": _delta_s(timing, "physical_audio_detected", "playback_stream_started"),
        "total_generation_s": generation_s,
        "playback_duration_s": _delta_s(timing, "playback_completed", "playback_stream_started"),
        "rtf": generation_s / audio_duration_s if generation_s is not None and audio_duration_s else None,
    }


class VLLMOmniTTSEngine:
    """OpenAI-compatible vLLM-Omni Qwen3-TTS client."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8091/v1",
        model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        speaker: str = "Ono_Anna",
        language: str = "Japanese",
        timeout_s: float = 300.0,
        initial_codec_chunk_frames: int | None = None,
        streaming: bool = False,
        client_factory: Callable[..., Any] = httpx.Client,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.speaker = speaker
        self.language = language
        self.timeout_s = timeout_s
        self.initial_codec_chunk_frames = initial_codec_chunk_frames
        self.streaming = streaming
        self.client_factory = client_factory

    def _speech_url(self) -> str:
        return f"{self.base_url}/audio/speech"

    def _voices_url(self) -> str:
        return f"{self.base_url}/audio/voices"

    def request_payload(
        self,
        text: str,
        *,
        stream: bool = False,
        stream_format: str | None = None,
        initial_codec_chunk_frames: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input": text,
            "model": self.model_name,
            "voice": self.speaker,
            "language": self.language,
            "task_type": "CustomVoice",
            "response_format": "pcm" if stream else "wav",
        }
        if stream:
            payload["stream"] = True
            payload["stream_format"] = stream_format or "audio"
        selected_initial = self.initial_codec_chunk_frames if initial_codec_chunk_frames is None else initial_codec_chunk_frames
        if selected_initial is not None:
            if selected_initial < 1:
                raise ValueError("initial_codec_chunk_frames must be positive")
            payload["initial_codec_chunk_frames"] = selected_initial
        return payload

    def health(self) -> dict[str, Any]:
        started = time.monotonic_ns()
        try:
            with self.client_factory(timeout=self.timeout_s) as client:
                response = client.get(self._voices_url())
                response.raise_for_status()
                body = response.json()
            return {
                "status": "ready",
                "http_status": response.status_code,
                "voices": body.get("voices", []) if isinstance(body, dict) else [],
                "elapsed_s": (time.monotonic_ns() - started) / 1e9,
            }
        except Exception as exc:
            return {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_s": (time.monotonic_ns() - started) / 1e9,
            }

    @staticmethod
    def _safe_headers(headers: httpx.Headers) -> dict[str, str]:
        allowed = {
            "content-type",
            "x-vllm-omni-input-tokens",
            "x-vllm-omni-output-tokens",
            "x-vllm-omni-total-tokens",
            "x-vllm-omni-input-text-tokens",
            "x-vllm-omni-input-audio-tokens",
        }
        return {key: value for key, value in headers.items() if key.lower() in allowed}

    def synthesize(
        self,
        text: str,
        *,
        output_path: str | Path,
        cancel_event: Any = None,
        event_log: EventLog | None = None,
        initial_codec_chunk_frames: int | None = None,
    ) -> dict[str, Any]:
        if self.streaming:
            raise RuntimeError("streaming backend requires synthesize_stream with persistent playback")
        started_ns = time.monotonic_ns()
        timing: dict[str, int | None] = {
            "request_start": started_ns,
            "server_request_received": None,
            "server_response_headers": None,
            "first_text_sent": None,
            "first_audio_chunk_received": None,
            "first_audio_chunk_queued": None,
            "playback_stream_started": None,
            "physical_audio_detected": None,
            "last_audio_chunk_received": None,
            "first_actual_speech_pcm": None,
            "playback_completed": None,
        }
        if event_log:
            event_log.mark_at("vllm_request_start", started_ns, mode="non_streaming")
        if cancel_event is not None and cancel_event.is_set():
            return {"status": "cancelled", "cancelled": True, "timing_ns": timing}
        payload = self.request_payload(text, stream=False, initial_codec_chunk_frames=initial_codec_chunk_frames)
        timing["first_text_sent"] = time.monotonic_ns()
        if event_log:
            event_log.mark_at("vllm_first_text_sent", timing["first_text_sent"])
        try:
            with self.client_factory(timeout=self.timeout_s) as client:
                response = client.post(self._speech_url(), json=payload)
                response.raise_for_status()
                response_headers_ns = time.monotonic_ns()
                body = response.content
            if body:
                timing["first_audio_chunk_received"] = response_headers_ns
                timing["last_audio_chunk_received"] = response_headers_ns
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(body)
            samples, sample_rate = sf.read(str(output), always_2d=False)
            metrics = audio_metrics_from_pcm(
                (np.asarray(samples, dtype=np.float32) * 32768.0).astype("<i2").tobytes(),
                sample_rate=int(sample_rate),
            )
            metrics["generated_audio_analysis"] = measure_generated_audio_leading_silence(np.asarray(samples), int(sample_rate))
            onset_s = (metrics.get("generated_audio_analysis") or {}).get("stable_speech_onset_s")
            first_audio_ns = timing.get("first_audio_chunk_received")
            if isinstance(onset_s, (int, float)) and isinstance(first_audio_ns, int):
                timing["first_actual_speech_pcm"] = first_audio_ns + int(float(onset_s) * 1e9)
            timing["playback_completed"] = time.monotonic_ns()
            derived = aggregate_stream_timing(timing, audio_duration_s=metrics["audio_duration_s"])
            result = {
                "status": "measured",
                "cancelled": False,
                "mode": "vllm_omni_non_streaming",
                "text": text,
                "path": str(output),
                "sample_rate": int(sample_rate),
                "audio_duration_s": metrics["audio_duration_s"],
                "generated_audio_analysis": metrics.get("generated_audio_analysis"),
                "metrics": metrics,
                "timing_ns": timing,
                "timing": derived,
                "response_status": response.status_code,
                "response_headers": self._safe_headers(response.headers),
                "request_payload": payload,
            }
            if event_log:
                last_audio_ns = timing["last_audio_chunk_received"]
                if last_audio_ns is not None:
                    event_log.mark_at("vllm_audio_complete", last_audio_ns, mode="non_streaming")
            return result
        except Exception as exc:
            return {
                "status": "error",
                "cancelled": False,
                "mode": "vllm_omni_non_streaming",
                "text": text,
                "timing_ns": timing,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "request_payload": payload,
            }

    def synthesize_stream(
        self,
        text: str,
        *,
        output_path: str | Path,
        playback: Any,
        cancel_event: Any = None,
        event_log: EventLog | None = None,
        initial_codec_chunk_frames: int | None = None,
    ) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        timing: dict[str, int | None] = {
            "request_start": started_ns,
            "server_request_received": None,
            "server_response_headers": None,
            "first_text_sent": None,
            "first_audio_chunk_received": None,
            "first_audio_chunk_queued": None,
            "playback_stream_started": None,
            "physical_audio_detected": None,
            "last_audio_chunk_received": None,
            "first_actual_speech_pcm": None,
            "playback_completed": None,
        }
        payload = self.request_payload(
            text,
            stream=True,
            stream_format="audio",
            initial_codec_chunk_frames=initial_codec_chunk_frames,
        )
        if event_log:
            event_log.mark_at("vllm_request_start", started_ns, mode="streaming")
        if cancel_event is not None and cancel_event.is_set():
            return {"status": "cancelled", "cancelled": True, "timing_ns": timing}
        pcm_parts: list[bytes] = []
        parser = PCMChunkParser()
        playback_started = False
        try:
            timing["first_text_sent"] = time.monotonic_ns()
            if event_log:
                event_log.mark_at("vllm_first_text_sent", timing["first_text_sent"])
            with self.client_factory(timeout=self.timeout_s) as client:
                with client.stream("POST", self._speech_url(), json=payload) as response:
                    response.raise_for_status()
                    timing["server_response_headers"] = time.monotonic_ns()
                    for network_chunk in response.iter_bytes():
                        if cancel_event is not None and cancel_event.is_set():
                            playback.cancel()
                            return {
                                "status": "cancelled",
                                "cancelled": True,
                                "mode": "vllm_omni_streaming",
                                "timing_ns": timing,
                                "request_payload": payload,
                            }
                        if not network_chunk:
                            continue
                        receive_ns = time.monotonic_ns()
                        if timing["first_audio_chunk_received"] is None:
                            timing["first_audio_chunk_received"] = receive_ns
                            if event_log:
                                event_log.mark_at("vllm_first_audio_chunk_received", receive_ns)
                        for pcm_chunk in parser.feed(network_chunk):
                            if not pcm_chunk:
                                continue
                            if not playback_started:
                                playback.start(sample_rate=VLLM_OMNI_SAMPLE_RATE, channels=1)
                                playback_started = True
                                timing["playback_stream_started"] = time.monotonic_ns()
                                if event_log:
                                    event_log.mark_at("vllm_playback_stream_started", timing["playback_stream_started"])
                            playback.queue(pcm_chunk)
                            if timing["first_audio_chunk_queued"] is None:
                                timing["first_audio_chunk_queued"] = getattr(playback, "last_queued_ns", time.monotonic_ns())
                                if event_log:
                                    event_log.mark_at("vllm_first_audio_chunk_queued", timing["first_audio_chunk_queued"])
                            pcm_parts.append(pcm_chunk)
                        timing["last_audio_chunk_received"] = receive_ns
                    parser.finish()
            if playback_started:
                playback_result = playback.finish()
                timing["playback_completed"] = time.monotonic_ns()
            else:
                playback_result = {"cancelled": False, "empty": True}
                timing["playback_completed"] = time.monotonic_ns()
            raw_pcm = b"".join(pcm_parts)
            output = Path(output_path)
            write_pcm_wav(output, raw_pcm)
            metrics = audio_metrics_from_pcm(raw_pcm)
            onset_s = (metrics.get("generated_audio_analysis") or {}).get("stable_speech_onset_s")
            first_audio_ns = timing.get("first_audio_chunk_received")
            if isinstance(onset_s, (int, float)) and isinstance(first_audio_ns, int):
                timing["first_actual_speech_pcm"] = first_audio_ns + int(float(onset_s) * 1e9)
            derived = aggregate_stream_timing(timing, audio_duration_s=metrics["audio_duration_s"])
            result = {
                "status": "measured" if raw_pcm else "error",
                "cancelled": False,
                "mode": "vllm_omni_streaming",
                "text": text,
                "path": str(output),
                "pcm_path": str(output.with_suffix(".pcm")),
                "sample_rate": VLLM_OMNI_SAMPLE_RATE,
                "audio_duration_s": metrics["audio_duration_s"],
                "pcm_bytes": len(raw_pcm),
                "generated_audio_analysis": metrics.get("generated_audio_analysis"),
                "metrics": metrics,
                "timing_ns": timing,
                "timing": derived,
                "network_chunks": len(pcm_parts),
                "playback": playback_result,
                "request_payload": payload,
            }
            output.with_suffix(".pcm").write_bytes(raw_pcm)
            if event_log:
                last_audio_ns = timing["last_audio_chunk_received"]
                if last_audio_ns is not None:
                    event_log.mark_at("vllm_last_audio_chunk_received", last_audio_ns)
                playback_completed_ns = timing["playback_completed"]
                if playback_completed_ns is not None:
                    event_log.mark_at("vllm_playback_completed", playback_completed_ns)
            return result
        except Exception as exc:
            if playback_started:
                playback.cancel()
            return {
                "status": "error",
                "cancelled": False,
                "mode": "vllm_omni_streaming",
                "text": text,
                "timing_ns": timing,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "request_payload": payload,
            }
