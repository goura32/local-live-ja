from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .telemetry import EventLog, ResourceMonitor


@dataclass
class TTSResult:
    text: str
    path: str
    sample_rate: int
    audio_seconds: float | None
    elapsed_seconds: float
    model_load_seconds: float
    inference_elapsed_seconds: float
    first_audio_equivalent_seconds: float
    warm_first_audio_equivalent_seconds: float
    audio_complete_seconds: float
    playback_possible_seconds: float
    timing_ns: dict[str, int]
    rtf: float | None
    device: str
    model: str
    speaker: str
    streaming_supported: bool = False
    gpu_memory_peak_mib: int | None = None
    gpu_memory_delta_peak_mib: int | None = None
    cpu_load_percent: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Qwen3TTSEngine:
    """Official Qwen3-TTS API adapter; sentence-level, not custom streaming."""

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        speaker: str = "Ono_Anna",
        language: str = "Japanese",
        device: str = "auto",
        max_new_tokens: int = 2048,
    ) -> None:
        self.model_name = model
        self.speaker = speaker
        self.language = language
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.resolved_device: str | None = None
        self._model: Any = None

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError("qwen-tts and torch are not installed; run uv sync --extra voice") from exc
        self.resolved_device = self.resolve_device()
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=self.resolved_device,
            dtype=torch.bfloat16 if self.resolved_device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa",
        )
        return self._model

    def unload(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def synthesize(
        self,
        text: str,
        *,
        output_path: str | Path,
        cancel_event: Any = None,
        event_log: EventLog | None = None,
    ) -> dict[str, Any]:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TTS cancelled before request")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if event_log:
            event_log.mark("tts_request", text_chars=len(text), model=self.model_name)
        started = time.monotonic_ns()
        load_started = time.monotonic_ns()
        model_was_loaded = self._model is not None
        with ResourceMonitor() as monitor:
            if event_log:
                event_log.mark("tts_model_load_start", cold=not model_was_loaded)
            model = self.load()
            model_loaded_ns = time.monotonic_ns()
            model_load_seconds = 0.0 if model_was_loaded else (model_loaded_ns - load_started) / 1e9
            if event_log:
                event_log.mark("tts_model_loaded", cold=not model_was_loaded)
            inference_started = time.monotonic_ns()
            if event_log:
                event_log.mark("tts_inference_start")
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language=self.language,
                speaker=self.speaker,
                max_new_tokens=self.max_new_tokens,
            )
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            generation_complete_ns = time.monotonic_ns()
            if event_log:
                event_log.mark("tts_generation_complete")
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("TTS cancelled after generation")
            waveform = np.asarray(wavs[0])
            sf.write(str(path), waveform, int(sample_rate))
            audio_complete_ns = time.monotonic_ns()
            if event_log:
                event_log.mark("tts_audio_complete", path=str(path))
            playback_possible_ns = time.monotonic_ns()
            if event_log:
                event_log.mark("tts_playback_possible", path=str(path))
        ended_ns = time.monotonic_ns()
        elapsed = (ended_ns - started) / 1e9
        inference_elapsed = (generation_complete_ns - inference_started) / 1e9
        first_audio = (generation_complete_ns - started) / 1e9
        warm_first_audio = (generation_complete_ns - inference_started) / 1e9
        audio_complete = (audio_complete_ns - started) / 1e9
        playback_possible = (playback_possible_ns - started) / 1e9
        audio_seconds = len(waveform) / int(sample_rate) if len(waveform) else None
        if event_log:
            event_log.mark("tts_end", path=str(path), sample_rate=int(sample_rate))
        return TTSResult(
            text=text,
            path=str(path),
            sample_rate=int(sample_rate),
            audio_seconds=audio_seconds,
            elapsed_seconds=elapsed,
            model_load_seconds=model_load_seconds,
            inference_elapsed_seconds=inference_elapsed,
            first_audio_equivalent_seconds=first_audio,
            warm_first_audio_equivalent_seconds=warm_first_audio,
            audio_complete_seconds=audio_complete,
            playback_possible_seconds=playback_possible,
            timing_ns={
                "request_start": started,
                "model_load_start": load_started,
                "model_loaded": model_loaded_ns,
                "inference_start": inference_started,
                "generation_complete": generation_complete_ns,
                "audio_complete": audio_complete_ns,
                "playback_possible": playback_possible_ns,
                "tts_end": ended_ns,
            },
            rtf=inference_elapsed / audio_seconds if audio_seconds else None,
            device=self.resolved_device or self.resolve_device(),
            model=self.model_name,
            speaker=self.speaker,
            gpu_memory_peak_mib=monitor.gpu_memory_peak_mib,
            gpu_memory_delta_peak_mib=monitor.gpu_memory_delta_peak_mib,
            cpu_load_percent=monitor.cpu_load_percent,
        ).to_dict()
