from __future__ import annotations

import gc
import importlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from .telemetry import EventLog, ResourceMonitor


def _cuda_library_paths() -> list[str]:
    paths: list[str] = []
    for module_name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
        try:
            module = importlib.import_module(module_name)
            module_file = getattr(module, "__file__", None)
            if module_file:
                candidate = Path(module_file).parent
            else:
                locations = list(getattr(module, "__path__", []))
                if not locations:
                    continue
                candidate = Path(locations[0])
        except (ImportError, AttributeError, TypeError):
            continue
        if candidate.is_dir() and str(candidate) not in paths:
            paths.append(str(candidate))
    return paths


def _prepare_cuda_libraries() -> list[str]:
    paths = _cuda_library_paths()
    if paths:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        entries = [entry for entry in existing.split(":") if entry]
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths + [entry for entry in entries if entry not in paths])
    return paths


@dataclass
class ASRResult:
    text: str
    model: str
    device: str
    compute_type: str
    language_requested: str
    detected_language: str | None
    language_probability: float | None
    audio_seconds: float | None
    elapsed_seconds: float
    rtf: float | None
    gpu_memory_peak_mib: int | None
    gpu_memory_delta_peak_mib: int | None
    cpu_load_percent: float | None
    segments: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class WhisperASR:
    """Lazy faster-whisper wrapper for utterance-final ASR."""

    def __init__(
        self,
        *,
        model: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "ja",
        beam_size: int = 5,
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self._model: Any = None

    def load(self) -> Any:
        if self._model is None:
            if self.device == "cuda":
                _prepare_cuda_libraries()
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("faster-whisper is not installed; run uv sync --extra voice") from exc
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
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

    def transcribe(self, audio_path: str | Path, *, event_log: EventLog | None = None) -> ASRResult:
        path = Path(audio_path)
        try:
            info = sf.info(str(path))
            audio_seconds = float(info.duration)
        except Exception:
            audio_seconds = None
        if event_log:
            event_log.mark("asr_start", model=self.model_name, device=self.device, compute_type=self.compute_type)
        started = time.monotonic_ns()
        with ResourceMonitor() as monitor:
            model = self.load()
            segments, info = model.transcribe(
                str(path),
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            materialized = list(segments)
        elapsed = (time.monotonic_ns() - started) / 1e9
        text = "".join(str(segment.text) for segment in materialized).strip()
        if event_log:
            event_log.mark("asr_final", text_chars=len(text), detected_language=getattr(info, "language", None))
        rtf = elapsed / audio_seconds if audio_seconds and audio_seconds > 0 else None
        return ASRResult(
            text=text,
            model=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            language_requested=self.language,
            detected_language=getattr(info, "language", None),
            language_probability=_float_or_none(getattr(info, "language_probability", None)),
            audio_seconds=audio_seconds,
            elapsed_seconds=elapsed,
            rtf=rtf,
            gpu_memory_peak_mib=monitor.gpu_memory_peak_mib,
            gpu_memory_delta_peak_mib=monitor.gpu_memory_delta_peak_mib,
            cpu_load_percent=monitor.cpu_load_percent,
            segments=[
                {
                    "start": float(getattr(segment, "start", 0.0)),
                    "end": float(getattr(segment, "end", 0.0)),
                    "text": str(getattr(segment, "text", "")),
                }
                for segment in materialized
            ],
        )


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
