from __future__ import annotations

from typing import Any

import numpy as np


def detect_speech_intervals(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: int = 20,
    min_speech_ms: int = 120,
    threshold: float = 0.015,
    hangover_ms: int = 100,
) -> list[tuple[float, float]]:
    """CPU energy VAD used to close an utterance before Whisper.

    It deliberately favors a simple deterministic gate over a full streaming
    decoder. A later production iteration can replace this function with
    WebRTC VAD without changing the pipeline contract.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    frame_length = max(1, int(sample_rate * frame_ms / 1000))
    frame_count = len(samples) // frame_length
    if frame_count == 0:
        return []
    rms = np.sqrt(
        np.mean(
            np.square(samples[: frame_count * frame_length].reshape(frame_count, frame_length)),
            axis=1,
        )
    )
    active = rms >= threshold
    min_frames = max(1, int(np.ceil(min_speech_ms / frame_ms)))
    hangover_frames = max(0, int(np.ceil(hangover_ms / frame_ms)))
    intervals: list[tuple[float, float]] = []
    index = 0
    while index < frame_count:
        if not active[index]:
            index += 1
            continue
        start = index
        last_active = index
        index += 1
        while index < frame_count:
            if active[index]:
                last_active = index
            elif index - last_active > hangover_frames:
                break
            index += 1
        end = min(frame_count, last_active + 1)
        if end - start >= min_frames:
            intervals.append((start * frame_ms / 1000.0, end * frame_ms / 1000.0))
    return intervals


def speech_bounds(intervals: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not intervals:
        return None
    return min(start for start, _ in intervals), max(end for _, end in intervals)


def trim_to_speech(audio: np.ndarray, sample_rate: int, intervals: list[tuple[float, float]]) -> np.ndarray:
    bounds = speech_bounds(intervals)
    if bounds is None:
        return np.asarray(audio)
    start, end = bounds
    return np.asarray(audio)[int(start * sample_rate) : int(end * sample_rate)]


def vad_summary(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    intervals = detect_speech_intervals(audio, sample_rate)
    return {"sample_rate": sample_rate, "intervals": intervals, "speech_seconds": sum(e - s for s, e in intervals)}
