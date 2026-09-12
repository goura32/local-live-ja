from __future__ import annotations

from typing import Any

import numpy as np


def _mono(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array
    return array.mean(axis=1)


def _resample(value: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return _mono(value)
    source = _mono(value)
    if len(source) < 2:
        return source
    target_length = max(1, round(len(source) * target_rate / source_rate))
    source_x = np.linspace(0.0, 1.0, len(source), endpoint=False)
    target_x = np.linspace(0.0, 1.0, target_length, endpoint=False)
    return np.interp(target_x, source_x, source).astype(np.float32)


def resample_mono(value: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Convert audio to mono at a requested rate for benchmark isolation."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("source_rate and target_rate must be positive")
    return _resample(value, source_rate, target_rate)


def _normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def signal_rms(value: np.ndarray) -> float:
    array = _mono(value)
    return float(np.sqrt(np.mean(np.square(array)))) if len(array) else 0.0


def clipping_ratio(value: np.ndarray, *, threshold: float = 0.98) -> float:
    """Return the fraction of samples at or beyond the inclusive clip gate."""
    if threshold <= 0 or threshold > 1:
        raise ValueError("clipping threshold must be in the interval (0, 1]")
    array = _mono(value)
    if len(array) == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(array) >= threshold) / len(array))


def noise_floor_rms(
    value: np.ndarray,
    sample_rate: int,
    *,
    window_s: float = 0.25,
) -> float:
    """Estimate RMS from the leading pre-playback window."""
    if sample_rate <= 0 or window_s <= 0:
        raise ValueError("sample_rate and window_s must be positive")
    array = _mono(value)
    count = min(len(array), max(1, int(round(sample_rate * window_s))))
    return signal_rms(array[:count]) if count else 0.0


def detect_acoustic_onset(
    recording: np.ndarray,
    sample_rate: int,
    *,
    search_start_s: float = 0.25,
    noise_window_s: float = 0.25,
    frame_ms: int = 10,
    min_consecutive_frames: int = 2,
    threshold_multiplier: float = 4.0,
    reference: np.ndarray | None = None,
    reference_rate: int | None = None,
) -> dict[str, Any]:
    """Detect the first sustained above-floor acoustic frame.

    The onset is derived from the raw microphone capture rather than a sleep
    estimate. Optional reference alignment is returned as a diagnostic proxy;
    it does not replace the noise-floor gate.
    """
    if sample_rate <= 0 or frame_ms <= 0 or min_consecutive_frames < 1:
        raise ValueError("invalid acoustic onset parameters")
    array = _mono(recording)
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000)))
    frame_count = len(array) // frame_length
    frames = array[: frame_count * frame_length].reshape(frame_count, frame_length) if frame_count else np.empty((0, frame_length))
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1)) if frame_count else np.empty(0, dtype=np.float32)
    noise_frames = min(frame_count, max(1, int(round(noise_window_s * 1000 / frame_ms))))
    noise_values = frame_rms[:noise_frames]
    noise = float(np.median(noise_values)) if len(noise_values) else 0.0
    mad = float(np.median(np.abs(noise_values - noise))) if len(noise_values) else 0.0
    threshold = max(noise * threshold_multiplier, noise + 6.0 * mad, 0.004)
    first_frame = max(0, int(np.floor(search_start_s * sample_rate / frame_length)))
    onset_frame: int | None = None
    for index in range(first_frame, max(first_frame, frame_count - min_consecutive_frames + 1)):
        window = frame_rms[index : index + min_consecutive_frames]
        if len(window) == min_consecutive_frames and bool(np.all(window >= threshold)):
            onset_frame = index
            break
    onset_s = onset_frame * frame_length / sample_rate if onset_frame is not None else None
    result: dict[str, Any] = {
        "detected": onset_frame is not None,
        "onset_s": onset_s,
        "onset_frame": onset_frame,
        "noise_floor_rms": noise,
        "threshold_rms": threshold,
        "sample_rate": sample_rate,
        "frame_ms": frame_ms,
        "search_start_s": search_start_s,
        "noise_window_s": noise_window_s,
        "min_consecutive_frames": min_consecutive_frames,
    }
    if reference is not None:
        result["reference_alignment"] = aligned_correlation(
            reference,
            array,
            sample_rate=reference_rate or sample_rate,
            recording_rate=sample_rate,
        )
    else:
        result["reference_alignment"] = None
    return result


def aligned_correlation(
    reference: np.ndarray,
    recording: np.ndarray,
    *,
    sample_rate: int,
    recording_rate: int | None = None,
    max_lag_s: float = 2.0,
) -> dict[str, Any]:
    """Align two signals and return correlation plus lag.

    This is a diagnostic proxy for room echo, not a calibrated acoustic
    measurement. Signals are downsampled for bounded CPU cost during search.
    """
    rec_rate = recording_rate or sample_rate
    ref = _resample(reference, sample_rate, 4000)
    rec = _resample(recording, rec_rate, 4000)
    if len(ref) == 0 or len(rec) == 0 or signal_rms(ref) <= 1e-9 or signal_rms(rec) <= 1e-9:
        return {"correlation": None, "lag_samples": None, "lag_seconds": None}
    step_rate = 4000
    max_lag = min(int(max_lag_s * step_rate), max(len(ref), len(rec)) - 1)
    # A bounded grid avoids an O(N^2) search on long recordings.
    best = (-1.0, 0)
    for lag in range(-max_lag, max_lag + 1, max(1, step_rate // 2000)):
        if lag >= 0:
            left = ref[lag:]
            right = rec[: len(left)]
        else:
            right = rec[-lag:]
            left = ref[: len(right)]
        length = min(len(left), len(right))
        if length < 32:
            continue
        score = abs(_normalized_correlation(left[:length], right[:length]))
        if score > best[0]:
            best = (score, lag)
    return {
        "correlation": best[0] if best[0] >= 0 else None,
        "lag_samples": int(round(best[1] * sample_rate / step_rate)) if best[0] >= 0 else None,
        "lag_seconds": (best[1] / step_rate) if best[0] >= 0 else None,
    }


def residual_echo_db(off: np.ndarray, on: np.ndarray) -> float | None:
    """Return attenuation in dB from AEC-off RMS to AEC-on RMS."""
    off_mono = _mono(off)
    on_mono = _mono(on)
    off_rms = signal_rms(off_mono)
    on_rms = signal_rms(on_mono)
    if off_rms <= 0.0 or on_rms <= 0.0:
        return None
    return float(20.0 * np.log10(off_rms / on_rms))
