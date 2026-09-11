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
