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


def measure_generated_audio_leading_silence(
    value: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: int = 10,
    min_consecutive_frames: int = 3,
    noise_window_s: float = 0.1,
    absolute_floor: float = 0.004,
    peak_relative_threshold: float = 0.02,
    noise_multiplier: float = 4.0,
) -> dict[str, Any]:
    """Measure low-energy audio before a stable generated-speech onset.

    A frame is active when its RMS is above the largest of an absolute floor,
    a peak-relative floor, and a noise-floor estimate.  The stable onset is
    the beginning of the first run of ``min_consecutive_frames`` active
    frames.  This intentionally leaves onset detection separate from
    trimming: callers can retain a pre-roll before the measured onset.
    """
    if sample_rate <= 0 or frame_ms <= 0 or min_consecutive_frames < 1:
        raise ValueError("invalid generated-audio onset parameters")
    if noise_window_s <= 0 or absolute_floor < 0 or peak_relative_threshold < 0:
        raise ValueError("invalid generated-audio threshold parameters")
    array = _mono(value)
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000)))
    frame_count = int(np.ceil(len(array) / frame_length)) if len(array) else 0
    if frame_count:
        padded = np.pad(array, (0, frame_count * frame_length - len(array)))
        frames = padded.reshape(frame_count, frame_length)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    else:
        frame_rms = np.empty(0, dtype=np.float32)
    peak = float(np.max(np.abs(array))) if len(array) else 0.0
    rms = signal_rms(array)
    noise_count = min(
        frame_count,
        max(1, int(round(noise_window_s * sample_rate / frame_length))),
    )
    noise_values = frame_rms[:noise_count]
    noise = float(np.median(noise_values)) if len(noise_values) else 0.0
    mad = float(np.median(np.abs(noise_values - noise))) if len(noise_values) else 0.0
    threshold = max(
        float(absolute_floor),
        peak * float(peak_relative_threshold),
        noise * float(noise_multiplier),
        noise + 6.0 * mad,
    )
    if peak > 0.0 and frame_count <= min_consecutive_frames:
        # A very short clip can have speech in the noise-estimation window.
        # Do not make an otherwise fully active clip impossible to detect.
        threshold = min(threshold, float(np.max(frame_rms)))
    active = frame_rms >= threshold
    first_low_frame = int(np.flatnonzero(active)[0]) if np.any(active) else None
    stable_frame: int | None = None
    required_frames = min(min_consecutive_frames, frame_count) if frame_count else min_consecutive_frames
    for index in range(max(0, frame_count - required_frames + 1)):
        if bool(np.all(active[index : index + required_frames])):
            stable_frame = index
            break
    nonzero = np.flatnonzero(np.abs(array) > 0.0)
    first_nonzero = int(nonzero[0]) if len(nonzero) else None
    stable_s = stable_frame * frame_length / sample_rate if stable_frame is not None else None
    return {
        "detected": stable_frame is not None,
        "first_nonzero_sample": first_nonzero,
        "first_nonzero_s": first_nonzero / sample_rate if first_nonzero is not None else None,
        "first_low_threshold_crossing_frame": first_low_frame,
        "first_low_threshold_crossing_s": first_low_frame * frame_length / sample_rate if first_low_frame is not None else None,
        "stable_speech_onset_frame": stable_frame,
        "stable_speech_onset_s": stable_s,
        "leading_silence_duration_s": stable_s,
        "peak": peak,
        "rms": rms,
        "audio_duration_s": len(array) / sample_rate,
        "sample_rate": sample_rate,
        "frame_ms": frame_ms,
        "min_consecutive_frames": min_consecutive_frames,
        "noise_window_s": noise_window_s,
        "noise_floor_rms": noise,
        "noise_floor_mad": mad,
        "absolute_floor": absolute_floor,
        "peak_relative_threshold": peak_relative_threshold,
        "noise_multiplier": noise_multiplier,
        "threshold_rms": threshold,
    }


def trim_leading_silence(
    value: np.ndarray,
    sample_rate: int,
    analysis: dict[str, Any] | None = None,
    *,
    pre_roll_s: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove only measured leading low-energy audio while retaining pre-roll."""
    if sample_rate <= 0 or pre_roll_s < 0:
        raise ValueError("sample_rate must be positive and pre_roll_s non-negative")
    original = np.asarray(value)
    measured = analysis or measure_generated_audio_leading_silence(original, sample_rate)
    onset = measured.get("stable_speech_onset_s") if measured.get("detected") else None
    if onset is None:
        start_sample = 0
    else:
        start_sample = max(0, int(np.floor((float(onset) - pre_roll_s) * sample_rate)))
        start_sample = min(start_sample, len(original))
    trimmed = original[start_sample:]
    return trimmed, {
        "trimmed": start_sample > 0,
        "pre_roll_s": pre_roll_s,
        "start_sample": start_sample,
        "start_s": start_sample / sample_rate,
        "removed_duration_s": start_sample / sample_rate,
        "output_duration_s": len(trimmed) / sample_rate,
        "input_duration_s": len(original) / sample_rate,
        "detected_onset_s": onset,
    }


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
