from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from .vad import detect_speech_intervals


@dataclass(frozen=True)
class EchoRejectionConfig:
    """Lightweight reference-aware post-VAD decision thresholds."""

    correlation_threshold: float = 0.9
    residual_energy_ratio_threshold: float = 0.3
    max_lag_s: float = 0.25
    energy_ratio_min: float = 0.0
    energy_ratio_max: float = 2.0
    window_s: float = 0.08
    hop_s: float = 0.04
    vad_threshold: float = 0.015
    calibrated_from: str = "default_not_calibrated"


def echo_config_from_mapping(mapping: Mapping[str, Any] | None) -> EchoRejectionConfig:
    values = dict(mapping or {})
    fields = {
        "correlation_threshold",
        "residual_energy_ratio_threshold",
        "max_lag_s",
        "energy_ratio_min",
        "energy_ratio_max",
        "window_s",
        "hop_s",
        "vad_threshold",
        "calibrated_from",
    }
    selected = {key: values[key] for key in fields if key in values}
    return EchoRejectionConfig(**selected)


def _mono(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value)))) if len(value) else 0.0


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    length = min(len(left), len(right))
    if length < 2:
        return 0.0
    left = left[:length] - np.mean(left[:length])
    right = right[:length] - np.mean(right[:length])
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(abs(np.dot(left, right) / denominator))


def _resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return signal.astype(np.float32, copy=False)
    if len(signal) < 2:
        return signal.astype(np.float32, copy=False)
    target_length = max(1, round(len(signal) * target_rate / source_rate))
    source_x = np.linspace(0.0, 1.0, len(signal), endpoint=False)
    target_x = np.linspace(0.0, 1.0, target_length, endpoint=False)
    return np.interp(target_x, source_x, signal).astype(np.float32)


def _aligned_segments(reference: np.ndarray, microphone: np.ndarray, lag_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if lag_samples >= 0:
        length = min(len(reference), len(microphone) - lag_samples)
        if length <= 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        return reference[:length], microphone[lag_samples : lag_samples + length]
    offset = -lag_samples
    length = min(len(reference) - offset, len(microphone))
    if length <= 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    return reference[offset : offset + length], microphone[:length]


def _best_alignment(reference: np.ndarray, microphone: np.ndarray, sample_rate: int, max_lag_s: float) -> tuple[float, int]:
    search_rate = min(4000, sample_rate)
    ref = _resample(reference, sample_rate, search_rate)
    mic = _resample(microphone, sample_rate, search_rate)
    max_lag = max(0, int(round(max_lag_s * search_rate)))
    best_score = -1.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        left, right = _aligned_segments(ref, mic, lag)
        if len(left) < max(32, search_rate // 100):
            continue
        score = _correlation(left, right)
        if score > best_score:
            best_score = score
            best_lag = lag
    if best_score < 0:
        return 0.0, 0
    return float(best_score), int(round(best_lag * sample_rate / search_rate))


def _window_metric(reference: np.ndarray, microphone: np.ndarray) -> dict[str, float]:
    reference_rms = _rms(reference)
    microphone_rms = _rms(microphone)
    if reference_rms <= 1e-9:
        return {
            "correlation": 0.0,
            "energy_ratio": microphone_rms,
            "residual_energy_ratio": 1.0 if microphone_rms > 1e-9 else 0.0,
            "reference_rms": reference_rms,
            "microphone_rms": microphone_rms,
        }
    centered_reference = reference - np.mean(reference)
    centered_microphone = microphone - np.mean(microphone)
    denominator = float(np.dot(centered_reference, centered_reference))
    scale = float(np.dot(centered_reference, centered_microphone) / denominator) if denominator > 1e-12 else 0.0
    residual = centered_microphone - scale * centered_reference
    return {
        "correlation": _correlation(reference, microphone),
        "energy_ratio": microphone_rms / reference_rms,
        "residual_energy_ratio": _rms(residual) / microphone_rms if microphone_rms > 1e-9 else 0.0,
        "reference_rms": reference_rms,
        "microphone_rms": microphone_rms,
    }


def analyze_echo_pair(
    reference: np.ndarray,
    microphone: np.ndarray,
    sample_rate: int,
    *,
    max_lag_s: float = 0.25,
    window_s: float = 0.08,
    hop_s: float = 0.04,
    energy_floor: float = 0.004,
) -> dict[str, Any]:
    """Measure reference correlation and unexplained short-window energy."""
    if sample_rate <= 0 or max_lag_s < 0 or window_s <= 0 or hop_s <= 0:
        raise ValueError("invalid echo analysis parameters")
    ref = _mono(reference)
    mic = _mono(microphone)
    if len(ref) == 0 or len(mic) == 0:
        return {
            "correlation": 0.0,
            "lag_samples": 0,
            "lag_seconds": 0.0,
            "energy_ratio": 0.0,
            "residual_energy_ratio": 1.0,
            "window_metrics": [],
        }
    correlation, lag_samples = _best_alignment(ref, mic, sample_rate, max_lag_s)
    aligned_ref, aligned_mic = _aligned_segments(ref, mic, lag_samples)
    window_length = max(1, int(round(window_s * sample_rate)))
    hop_length = max(1, int(round(hop_s * sample_rate)))
    windows: list[dict[str, float]] = []
    for start in range(0, max(0, len(aligned_ref) - window_length + 1), hop_length):
        end = start + window_length
        metric = _window_metric(aligned_ref[start:end], aligned_mic[start:end])
        if max(metric["reference_rms"], metric["microphone_rms"]) >= energy_floor:
            metric["start_s"] = start / sample_rate
            windows.append(metric)
    if not windows:
        windows = [_window_metric(aligned_ref, aligned_mic)]
    return {
        "correlation": correlation,
        "lag_samples": lag_samples,
        "lag_seconds": lag_samples / sample_rate,
        "energy_ratio": _rms(mic) / max(_rms(ref), 1e-9),
        "residual_energy_ratio": float(statistics.median(item["residual_energy_ratio"] for item in windows)),
        "min_window_correlation": float(min(item["correlation"] for item in windows)),
        "max_window_residual_energy_ratio": float(max(item["residual_energy_ratio"] for item in windows)),
        "window_metrics": windows,
    }


def _is_echo(metrics: Mapping[str, Any], config: EchoRejectionConfig) -> bool:
    energy_ratio = float(metrics.get("energy_ratio", 0.0))
    lag_seconds = abs(float(metrics.get("lag_seconds", 0.0)))
    correlation = float(metrics.get("correlation", 0.0))
    residual = float(metrics.get("max_window_residual_energy_ratio", metrics.get("residual_energy_ratio", 1.0)))
    return (
        correlation >= config.correlation_threshold
        and lag_seconds <= config.max_lag_s
        and residual <= config.residual_energy_ratio_threshold
        and config.energy_ratio_min <= energy_ratio <= config.energy_ratio_max
    )


def classify_echo_candidate(
    reference: np.ndarray | None,
    microphone: np.ndarray,
    sample_rate: int,
    *,
    playback_active: bool,
    config: EchoRejectionConfig | None = None,
) -> dict[str, Any]:
    selected = config or EchoRejectionConfig()
    if not playback_active:
        return {"decision": "possible_user_speech", "reject": False, "reason": "playback_inactive", "metrics": None}
    if reference is None or len(_mono(reference)) == 0:
        return {"decision": "possible_user_speech", "reject": False, "reason": "no_reference", "metrics": None}
    metrics = analyze_echo_pair(
        _mono(reference),
        _mono(microphone),
        sample_rate,
        max_lag_s=selected.max_lag_s,
        window_s=selected.window_s,
        hop_s=selected.hop_s,
    )
    echo = _is_echo(metrics, selected)
    return {
        "decision": "probable_self_echo" if echo else "possible_user_speech",
        "reject": echo,
        "reason": "reference_explained" if echo else "unexplained_energy_or_mismatch",
        "metrics": metrics,
        "thresholds": selected.__dict__.copy(),
    }


def calibrate_thresholds(fixtures: list[Mapping[str, Any]], sample_rate: int) -> EchoRejectionConfig:
    """Choose thresholds from measured assistant-only/user-like fixture metrics."""
    measured: list[tuple[str, dict[str, Any]]] = []
    for fixture in fixtures:
        reference = fixture.get("reference")
        microphone = fixture.get("microphone")
        if reference is None or microphone is None:
            continue
        measured.append(
            (
                str(fixture.get("label", "")),
                analyze_echo_pair(_mono(reference), _mono(microphone), sample_rate),
            )
        )
    assistant = [metrics for label, metrics in measured if label == "assistant_only"]
    users = [metrics for label, metrics in measured if label == "synthetic_user_like"]
    if not assistant or not users:
        return EchoRejectionConfig(calibrated_from="insufficient_fixture_classes")
    energy_values = [float(item["energy_ratio"]) for item in assistant]
    energy_min = max(0.0, min(energy_values) * 0.7)
    energy_max = max(energy_values) * 1.35 + 0.02
    lag_max = max(abs(float(item["lag_seconds"])) for _, item in measured) + 0.01
    best: tuple[float, float, float, float] | None = None
    selected = EchoRejectionConfig(calibrated_from="calibration_failed")
    for correlation_threshold in np.linspace(0.5, 0.99, 20):
        for residual_threshold in np.linspace(0.1, 0.9, 17):
            config = EchoRejectionConfig(
                correlation_threshold=float(correlation_threshold),
                residual_energy_ratio_threshold=float(residual_threshold),
                max_lag_s=float(lag_max),
                energy_ratio_min=float(energy_min),
                energy_ratio_max=float(energy_max),
                calibrated_from="synthetic_fixture_grid_search",
            )
            assistant_false_accept = sum(not _is_echo(item, config) for item in assistant) / len(assistant)
            user_acceptance = sum(not _is_echo(item, config) for item in users) / len(users)
            objective = assistant_false_accept + (1.0 - user_acceptance)
            feasible_bonus = 0.0 if assistant_false_accept <= 0.1 and user_acceptance >= 0.9 else 1.0
            score = (feasible_bonus, objective, -float(correlation_threshold), -float(residual_threshold))
            if best is None or score < best:
                best = score
                selected = config
    if best is None:
        return EchoRejectionConfig(calibrated_from="calibration_failed")
    return selected


def _vad_positive_frames(audio: np.ndarray, sample_rate: int, threshold: float) -> int:
    intervals = detect_speech_intervals(audio, sample_rate, threshold=threshold)
    return sum(max(1, round((end - start) * 1000 / 20)) for start, end in intervals)


def evaluate_echo_rejection(
    fixtures: list[Mapping[str, Any]],
    sample_rate: int,
    *,
    config: EchoRejectionConfig | None = None,
) -> dict[str, Any]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    selected = config or calibrate_thresholds(fixtures, sample_rate)
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        reference = fixture.get("reference")
        microphone = _mono(fixture.get("microphone"))
        label = str(fixture.get("label", "unknown"))
        decision = classify_echo_candidate(
            _mono(reference) if reference is not None else None,
            microphone,
            sample_rate,
            playback_active=bool(fixture.get("playback_active", True)),
            config=selected,
        )
        vad_frames = _vad_positive_frames(microphone, sample_rate, selected.vad_threshold)
        rows.append(
            {
                "name": str(fixture.get("name", f"fixture_{len(rows)}")),
                "label": label,
                "vad_positive": vad_frames > 0,
                "vad_positive_frames": vad_frames,
                "decision": decision["decision"],
                "reject": decision["reject"],
                "metrics": decision["metrics"],
            }
        )
    assistant_rows = [row for row in rows if row["label"] == "assistant_only"]
    user_rows = [row for row in rows if row["label"] == "synthetic_user_like"]
    assistant_positive = [row for row in assistant_rows if row["vad_positive"]]
    user_positive = [row for row in user_rows if row["vad_positive"]]
    assistant_accepted = [row for row in assistant_positive if not row["reject"]]
    user_accepted = [row for row in user_positive if not row["reject"]]

    def distribution(key: str) -> dict[str, Any]:
        values = [float(row["metrics"][key]) for row in rows if row.get("metrics") and isinstance(row["metrics"].get(key), (int, float))]
        return {"values": values, "count": len(values), "median": statistics.median(values) if values else None, "min": min(values) if values else None, "max": max(values) if values else None}

    assistant_total = len(assistant_positive)
    user_total = len(user_positive)
    false_accept_rate = len(assistant_accepted) / assistant_total if assistant_total else None
    acceptance_rate = len(user_accepted) / user_total if user_total else None
    return {
        "status": "measured",
        "sample_rate": sample_rate,
        "thresholds": selected.__dict__.copy(),
        "rows": rows,
        "assistant_only": {
            "vad_positives": assistant_total,
            "vad_positive_frames": sum(row["vad_positive_frames"] for row in assistant_rows),
            "echo_reject_count": sum(row["reject"] for row in assistant_positive),
            "false_accept_count": len(assistant_accepted),
            "false_accept_rate": false_accept_rate,
        },
        "synthetic_user_like": {
            "vad_positives": user_total,
            "vad_positive_frames": sum(row["vad_positive_frames"] for row in user_rows),
            "accepted_count": len(user_accepted),
            "acceptance_rate": acceptance_rate,
            "false_reject_rate": 1.0 - acceptance_rate if acceptance_rate is not None else None,
        },
        "correlation_distribution": distribution("correlation"),
        "energy_ratio_distribution": distribution("energy_ratio"),
        "residual_energy_ratio_distribution": distribution("max_window_residual_energy_ratio"),
        "lag_seconds_distribution": distribution("lag_seconds"),
        "mute_reference": {
            "assistant_only_false_accept_rate": 0.0,
            "adopted": False,
            "reason": "VAD disabled during playback would prevent future barge-in",
        },
    }
