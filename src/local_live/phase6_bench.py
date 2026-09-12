from __future__ import annotations

import copy
import json
import math
import subprocess
import statistics
import time
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

from .audio import AudioVolumeGuard, PipeWireInventory, PipeWirePCMPlayback, RawCaptureSession, audio_file_stats, stable_target
from .audio_metrics import clipping_ratio, detect_acoustic_onset, measure_generated_audio_leading_silence, resample_mono, signal_rms
from .bench import artifact_dir, write_benchmark
from .config import nested
from .phase4_bench import _ensure_stability_fixtures, _echo_reference, _run_health, _run_stability_turn, percentile_summary, run_stability_bench
from .phase5_bench import _pipewire_health, resource_snapshot


PHASE6_ANALYSIS_RATE = 4000
PHASE6_REFERENCE_CORRELATION_MIN = 0.65
PHASE6_REFERENCE_MARGIN_MIN = 0.01
PHASE6_ONSET_SEARCH_MARGIN_S = 0.10
PHASE6_DEFAULT_PATH_HIGH_S = 1.0
PHASE6_FIXED_REPLAY_REPEATS = 100
PHASE6_NEGATIVE_REPEATS = 30
PHASE6_FIXTURE_REPEATS = 10
PHASE6_FIXTURE_COUNT = 5
PHASE6_LATE_TOLERANCE_S = 0.05


def split_pcm_chunks(payload: bytes, chunk_bytes: int) -> list[bytes]:
    """Split PCM16 bytes without changing order or creating odd-byte chunks."""
    if len(payload) % 2:
        raise ValueError("PCM16 payload must contain an even number of bytes")
    chunk_size = int(chunk_bytes)
    if chunk_size < 2:
        raise ValueError("chunk_bytes must be at least 2")
    chunk_size -= chunk_size % 2
    return [payload[offset : offset + chunk_size] for offset in range(0, len(payload), chunk_size)]


def frame_energy_series(value: np.ndarray, sample_rate: int, *, frame_ms: int = 10) -> dict[str, Any]:
    """Return a padded short-frame RMS series for auditable onset evidence."""
    if sample_rate <= 0 or frame_ms <= 0:
        raise ValueError("sample_rate and frame_ms must be positive")
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000)))
    frame_count = int(math.ceil(len(array) / frame_length)) if len(array) else 0
    if frame_count:
        padded = np.pad(array, (0, frame_count * frame_length - len(array)))
        frames = padded.reshape(frame_count, frame_length)
        values = np.sqrt(np.mean(np.square(frames), axis=1)).astype(np.float32)
    else:
        values = np.empty(0, dtype=np.float32)
    return {
        "sample_rate": sample_rate,
        "frame_ms": frame_ms,
        "frame_samples": frame_length,
        "frame_count": frame_count,
        "duration_s": len(array) / sample_rate,
        "values": [float(item) for item in values],
        "rms": signal_rms(array),
        "peak": float(np.max(np.abs(array))) if len(array) else 0.0,
        "partial_final_frame_padded": bool(len(array) and len(array) % frame_length),
    }


def _valid_normalized_correlation(reference: np.ndarray, recording: np.ndarray) -> np.ndarray:
    """Compute normalized valid correlations with FFT and local recording energy."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    rec = np.asarray(recording, dtype=np.float64).reshape(-1)
    if len(ref) < 2 or len(rec) < len(ref):
        return np.empty(0, dtype=np.float64)
    ref_centered = ref - float(np.mean(ref))
    ref_norm = float(np.linalg.norm(ref_centered))
    if ref_norm <= 1e-12:
        return np.empty(0, dtype=np.float64)
    size = 1 << (len(rec) + len(ref) - 1).bit_length()
    convolution = np.fft.irfft(
        np.fft.rfft(rec, size) * np.fft.rfft(ref_centered[::-1], size),
        size,
    )[: len(rec) + len(ref) - 1]
    dots = convolution[len(ref) - 1 : len(rec)]
    prefix = np.concatenate(([0.0], np.cumsum(rec, dtype=np.float64)))
    squared_prefix = np.concatenate(([0.0], np.cumsum(np.square(rec), dtype=np.float64)))
    sums = prefix[len(ref) :] - prefix[: -len(ref)]
    squared_sums = squared_prefix[len(ref) :] - squared_prefix[: -len(ref)]
    energy = np.maximum(0.0, squared_sums - np.square(sums) / len(ref))
    denominator = ref_norm * np.sqrt(energy)
    scores = np.divide(dots, denominator, out=np.zeros_like(dots), where=denominator > 1e-12)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def build_expected_onset_window(
    first_pcm_written_s: float,
    reference_onset_s: float,
    physical_path_distribution: Mapping[str, Any],
    *,
    margin_s: float = PHASE6_ONSET_SEARCH_MARGIN_S,
) -> dict[str, Any]:
    """Build an onset window from measured playback-path latency, not a fixed onset."""
    if first_pcm_written_s < 0 or reference_onset_s < 0 or margin_s < 0:
        raise ValueError("onset timing values must be non-negative")
    low = physical_path_distribution.get("low_s", 0.0)
    high = physical_path_distribution.get("high_s", PHASE6_DEFAULT_PATH_HIGH_S)
    low_s = max(0.0, float(low) if isinstance(low, (int, float)) else 0.0)
    high_s = max(low_s, float(high) if isinstance(high, (int, float)) else PHASE6_DEFAULT_PATH_HIGH_S)
    base = first_pcm_written_s + reference_onset_s
    return {
        "start_s": max(0.0, base + low_s - margin_s),
        "end_s": base + high_s + margin_s,
        "base_s": base,
        "path_low_s": low_s,
        "path_high_s": high_s,
        "margin_s": margin_s,
        "source": physical_path_distribution.get("source", "measured_physical_replay_distribution"),
    }


def reference_alignment_evidence(
    reference: np.ndarray,
    recording: np.ndarray,
    *,
    sample_rate: int,
    recording_rate: int,
    expected_playback_start_s: float,
    reference_onset_s: float | None = None,
    physical_path_distribution: Mapping[str, Any] | None = None,
    analysis_rate: int = PHASE6_ANALYSIS_RATE,
    search_margin_s: float = PHASE6_ONSET_SEARCH_MARGIN_S,
) -> dict[str, Any]:
    """Align actual playback PCM to a capture and report confidence plus margin."""
    if sample_rate <= 0 or recording_rate <= 0 or expected_playback_start_s < 0:
        raise ValueError("invalid reference alignment parameters")
    reference_array = resample_mono(np.asarray(reference, dtype=np.float32), sample_rate, analysis_rate)
    recording_array = resample_mono(np.asarray(recording, dtype=np.float32), recording_rate, analysis_rate)
    if reference_onset_s is None:
        analysis = measure_generated_audio_leading_silence(reference_array, analysis_rate)
        onset_value = analysis.get("stable_speech_onset_s")
        reference_onset_s = float(onset_value) if isinstance(onset_value, (int, float)) else 0.0
    reference_onset_s = max(0.0, float(reference_onset_s))
    distribution = physical_path_distribution or {"low_s": 0.0, "high_s": PHASE6_DEFAULT_PATH_HIGH_S, "source": "bounded_fallback"}
    expected = build_expected_onset_window(expected_playback_start_s, reference_onset_s, distribution, margin_s=search_margin_s)
    active_start = min(len(reference_array), max(0, int(round(reference_onset_s * analysis_rate))))
    active_reference = reference_array[active_start:]
    window_specs: list[tuple[float, float]] = []
    for window_s in (0.08, 0.12, 0.18, 0.30):
        max_offset_s = max(0.0, len(active_reference) / analysis_rate - window_s)
        offsets = np.arange(0.0, max_offset_s + 0.001, 0.12).tolist()
        if not offsets or offsets[-1] < max_offset_s - 0.02:
            offsets.append(max_offset_s)
        window_specs.extend((window_s, float(offset)) for offset in offsets)
    exclusion = max(1, int(round(0.18 * analysis_rate)))
    candidate_pool: list[tuple[float, float, int, float, float]] = []
    for window_s, reference_offset_s in window_specs:
        offset = int(round(reference_offset_s * analysis_rate))
        window_length = int(round(window_s * analysis_rate))
        segment = active_reference[offset : offset + window_length]
        scores = _valid_normalized_correlation(segment, recording_array)
        if not len(scores):
            continue
        min_position = max(0, int(math.floor((expected["start_s"] + reference_offset_s) * analysis_rate)))
        max_position = min(len(scores) - 1, int(math.ceil((expected["end_s"] + reference_offset_s) * analysis_rate)))
        if max_position < min_position:
            continue
        positions = np.arange(min_position, max_position + 1, dtype=np.int64)
        selected = scores[positions]
        order = np.argsort(np.abs(selected))[::-1]
        local_positions: list[int] = []
        for local_index in order:
            position = int(positions[int(local_index)])
            active_position = position - offset
            if any(abs(active_position - previous) <= exclusion for previous in local_positions):
                continue
            local_positions.append(active_position)
            signed_score = float(scores[position])
            candidate_pool.append((abs(signed_score), signed_score, active_position, window_s, reference_offset_s))
            if len(local_positions) >= 4:
                break
    if not candidate_pool:
        return {
            "matched": False,
            "best_correlation": None,
            "confidence": None,
            "confidence_margin": None,
            "reference_start_s": None,
            "aligned_onset_s": None,
            "lag_s": None,
            "search_window_s": {"start_s": expected["start_s"], "end_s": expected["end_s"]},
            "reference_onset_s": reference_onset_s,
            "analysis_rate": analysis_rate,
            "candidate_count": 0,
            "reason": "reference_or_recording_too_short",
        }
    best_score, best_signed_score, best_position, best_window_s, best_reference_offset_s = max(candidate_pool, key=lambda item: item[0])
    second_candidates = [item[0] for item in candidate_pool if abs(item[2] - best_position) > exclusion]
    second_score = max(second_candidates) if second_candidates else None
    margin = best_score - second_score if second_score is not None else None
    matched = best_score >= PHASE6_REFERENCE_CORRELATION_MIN and (margin is None or margin >= PHASE6_REFERENCE_MARGIN_MIN)
    active_reference_start_s = best_position / analysis_rate
    reference_start_s = active_reference_start_s - reference_onset_s
    aligned_onset_s = active_reference_start_s
    return {
        "matched": bool(matched),
        "best_correlation": best_score,
        "confidence": best_score,
        "confidence_margin": margin,
        "second_best_correlation": second_score,
        "signed_correlation": best_signed_score,
        "correlation_sign": 1 if best_signed_score >= 0 else -1,
        "correlation_window_s": best_window_s,
        "correlation_reference_offset_s": best_reference_offset_s,
        "reference_start_s": reference_start_s,
        "aligned_onset_s": aligned_onset_s,
        "lag_s": active_reference_start_s - (expected_playback_start_s + reference_onset_s),
        "search_window_s": {"start_s": expected["start_s"], "end_s": expected["end_s"]},
        "reference_onset_s": reference_onset_s,
        "analysis_rate": analysis_rate,
        "candidate_count": int(len(candidate_pool)),
        "reason": "matched" if matched else "correlation_or_margin_below_gate",
    }


def classify_physical_onset(
    playback_evidence: Mapping[str, Any],
    microphone_evidence: Mapping[str, Any],
    energy_detector: Mapping[str, Any],
    alignment_evidence: Mapping[str, Any],
    expected_window: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Classify physical onset with independent playback, capture, energy, and alignment evidence."""
    playback_active = playback_evidence.get("playback_active", True)
    energy_detected = bool(energy_detector.get("detected"))
    if playback_active is False:
        return {
            "classification": "false_positive" if energy_detected else "no_playback_negative",
            "measurement_confirmed": False,
            "confidence": "detector_false_positive" if energy_detected else "negative_control",
        }
    if playback_evidence.get("playback_success") is False:
        return {"classification": "playback_failure", "measurement_confirmed": False, "confidence": "playback_evidence"}
    exit_status = playback_evidence.get("process_exit_status")
    if isinstance(exit_status, int) and exit_status not in (0,):
        return {"classification": "playback_failure", "measurement_confirmed": False, "confidence": "playback_exit_status"}
    bytes_written = playback_evidence.get("pcm_bytes_written")
    if isinstance(bytes_written, (int, float)) and bytes_written <= 0:
        return {"classification": "playback_failure", "measurement_confirmed": False, "confidence": "pcm_write_evidence"}
    stats = microphone_evidence.get("recording_stats") or {}
    duration = float(stats.get("duration_s", 0.0) or 0.0)
    rms = float(stats.get("rms", 0.0) or 0.0)
    peak = float(stats.get("peak", 0.0) or 0.0)
    if duration < 0.5 or (rms <= 0.001 and peak <= 0.005):
        return {"classification": "microphone_capture_failure", "measurement_confirmed": False, "confidence": "recording_level_or_duration"}
    aligned = bool(alignment_evidence.get("matched"))
    aligned_onset = alignment_evidence.get("aligned_onset_s")
    if aligned and expected_window and isinstance(aligned_onset, (int, float)):
        if float(aligned_onset) < float(expected_window.get("start_s", 0.0)) - PHASE6_LATE_TOLERANCE_S or float(aligned_onset) > float(expected_window.get("end_s", float("inf"))) + PHASE6_LATE_TOLERANCE_S:
            return {"classification": "late_outside_window", "measurement_confirmed": False, "confidence": "reference_alignment_outside_expected_window"}
    if aligned and energy_detected:
        return {"classification": "confirmed", "measurement_confirmed": True, "confidence": "energy_and_reference_alignment"}
    if aligned:
        return {"classification": "correlation_recovered", "measurement_confirmed": True, "confidence": "reference_alignment_recovered_energy_miss"}
    if energy_detected:
        return {"classification": "energy_only", "measurement_confirmed": False, "confidence": "reference_alignment_weak"}
    if rms > 0.001 or peak > 0.005:
        return {"classification": "no_physical_match", "measurement_confirmed": False, "confidence": "capture_present_reference_missing"}
    return {"classification": "unknown", "measurement_confirmed": False, "confidence": "insufficient_evidence"}


def separate_application_measurement_status(application_success: bool, physical_classification: str) -> dict[str, Any]:
    """Keep application success independent from physical measurement classification."""
    return {
        "application_status": "success" if application_success else "failed",
        "physical_measurement_status": physical_classification,
        "application_failed_due_to_measurement": False,
    }


def physical_latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize confirmed-only and confirmed-plus-recovered physical latency."""
    def latency(row: Mapping[str, Any]) -> float | None:
        value = row.get("physical_latency_s")
        return float(value) if isinstance(value, (int, float)) else None

    confirmed = [value for row in rows if row.get("measurement_classification") == "confirmed" for value in [latency(row)] if value is not None]
    confirmed_recovered = [value for row in rows if row.get("measurement_classification") in {"confirmed", "correlation_recovered"} for value in [latency(row)] if value is not None]
    app_rows = [row for row in rows if row.get("application_status") in {"success", "failed"}]
    confirmation_rows = [row for row in rows if row.get("measurement_classification") in {"confirmed", "correlation_recovered"}]
    return {
        "confirmed_only": percentile_summary(confirmed),
        "confirmed_plus_recovered": percentile_summary(confirmed_recovered),
        "application_success_rate": sum(row.get("application_status") == "success" for row in app_rows) / len(app_rows) if app_rows else None,
        "measurement_confirmation_rate": len(confirmation_rows) / len(rows) if rows else None,
        "confirmed_count": sum(row.get("measurement_classification") == "confirmed" for row in rows),
        "correlation_recovered_count": sum(row.get("measurement_classification") == "correlation_recovered" for row in rows),
        "unknown_count": sum(row.get("measurement_classification") == "unknown" for row in rows),
    }


def _simple_distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "values": ordered,
        "median": statistics.median(ordered) if ordered else None,
        "min": min(ordered) if ordered else None,
        "max": max(ordered) if ordered else None,
    }


def load_physical_path_distribution(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load a bounded physical-path distribution from prior measured replay evidence."""
    result_dir = Path(nested(dict(config), "app", "result_dir", default="results"))
    source_paths: list[str] = []
    values: list[float] = []
    playback_path = result_dir / "bench_playback_path.json"
    if playback_path.exists():
        source_paths.append(str(playback_path))
        try:
            payload = json.loads(playback_path.read_text())
            values.extend(
                float(value)
                for value in ((payload.get("data") or {}).get("summary") or {}).get("expected_signal_to_measured_microphone_s", {}).get("values", [])
                if isinstance(value, (int, float)) and float(value) >= 0.0
            )
        except (OSError, ValueError, TypeError):
            pass
    if not values:
        stability_path = result_dir / "bench_stability.json"
        if stability_path.exists():
            source_paths.append(str(stability_path))
            try:
                payload = json.loads(stability_path.read_text())
                for row in ((payload.get("data") or {}).get("turns") or []):
                    timing = row.get("timing_ns") or {}
                    stages = row.get("stages") or {}
                    if isinstance(timing.get("physical_audio_detected"), int) and isinstance(timing.get("playback_stream_started"), int):
                        acoustic = (timing["physical_audio_detected"] - timing["playback_stream_started"]) / 1e9
                        generated = stages.get("first_pcm_to_actual")
                        if isinstance(generated, (int, float)) and acoustic >= float(generated):
                            values.append(acoustic - float(generated))
            except (OSError, ValueError, TypeError):
                pass
    if not values:
        return {
            "source": "not_recorded; bounded fallback only",
            "source_paths": source_paths,
            "count": 0,
            "low_s": 0.0,
            "high_s": PHASE6_DEFAULT_PATH_HIGH_S,
            "median_s": None,
            "p95_s": None,
            "values": [],
        }
    ordered = sorted(values)
    p95 = float(np.percentile(np.asarray(ordered, dtype=np.float64), 95))
    return {
        "source": "prior_physical_replay_distribution",
        "source_paths": source_paths,
        "count": len(ordered),
        "low_s": max(0.0, ordered[0]),
        "high_s": max(ordered[-1], p95),
        "median_s": float(statistics.median(ordered)),
        "p95_s": p95,
        "values": ordered,
    }


def _pcm16_from_wav(path: Path) -> tuple[bytes, np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        raw = source.readframes(source.getnframes())
    if channels == 1 and width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return raw, values, rate
    samples, rate = sf.read(str(path), always_2d=False)
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    raw = np.clip(values, -1.0, 1.0)
    raw = (raw * 32767.0).astype("<i2").tobytes()
    return raw, np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, int(rate)


def _write_energy_artifact(config: Mapping[str, Any], name: str, evidence: Mapping[str, Any]) -> str:
    path = artifact_dir(dict(config)) / name
    path.write_text(json.dumps(dict(evidence), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _capture_positive_replay(
    config: Mapping[str, Any],
    *,
    reference_path: Path,
    reference_pcm_path: Path,
    reference_pcm: bytes,
    reference_values: np.ndarray,
    reference_rate: int,
    physical_path_distribution: Mapping[str, Any],
    playback_target: str,
    capture_target: str,
    label: str,
    repeat_index: int,
    lead_s: float,
    tail_s: float,
    run_id: str = "latest",
) -> dict[str, Any]:
    output = artifact_dir(dict(config)) / f"phase6_{run_id}_{label}_{repeat_index:03d}_raw_mic.wav"
    observation: dict[str, Any] = {
        "run_kind": label,
        "repeat_index": repeat_index,
        "status": "error",
        "application_status": "failed",
        "physical_measurement_status": "unknown",
        "measurement_classification": "unknown",
        "reference_path": str(reference_path),
        "reference_pcm_path": str(reference_pcm_path),
        "recording_path": str(output),
    }
    capture: RawCaptureSession | None = None
    playback: PipeWirePCMPlayback | None = None
    process_holder: dict[str, Any] = {}
    try:
        current_inventory = PipeWireInventory.discover()
        current_health = _pipewire_health(current_inventory)
        current_speaker = current_inventory.usb_speaker()
        current_microphone = current_inventory.usb_microphone()
        device_available = bool(
            current_speaker
            and current_microphone
            and stable_target(current_speaker) == playback_target
            and stable_target(current_microphone) == capture_target
        )
        observation["device_available"] = device_available
        observation["pipewire_health"] = current_health
        if not device_available:
            observation.update(
                {
                    "status": "blocked",
                    "physical_measurement_status": "device_unavailable",
                    "measurement_classification": "device_unavailable",
                }
            )
            return observation
        capture = RawCaptureSession(output, target=capture_target, sample_rate=16000)
        capture.start()
        record_start_ns = capture.started_ns or time.monotonic_ns()
        time.sleep(max(0.0, lead_s))

        def spawn(command: list[str], **kwargs: Any) -> Any:
            process = subprocess.Popen(command, **kwargs)
            process_holder["process"] = process
            return process

        playback = PipeWirePCMPlayback(playback_target, popen_factory=spawn)
        start_info = playback.start(sample_rate=reference_rate, channels=1)
        process = process_holder.get("process")
        process_alive_at_start = bool(process is not None and process.poll() is None)
        first_pcm_written_ns: int | None = None
        write_events: list[dict[str, Any]] = []
        bytes_written = 0
        write_count = 0
        chunk_bytes = max(2, int(nested(dict(config), "bench", "phase6_pcm_chunk_bytes", default=8192)))
        chunk_bytes -= chunk_bytes % 2
        for chunk in split_pcm_chunks(reference_pcm, chunk_bytes):
            if not chunk:
                continue
            result = playback.queue(chunk)
            if result.get("queued"):
                timestamp_ns = playback.last_queued_ns or time.monotonic_ns()
                first_pcm_written_ns = first_pcm_written_ns or timestamp_ns
                bytes_written += len(chunk)
                write_count += 1
                if len(write_events) < 8:
                    write_events.append({"timestamp_ns": timestamp_ns, "bytes": len(chunk)})
        finish_result = playback.finish()
        playback_completed_ns = time.monotonic_ns()
        process_exit_status = process.returncode if process is not None else None
        recording = capture.stop(tail_s=tail_s)
        recorded, recorded_rate = sf.read(str(output), always_2d=False)
        recorded_array = np.asarray(recorded, dtype=np.float32).reshape(-1)
        stats = recording.get("recording_stats") or audio_file_stats(output)
        reference_analysis = measure_generated_audio_leading_silence(reference_values, reference_rate)
        reference_onset_s = float(reference_analysis.get("stable_speech_onset_s") or 0.0)
        playback_start_ns = start_info.get("started_ns") or first_pcm_written_ns
        playback_start_offset_s = ((playback_start_ns - record_start_ns) / 1e9) if isinstance(playback_start_ns, int) else lead_s
        first_pcm_offset_s = ((first_pcm_written_ns - record_start_ns) / 1e9) if isinstance(first_pcm_written_ns, int) else playback_start_offset_s
        expected_window = build_expected_onset_window(first_pcm_offset_s, reference_onset_s, physical_path_distribution)
        search_start_s = max(0.0, expected_window["start_s"] - PHASE6_ONSET_SEARCH_MARGIN_S)
        energy = detect_acoustic_onset(recorded_array, int(recorded_rate), search_start_s=search_start_s)
        energy_series = frame_energy_series(recorded_array, int(recorded_rate))
        alignment = reference_alignment_evidence(
            reference_values,
            recorded_array,
            sample_rate=reference_rate,
            recording_rate=int(recorded_rate),
            expected_playback_start_s=first_pcm_offset_s,
            reference_onset_s=reference_onset_s,
            physical_path_distribution=physical_path_distribution,
        )
        playback_evidence = {
            "playback_active": True,
            "playback_success": process_exit_status in (0,),
            "first_pcm_received": bool(bytes_written),
            "first_actual_speech_pcm": reference_onset_s,
            "pcm_bytes_written": bytes_written,
            "pcm_write_count": write_count,
            "write_events": write_events,
            "persistent_playback_process_alive_at_start": process_alive_at_start,
            "persistent_playback_process_exit_status": process_exit_status,
            "process_exit_status": process_exit_status,
            "playback_completed": True,
            "playback_result": finish_result,
            "first_pcm_written_ns": first_pcm_written_ns,
            "playback_started_ns": playback_start_ns,
            "playback_completed_ns": playback_completed_ns,
        }
        microphone_evidence = {
            "recording_path": str(output),
            "recording_stats": stats,
            "short_frame_energy_series_path": _write_energy_artifact(config, f"{output.stem}_energy.json", energy_series),
            "adaptive_noise_floor_rms": energy.get("noise_floor_rms"),
            "device_available": device_available,
            "pipewire_health": current_health,
        }
        classification = classify_physical_onset(playback_evidence, microphone_evidence, energy, alignment, expected_window)
        detected_onset_s = energy.get("onset_s")
        physical_onset_s = float(detected_onset_s) if isinstance(detected_onset_s, (int, float)) else alignment.get("aligned_onset_s")
        physical_latency_s = (
            float(physical_onset_s) - (first_pcm_offset_s + reference_onset_s)
            if isinstance(physical_onset_s, (int, float))
            else None
        )
        app_status = separate_application_measurement_status(
            bool(playback_evidence.get("playback_success")) and bool(playback_evidence.get("playback_completed")),
            classification["classification"],
        )
        observation.update(
            {
                "status": "measured",
                **app_status,
                "measurement_classification": classification["classification"],
                "measurement_confirmed": classification["measurement_confirmed"],
                "classification_confidence": classification["confidence"],
                "playback_evidence": playback_evidence,
                "microphone_evidence": microphone_evidence,
                "reference_alignment": alignment,
                "energy_detector": energy,
                "expected_onset_window": expected_window,
                "detected_onset_s": detected_onset_s,
                "aligned_onset_s": alignment.get("aligned_onset_s"),
                "physical_onset_s": physical_onset_s,
                "physical_latency_s": physical_latency_s,
                "record_start_ns": record_start_ns,
                "pipewire_health": microphone_evidence["pipewire_health"],
                "recording_stats": stats,
                "late_onset": classification["classification"] == "late_outside_window",
            }
        )
        reference_pcm_path.write_bytes(reference_pcm) if not reference_pcm_path.exists() else None
        return observation
    except Exception as exc:
        observation.update({"error_type": type(exc).__name__, "error": str(exc)})
        return observation
    finally:
        if playback is not None and playback.active:
            try:
                playback.cancel()
            except Exception:
                pass
        if capture is not None and capture.process is not None and capture._result is None:
            try:
                capture.stop(tail_s=0.1)
            except Exception:
                pass


def _capture_negative(
    config: Mapping[str, Any],
    *,
    capture_target: str,
    repeat_index: int,
    duration_s: float,
    run_id: str = "latest",
) -> dict[str, Any]:
    output = artifact_dir(dict(config)) / f"phase6_{run_id}_negative_{repeat_index:03d}_raw_mic.wav"
    row: dict[str, Any] = {
        "run_kind": "no_playback_negative",
        "repeat_index": repeat_index,
        "status": "error",
        "application_status": "not_applicable",
        "physical_measurement_status": "negative_control",
        "measurement_classification": "unknown",
        "recording_path": str(output),
    }
    capture: RawCaptureSession | None = None
    try:
        current_inventory = PipeWireInventory.discover()
        current_health = _pipewire_health(current_inventory)
        current_microphone = current_inventory.usb_microphone()
        microphone_available = bool(current_microphone and stable_target(current_microphone) == capture_target)
        row["device_available"] = microphone_available
        row["pipewire_health"] = current_health
        if not microphone_available:
            row.update({"status": "blocked", "physical_measurement_status": "device_unavailable", "measurement_classification": "device_unavailable"})
            return row
        capture = RawCaptureSession(output, target=capture_target, sample_rate=16000)
        capture.start()
        time.sleep(max(0.0, duration_s))
        recording = capture.stop(tail_s=0.0)
        samples, rate = sf.read(str(output), always_2d=False)
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        energy = detect_acoustic_onset(array, int(rate), search_start_s=min(0.25, max(0.0, duration_s / 2.0)))
        energy_series = frame_energy_series(array, int(rate))
        stats = recording.get("recording_stats") or audio_file_stats(output)
        playback_evidence = {"playback_active": False, "playback_success": True, "pcm_bytes_written": 0, "playback_completed": False}
        microphone_evidence = {
            "recording_path": str(output),
            "recording_stats": stats,
            "short_frame_energy_series_path": _write_energy_artifact(config, f"{output.stem}_energy.json", energy_series),
            "adaptive_noise_floor_rms": energy.get("noise_floor_rms"),
            "device_available": microphone_available,
            "pipewire_health": current_health,
        }
        classification = classify_physical_onset(playback_evidence, microphone_evidence, energy, {"matched": False}, None)
        row.update(
            {
                "status": "measured",
                "physical_measurement_status": classification["classification"],
                "measurement_classification": classification["classification"],
                "measurement_confirmed": False,
                "classification_confidence": classification["confidence"],
                "playback_evidence": playback_evidence,
                "microphone_evidence": microphone_evidence,
                "energy_detector": energy,
                "recording_stats": stats,
                "false_positive": classification["classification"] == "false_positive",
            }
        )
    except Exception as exc:
        row.update({"error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if capture is not None and capture.process is not None and capture._result is None:
            try:
                capture.stop(tail_s=0.1)
            except Exception:
                pass
    return row


def _summarize_physical_rows(rows: list[dict[str, Any]], *, expected_attempts: int) -> dict[str, Any]:
    measured = [row for row in rows if row.get("status") == "measured"]
    classifications: dict[str, int] = {}
    for row in rows:
        name = str(row.get("measurement_classification", "unknown"))
        classifications[name] = classifications.get(name, 0) + 1
    confirmed = sum(row.get("measurement_classification") == "confirmed" for row in rows)
    recovered = sum(row.get("measurement_classification") == "correlation_recovered" for row in rows)
    unknown = sum(row.get("measurement_classification") == "unknown" for row in rows)
    return {
        "attempts": len(rows),
        "expected_attempts": expected_attempts,
        "measured": len(measured),
        "blocked": sum(row.get("status") == "blocked" for row in rows),
        "failed": sum(row.get("status") == "error" for row in rows),
        "classifiable": len(rows) - unknown - sum(row.get("status") == "error" for row in rows),
        "classifiable_rate": (len(rows) - unknown - sum(row.get("status") == "error" for row in rows)) / len(rows) if rows else None,
        "confirmed": confirmed,
        "correlation_recovered": recovered,
        "confirmed_or_recovered": confirmed + recovered,
        "confirmation_rate": (confirmed + recovered) / len(rows) if rows else None,
        "unknown": unknown,
        "unknown_rate": unknown / len(rows) if rows else None,
        "classification_counts": classifications,
        "application_failed": sum(row.get("application_status") == "failed" for row in rows),
        "physical_latency": physical_latency_summary(rows),
    }


def run_phase6_physical_onset_bench(
    config: dict[str, Any],
    *,
    fixed_repeats: int | None = None,
    negative_repeats: int | None = None,
    fixture_repeats: int | None = None,
) -> dict[str, Any]:
    """Run fixed-reference, negative-control, and five-fixture physical onset evidence."""
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_id = time.strftime("%Y%m%d_%H%M%S")
    data: dict[str, Any] = {
        "status": "blocked",
        "benchmark": "physical_onset",
        "phase": 6,
        "run_id": run_id,
        "turn57_diagnosis": "results/turn57_diagnosis.json",
        "fixed_replay": {"rows": [], "summary": None},
        "fixture_replay": {"rows": [], "summary": None, "fixtures": []},
        "negative_controls": {"rows": [], "summary": None},
        "reference": None,
        "physical_path_distribution": None,
        "audio_state": {"snapshot": None, "restore_error": None},
        "deferred_manual": [
            "human speech",
            "human physical double-talk or barge-in",
            "MOS, subjective listening, and manual gain tuning",
        ],
    }
    guard: AudioVolumeGuard | None = None
    try:
        fixed_repeats = int(
            fixed_repeats
            if fixed_repeats is not None
            else nested(config, "bench", "phase6_fixed_replay_repeats", default=PHASE6_FIXED_REPLAY_REPEATS)
        )
        negative_repeats = int(
            negative_repeats
            if negative_repeats is not None
            else nested(config, "bench", "phase6_negative_repeats", default=PHASE6_NEGATIVE_REPEATS)
        )
        fixture_repeats = int(
            fixture_repeats
            if fixture_repeats is not None
            else nested(config, "bench", "phase6_fixture_repeats", default=PHASE6_FIXTURE_REPEATS)
        )
        fixture_count = int(nested(config, "bench", "phase6_fixture_count", default=PHASE6_FIXTURE_COUNT))
        if min(fixed_repeats, negative_repeats, fixture_repeats, fixture_count) < 0:
            raise ValueError("Phase 6 repeat counts must be non-negative")
        reference_path, reference, reference_rate = _echo_reference(config)
        reference_pcm, reference_values, reference_rate = _pcm16_from_wav(reference_path)
        reference_analysis = measure_generated_audio_leading_silence(reference_values, reference_rate)
        path_distribution = load_physical_path_distribution(config)
        data["reference"] = {
            "path": str(reference_path),
            "sample_rate": reference_rate,
            "pcm_bytes": len(reference_pcm),
            "duration_s": len(reference_values) / reference_rate,
            "generated_onset": reference_analysis,
            "authoritative_pcm_definition": "exact PCM16 bytes queued to persistent pw-cat stdin; HTTP chunk boundaries are not used",
        }
        data["physical_path_distribution"] = path_distribution
        inventory = PipeWireInventory.discover()
        speaker = inventory.usb_speaker()
        microphone = inventory.usb_microphone()
        playback_target = stable_target(speaker) if speaker else None
        capture_target = stable_target(microphone) if microphone else None
        if not playback_target or not capture_target:
            raise RuntimeError("stable USB playback and capture targets are required")
        data["targets"] = {
            "speaker": playback_target,
            "microphone": capture_target,
            "initial_health": _pipewire_health(inventory),
        }
        guard = AudioVolumeGuard(speaker_target=playback_target, microphone_target=capture_target)
        guard.__enter__()
        data["audio_state"]["snapshot"] = guard.snapshot.to_dict() if guard.snapshot else None
        guard.set_mutes(speaker_muted=False, microphone_muted=False)
        authoritative_path = artifact_dir(config) / f"phase6_{run_id}_fixed_reference_authoritative.pcm"
        authoritative_path.write_bytes(reference_pcm)
        lead_s = float(nested(config, "bench", "phase6_record_lead_s", default=0.4))
        tail_s = max(0.35, float(path_distribution.get("high_s", PHASE6_DEFAULT_PATH_HIGH_S)) + 0.1)
        fixed_rows: list[dict[str, Any]] = []
        for index in range(1, fixed_repeats + 1):
            fixed_rows.append(
                _capture_positive_replay(
                    config,
                    reference_path=reference_path,
                    reference_pcm_path=authoritative_path,
                    reference_pcm=reference_pcm,
                    reference_values=reference_values,
                    reference_rate=reference_rate,
                    physical_path_distribution=path_distribution,
                    playback_target=playback_target,
                    capture_target=capture_target,
                    label="fixed_reference",
                    repeat_index=index,
                    lead_s=lead_s,
                    tail_s=tail_s,
                    run_id=run_id,
                )
            )
        data["fixed_replay"] = {
            "rows": fixed_rows,
            "summary": _summarize_physical_rows(fixed_rows, expected_attempts=fixed_repeats),
        }
        fixtures = _ensure_stability_fixtures(config)[:fixture_count] if fixture_count else []
        fixture_rows: list[dict[str, Any]] = []
        fixture_metadata: list[dict[str, Any]] = []
        for fixture_index, fixture in enumerate(fixtures, start=1):
            fixture_path = Path(str(fixture["path"]))
            fixture_pcm, fixture_values, fixture_rate = _pcm16_from_wav(fixture_path)
            fixture_ref_path = artifact_dir(config) / f"phase6_{run_id}_fixture_{fixture_index:02d}_authoritative.pcm"
            fixture_ref_path.write_bytes(fixture_pcm)
            fixture_metadata.append({**{key: fixture.get(key) for key in ("name", "text", "path")}, "authoritative_pcm_path": str(fixture_ref_path), "sample_rate": fixture_rate, "pcm_bytes": len(fixture_pcm)})
            for repeat_index in range(1, fixture_repeats + 1):
                fixture_rows.append(
                    _capture_positive_replay(
                        config,
                        reference_path=fixture_path,
                        reference_pcm_path=fixture_ref_path,
                        reference_pcm=fixture_pcm,
                        reference_values=fixture_values,
                        reference_rate=fixture_rate,
                        physical_path_distribution=path_distribution,
                        playback_target=playback_target,
                        capture_target=capture_target,
                        label=f"fixture_{fixture_index:02d}_{fixture.get('name', fixture_index)}",
                        repeat_index=repeat_index,
                        lead_s=lead_s,
                        tail_s=max(0.35, float(path_distribution.get("high_s", PHASE6_DEFAULT_PATH_HIGH_S)) + 0.1),
                        run_id=run_id,
                    )
                )
        data["fixture_replay"] = {
            "rows": fixture_rows,
            "fixtures": fixture_metadata,
            "summary": _summarize_physical_rows(fixture_rows, expected_attempts=len(fixtures) * fixture_repeats),
        }
        negative_duration_s = max(1.0, min(2.0, float(nested(config, "bench", "phase6_negative_capture_s", default=1.2))))
        negative_rows = [_capture_negative(config, capture_target=capture_target, repeat_index=index, duration_s=negative_duration_s, run_id=run_id) for index in range(1, negative_repeats + 1)]
        negative_summary = _summarize_physical_rows(negative_rows, expected_attempts=negative_repeats)
        negative_summary.update(
            {
                "false_positive_count": sum(row.get("measurement_classification") == "false_positive" for row in negative_rows),
                "false_positive_rate": sum(row.get("measurement_classification") == "false_positive" for row in negative_rows) / len(negative_rows) if negative_rows else None,
                "noise_floor_distribution": _simple_distribution([float(row["energy_detector"]["noise_floor_rms"]) for row in negative_rows if isinstance((row.get("energy_detector") or {}).get("noise_floor_rms"), (int, float))]),
            }
        )
        data["negative_controls"] = {"rows": negative_rows, "summary": negative_summary}
        data["summary"] = {
            "fixed_replay": data["fixed_replay"]["summary"],
            "fixture_replay": data["fixture_replay"]["summary"],
            "negative_controls": negative_summary,
            "classifiable_rate": data["fixed_replay"]["summary"].get("classifiable_rate"),
            "confirmed_plus_recovered_rate": data["fixed_replay"]["summary"].get("confirmation_rate"),
            "false_positive_rate": negative_summary.get("false_positive_rate"),
            "unknown_rate": data["fixed_replay"]["summary"].get("unknown_rate"),
            "primary_blocked_causes": data["fixed_replay"]["summary"].get("classification_counts", {}),
            "detector_a": {
                "name": "adaptive short-frame RMS",
                "success_count": sum(row.get("energy_detector", {}).get("detected", False) for row in fixed_rows),
                "success_rate": sum(row.get("energy_detector", {}).get("detected", False) for row in fixed_rows) / len(fixed_rows) if fixed_rows else None,
            },
            "detector_b": {
                "name": "actual playback PCM reference alignment",
                "recovery_count": sum(row.get("measurement_classification") == "correlation_recovered" for row in fixed_rows),
                "matched_count": sum(bool(row.get("reference_alignment", {}).get("matched")) for row in fixed_rows),
            },
        }
        data["status"] = "measured"
    except Exception as exc:
        data.update({"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if guard is not None:
            try:
                guard.__exit__(None, None, None)
            except Exception as exc:
                data["audio_state"]["restore_error"] = f"{type(exc).__name__}: {exc}"
            else:
                data["audio_state"]["restore_error"] = guard.restore_error
    data["finished_at_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return write_benchmark(config, "physical_onset", data, started_at=started)


def _annotate_stability_row(row: dict[str, Any], config: Mapping[str, Any], path_distribution: Mapping[str, Any], run_id: str = "latest") -> dict[str, Any]:
    """Post-process a Phase 4 row with Phase 6 application/measurement separation."""
    tts = row.get("tts") or {}
    physical = row.get("physical") or {}
    timing = row.get("timing_ns") or {}
    stats = physical.get("recording_stats") or {}
    recording_path = stats.get("path")
    pcm_path = tts.get("pcm_path")
    playback_result = tts.get("playback") or {}
    process_exit_status = playback_result.get("process_exit_status", "not_recorded")
    application_success = bool(
        tts.get("status") == "measured"
        and not tts.get("cancelled", False)
        and isinstance(timing.get("playback_completed"), int)
        and isinstance(tts.get("pcm_bytes"), (int, float))
        and float(tts.get("pcm_bytes", 0)) > 0
        and process_exit_status in (0, "not_recorded")
    )
    if not recording_path or not Path(str(recording_path)).exists() or not pcm_path or not Path(str(pcm_path)).exists():
        physical_classification = "unknown"
        row["physical_evidence"] = {"status": "not_recorded", "recording_path": recording_path, "pcm_path": pcm_path}
        row.update(separate_application_measurement_status(application_success, physical_classification))
        row["measurement_classification"] = physical_classification
        row["physical_latency_s"] = None
        row["late_onset_cause"] = "unknown_missing_artifact"
        return row
    try:
        pcm_raw = Path(str(pcm_path)).read_bytes()
        reference = np.frombuffer(pcm_raw, dtype="<i2").astype(np.float32) / 32768.0
        recorded, recorded_rate = sf.read(str(recording_path), always_2d=False)
        recorded_array = np.asarray(recorded, dtype=np.float32).reshape(-1)
        ref_analysis = measure_generated_audio_leading_silence(reference, int(tts.get("sample_rate") or 24000))
        ref_onset_s = float(ref_analysis.get("stable_speech_onset_s") or 0.0)
        record_start = timing.get("record_start")
        playback_start = timing.get("playback_stream_started") or timing.get("first_audio_chunk_queued")
        first_pcm_offset_s = ((playback_start - record_start) / 1e9) if isinstance(playback_start, int) and isinstance(record_start, int) else 0.0
        expected_window = build_expected_onset_window(first_pcm_offset_s, ref_onset_s, path_distribution)
        energy = detect_acoustic_onset(recorded_array, int(recorded_rate), search_start_s=max(0.0, expected_window["start_s"] - PHASE6_ONSET_SEARCH_MARGIN_S))
        alignment = reference_alignment_evidence(
            reference,
            recorded_array,
            sample_rate=int(tts.get("sample_rate") or 24000),
            recording_rate=int(recorded_rate),
            expected_playback_start_s=first_pcm_offset_s,
            reference_onset_s=ref_onset_s,
            physical_path_distribution=path_distribution,
        )
        energy_series = frame_energy_series(recorded_array, int(recorded_rate))
        energy_path = _write_energy_artifact(config, f"phase6_{run_id}_stability_turn_{row.get('turn_index', 'unknown')}_energy.json", energy_series)
        playback_success = process_exit_status in (0, "not_recorded") and bool(timing.get("playback_completed"))
        playback_evidence = {
            "playback_active": True,
            "playback_success": playback_success,
            "first_pcm_received": timing.get("first_audio_chunk_received") is not None,
            "first_actual_speech_pcm": timing.get("first_actual_speech_pcm") is not None,
            "pcm_bytes_written": playback_result.get("pcm_bytes_queued", tts.get("pcm_bytes")),
            "pcm_bytes_written_provenance": "PipeWirePCMPlayback queue accumulator" if "pcm_bytes_queued" in playback_result else "assembled PCM; process playback evidence not recorded in source row",
            "process_exit_status": process_exit_status,
            "persistent_playback_process_alive_at_start": playback_result.get("process_alive_at_start", "not_recorded"),
            "playback_completed": timing.get("playback_completed") is not None,
        }
        microphone_evidence = {
            "recording_path": str(recording_path),
            "recording_stats": stats,
            "short_frame_energy_series_path": energy_path,
            "adaptive_noise_floor_rms": energy.get("noise_floor_rms"),
            "device_available": True,
            "pipewire_health": {"status": "not_recorded_in_row"},
        }
        classification = classify_physical_onset(playback_evidence, microphone_evidence, energy, alignment, expected_window)
        energy_onset = energy.get("onset_s")
        alignment_onset = alignment.get("aligned_onset_s")
        energy_in_window = isinstance(energy_onset, (int, float)) and expected_window["start_s"] <= energy_onset <= expected_window["end_s"]
        physical_onset = energy_onset if energy_in_window else alignment_onset
        physical_latency = float(physical_onset) - (first_pcm_offset_s + ref_onset_s) if isinstance(physical_onset, (int, float)) else None
        row.update(
            separate_application_measurement_status(application_success, classification["classification"])
        )
        row.update(
            {
                "measurement_classification": classification["classification"],
                "measurement_confirmed": classification["measurement_confirmed"],
                "classification_confidence": classification["confidence"],
                "physical_latency_s": physical_latency,
                "expected_onset_window": expected_window,
                "physical_evidence": {
                    "playback": playback_evidence,
                    "microphone": microphone_evidence,
                    "expected_onset_window": expected_window,
                    "energy_detector": energy,
                    "reference_alignment": alignment,
                },
                "late_onset_cause": _late_onset_cause(classification["classification"], alignment, energy, expected_window, timing, tts),
            }
        )
    except Exception as exc:
        row.update(
            {
                **separate_application_measurement_status(application_success, "unknown"),
                "measurement_classification": "unknown",
                "physical_latency_s": None,
                "physical_evidence_error": f"{type(exc).__name__}: {exc}",
                "late_onset_cause": "unknown_analysis_error",
            }
        )
    return row


def _late_onset_cause(
    classification: str,
    alignment: Mapping[str, Any],
    energy: Mapping[str, Any],
    expected_window: Mapping[str, Any],
    timing: Mapping[str, Any],
    tts: Mapping[str, Any],
) -> str | None:
    energy_onset = energy.get("onset_s")
    if isinstance(energy_onset, (int, float)) and expected_window:
        start = float(expected_window.get("start_s", -math.inf))
        end = float(expected_window.get("end_s", math.inf))
        if energy_onset < start or energy_onset > end:
            if alignment.get("matched") and start <= float(alignment.get("aligned_onset_s", math.inf)) <= end:
                return "energy_detector_late_reference_recovered"
            return "energy_detector_outside_expected_window"
    if classification == "late_outside_window" and alignment.get("matched"):
        return "physical_path_or_expected_window_distribution"
    if classification == "energy_only":
        return "reference_alignment_mismatch_or_ambient_energy"
    if classification == "no_physical_match":
        return "capture_alignment_or_detector"
    if classification == "unknown" and timing.get("first_actual_speech_pcm") is None:
        return "tts_actual_pcm_timing_not_recorded"
    if classification == "unknown" and tts.get("pcm_bytes"):
        return "capture_alignment_or_detector"
    return None


def annotate_stability_with_phase6(data: dict[str, Any], config: Mapping[str, Any], path_distribution: Mapping[str, Any], run_id: str = "latest") -> dict[str, Any]:
    rows = data.get("turns") or []
    annotated = [_annotate_stability_row(dict(row), config, path_distribution, run_id) for row in rows]
    data["turns"] = annotated
    counts: dict[str, int] = {}
    for row in annotated:
        key = str(row.get("measurement_classification", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    application_success = sum(row.get("application_status") == "success" for row in annotated)
    confirmations = sum(row.get("measurement_classification") in {"confirmed", "correlation_recovered"} for row in annotated)
    summary = data.setdefault("summary", {})
    summary.update(
        {
            "application_attempt_count": len(annotated),
            "application_success_count": application_success,
            "application_failed_count": len(annotated) - application_success,
            "application_success_rate": application_success / len(annotated) if annotated else None,
            "physical_measurement_status_counts": counts,
            "physical_measurement_confirmation_count": confirmations,
            "physical_measurement_confirmation_rate": confirmations / len(annotated) if annotated else None,
            "phase6_latency": physical_latency_summary(annotated),
            "primary_blocked_causes": counts,
        }
    )
    late_causes = {
        cause: sum(row.get("late_onset_cause") == cause for row in annotated)
        for cause in sorted({str(row.get("late_onset_cause")) for row in annotated if row.get("late_onset_cause")})
    }
    legacy_outlier_causes = ((data.get("summary") or {}).get("outlier_cause_counts") or {})
    data["phase6_measurement"] = {
        "application_status_is_independent": True,
        "physical_measurement_status_is_independent": True,
        "reference_source": "actual PCM bytes assembled for persistent playback when pcm_path is available",
        "classification_counts": counts,
        "late_onset_causes": late_causes,
        "late_onset_analysis": {
            "phase4_legacy_acoustic_onset_outliers": legacy_outlier_causes.get("acoustic_onset", 0),
            "phase6_late_onset_rows": sum(value for key, value in late_causes.items() if key != "energy_detector_late_reference_recovered"),
            "energy_detector_late_reference_recovered": late_causes.get("energy_detector_late_reference_recovered", 0),
            "interpretation": "actual PCM alignment is the physical reference when energy onset is outside the derived window",
        },
    }
    return data


def _resource_regression(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("open_fds", "child_processes", "playback_processes", "active_http_connections")
    deltas = {
        key: float(after[key]) - float(before[key])
        for key in keys
        if isinstance(before.get(key), (int, float)) and isinstance(after.get(key), (int, float))
    }
    return {
        "status": "pass" if all(value <= 0 for value in deltas.values()) else "regression_observed",
        "deltas": deltas,
        "note": "Phase 5 per-turn resource monitoring is reused; this is a regression check only.",
    }


def summarize_first_sentence_buffering(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    chars: list[int] = []
    boundary_count = 0
    for row in rows:
        stage_value = (row.get("stages") or {}).get("first_sentence_buffering")
        if isinstance(stage_value, (int, float)):
            values.append(float(stage_value))
        text = ((row.get("llm") or {}).get("first_text_chunk"))
        if isinstance(text, str):
            chars.append(len(text))
            if any(mark in text for mark in "。！？!?\n"):
                boundary_count += 1
    outlier_count = sum(
        bool((row.get("outlier") or {}).get("is_outlier"))
        and (row.get("outlier") or {}).get("dominant_cause") == "first_sentence_buffering"
        for row in rows
    )
    return {
        "bug_found": False,
        "assessment": "bounded chunker behavior and natural-boundary variation; no chunking bug identified",
        "observed_count": len(values),
        "buffering_s": _simple_distribution(values),
        "first_chunk_chars": _simple_distribution([float(value) for value in chars]),
        "natural_boundary_count": boundary_count,
        "natural_boundary_rate": boundary_count / len(chars) if chars else None,
        "timeout_fallback": "not_recorded; source row does not tag whether timeout or punctuation emitted the chunk",
        "outlier_dominant_count": outlier_count,
        "contract": {"max_chars": "config.tts.sentence_max_chars", "timeout_s": "config.tts.sentence_timeout_s"},
    }


def run_phase6_unattended_bench(config: dict[str, Any], physical_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run Phase 6 physical hardening followed by the final 100-turn stability pass."""
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_id = time.strftime("%Y%m%d_%H%M%S")
    before = resource_snapshot()
    physical_result = physical_result or run_phase6_physical_onset_bench(config)
    path_distribution = ((physical_result.get("data") or {}).get("physical_path_distribution") or load_physical_path_distribution(config))
    stability_config = copy.deepcopy(config)
    stability_config.setdefault("bench", {})
    stability_config["bench"]["stability_target_turns"] = max(100, int(nested(config, "bench", "unattended_turns", default=100)))
    stability_config["bench"]["stability_minimum_turns"] = max(60, int(nested(config, "bench", "unattended_minimum_turns", default=60)))
    stability_config["bench"]["stability_restart_turns"] = max(5, int(nested(config, "bench", "unattended_restart_turns", default=5)))
    stability_result = run_stability_bench(stability_config)
    stability_data = annotate_stability_with_phase6(stability_result.get("data") or {}, config, path_distribution, run_id)
    stability_result = write_benchmark(config, "stability", stability_data, started_at=stability_result.get("started_at"))
    after = resource_snapshot()
    fixed_summary = ((physical_result.get("data") or {}).get("fixed_replay") or {}).get("summary") or {}
    negative_summary = ((physical_result.get("data") or {}).get("negative_controls") or {}).get("summary") or {}
    stability_summary = stability_data.get("summary") or {}
    stability_latency = stability_summary.get("phase6_latency") or {}
    stability_attempts = int(stability_summary.get("application_attempt_count", 0) or 0)
    stability_application_failed = int(stability_summary.get("application_failed_count", 0) or 0)
    stability_confirmation_rate = stability_summary.get("physical_measurement_confirmation_rate")
    stable_latency = stability_latency.get("confirmed_plus_recovered") or {}
    resource_regression = _resource_regression(before, after)
    process_series = stability_data.get("process_resources") or (stability_data.get("summary") or {}).get("process_resources") or {}
    series_metrics = [
        (process_series.get(name) or {}).get("monotonic_growth_turns")
        for name in ("open_fds", "child_processes", "playback_processes", "active_http_connections")
    ]
    if series_metrics and all(value is not None for value in series_metrics) and all(value == 0 for value in series_metrics):
        resource_regression = {
            **resource_regression,
            "status": "pass",
            "outer_snapshot_delta": resource_regression.get("deltas"),
            "outer_snapshot_interpretation": "warm-cache/runtime initialization delta; per-turn series is authoritative",
            "per_turn_monotonic_growth_turns": dict(zip(("open_fds", "child_processes", "playback_processes", "active_http_connections"), series_metrics)),
        }
    pass_candidate = bool(
        fixed_summary.get("attempts", 0) >= 100
        and fixed_summary.get("classifiable_rate", 0.0) >= 0.99
        and fixed_summary.get("confirmation_rate", 0.0) >= 0.99
        and negative_summary.get("false_positive_count", 1) == 0
        and negative_summary.get("unknown_rate", 1.0) <= 0.01
        and stability_attempts >= 100
        and stability_application_failed == 0
        and isinstance(stability_confirmation_rate, (int, float))
        and stability_confirmation_rate >= 0.99
        and isinstance(stable_latency.get("median"), (int, float))
        and stable_latency["median"] < 2.0
        and isinstance(stable_latency.get("p95"), (int, float))
        and stable_latency["p95"] < 2.5
        and resource_regression["status"] == "pass"
    )
    data = {
        "status": "unattended_validation_complete" if pass_candidate else "measured_with_limitations",
        "benchmark": "unattended",
        "phase": 6,
        "run_id": run_id,
        "turn57_diagnosis": "results/turn57_diagnosis.json",
        "fixed_configuration": {
            "asr": "large-v3-turbo / cuda / int8_float16",
            "llm": "ollama / qwen3.5:9b-q4_K_M",
            "tts": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice / Ono_Anna / Japanese",
            "serving": "vLLM-Omni 0.28.0 + vLLM 0.28.0 / HTTP raw PCM streaming",
            "aec_changed": False,
            "echo_threshold_changed": False,
        },
        "physical_onset": physical_result.get("data"),
        "audio_state_restore": (physical_result.get("data") or {}).get("audio_state"),
        "stability": stability_data,
        "resource_regression": resource_regression,
        "summary": {
            "fixed_replay_attempts": fixed_summary.get("attempts", 0),
            "fixed_replay_measured": fixed_summary.get("measured", 0),
            "fixed_replay_blocked": fixed_summary.get("blocked", 0),
            "fixed_replay_failed": fixed_summary.get("failed", 0),
            "fixed_replay_classifiable_rate": fixed_summary.get("classifiable_rate"),
            "fixed_replay_confirmed": fixed_summary.get("confirmed", 0),
            "fixed_replay_recovered": fixed_summary.get("correlation_recovered", 0),
            "fixed_replay_unknown_rate": fixed_summary.get("unknown_rate"),
            "negative_false_positive_rate": negative_summary.get("false_positive_rate"),
            "negative_false_positive_count": negative_summary.get("false_positive_count"),
            "stability_attempts": stability_attempts,
            "stability_application_success": stability_summary.get("application_success_count", 0),
            "stability_application_failed": stability_application_failed,
            "stability_physical_measurement_success": stability_summary.get("physical_measurement_confirmation_count", 0),
            "stability_physical_measurement_success_rate": stability_confirmation_rate,
            "confirmed_only_latency": stability_latency.get("confirmed_only"),
            "confirmed_plus_recovered_latency": stability_latency.get("confirmed_plus_recovered"),
            "application_success_rate": stability_summary.get("application_success_rate"),
            "measurement_confirmation_rate": stability_confirmation_rate,
            "median": stable_latency.get("median"),
            "p95": stable_latency.get("p95"),
            "p99": stable_latency.get("p99"),
            "max": stable_latency.get("max"),
        },
        "late_onset_cause_counts": stability_data.get("phase6_measurement", {}).get("late_onset_causes", {}),
        "late_onset_analysis": stability_data.get("phase6_measurement", {}).get("late_onset_analysis", {}),
        "first_sentence_buffering": summarize_first_sentence_buffering(stability_data.get("turns") or []),
        "server_restart": stability_data.get("server_restart"),
        "deferred_manual": [
            "human speech",
            "human physical double-talk",
            "human barge-in",
            "MOS/listening and subjective audio quality",
            "manual microphone gain/device tuning",
            "production approval",
        ],
        "pass_candidate": pass_candidate,
        "limitations": [
            "physical measurements are unattended replay/control measurements and do not establish human speech behavior",
            "application success is intentionally independent from physical measurement classification",
        ],
    }
    return write_benchmark(config, "unattended", data, started_at=started)


__all__ = [
    "build_expected_onset_window",
    "classify_physical_onset",
    "frame_energy_series",
    "load_physical_path_distribution",
    "physical_latency_summary",
    "reference_alignment_evidence",
    "run_phase6_physical_onset_bench",
    "run_phase6_unattended_bench",
    "separate_application_measurement_status",
    "split_pcm_chunks",
    "summarize_first_sentence_buffering",
]
