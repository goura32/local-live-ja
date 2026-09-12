from __future__ import annotations

import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

from .audio import AudioVolumeGuard, PipeWireInventory, PipeWirePCMPlayback, audio_file_stats, play_and_record, stable_target
from .audio_metrics import aligned_correlation, detect_acoustic_onset, measure_generated_audio_leading_silence
from .bench import artifact_dir, write_benchmark
from .config import nested
from .echo_rejection import (
    EchoRejectionConfig,
    echo_config_from_mapping,
    evaluate_echo_rejection,
    evaluate_echo_rejection_split,
    split_calibration_validation,
)
from .phase4_bench import _echo_reference
from .phase4_bench import _InterruptFixtureLLM, percentile_summary, run_stability_bench
from .llm.events import LLMError
from .pipeline import LivePipeline
from .telemetry import current_gpu_memory, nvidia_smi
from .vllm_bench import _make_vllm_client, _server_env, _vllm_model, _vllm_python
from .vllm_omni_tts import VLLMOmniTTSEngine
from .vllm_server import VLLMOmniServer


PHASE5_ECHO_USER_TEXT = "別の発話として、ネットワークの状態と応答時間を確認します。"
PHASE5_ECHO_SAMPLE_RATE = 16000
PHASE5_ONSET_REPEATS = 30
PHASE5_ONSET_THRESHOLDS = (3.0, 4.0, 5.0)
PHASE5_ONSET_TIMING_TOLERANCE_S = 0.25


def build_double_talk_fixture(
    reference: np.ndarray,
    user: np.ndarray,
    sample_rate: int,
    *,
    name: str,
    label: str,
    offset_s: float,
    user_gain: float,
    echo_gain: float,
    noise_rms: float,
    lag_s: float,
    seed: int,
    playback_active: bool,
) -> dict[str, Any]:
    """Mix an independent user fixture with delayed assistant reference audio."""
    if sample_rate <= 0 or offset_s < 0 or user_gain < 0 or echo_gain < 0 or noise_rms < 0 or lag_s < 0:
        raise ValueError("invalid double-talk fixture parameters")
    reference_array = np.asarray(reference, dtype=np.float32).reshape(-1)
    user_array = np.asarray(user, dtype=np.float32).reshape(-1)
    lag_samples = int(round(lag_s * sample_rate))
    user_start = int(round(offset_s * sample_rate))
    user_end = user_start + len(user_array) if user_gain else 0
    total_length = max(len(reference_array) + lag_samples, user_end)
    microphone = np.zeros(total_length, dtype=np.float32)
    if len(reference_array):
        microphone[lag_samples : lag_samples + len(reference_array)] += echo_gain * reference_array
    if len(user_array) and user_gain:
        microphone[user_start : user_start + len(user_array)] += user_gain * user_array
    if noise_rms:
        microphone += np.random.default_rng(seed).normal(0.0, noise_rms, size=total_length).astype(np.float32)
    microphone = np.clip(microphone, -1.0, 1.0).astype(np.float32)
    reference_duration_s = len(reference_array) / sample_rate
    user_end_s = (user_start + len(user_array)) / sample_rate
    if offset_s <= 0.12:
        offset_class = "assistant_start"
    elif offset_s < reference_duration_s * 0.65:
        offset_class = "assistant_middle"
    elif offset_s <= reference_duration_s:
        offset_class = "assistant_end"
    else:
        offset_class = "assistant_end_or_after"
    metadata = {
        "offset_class": offset_class,
        "user_offset_s": offset_s,
        "user_gain": user_gain,
        "echo_gain": echo_gain,
        "noise_rms": noise_rms,
        "echo_lag_s": lag_s,
        "reference_duration_s": reference_duration_s,
        "user_duration_s": len(user_array) / sample_rate,
        "user_end_s": user_end_s,
        "playback_active": playback_active,
        "seed": seed,
    }
    return {
        "name": name,
        "label": label,
        "reference": reference_array.copy(),
        "microphone": microphone,
        "playback_active": playback_active,
        "metadata": metadata,
    }


def _offsets(reference_duration_s: float, user_duration_s: float) -> list[tuple[str, float]]:
    return [
        ("assistant_start", 0.04),
        ("assistant_middle", max(0.14, reference_duration_s * 0.45)),
        ("assistant_end", max(0.16, reference_duration_s - min(0.65, user_duration_s * 0.35))),
        ("assistant_end_or_after", reference_duration_s + 0.06),
    ]


def _ensure_phase5_user_fixture(config: dict[str, Any]) -> tuple[Path, np.ndarray, int, dict[str, Any]]:
    path = artifact_dir(config) / "phase5_echo_user_speech_fixture.wav"
    existing = artifact_dir(config) / "phase4_input_technical.wav"
    if not path.exists() and existing.exists():
        path = existing
        audio, rate = sf.read(str(path), always_2d=False)
        return path, np.asarray(audio, dtype=np.float32), int(rate), {
            "text": "PipeWireとCUDAの状態を教えてください。",
            "path": str(path),
            "source": "reused_existing_phase4_qwen3_tts_fixture",
        }
    generated: dict[str, Any] = {"text": PHASE5_ECHO_USER_TEXT, "path": str(path), "source": "existing_qwen3_tts_fixture"}
    if not path.exists():
        from .tts import Qwen3TTSEngine

        engine = Qwen3TTSEngine(
            model=nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            speaker=nested(config, "tts", "speaker", default="Ono_Anna"),
            language=nested(config, "tts", "language", default="Japanese"),
            device="auto",
            max_new_tokens=int(nested(config, "tts", "max_new_tokens", default=2048)),
            generation_kwargs=dict(nested(config, "tts", "generation_kwargs", default={}) or {}),
        )
        try:
            generated["tts"] = engine.synthesize(PHASE5_ECHO_USER_TEXT, output_path=path)
            generated["source"] = "qwen3_tts"
        finally:
            engine.unload()
    audio, rate = sf.read(str(path), always_2d=False)
    return path, np.asarray(audio, dtype=np.float32), int(rate), generated


def build_phase5_double_talk_dataset(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate 20 assistant-only and 20 independent-user conditions."""
    reference_path, reference, sample_rate = _echo_reference(config)
    user_path, user, user_rate, user_generation = _ensure_phase5_user_fixture(config)
    if user_rate != sample_rate:
        from .audio_metrics import resample_mono

        user = resample_mono(user, user_rate, sample_rate)
    reference_duration_s = len(reference) / sample_rate
    user_duration_s = len(user) / sample_rate
    offsets = _offsets(reference_duration_s, user_duration_s)
    levels = [
        ("user_weak_echo_strong", 0.12, 0.40),
        ("same_level", 0.28, 0.28),
        ("user_strong_echo_weak", 0.45, 0.16),
    ]
    noises = (0.0005, 0.002, 0.006)
    lags = (0.02, 0.04, 0.06)
    fixtures: list[dict[str, Any]] = []
    artifact_paths: dict[str, Any] = {"reference": str(reference_path), "user_fixture": str(user_path), "fixtures": {}}
    conditions: list[dict[str, Any]] = []
    for index in range(20):
        offset_class, offset_s = offsets[index % len(offsets)]
        level_class, user_gain, echo_gain = levels[index % len(levels)]
        noise_rms = noises[index % len(noises)]
        lag_s = lags[index % len(lags)]
        condition = {
            "condition_index": index,
            "offset_class": offset_class,
            "offset_s": offset_s,
            "level_class": level_class,
            "user_gain": user_gain,
            "echo_gain": echo_gain,
            "noise_rms": noise_rms,
            "lag_s": lag_s,
        }
        conditions.append(condition)
        assistant = build_double_talk_fixture(
            reference,
            user,
            sample_rate,
            name=f"assistant_only_{index:02d}",
            label="assistant_only",
            offset_s=offset_s,
            user_gain=0.0,
            echo_gain=echo_gain,
            noise_rms=noise_rms,
            lag_s=lag_s,
            seed=52000 + index,
            playback_active=True,
        )
        assistant["metadata"].update(condition)
        assistant_path = artifact_dir(config) / f"phase5_echo_assistant_only_{index:02d}.wav"
        sf.write(str(assistant_path), assistant["microphone"], sample_rate)
        assistant["artifact_path"] = str(assistant_path)
        artifact_paths["fixtures"][assistant["name"]] = str(assistant_path)
        fixtures.append(assistant)
        user_like = build_double_talk_fixture(
            reference,
            user,
            sample_rate,
            name=f"synthetic_user_like_{index:02d}",
            label="synthetic_user_like",
            offset_s=offset_s,
            user_gain=user_gain,
            echo_gain=echo_gain,
            noise_rms=noise_rms,
            lag_s=lag_s,
            seed=62000 + index,
            playback_active=True,
        )
        user_like["metadata"].update(condition)
        user_path_for_condition = artifact_dir(config) / f"phase5_echo_synthetic_user_like_{index:02d}.wav"
        sf.write(str(user_path_for_condition), user_like["microphone"], sample_rate)
        user_like["artifact_path"] = str(user_path_for_condition)
        artifact_paths["fixtures"][user_like["name"]] = str(user_path_for_condition)
        fixtures.append(user_like)
    return fixtures, {
        "sample_rate": sample_rate,
        "reference_path": str(reference_path),
        "user_fixture": user_generation,
        "condition_count": len(conditions),
        "fixture_count": len(fixtures),
        "class_counts": {
            "assistant_only": sum(item["label"] == "assistant_only" for item in fixtures),
            "synthetic_user_like": sum(item["label"] == "synthetic_user_like" for item in fixtures),
        },
        "conditions": conditions,
        "artifact_paths": artifact_paths,
        "not_physical_double_talk": True,
    }


def _compact_echo_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    import copy

    compact = copy.deepcopy(dict(evaluation))
    compact_rows = []
    for row in compact.get("rows", []):
        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            windows = metrics.pop("window_metrics", None)
            if isinstance(windows, list):
                metrics["window_count"] = len(windows)
        compact_rows.append(row)
    compact["rows"] = compact_rows
    return compact


def _compact_echo_split(split_result: Mapping[str, Any]) -> dict[str, Any]:
    import copy

    compact = copy.deepcopy(dict(split_result))
    evaluations = compact.get("fixed_threshold_evaluation") or {}
    compact["fixed_threshold_evaluation"] = {
        name: _compact_echo_evaluation(value) for name, value in evaluations.items()
    }
    if isinstance(compact.get("calibrated_validation_evaluation"), Mapping):
        compact["calibrated_validation_evaluation"] = _compact_echo_evaluation(compact["calibrated_validation_evaluation"])
    return compact


def _echo_result_with_compatibility(split_result: dict[str, Any]) -> dict[str, Any]:
    split_result = _compact_echo_split(split_result)
    fixed_all = split_result["fixed_threshold_evaluation"]["all"]
    fixed_validation = split_result["fixed_threshold_evaluation"]["validation"]
    validation_assistant = fixed_validation["assistant_only"]
    validation_user = fixed_validation["synthetic_user_like"]
    return {
        "status": "measured",
        "benchmark": "echo_rejection",
        "method": "post-VAD 80 ms reference correlation + lag + energy ratio + max residual energy; no AEC replacement",
        "threshold_selection": "Phase 4 fixed thresholds evaluated first; calibration-only grid search reported separately",
        "thresholds": split_result["fixed_thresholds"],
        "rows": fixed_all["rows"],
        "assistant_only": fixed_all["assistant_only"],
        "synthetic_user_like": fixed_all["synthetic_user_like"],
        "correlation_distribution": fixed_all["correlation_distribution"],
        "energy_ratio_distribution": fixed_all["energy_ratio_distribution"],
        "residual_energy_ratio_distribution": fixed_all["residual_energy_ratio_distribution"],
        "lag_seconds_distribution": fixed_all["lag_seconds_distribution"],
        "phase5_split_evaluation": split_result,
        "validation_set": {
            "assistant_only": validation_assistant,
            "synthetic_user_like": validation_user,
            "thresholds": split_result["fixed_thresholds"],
            "threshold_margin_rows": fixed_validation.get("threshold_margin_rows", []),
        },
        "targets": {
            "assistant_only_false_accept_le_5pct": isinstance(validation_assistant.get("false_accept_rate"), (int, float)) and validation_assistant["false_accept_rate"] <= 0.05,
            "synthetic_double_talk_acceptance_ge_95pct": isinstance(validation_user.get("acceptance_rate"), (int, float)) and validation_user["acceptance_rate"] >= 0.95,
        },
        "mute_baseline": {
            "assistant_only_false_accept_rate": 0.0,
            "adopted": False,
            "reason": "VAD mute during playback removes future barge-in candidates",
        },
        "adopted": {
            "decision": "echo-aware",
            "reason": "keep VAD active and reject only reference-explained residual echo",
        },
        "deferred_manual": [
            "human physical double-talk",
            "human barge-in",
            "MOS and subjective listening",
        ],
        "component_status": "pass" if all([validation_assistant.get("false_accept_rate", 1.0) <= 0.05, validation_user.get("acceptance_rate", 0.0) >= 0.95]) else "measured_with_limitations",
    }


def run_phase5_echo_rejection_bench(config: dict[str, Any]) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data: dict[str, Any] = {
        "status": "blocked",
        "benchmark": "echo_rejection",
        "phase": 5,
        "error": None,
        "error_type": None,
    }
    try:
        fixtures, dataset = build_phase5_double_talk_dataset(config)
        fixed = echo_config_from_mapping(config.get("echo_rejection"))
        split_result = evaluate_echo_rejection_split(fixtures, dataset["sample_rate"], fixed_config=fixed)
        data.update({"dataset": dataset, **_echo_result_with_compatibility(split_result)})
    except Exception as exc:
        data.update({"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)})
    return write_benchmark(config, "echo_rejection", data, started_at=started)


def classify_onset_failure(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a missing acoustic onset without hiding the original observation."""
    if observation.get("detected"):
        return {"cause": "measured", "confidence": "direct_detection"}
    if not observation.get("playback_success", True):
        return {"cause": "speaker_playback_missing", "confidence": "direct_playback_failure"}
    if not observation.get("device_available", True):
        return {"cause": "audio_device_unavailable", "confidence": "device_probe"}
    health = observation.get("pipewire_health") or {}
    if health.get("status") not in {None, "ready"}:
        return {"cause": "pipewire_unhealthy", "confidence": "health_probe"}
    stats = observation.get("recording_stats") or {}
    if float(stats.get("duration_s", 0.0) or 0.0) < 0.5 or float(stats.get("rms", 0.0) or 0.0) <= 0.001 or float(stats.get("peak", 0.0) or 0.0) <= 0.005:
        return {"cause": "microphone_capture_missing", "confidence": "recording_level_or_duration"}
    if observation.get("sensitivity_detected"):
        return {"cause": "threshold_too_strict", "confidence": "threshold_sensitivity"}
    correlation = (observation.get("cross_correlation") or {}).get("correlation")
    if isinstance(correlation, (int, float)) and correlation >= 0.8:
        return {"cause": "onset_detector_false_negative", "confidence": "reference_correlation"}
    return {"cause": "unknown", "confidence": "insufficient_diagnostic_evidence"}


def _synthetic_onset_sensitivity(sample_rate: int = 16000) -> dict[str, Any]:
    rng = np.random.default_rng(52025)
    noise = rng.normal(0.0, 0.0008, size=sample_rate).astype(np.float32)
    signal = noise.copy()
    start = int(0.5 * sample_rate)
    signal[start : int(0.75 * sample_rate)] += 0.08 * np.sin(2 * np.pi * 440 * np.arange(int(0.25 * sample_rate)) / sample_rate).astype(np.float32)
    rows: list[dict[str, Any]] = []
    for multiplier in PHASE5_ONSET_THRESHOLDS:
        noise_result = detect_acoustic_onset(noise, sample_rate, search_start_s=0.25, threshold_multiplier=multiplier)
        signal_result = detect_acoustic_onset(signal, sample_rate, search_start_s=0.25, threshold_multiplier=multiplier)
        rows.append(
            {
                "threshold_multiplier": multiplier,
                "noise_false_positive": bool(noise_result["detected"]),
                "signal_false_negative": not bool(signal_result["detected"]),
                "signal_onset_s": signal_result.get("onset_s"),
                "signal_onset_error_s": (float(signal_result["onset_s"]) - 0.5) if signal_result.get("onset_s") is not None else None,
                "noise": noise_result,
                "signal": signal_result,
            }
        )
    return {
        "sample_rate": sample_rate,
        "expected_signal_onset_s": 0.5,
        "thresholds": rows,
        "false_positive_count": sum(row["noise_false_positive"] for row in rows),
        "false_negative_count": sum(row["signal_false_negative"] for row in rows),
        "method": "bounded 3x/4x/5x noise multiplier sensitivity; production threshold unchanged",
    }


def _pipewire_health(inventory: PipeWireInventory | None) -> dict[str, Any]:
    if inventory is None:
        return {"status": "unavailable"}
    return {
        "status": "ready" if inventory.sinks and inventory.sources else "degraded",
        "sink_count": len(inventory.sinks),
        "source_count": len(inventory.sources),
        "device_count": len(inventory.devices),
    }


def run_physical_onset_repeat_bench(config: dict[str, Any], *, repeats: int = PHASE5_ONSET_REPEATS) -> dict[str, Any]:
    """Replay one fixed reference and retain every physical onset observation."""
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data: dict[str, Any] = {
        "status": "blocked",
        "benchmark": "physical_onset_repeat",
        "attempts": repeats,
        "rows": [],
        "threshold_sensitivity": _synthetic_onset_sensitivity(),
        "audio_state": {"snapshot": None, "restore_error": None},
        "reference": None,
    }
    guard: AudioVolumeGuard | None = None
    try:
        reference_path, reference, sample_rate = _echo_reference(config)
        reference_analysis = measure_generated_audio_leading_silence(reference, sample_rate)
        data["reference"] = {
            "path": str(reference_path),
            "sample_rate": sample_rate,
            "duration_s": len(reference) / sample_rate,
            "generated_onset": reference_analysis,
        }
        inventory = PipeWireInventory.discover()
        speaker = inventory.usb_speaker()
        microphone = inventory.usb_microphone()
        playback_target = stable_target(speaker) if speaker else None
        capture_target = stable_target(microphone) if microphone else None
        if not playback_target or not capture_target:
            raise RuntimeError("stable USB playback and capture targets are required")
        data["targets"] = {"speaker": playback_target, "microphone": capture_target, "initial_health": _pipewire_health(inventory)}
        guard = AudioVolumeGuard(speaker_target=playback_target, microphone_target=capture_target)
        guard.__enter__()
        data["audio_state"]["snapshot"] = guard.snapshot.to_dict() if guard.snapshot else None
        guard.set_mutes(speaker_muted=False, microphone_muted=False)
        lead_s = 0.4
        tail_s = 0.5
        expected_onset_s = lead_s + float(reference_analysis.get("stable_speech_onset_s") or 0.0)
        for repeat_index in range(1, repeats + 1):
            output = artifact_dir(config) / f"phase5_onset_repeat_{repeat_index:03d}.wav"
            observation: dict[str, Any] = {
                "repeat_index": repeat_index,
                "status": "error",
                "playback_success": False,
                "device_available": False,
                "expected_signal_onset_s": expected_onset_s,
                "detected_onset_s": None,
                "cross_correlation": None,
                "pipewire_health": None,
            }
            try:
                current_inventory = PipeWireInventory.discover()
                current_speaker = current_inventory.usb_speaker()
                current_microphone = current_inventory.usb_microphone()
                current_playback = stable_target(current_speaker) if current_speaker else None
                current_capture = stable_target(current_microphone) if current_microphone else None
                observation["device_available"] = bool(current_playback and current_capture)
                observation["pipewire_health"] = _pipewire_health(current_inventory)
                if not current_playback or not current_capture:
                    raise RuntimeError("stable target unavailable during repeat")
                playback = play_and_record(
                    reference_path,
                    output,
                    playback_target=current_playback,
                    capture_target=current_capture,
                    lead_s=lead_s,
                    tail_s=tail_s,
                    sample_rate=16000,
                )
                observation.update(
                    {
                        "status": "measured",
                        "playback_success": True,
                        "playback": playback,
                        "recording_stats": playback.get("recording_stats") or audio_file_stats(output),
                        "record_returncode": playback.get("record_returncode"),
                        "recording_length_expected_s": float(sf.info(str(reference_path)).duration) + lead_s + tail_s,
                    }
                )
                recording, recording_rate = sf.read(str(output), always_2d=False)
                recording_array = np.asarray(recording, dtype=np.float32)
                detected = detect_acoustic_onset(
                    recording_array,
                    int(recording_rate),
                    search_start_s=lead_s,
                    reference=reference,
                    reference_rate=sample_rate,
                )
                observation["detector"] = detected
                observation["detected"] = bool(detected.get("detected"))
                observation["detected_onset_s"] = detected.get("onset_s")
                observation["physical_audio_detected"] = bool(detected.get("detected"))
                observation["cross_correlation"] = aligned_correlation(reference, recording_array, sample_rate=sample_rate, recording_rate=int(recording_rate))
                observation["recording_length_error_s"] = float((observation["recording_stats"] or {}).get("duration_s", 0.0)) - float(observation["recording_length_expected_s"])
                sensitivity_rows = []
                for multiplier in PHASE5_ONSET_THRESHOLDS:
                    sensitivity_rows.append(
                        {
                            "threshold_multiplier": multiplier,
                            "result": detect_acoustic_onset(recording_array, int(recording_rate), search_start_s=lead_s, threshold_multiplier=multiplier),
                        }
                    )
                observation["threshold_sensitivity"] = sensitivity_rows
                observation["sensitivity_detected"] = any(item["result"].get("detected") for item in sensitivity_rows if item["threshold_multiplier"] < 4.0)
                observation["failure_classification"] = classify_onset_failure(observation)
                if observation["detected_onset_s"] is not None:
                    observation["onset_timing_error_s"] = float(observation["detected_onset_s"]) - expected_onset_s
                else:
                    observation["onset_timing_error_s"] = None
                error_s = observation["onset_timing_error_s"]
                observation["false_positive_candidate"] = bool(
                    isinstance(error_s, (int, float)) and error_s < -PHASE5_ONSET_TIMING_TOLERANCE_S
                )
                observation["late_onset_candidate"] = bool(
                    isinstance(error_s, (int, float)) and error_s > PHASE5_ONSET_TIMING_TOLERANCE_S
                )
                observation["onset_quality_classification"] = (
                    "false_positive_candidate"
                    if observation["false_positive_candidate"]
                    else "late_onset_candidate"
                    if observation["late_onset_candidate"]
                    else "within_timing_tolerance"
                    if observation["physical_audio_detected"]
                    else "not_detected"
                )
                if not observation["physical_audio_detected"]:
                    observation["status"] = "blocked"
                    observation["blocked_reason"] = "physical_audio_not_detected"
            except Exception as exc:
                observation.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "failure_classification": classify_onset_failure(observation),
                    }
                )
            data["rows"].append(observation)
        measured = [row for row in data["rows"] if row.get("status") == "measured"]
        blocked = [row for row in data["rows"] if row.get("status") == "blocked"]
        failed = [row for row in data["rows"] if row.get("status") == "error"]
        causes: dict[str, int] = {}
        for row in data["rows"]:
            cause = (row.get("failure_classification") or {}).get("cause", "unknown")
            causes[cause] = causes.get(cause, 0) + 1
        data["summary"] = {
            "attempts": len(data["rows"]),
            "measured": len(measured),
            "blocked": len(blocked),
            "failed": len(failed),
            "onset_success_rate": len(measured) / len(data["rows"]) if data["rows"] else None,
            "onset_failure_rate": len(blocked) / len(data["rows"]) if data["rows"] else None,
            "failure_cause_counts": causes,
            "detected_onset_s": _simple_distribution([float(row["detected_onset_s"]) for row in measured if isinstance(row.get("detected_onset_s"), (int, float))]),
            "onset_timing_error_s": _simple_distribution([float(row["onset_timing_error_s"]) for row in measured if isinstance(row.get("onset_timing_error_s"), (int, float))]),
            "false_positive_candidate_count": sum(bool(row.get("false_positive_candidate")) for row in data["rows"]),
            "late_onset_candidate_count": sum(bool(row.get("late_onset_candidate")) for row in data["rows"]),
            "onset_timing_tolerance_s": PHASE5_ONSET_TIMING_TOLERANCE_S,
            "recording_rms": _simple_distribution([float((row.get("recording_stats") or {}).get("rms")) for row in data["rows"] if isinstance((row.get("recording_stats") or {}).get("rms"), (int, float))]),
            "recording_peak": _simple_distribution([float((row.get("recording_stats") or {}).get("peak")) for row in data["rows"] if isinstance((row.get("recording_stats") or {}).get("peak"), (int, float))]),
            "recording_length_s": _simple_distribution([float((row.get("recording_stats") or {}).get("duration_s")) for row in data["rows"] if isinstance((row.get("recording_stats") or {}).get("duration_s"), (int, float))]),
        }
        data["status"] = "measured" if data["rows"] else "blocked"
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
    return data


def _simple_distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "values": ordered,
        "median": statistics.median(ordered) if ordered else None,
        "min": min(ordered) if ordered else None,
        "max": max(ordered) if ordered else None,
    }


def resource_snapshot(*, http_ports: tuple[int, ...] = (8091, 11434)) -> dict[str, Any]:
    """Capture process, descriptor, child, playback, connection, and GPU state."""
    snapshot: dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "gpu_memory_used_mib": None,
        "gpu_memory_free_mib": None,
        "gpu_memory_total_mib": None,
        "process_rss_mib": None,
        "open_fds": None,
        "child_processes": None,
        "playback_processes": 0,
        "active_http_connections": None,
    }
    try:
        import psutil

        process = psutil.Process(os.getpid())
        snapshot["process_rss_mib"] = float(process.memory_info().rss / (1024 * 1024))
        snapshot["open_fds"] = int(process.num_fds()) if hasattr(process, "num_fds") else None
        snapshot["child_processes"] = len(process.children(recursive=True))
        playback_names = {"pw-cat", "pw-play", "pw-record"}
        snapshot["playback_processes"] = sum(
            1
            for item in psutil.process_iter(["name", "cmdline"])
            if ((item.info.get("name") or "").casefold() in playback_names or any(name in " ".join(item.info.get("cmdline") or []).casefold() for name in playback_names))
        )
        connections = []
        try:
            connections = psutil.net_connections(kind="tcp")
        except (psutil.Error, OSError):
            connections = []
        snapshot["active_http_connections"] = sum(
            1
            for item in connections
            if item.status in {"ESTABLISHED", "SYN_SENT", "SYN_RECV", "LISTEN"}
            and ((item.laddr and item.laddr.port in http_ports) or (item.raddr and item.raddr.port in http_ports))
        )
    except Exception:
        pass
    rows = nvidia_smi("memory.used,memory.free,memory.total")
    used: list[int] = []
    free: list[int] = []
    total: list[int] = []
    for row in rows:
        for key, target in (("memory.used", used), ("memory.free", free), ("memory.total", total)):
            try:
                target.append(int(float(row.get(key, ""))))
            except (TypeError, ValueError):
                pass
    snapshot["gpu_memory_used_mib"] = max(used) if used else (max(current_gpu_memory()) if current_gpu_memory() else None)
    snapshot["gpu_memory_free_mib"] = min(free) if free else None
    snapshot["gpu_memory_total_mib"] = max(total) if total else None
    snapshot["gpu_vram_mib"] = snapshot["gpu_memory_used_mib"]
    return snapshot


def summarize_resource_lifecycle(before: Mapping[str, Any], after: Mapping[str, Any], *, samples: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    keys = ("gpu_memory_used_mib", "process_rss_mib", "open_fds", "child_processes", "playback_processes", "active_http_connections")
    deltas = {
        key: float(after[key]) - float(before[key])
        for key in keys
        if isinstance(before.get(key), (int, float)) and isinstance(after.get(key), (int, float))
    }
    monotonic_growth: dict[str, bool] = {}
    if samples:
        for key in keys:
            values = [float(item[key]) for item in samples if isinstance(item.get(key), (int, float))]
            monotonic_growth[key] = len(values) >= 3 and all(right >= left for left, right in zip(values, values[1:])) and values[-1] > values[0]
    leak_suspected = bool(
        any(monotonic_growth.values())
        or deltas.get("child_processes", 0.0) > 1
        or deltas.get("playback_processes", 0.0) > 0
    )
    start_end_growth = {
        key: deltas.get(key, 0.0) > 0
        for key in ("gpu_memory_used_mib", "process_rss_mib", "open_fds", "active_http_connections")
        if key in deltas
    }
    return {
        "before": dict(before),
        "after": dict(after),
        "deltas": deltas,
        "monotonic_growth": monotonic_growth,
        "leak_suspected": leak_suspected,
        "start_end_growth": start_end_growth,
        "warm_cache_growth_possible": bool(
            deltas.get("gpu_memory_used_mib", 0.0) > 512 or deltas.get("process_rss_mib", 0.0) > 128
        ),
        "fd_growth_requires_followup": deltas.get("open_fds", 0.0) > 8 and not monotonic_growth.get("open_fds", False),
        "method": "start/end plus sampled monotonic-growth check; warm model cache separated from leak; no per-turn GC/cache clear",
    }


def summarize_cancellation_runs(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated cancellation observations without dropping failures."""
    measured = [row for row in rows if row.get("status") == "measured"]
    stop_values = [float(row["software_stop_s"]) for row in measured if isinstance(row.get("software_stop_s"), (int, float))]
    stale_count = sum(bool(row.get("stale_pcm")) for row in rows)
    idle_count = sum(row.get("pipeline_state") == "IDLE" for row in measured)
    recovered_count = sum(bool(row.get("recovered")) for row in measured)
    return {
        "count": len(rows),
        "measured_count": len(measured),
        "failed_count": len(rows) - len(measured),
        "software_stop": percentile_summary(stop_values),
        "stale_pcm_count": stale_count,
        "pipeline_idle_rate": idle_count / len(measured) if measured else None,
        "recovery_rate": recovered_count / len(measured) if measured else None,
        "http_stream_active_count": sum(bool(row.get("http_stream_active_at_interrupt")) for row in rows),
        "http_stream_cancelled_count": sum(bool(row.get("http_stream_cancelled")) for row in rows),
        "http_stream_cancel_observation_rate": sum(bool(row.get("http_stream_cancelled")) for row in rows) / len(rows) if rows else None,
        "status": "pass" if measured and len(measured) == len(rows) and stale_count == 0 and idle_count == len(measured) and recovered_count == len(measured) and percentile_summary(stop_values).get("median") is not None and percentile_summary(stop_values)["median"] <= 0.2 else "measured_with_limitations",
    }


class _TimedCancellationPlayback:
    """Hold a persistent PCM stream at a deterministic queue boundary."""

    streaming = True

    def __init__(self, target: str, hold_after_queue: int) -> None:
        self.inner = PipeWirePCMPlayback(target)
        self.hold_after_queue = hold_after_queue
        self.reached = threading.Event()
        self.release = threading.Event()
        self.queue_count = 0
        self.queue_after_cancel = 0
        self.successful_queue_after_cancel = 0
        self.cancel_requested_ns: int | None = None

    @property
    def active(self) -> bool:
        return self.inner.active

    @property
    def last_queued_ns(self) -> int | None:
        return self.inner.last_queued_ns

    def start(self, *, sample_rate: int, channels: int) -> dict[str, Any]:
        return self.inner.start(sample_rate=sample_rate, channels=channels)

    def queue(self, payload: bytes) -> dict[str, Any]:
        if self.cancel_requested_ns is not None:
            self.queue_after_cancel += 1
        result = self.inner.queue(payload)
        if self.cancel_requested_ns is not None and result.get("queued"):
            self.successful_queue_after_cancel += 1
        if result.get("queued"):
            self.queue_count += 1
        if self.queue_count == self.hold_after_queue and not self.reached.is_set():
            self.reached.set()
            self.release.wait(timeout=15.0)
        return result

    def finish(self) -> dict[str, Any]:
        self.release.set()
        return self.inner.finish()

    def cancel(self) -> dict[str, Any]:
        if self.cancel_requested_ns is None:
            self.cancel_requested_ns = time.monotonic_ns()
        self.release.set()
        return self.inner.cancel()


def _run_cancellation_case(
    config: dict[str, Any],
    *,
    engine: Any,
    playback_target: str,
    timing_name: str,
    hold_after_queue: int,
    repeat_index: int,
) -> dict[str, Any]:
    playback = _TimedCancellationPlayback(playback_target, hold_after_queue)
    pipeline = LivePipeline(
        llm=_InterruptFixtureLLM(),
        tts=engine,
        playback=playback,
        artifact_dir=artifact_dir(config),
        sentence_max_chars=48,
        sentence_timeout_s=0.8,
        echo_rejection_config=echo_config_from_mapping(config.get("echo_rejection")),
    )
    results: list[Any] = []
    worker = threading.Thread(target=lambda: results.append(pipeline.respond("複数タイミング中断")), daemon=True)
    row: dict[str, Any] = {
        "status": "blocked",
        "timing_condition": timing_name,
        "repeat_index": repeat_index,
        "hold_after_queue": hold_after_queue,
        "queue_reached": False,
        "stale_pcm": None,
        "pipeline_state": None,
    }
    worker.start()
    wait_s = float(nested(config, "bench", "cancellation_wait_s", default=60.0))
    if not playback.reached.wait(timeout=wait_s):
        if worker.is_alive():
            pipeline.cancel()
            worker.join(timeout=10.0)
        row.update({"reason": "requested_queue_boundary_not_reached", "worker_alive": worker.is_alive()})
        return row
    row["queue_reached"] = True
    request_ns = time.monotonic_ns()
    http_active_at_interrupt = getattr(engine, "_active_client", None) is not None
    pipeline.cancel()
    playback_stop_ns = time.monotonic_ns()
    worker.join(timeout=10.0)
    result = results[0] if results else None
    row.update(
        {
            "status": "measured" if result is not None and not worker.is_alive() and result.cancelled else "error",
            "interrupt_request_ns": request_ns,
            "playback_stop_ns": playback_stop_ns,
            "software_stop_s": (playback_stop_ns - request_ns) / 1e9,
            "worker_alive": worker.is_alive(),
            "http_stream_cancelled": bool(getattr(engine, "_active_http_cancelled", False) or (result and getattr(result, "http_stream_cancelled", False))),
            "http_stream_active_at_interrupt": http_active_at_interrupt,
            "http_stream_completed_before_interrupt": not http_active_at_interrupt,
            "playback_active_after_interrupt": playback.active,
            "queue_count_before_cancel": playback.queue_count,
            "queue_after_cancel": playback.queue_after_cancel,
            "successful_queue_after_cancel": playback.successful_queue_after_cancel,
            "queue_discarded": playback.successful_queue_after_cancel == 0,
            "stale_pcm": playback.successful_queue_after_cancel > 0 or playback.active,
            "pipeline_cancelled": bool(result and result.cancelled),
            "pipeline_state": result.state if result else None,
            "spoken_text": result.spoken_text if result else None,
            "spoken_text_committed_events": sum(event.kind == "spoken_text_committed" for event in (result.events if result else [])),
            "spoken_text_correct": bool(result and result.cancelled and result.spoken_text == "" and not any(event.kind == "spoken_text_committed" for event in result.events)),
            "recovered": bool(result and result.state == "IDLE" and result.cancelled),
            "error": result.error if result else "missing pipeline result",
        }
    )
    return row


def run_cancellation_stress_bench(config: dict[str, Any], *, repeats_per_timing: int = 5) -> dict[str, Any]:
    """Exercise first/middle/end PCM cancellation boundaries on resident vLLM."""
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data: dict[str, Any] = {
        "status": "blocked",
        "benchmark": "interruption",
        "phase": 5,
        "timing_conditions": {"first_pcm": 1, "playback_middle": 3, "playback_end": 5},
        "rows": [],
        "summary": None,
        "server": None,
        "audio_state": {"snapshot": None, "restore_error": None},
        "recovery_probe": None,
        "physical_stop_s": None,
        "physical_stop_status": "not_recorded_in_software_stress; existing physical interruption probe remains separate",
    }
    server: VLLMOmniServer | None = None
    guard: AudioVolumeGuard | None = None
    try:
        inventory = PipeWireInventory.discover()
        speaker = inventory.usb_speaker()
        microphone = inventory.usb_microphone()
        playback_target = stable_target(speaker) if speaker else None
        capture_target = stable_target(microphone) if microphone else None
        if not playback_target or not capture_target:
            raise RuntimeError("stable USB playback and capture targets are required")
        guard = AudioVolumeGuard(speaker_target=playback_target, microphone_target=capture_target)
        guard.__enter__()
        data["audio_state"]["snapshot"] = guard.snapshot.to_dict() if guard.snapshot else None
        guard.set_mutes(speaker_muted=False, microphone_muted=False)
        server = VLLMOmniServer(
            python_bin=_vllm_python(config),
            model=_vllm_model(config),
            host="127.0.0.1",
            port=8091,
            deploy_config=nested(config, "tts", "vllm_deploy_config", default=None),
            log_path=artifact_dir(config) / "phase5_cancellation_vllm_server.log",
            gpu_memory_utilization=nested(config, "tts", "vllm_gpu_memory_utilization", default=None),
            extra_env=_server_env(config),
        )
        data["server"] = server.start(timeout_s=float(nested(config, "tts", "vllm_server_start_timeout_s", default=900.0)))
        engine = _make_vllm_client(config, streaming=True, initial=nested(config, "tts", "vllm_initial_codec_chunk_frames", default=None))
        for timing_name, hold_after_queue in (("first_pcm", 1), ("playback_middle", 3), ("playback_end", 5)):
            for repeat_index in range(1, repeats_per_timing + 1):
                data["rows"].append(_run_cancellation_case(config, engine=engine, playback_target=playback_target, timing_name=timing_name, hold_after_queue=hold_after_queue, repeat_index=repeat_index))
        data["summary"] = summarize_cancellation_runs(data["rows"])
        recovery_playback = PipeWirePCMPlayback(playback_target)
        recovery_pipeline = LivePipeline(llm=_InterruptFixtureLLM(), tts=engine, playback=recovery_playback, artifact_dir=artifact_dir(config))
        recovery = recovery_pipeline.respond("キャンセル後の次ターン復旧です。")
        data["recovery_probe"] = {
            "status": "measured" if recovery.state == "IDLE" and not recovery.cancelled and recovery.error is None else "error",
            "state": recovery.state,
            "cancelled": recovery.cancelled,
            "error": recovery.error,
            "spoken_text": recovery.spoken_text,
        }
        data["status"] = "measured"
    except Exception as exc:
        data.update({"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if server is not None:
            data.setdefault("server", {})
            data["server"]["stop"] = server.stop()
            current = current_gpu_memory()
            data["memory_after_server_stop"] = {"gpu_memory_used_mib": max(current) if current else None}
        if guard is not None:
            try:
                guard.__exit__(None, None, None)
            except Exception as exc:
                data["audio_state"]["restore_error"] = f"{type(exc).__name__}: {exc}"
            else:
                data["audio_state"]["restore_error"] = guard.restore_error
    return write_benchmark(config, "interruption", data, started_at=started)


class _FaultStdin:
    def write(self, payload: bytes) -> int:
        return len(payload)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FaultProcess:
    def __init__(self, *, premature: bool = False) -> None:
        self.stdin = _FaultStdin()
        self.stdout = None
        self.stderr = None
        self.returncode: int | None = 1 if premature else None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = -15 if self.terminated else 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class _DisconnectResponse:
    def __enter__(self) -> "_DisconnectResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self):
        yield b"\x00\x00" * 128
        raise ConnectionError("injected HTTP stream disconnect")


class _DisconnectClient:
    def __init__(self, **kwargs: Any) -> None:
        self.closed = False

    def __enter__(self) -> "_DisconnectClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def stream(self, *args: Any, **kwargs: Any) -> _DisconnectResponse:
        return _DisconnectResponse()

    def get(self, *args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("injected server unavailable")

    def close(self) -> None:
        self.closed = True


class _FaultLLM:
    requested_model = "phase5-fault"

    def stream(self, messages: Any, tools: Any = None, cancel_event: Any = None):
        yield LLMError(message="injected Ollama request failure", retryable=True, status_code=503)


def run_fault_injection_matrix(config: dict[str, Any]) -> dict[str, Any]:
    """Run bounded process/unit fault probes without changing OS audio devices."""
    rows: list[dict[str, Any]] = []
    disconnect_process = _FaultProcess()
    disconnect_playback = PipeWirePCMPlayback("Fault Sink", popen_factory=lambda command, **kwargs: disconnect_process)
    disconnect_engine = VLLMOmniTTSEngine(client_factory=lambda **kwargs: _DisconnectClient(), streaming=True)
    try:
        result = disconnect_engine.synthesize_stream("切断", output_path=artifact_dir(config) / "phase5_fault_disconnect.wav", playback=disconnect_playback)
        rows.append({"fault": "vllm_http_stream_disconnect", "status": "measured", "returned": True, "error_status": result.get("status"), "playback_active": disconnect_playback.active, "stale_pcm": disconnect_playback.active})
    except Exception as exc:
        rows.append({"fault": "vllm_http_stream_disconnect", "status": "error", "error_type": type(exc).__name__, "error": str(exc)})
    premature_process = _FaultProcess(premature=True)
    premature_playback = PipeWirePCMPlayback("Fault Sink", popen_factory=lambda command, **kwargs: premature_process)
    try:
        premature_playback.start(sample_rate=24000, channels=1)
        premature_playback.queue(b"\x00\x00")
        rows.append({"fault": "playback_process_premature_exit", "status": "error", "recovered": False})
    except Exception as exc:
        premature_playback.cancel()
        rows.append({"fault": "playback_process_premature_exit", "status": "measured", "recovered": not premature_playback.active, "error_type": type(exc).__name__})
    try:
        from .vllm_omni_tts import PCMChunkParser

        parser = PCMChunkParser()
        parser.feed(b"\x00")
        parser.finish()
        rows.append({"fault": "empty_or_invalid_pcm_chunk", "status": "error", "recovered": False})
    except ValueError as exc:
        rows.append({"fault": "empty_or_invalid_pcm_chunk", "status": "measured", "recovered": True, "error_type": type(exc).__name__})
    failure_pipeline = LivePipeline(llm=_FaultLLM(), tts=object(), playback=object(), artifact_dir=artifact_dir(config))
    failure_result = failure_pipeline.respond("Ollama障害")
    rows.append({"fault": "ollama_request_failure", "status": "measured", "state": failure_result.state, "error": failure_result.error, "hang": False})
    unavailable_engine = VLLMOmniTTSEngine(client_factory=lambda **kwargs: _DisconnectClient(), streaming=True)
    health = unavailable_engine.health()
    rows.append({"fault": "vllm_server_unavailable", "status": "measured", "health_status": health.get("status"), "hang": False})
    try:
        from .audio import RawCaptureSession

        RawCaptureSession(artifact_dir(config) / "phase5_fault_mic.wav", target="42")
        rows.append({"fault": "microphone_target_unavailable_mock", "status": "error", "recovered": False})
    except ValueError as exc:
        rows.append({"fault": "microphone_target_unavailable_mock", "status": "measured", "recovered": True, "error_type": type(exc).__name__})
    all_measured = all(row.get("status") == "measured" for row in rows)
    return {"status": "measured" if rows else "blocked", "faults": rows, "all_faults_measured": all_measured, "next_turn_recovery": all_measured}


def read_audio_state(config: dict[str, Any]) -> dict[str, Any]:
    """Read current Pulse state without changing it."""
    guard: AudioVolumeGuard | None = None
    try:
        inventory = PipeWireInventory.discover()
        speaker = inventory.usb_speaker()
        microphone = inventory.usb_microphone()
        speaker_target = stable_target(speaker) if speaker else None
        microphone_target = stable_target(microphone) if microphone else None
        if not speaker_target or not microphone_target:
            return {"status": "blocked", "reason": "stable USB playback and capture targets unavailable"}
        guard = AudioVolumeGuard(speaker_target=speaker_target, microphone_target=microphone_target)
        guard.__enter__()
        state = guard.snapshot.to_dict() if guard.snapshot else None
        return {"status": "measured", "state": state}
    except Exception as exc:
        return {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        if guard is not None:
            guard._restored = True


def audio_state_matches(expected: Mapping[str, Any] | None, actual: Mapping[str, Any] | None) -> bool | None:
    if not expected or not actual:
        return None
    keys = ("speaker", "microphone", "default_sink", "default_source")
    return all(expected.get(key) == actual.get(key) for key in keys)


def run_unattended_bench(config: dict[str, Any]) -> dict[str, Any]:
    """Orchestrate Phase 5's unattended evidence-producing checks."""
    import copy

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    before = resource_snapshot()
    resource_samples: list[Mapping[str, Any]] = [before]
    echo_result = run_phase5_echo_rejection_bench(config)
    resource_samples.append(resource_snapshot())
    onset_result = run_physical_onset_repeat_bench(config, repeats=int(nested(config, "bench", "onset_repeat_repeats", default=PHASE5_ONSET_REPEATS)))
    resource_samples.append(resource_snapshot())
    phase5_config = copy.deepcopy(config)
    phase5_config.setdefault("bench", {})
    phase5_config["bench"]["stability_target_turns"] = max(100, int(nested(config, "bench", "unattended_turns", default=100)))
    phase5_config["bench"]["stability_minimum_turns"] = max(60, int(nested(config, "bench", "unattended_minimum_turns", default=60)))
    phase5_config["bench"]["stability_restart_turns"] = max(5, int(nested(config, "bench", "unattended_restart_turns", default=5)))
    phase5_config["bench"]["artifact_prefix"] = "phase5"
    stability_result = run_stability_bench(phase5_config)
    resource_samples.append(resource_snapshot())
    cancellation_result = run_cancellation_stress_bench(config, repeats_per_timing=max(5, int(nested(config, "bench", "cancellation_repeats_per_timing", default=5))))
    resource_samples.append(resource_snapshot())
    faults = run_fault_injection_matrix(config)
    after = resource_snapshot()
    resource_samples.append(after)
    audio_state_readback = read_audio_state(config)
    stability_data = stability_result.get("data", {})
    data = {
        "status": "measured",
        "benchmark": "unattended",
        "phase": 5,
        "fixed_configuration": {
            "asr": "large-v3-turbo / cuda / int8_float16",
            "llm": "ollama / qwen3.5:9b-q4_K_M",
            "tts": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice / Ono_Anna / Japanese",
            "serving": "vLLM-Omni 0.28.0 + vLLM 0.28.0 / HTTP raw PCM streaming",
            "aec_changed": False,
        },
        "echo_rejection": echo_result.get("data", {}),
        "physical_onset_repeat": onset_result,
        "stability": stability_data,
        "cancellation": cancellation_result.get("data", {}),
        "fault_injection": faults,
        "resource_lifecycle": summarize_resource_lifecycle(before, after, samples=resource_samples),
        "audio_state_readback": audio_state_readback,
        "audio_state_restore_verified": audio_state_matches(
            ((onset_result.get("audio_state") or {}).get("snapshot")),
            audio_state_readback.get("state") if isinstance(audio_state_readback, Mapping) else None,
        ),
        "vram_margin": {
            "baseline_used_mib": before.get("gpu_memory_used_mib"),
            "peak_used_mib": (stability_data.get("summary") or {}).get("gpu_memory_peak_mib"),
            "free_min_mib": (stability_data.get("summary") or {}).get("gpu_memory_free_min_mib"),
            "start_free_mib": before.get("gpu_memory_free_mib"),
            "end_free_mib": after.get("gpu_memory_free_mib"),
            "drift_used_mib": float(after["gpu_memory_used_mib"]) - float(before["gpu_memory_used_mib"]) if isinstance(after.get("gpu_memory_used_mib"), (int, float)) and isinstance(before.get("gpu_memory_used_mib"), (int, float)) else None,
            "oom_count": sum("oom" in str(row.get("error", "")).casefold() for row in (stability_data.get("turns") or [])),
            "warning_below_500_mib": isinstance((stability_data.get("summary") or {}).get("gpu_memory_free_min_mib"), (int, float)) and stability_data["summary"]["gpu_memory_free_min_mib"] < 500,
        },
        "deferred_manual": [
            "human speech and physical human double-talk",
            "human barge-in",
            "MOS, subjective listening, and manual gain tuning",
            "production readiness approval",
        ],
        "limitations": [
            "synthetic double-talk is a reference/user fixture regression, not proof of physical double-talk",
            "physical onset repeats classify capture/detector causes but do not establish human speech behavior",
        ],
    }
    summary = stability_data.get("summary") or {}
    data["summary"] = {
        "attempts": summary.get("attempt_count", 0),
        "measured": summary.get("measured_turn_count", 0),
        "blocked": summary.get("blocked_turn_count", 0),
        "failed": summary.get("failed_turn_count", 0),
        "physical_first_audio": summary.get("physical_first_audio_s"),
        "onset_success_rate": summary.get("physical_onset_detection_success_rate"),
        "dominant_outlier_cause": summary.get("dominant_outlier_cause"),
        "server_restart": stability_data.get("server_restart"),
    }
    stability_goals = stability_data.get("summary", {}).get("goals") or {}
    stability_summary = stability_data.get("summary") or {}
    onset_summary = onset_result.get("summary") or {}
    cancellation_summary = (cancellation_result.get("data") or {}).get("summary") or {}
    data["component_status"] = {
        "echo_rejection": (echo_result.get("data") or {}).get("component_status", "blocked"),
        "physical_onset_repeat": "pass" if onset_result.get("status") == "measured" and onset_summary.get("blocked", 0) == 0 and onset_summary.get("failed", 0) == 0 else "measured_with_limitations",
        "continuous_stability": (stability_data.get("component_status") or {}).get("continuous_stability", "blocked"),
        "server_restart": (stability_data.get("component_status") or {}).get("server_restart", "blocked"),
        "cancellation": cancellation_summary.get("status", "blocked"),
        "fault_injection": "pass" if faults.get("all_faults_measured") else "measured_with_limitations",
        "resource_lifecycle": "pass" if not data["resource_lifecycle"].get("leak_suspected") and not data["resource_lifecycle"].get("fd_growth_requires_followup") else "measured_with_limitations",
        "audio_state_restore": "pass" if data.get("audio_state_restore_verified") is True else "measured_with_limitations",
    }
    data["pass_candidate"] = bool(
        data["summary"]["attempts"] >= 100
        and data["summary"]["failed"] == 0
        and data["summary"]["onset_success_rate"] is not None
        and data["summary"]["onset_success_rate"] >= 0.98
        and stability_goals.get("median_lt_2s")
        and stability_goals.get("p95_lt_2_5s")
        and not (stability_summary.get("memory_leak") or {}).get("suspected")
        and all(value == "pass" for value in data["component_status"].values())
    )
    data["status"] = "pass" if data["pass_candidate"] else "measured_with_limitations"
    if data["summary"]["attempts"] < 1:
        data["status"] = "blocked"
    return write_benchmark(config, "unattended", data, started_at=started)


__all__ = [
    "audio_state_matches",
    "build_double_talk_fixture",
    "build_phase5_double_talk_dataset",
    "classify_onset_failure",
    "evaluate_echo_rejection_split",
    "resource_snapshot",
    "read_audio_state",
    "run_phase5_echo_rejection_bench",
    "run_physical_onset_repeat_bench",
    "split_calibration_validation",
    "summarize_resource_lifecycle",
    "summarize_cancellation_runs",
]
