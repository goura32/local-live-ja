#!/usr/bin/env python3
"""Build a compact, machine-readable summary from local-live benchmark JSON."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        return {"status": "missing", "path": str(path.relative_to(ROOT))}
    return json.loads(path.read_text(encoding="utf-8"))


def pick(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: row.get(key) for key in keys}


def compact_tts_row(row: dict[str, Any]) -> dict[str, Any]:
    return pick(
        row,
        "status",
        "chunk",
        "phase",
        "run_number",
        "text_chars",
        "device",
        "audio_seconds",
        "elapsed_seconds",
        "model_load_seconds",
        "inference_elapsed_seconds",
        "first_audio_equivalent_seconds",
        "warm_first_audio_equivalent_seconds",
        "audio_complete_seconds",
        "playback_possible_seconds",
        "rtf",
        "gpu_memory_peak_mib",
        "gpu_memory_delta_peak_mib",
        "generated_audio_analysis",
        "error_type",
        "error",
    )


def compact_e2e_run(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **pick(
            row,
            "status",
            "run_number",
            "warmup",
            "requested_model",
            "actual_model",
            "asr_duration_s",
            "llm_ttft_s",
            "llm_total_duration_s",
            "tts_duration_s",
            "playback_roundtrip_duration_s",
            "total_e2e_duration_s",
            "peak_vram_mib",
            "assistant_cer",
            "outlier_flags",
            "dominant_stage",
            "error_type",
            "error",
        ),
        "stage_durations": row.get("stage_durations"),
        "assistant_roundtrip": row.get("assistant_roundtrip"),
        "timing_ns": row.get("timing_ns"),
        "llm": pick(row.get("llm") or {}, "text", "ttft_s", "elapsed_s", "completion", "error", "error_details", "cancelled"),
        "tts": [compact_tts_row(item) for item in row.get("tts", [])],
        "assistant_asr": [pick(item, "text", "elapsed_seconds", "rtf", "device", "compute_type") for item in row.get("assistant_asr", [])],
    }


def compact_phase4(
    stability: dict[str, Any],
    echo_rejection: dict[str, Any],
    mic_readiness: dict[str, Any],
    interruption: dict[str, Any],
) -> dict[str, Any]:
    """Keep aggregate Phase 4 facts here; detailed rows stay in result files."""
    stability_server = stability.get("server") or {}
    restart = stability.get("server_restart") or {}
    interruption_data = interruption.get("interruption") or {}
    mic_asr = (mic_readiness.get("asr") or {}).get("result") or {}
    return {
        "stability": {
            "status": stability.get("status"),
            "configuration": stability.get("configuration"),
            "summary": stability.get("summary"),
            "warm_state_windows": stability.get("warm_state_windows"),
            "component_status": stability.get("component_status"),
            "server": pick(stability_server, "model", "base_url", "host", "port", "python", "server_start_to_ready_s", "model_load_s", "health", "gpu_memory_at_ready_mib", "environment_overrides", "stop", "versions"),
            "server_restart": pick(restart, "server_start_to_ready_s", "health", "gpu_memory_at_ready_mib", "restart_elapsed_s", "completed_runs", "stop"),
            "memory": stability.get("memory"),
            "interruption": {"status": interruption_data.get("status"), "software_playback_stop_s": interruption_data.get("software_playback_stop_s"), "physical_stop_s": interruption_data.get("physical_stop_s"), "persistent_pcm_active_after_interrupt": interruption_data.get("persistent_pcm_active_after_interrupt"), "vllm_http_stream_cancelled": interruption_data.get("vllm_http_stream_cancelled"), "pipeline": interruption_data.get("pipeline")},
            "phase3b_outlier_reference": stability.get("phase3b_outlier_reference"),
        },
        "echo_rejection": {
            "status": echo_rejection.get("status"),
            "method": echo_rejection.get("method"),
            "thresholds": echo_rejection.get("thresholds"),
            "assistant_only": echo_rejection.get("assistant_only"),
            "synthetic_user_like": echo_rejection.get("synthetic_user_like"),
            "correlation_distribution": echo_rejection.get("correlation_distribution"),
            "energy_ratio_distribution": echo_rejection.get("energy_ratio_distribution"),
            "residual_energy_ratio_distribution": echo_rejection.get("residual_energy_ratio_distribution"),
            "lag_seconds_distribution": echo_rejection.get("lag_seconds_distribution"),
            "component_status": echo_rejection.get("component_status"),
            "mute_reference": echo_rejection.get("mute_reference"),
        },
        "mic_readiness": {
            "status": mic_readiness.get("status"),
            "raw_usb_microphone": {"target": (mic_readiness.get("raw_usb_microphone") or {}).get("target"), "stats": (mic_readiness.get("raw_usb_microphone") or {}).get("stats")},
            "aec_source": {"target": (mic_readiness.get("aec_source") or {}).get("target"), "sink_master": (mic_readiness.get("aec_source") or {}).get("sink_master"), "stats": (mic_readiness.get("aec_source") or {}).get("stats")},
            "vad": mic_readiness.get("vad"),
            "asr": {"status": (mic_readiness.get("asr") or {}).get("status"), "result": pick(mic_asr, "model", "device", "compute_type", "text", "elapsed_seconds", "gpu_memory_peak_mib")},
        },
        "interruption": {"status": interruption.get("status"), "server": pick(interruption.get("server") or {}, "server_start_to_ready_s", "health", "gpu_memory_at_ready_mib", "stop"), "interruption": interruption_data},
    }


def compact_phase5(unattended: dict[str, Any], echo_rejection: dict[str, Any], interruption: dict[str, Any]) -> dict[str, Any]:
    """Keep Phase 5 evidence compact; detailed turn rows remain in bench files."""
    stability = unattended.get("stability") or {}
    stability_summary = stability.get("summary") or {}
    echo = unattended.get("echo_rejection") or echo_rejection
    split = echo.get("phase5_split_evaluation") or {}
    fixed_validation = ((split.get("fixed_threshold_evaluation") or {}).get("validation") or {})
    calibrated_validation = split.get("calibrated_validation_evaluation") or {}
    onset = unattended.get("physical_onset_repeat") or {}
    cancellation = unattended.get("cancellation") or interruption
    return {
        "status": unattended.get("status"),
        "summary": unattended.get("summary"),
        "stability": {
            "status": stability.get("status"),
            "configuration": stability.get("configuration"),
            "summary": stability_summary,
            "warm_state_windows": stability.get("warm_state_windows"),
            "component_status": stability.get("component_status"),
            "server_restart": stability.get("server_restart"),
            "memory": stability.get("memory"),
        },
        "echo_rejection": {
            "status": echo.get("status"),
            "dataset": echo.get("dataset"),
            "fixed_thresholds": split.get("fixed_thresholds"),
            "fixed_validation": {key: fixed_validation.get(key) for key in ("assistant_only", "synthetic_user_like", "correlation_distribution", "energy_ratio_distribution", "lag_seconds_distribution")},
            "calibrated_thresholds_from_calibration": split.get("calibrated_thresholds_from_calibration"),
            "calibrated_validation": {key: calibrated_validation.get(key) for key in ("assistant_only", "synthetic_user_like", "correlation_distribution", "energy_ratio_distribution", "lag_seconds_distribution")},
            "targets": echo.get("targets"),
            "component_status": echo.get("component_status"),
        },
        "physical_onset_repeat": {
            "status": onset.get("status"),
            "summary": onset.get("summary"),
            "threshold_sensitivity": onset.get("threshold_sensitivity"),
            "targets": onset.get("targets"),
        },
        "cancellation": {
            "status": cancellation.get("status"),
            "summary": cancellation.get("summary"),
            "recovery_probe": cancellation.get("recovery_probe"),
            "audio_state": cancellation.get("audio_state"),
        },
        "fault_injection": unattended.get("fault_injection"),
        "resource_lifecycle": unattended.get("resource_lifecycle"),
        "vram_margin": unattended.get("vram_margin"),
        "deferred_manual": unattended.get("deferred_manual", []),
    }


def compact_phase6(physical_onset: dict[str, Any], unattended: dict[str, Any]) -> dict[str, Any]:
    """Keep Phase 6 measurement-hardening evidence compact; rows remain in bench files."""
    phase6 = unattended if unattended.get("phase") == 6 else {}
    physical = phase6.get("physical_onset") or physical_onset
    stability = phase6.get("stability") or {}
    stability_summary = stability.get("summary") or {}
    return {
        "status": phase6.get("status") or physical.get("status"),
        "turn57_diagnosis": phase6.get("turn57_diagnosis") or physical.get("turn57_diagnosis"),
        "physical_path_distribution": physical.get("physical_path_distribution"),
        "fixed_replay": {
            "status": physical.get("status"),
            "summary": (physical.get("fixed_replay") or {}).get("summary"),
        },
        "fixture_replay": {
            "summary": (physical.get("fixture_replay") or {}).get("summary"),
            "fixtures": (physical.get("fixture_replay") or {}).get("fixtures"),
        },
        "negative_controls": {
            "summary": (physical.get("negative_controls") or {}).get("summary"),
        },
        "stability": {
            "status": stability.get("status"),
            "attempt_count": stability_summary.get("attempt_count"),
            "application_success_count": stability_summary.get("application_success_count"),
            "application_failed_count": stability_summary.get("application_failed_count"),
            "physical_measurement_confirmation_count": stability_summary.get("physical_measurement_confirmation_count"),
            "physical_measurement_confirmation_rate": stability_summary.get("physical_measurement_confirmation_rate"),
            "summary": stability_summary,
            "phase6_measurement": stability.get("phase6_measurement"),
            "server_restart": stability.get("server_restart"),
        },
        "summary": phase6.get("summary"),
        "late_onset_cause_counts": phase6.get("late_onset_cause_counts"),
        "late_onset_analysis": phase6.get("late_onset_analysis"),
        "audio_state_restore": phase6.get("audio_state_restore"),
        "resource_regression": phase6.get("resource_regression"),
        "first_sentence_buffering": phase6.get("first_sentence_buffering"),
        "deferred_manual": phase6.get("deferred_manual", physical.get("deferred_manual", [])),
        "pass_candidate": phase6.get("pass_candidate"),
    }


def build_summary() -> dict[str, Any]:
    doctor = load("doctor.json")
    asr = load("bench_asr.json")
    tts = load("bench_tts.json")
    llm = load("bench_llm.json")
    e2e = load("bench_e2e.json")
    aec = load("bench_aec.json")
    live_latency = load("bench_live_latency.json")
    aec_matrix = load("bench_aec_matrix.json")
    playback_path = load("bench_playback_path.json")
    tts_serving = load("bench_tts_serving.json")
    stability = load("bench_stability.json")
    echo_rejection = load("bench_echo_rejection.json")
    mic_readiness = load("bench_mic_readiness.json")
    interruption = load("bench_interruption.json")
    unattended = load("bench_unattended.json")
    physical_onset = load("bench_physical_onset.json")
    run = load("run_latest.json")
    pytest_final = load("pytest_final.json")

    asr_data = asr.get("data", {})
    tts_data = tts.get("data", {})
    llm_data = llm.get("data", {})
    e2e_data = e2e.get("data", {})
    aec_data = aec.get("data", {})
    live_latency_data = live_latency.get("data", {})
    aec_matrix_data = aec_matrix.get("data", {})
    playback_path_data = playback_path.get("data", {})
    tts_serving_data = tts_serving.get("data", {})
    stability_data = stability.get("data", {})
    echo_rejection_data = echo_rejection.get("data", {})
    mic_readiness_data = mic_readiness.get("data", {})
    interruption_data = interruption.get("data", {})
    unattended_data = unattended.get("data", {})
    physical_onset_data = physical_onset.get("data", {})

    return {
        "schema": "local-live-ja/summary/v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": doctor.get("environment") or asr.get("environment") or None,
        "benchmark_paths": {
            "doctor": "results/doctor.json",
            "asr": "results/bench_asr.json",
            "tts": "results/bench_tts.json",
            "llm": "results/bench_llm.json",
            "e2e": "results/bench_e2e.json",
            "aec": "results/bench_aec.json",
            "live_latency": "results/bench_live_latency.json",
            "aec_matrix": "results/bench_aec_matrix.json",
            "playback_path": "results/bench_playback_path.json",
            "tts_serving": "results/bench_tts_serving.json",
            "stability": "results/bench_stability.json",
            "echo_rejection": "results/bench_echo_rejection.json",
            "mic_readiness": "results/bench_mic_readiness.json",
            "interruption": "results/bench_interruption.json",
            "unattended": "results/bench_unattended.json",
            "physical_onset": "results/bench_physical_onset.json",
            "live_latency_python_reference": "results/bench_live_latency_python.json",
            "run": "results/run_latest.json",
        },
        "doctor": {
            "status": doctor.get("status"),
            "warnings": doctor.get("warnings", []),
            "failures": doctor.get("failures", []),
            "checks": doctor.get("checks", {}),
        },
        "asr": {
            "metric_name": asr_data.get("metric_name"),
            "same_wav_for_all_modes": asr_data.get("same_wav_for_all_modes"),
            "reference_text": asr_data.get("reference_text"),
            "modes": {name: pick(row, "status", "device", "compute_type", "text", "cer", "rtf", "elapsed_seconds", "gpu_memory_peak_mib", "gpu_memory_delta_peak_mib", "cpu_load_percent", "error_type", "error") for name, row in asr_data.get("modes", {}).items()},
            "cpu_resident_profile": asr_data.get("cpu_resident_profile"),
        },
        "tts": {
            "model": tts_data.get("model"),
            "speaker": tts_data.get("speaker"),
            "language": tts_data.get("language"),
            "streaming_supported_by_official_python_api": tts_data.get("streaming_supported_by_official_python_api"),
            "runs": {
                name: {
                    **compact_tts_row(row),
                    "cold_start": compact_tts_row(row["cold_start"]) if isinstance(row.get("cold_start"), dict) else None,
                    "warm_start": {
                        chunk: {
                            "text_chars": chunk_data.get("text_chars"),
                            "model_resident_for_all_runs": chunk_data.get("model_resident_for_all_runs"),
                            "runs": [compact_tts_row(item) for item in chunk_data.get("runs", [])],
                            "median": chunk_data.get("median"),
                        }
                        for chunk, chunk_data in row.get("warm_start", {}).items()
                    },
                    "latency_matrix": pick(row.get("latency_matrix") or {}, "status", "device", "warm_repeat_target", "cold_definition", "warm_definition"),
                    "trim_quality": row.get("trim_quality"),
                    "generation_policy_comparison": row.get("generation_policy_comparison"),
                }
                for name, row in tts_data.get("runs", {}).items()
            },
        },
        "llm": {
            "requested_openrouter_model": llm_data.get("requested_openrouter_model"),
            "context_target": llm_data.get("context_target"),
            "providers": {
                name: {
                    "requested_model": row.get("requested_model"),
                    "actual_model": row.get("actual_model"),
                    "normal_stream": pick(row.get("normal_stream", {}), "text", "events", "completion", "error", "cancelled", "ttft_s", "elapsed_s"),
                    "tool_calling": pick(row.get("tool_calling", {}), "status", "tool_call_count", "tool_round_count", "final_text", "error_type", "error"),
                    "cancel": pick(row.get("cancel", {}), "pre_cancel", "midstream_cancel_observed", "turn"),
                }
                for name, row in llm_data.get("providers", {}).items()
            },
        },
        "e2e": {
            "status": e2e_data.get("status"),
            "repeat_target": e2e_data.get("repeat_target"),
            "gpu_compute_type": e2e_data.get("gpu_compute_type"),
            "configurations": {
                name: {
                    "status": row.get("status"),
                    "provider": row.get("provider"),
                    "requested_model": row.get("requested_model"),
                    "actual_model": row.get("actual_model"),
                    "actual_models_by_run": row.get("actual_models_by_run"),
                    "compute_type": row.get("compute_type"),
                    "repeat_target": row.get("repeat_target"),
                    "max_attempts": row.get("max_attempts"),
                    "attempt_count": row.get("attempt_count"),
                    "measured_run_count": row.get("measured_run_count"),
                    "warmup": compact_e2e_run(row["warmup"]) if isinstance(row.get("warmup"), dict) else None,
                    "warmup_recovery": compact_tts_row(row["warmup_recovery"]) if isinstance(row.get("warmup_recovery"), dict) else None,
                    "runs": [compact_e2e_run(item) for item in row.get("runs", [])],
                    "median": row.get("median"),
                    "outlier_policy": row.get("outlier_policy"),
                    "error_type": row.get("error_type"),
                    "error": row.get("error"),
                }
                for name, row in e2e_data.get("configurations", {}).items()
            },
        },
        "aec": aec_data,
        "live_latency": {
            "status": live_latency_data.get("status"),
            "backend": live_latency_data.get("backend"),
            "streaming": live_latency_data.get("streaming"),
            "metric_name": live_latency_data.get("metric_name"),
            "metric_definition": live_latency_data.get("metric_definition"),
            "event_definition": live_latency_data.get("event_definition"),
            "targets": live_latency_data.get("targets"),
            "volume_snapshot": live_latency_data.get("volume_snapshot"),
            "warmup": live_latency_data.get("warmup"),
            "summary": live_latency_data.get("summary"),
            "runs": live_latency_data.get("runs"),
            "chunking_comparison": live_latency_data.get("chunking_comparison"),
            "first_chunk_policy": live_latency_data.get("first_chunk_policy"),
            "first_chunk_policy_comparison": live_latency_data.get("first_chunk_policy_comparison"),
            "trim_policy": live_latency_data.get("trim_policy"),
            "latency_budget": live_latency_data.get("latency_budget"),
            "server": live_latency_data.get("server"),
            "memory": live_latency_data.get("memory"),
            "python_baseline_reference": live_latency_data.get("python_baseline_reference"),
            "volume_restore_error": live_latency_data.get("volume_restore_error"),
            "error_type": live_latency_data.get("error_type"),
            "error": live_latency_data.get("error"),
        },
        "aec_matrix": {
            "status": aec_matrix_data.get("status"),
            "matrix": aec_matrix_data.get("matrix"),
            "sequence": aec_matrix_data.get("sequence"),
            "clipping_definition": aec_matrix_data.get("clipping_definition"),
            "volume_snapshot": aec_matrix_data.get("volume_snapshot"),
            "matrix_summary": aec_matrix_data.get("matrix_summary"),
            "best_echo_only_operating_envelope": aec_matrix_data.get("best_echo_only_operating_envelope"),
            "best_candidates": aec_matrix_data.get("best_candidates"),
            "best_condition_remeasurements": aec_matrix_data.get("best_condition_remeasurements"),
            "baseline_attenuation_db": aec_matrix_data.get("baseline_attenuation_db"),
            "error_type": aec_matrix_data.get("error_type"),
            "error": aec_matrix_data.get("error"),
        },
        "playback_path": playback_path_data,
        "tts_serving": {
            "status": tts_serving_data.get("status"),
            "model": tts_serving_data.get("model"),
            "speaker": tts_serving_data.get("speaker"),
            "language": tts_serving_data.get("language"),
            "sample_rate_hz": tts_serving_data.get("sample_rate_hz"),
            "repeat_target_per_text_per_mode": tts_serving_data.get("repeat_target_per_text_per_mode"),
            "responses": tts_serving_data.get("responses"),
            "modes": tts_serving_data.get("modes"),
            "initial_codec_chunk_frames_comparison": tts_serving_data.get("initial_codec_chunk_frames_comparison"),
            "async_chunk": tts_serving_data.get("async_chunk"),
            "server": tts_serving_data.get("server"),
            "quality": tts_serving_data.get("quality"),
            "initial_codec_chunk_frames_quality": tts_serving_data.get("initial_codec_chunk_frames_quality"),
            "memory": tts_serving_data.get("memory"),
            "targets": tts_serving_data.get("targets"),
            "volume_restore_error": tts_serving_data.get("volume_restore_error"),
        },
        "phase4": compact_phase4(stability_data, echo_rejection_data, mic_readiness_data, interruption_data),
        "phase5": compact_phase5(unattended_data, echo_rejection_data, interruption_data),
        "phase6": compact_phase6(physical_onset_data, unattended_data),
        "run": {
            "status": run.get("status"),
            "assistant_text": run.get("spoken_text", run.get("assistant_text")),
            "spoken_text": run.get("spoken_text"),
            "audio_paths_count": len(run.get("audio_paths", [])),
            "error": run.get("error"),
            "timing": run.get("timing"),
        },
        "pytest": pytest_final,
        "component_judgement": component_judgement(asr_data, tts_data, llm_data, e2e_data, aec_data, pytest_final, live_latency_data, aec_matrix_data, playback_path_data, tts_serving_data, stability_data, echo_rejection_data, mic_readiness_data, interruption_data),
    }


def component_judgement(
    asr_data: dict[str, Any],
    tts_data: dict[str, Any],
    llm_data: dict[str, Any],
    e2e_data: dict[str, Any],
    aec_data: dict[str, Any],
    pytest_data: dict[str, Any],
    live_latency_data: dict[str, Any] | None = None,
    aec_matrix_data: dict[str, Any] | None = None,
    playback_path_data: dict[str, Any] | None = None,
    tts_serving_data: dict[str, Any] | None = None,
    stability_data: dict[str, Any] | None = None,
    echo_rejection_data: dict[str, Any] | None = None,
    mic_readiness_data: dict[str, Any] | None = None,
    interruption_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return component states without collapsing blocked work into PASS/FAIL."""
    asr_rows = list(asr_data.get("modes", {}).values())
    tts_rows = list(tts_data.get("runs", {}).values())
    providers = llm_data.get("providers", {})
    e2e_rows = list(e2e_data.get("configurations", {}).values())
    live_latency_data = live_latency_data or {}
    aec_matrix_data = aec_matrix_data or {}
    playback_path_data = playback_path_data or {}
    tts_serving_data = tts_serving_data or {}
    stability_data = stability_data or {}
    echo_rejection_data = echo_rejection_data or {}
    mic_readiness_data = mic_readiness_data or {}
    interruption_data = interruption_data or {}
    states = {
        "asr": "measured" if asr_rows and all(row.get("status") == "measured" for row in asr_rows) else "partial",
        "tts": "measured" if any(row.get("status") == "measured" for row in tts_rows) else "partial",
        "local_llm": "measured" if (providers.get("local", {}).get("normal_stream", {}).get("text")) else "partial",
        "openrouter_llm": "measured" if (providers.get("openrouter", {}).get("normal_stream", {}).get("text")) else "partial",
        "tool_calling_local": providers.get("local", {}).get("tool_calling", {}).get("status", "partial"),
        "tool_calling_openrouter": providers.get("openrouter", {}).get("tool_calling", {}).get("status", "partial"),
        "e2e": "measured" if e2e_rows and all(row.get("status") == "measured" for row in e2e_rows) else "partial",
        "aec": aec_data.get("status", "partial"),
        "live_latency": live_latency_data.get("status", "missing"),
        "tts_serving": tts_serving_data.get("status", "missing"),
        "aec_matrix": aec_matrix_data.get("status", "missing"),
        "playback_path": playback_path_data.get("status", "missing"),
        "continuous_stability": (stability_data.get("component_status") or {}).get("continuous_stability", stability_data.get("status", "missing")),
        "server_restart": (stability_data.get("component_status") or {}).get("server_restart", "missing"),
        "echo_rejection": echo_rejection_data.get("component_status", echo_rejection_data.get("status", "missing")),
        "microphone_readiness": mic_readiness_data.get("status", "missing"),
        "interruption": (stability_data.get("component_status") or {}).get("interruption", interruption_data.get("status", "missing")),
        "cpu_asr_profile": ("measured" if asr_data.get("cpu_resident_profile", {}).get("status") == "measured" else "partial"),
        "test_reproducibility": "pass" if pytest_data.get("exit_code") == 0 else "unverified",
    }
    live_summary = live_latency_data.get("summary") or {}
    live_median = (live_summary.get("speech_end_to_first_physical_audio_s") or {}).get("median")
    if states["live_latency"] == "measured" and isinstance(live_median, (int, float)) and live_median >= 2.0:
        states["live_latency"] = "measured_target_missed"
    repeat_summaries = aec_matrix_data.get("best_condition_remeasurements") or {}
    if states["aec_matrix"] == "measured" and any(
        ((item.get("summary") or {}).get("vad_false_trigger_rate") or 0) > 0
        for item in repeat_summaries.values()
    ):
        states["aec_matrix"] = "measured_with_false_triggers"
    limitations = [
        name
        for name, state in states.items()
        if name != "overall" and state not in {"measured", "pass", "success"}
    ]
    states["limitations"] = limitations
    states["overall"] = "measured_with_blockers" if "blocked" in states.values() or "partial" in states.values() else (
        "measured_with_limitations" if limitations else "measured"
    )
    return states


def main() -> None:
    output = RESULTS / "summary.json"
    output.write_text(json.dumps(build_summary(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
