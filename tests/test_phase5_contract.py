from __future__ import annotations

import numpy as np
import pytest

from local_live.audio_metrics import detect_acoustic_onset
from local_live.echo_rejection import (
    EchoRejectionConfig,
    evaluate_echo_rejection,
    evaluate_echo_rejection_split,
)
from local_live.phase4_bench import _process_resource_series_summary
from local_live.phase5_bench import (
    audio_state_matches,
    build_double_talk_fixture,
    classify_onset_failure,
    split_calibration_validation,
    summarize_cancellation_runs,
    summarize_resource_lifecycle,
    run_fault_injection_matrix,
)
from local_live.telemetry import ResourceMonitor
from local_live import telemetry
from local_live.vllm_omni_tts import VLLMOmniTTSEngine


def _reference(sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(1.2 * sample_rate), dtype=np.float32) / sample_rate
    envelope = np.zeros_like(t)
    envelope[int(0.08 * sample_rate) : int(0.42 * sample_rate)] = 1.0
    envelope[int(0.58 * sample_rate) : int(1.08 * sample_rate)] = 1.0
    return (0.25 * envelope * (np.sin(2 * np.pi * 233 * t) + 0.5 * np.sin(2 * np.pi * 377 * t))).astype(np.float32)


def _user(sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(0.55 * sample_rate), dtype=np.float32) / sample_rate
    envelope = np.zeros_like(t)
    envelope[int(0.04 * sample_rate) : int(0.50 * sample_rate)] = 1.0
    return (0.3 * envelope * (np.sin(2 * np.pi * 911 * t) + 0.35 * np.sin(2 * np.pi * 1237 * t))).astype(np.float32)


def test_double_talk_mixer_keeps_metadata_and_user_after_reference():
    fixture = build_double_talk_fixture(
        _reference(),
        _user(),
        16000,
        name="after_weak_user",
        label="synthetic_user_like",
        offset_s=1.35,
        user_gain=0.2,
        echo_gain=0.4,
        noise_rms=0.001,
        lag_s=0.04,
        seed=7,
        playback_active=True,
    )
    assert fixture["metadata"]["offset_class"] == "assistant_end_or_after"
    assert fixture["metadata"]["user_offset_s"] == pytest.approx(1.35)
    assert fixture["metadata"]["echo_lag_s"] == pytest.approx(0.04)
    assert len(fixture["microphone"]) > len(fixture["reference"])
    assert fixture["metadata"]["user_end_s"] > fixture["metadata"]["reference_duration_s"]
    assert fixture["microphone"].dtype == np.float32


def test_calibration_validation_split_is_stratified_and_deterministic():
    fixtures = [
        {"name": f"a{i}", "label": "assistant_only"} for i in range(10)
    ] + [
        {"name": f"u{i}", "label": "synthetic_user_like"} for i in range(10)
    ]
    first = split_calibration_validation(fixtures, fraction=0.5)
    second = split_calibration_validation(fixtures, fraction=0.5)
    assert [row["name"] for row in first["calibration"]] == [row["name"] for row in second["calibration"]]
    assert len(first["calibration"]) == len(first["validation"]) == 10
    assert {row["label"] for row in first["calibration"]} == {"assistant_only", "synthetic_user_like"}
    assert {row["label"] for row in first["validation"]} == {"assistant_only", "synthetic_user_like"}
    assert {row["name"] for row in first["calibration"]}.isdisjoint(row["name"] for row in first["validation"])


def test_echo_split_reports_fixed_thresholds_and_validation_margins():
    reference = _reference()
    fixtures = []
    for index in range(4):
        common = {
            "reference": reference,
            "playback_active": True,
            "metadata": {"condition_index": index},
        }
        echo = build_double_talk_fixture(
            reference,
            _user(),
            16000,
            name=f"a{index}",
            label="assistant_only",
            offset_s=0.05,
            user_gain=0.0,
            echo_gain=0.25,
            noise_rms=0.0002,
            lag_s=0.03,
            seed=index,
            playback_active=True,
        )
        echo.update(common)
        user = build_double_talk_fixture(
            reference,
            _user(),
            16000,
            name=f"u{index}",
            label="synthetic_user_like",
            offset_s=0.35,
            user_gain=0.35,
            echo_gain=0.25,
            noise_rms=0.0002,
            lag_s=0.03,
            seed=100 + index,
            playback_active=True,
        )
        user.update(common)
        fixtures.extend([echo, user])
    config = EchoRejectionConfig(
        correlation_threshold=0.90,
        residual_energy_ratio_threshold=0.40,
        max_lag_s=0.10,
        energy_ratio_min=0.0,
        energy_ratio_max=2.0,
    )
    result = evaluate_echo_rejection_split(fixtures, 16000, fixed_config=config)
    assert result["split"]["calibration_count"] == 4
    assert result["split"]["validation_count"] == 4
    assert result["fixed_threshold_evaluation"]["validation"]["thresholds"] == config.__dict__
    assert result["fixed_threshold_evaluation"]["validation"]["threshold_margin_rows"]
    assert result["calibrated_validation_evaluation"]["thresholds"]
    assert result["threshold_selection_order"][0] == "fixed_phase4_thresholds_first"


def test_onset_failure_classifier_distinguishes_capture_and_detector():
    base = {
        "playback_success": True,
        "device_available": True,
        "pipewire_health": {"status": "ready"},
        "recording_stats": {"duration_s": 2.0, "rms": 0.03, "peak": 0.2},
    }
    assert classify_onset_failure({**base, "detected": True})["cause"] == "measured"
    assert classify_onset_failure({**base, "detected": False, "sensitivity_detected": True})["cause"] == "threshold_too_strict"
    assert classify_onset_failure({**base, "detected": False, "sensitivity_detected": False, "cross_correlation": {"correlation": 0.95}})["cause"] == "onset_detector_false_negative"
    assert classify_onset_failure({**base, "recording_stats": {"duration_s": 0.1, "rms": 0.0, "peak": 0.0}, "detected": False})["cause"] == "microphone_capture_missing"
    assert classify_onset_failure({**base, "playback_success": False, "detected": False})["cause"] == "speaker_playback_missing"


def test_onset_sensitivity_has_low_false_positive_for_synthetic_noise():
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.0008, size=16000).astype(np.float32)
    signal = noise.copy()
    signal[8000:12000] += 0.08 * np.sin(2 * np.pi * 440 * np.arange(4000) / 16000).astype(np.float32)
    noise_result = detect_acoustic_onset(noise, 16000, search_start_s=0.25, threshold_multiplier=3.0)
    signal_result = detect_acoustic_onset(signal, 16000, search_start_s=0.25, threshold_multiplier=3.0)
    assert noise_result["detected"] is False
    assert signal_result["detected"] is True
    assert signal_result["onset_s"] == pytest.approx(0.5, abs=0.03)


def test_resource_lifecycle_reports_deltas_and_leak_flag():
    before = {"gpu_vram_mib": 10, "process_rss_mib": 100.0, "open_fds": 4, "child_processes": 1, "playback_processes": 0, "active_http_connections": 0}
    after = {"gpu_vram_mib": 11, "process_rss_mib": 101.0, "open_fds": 4, "child_processes": 1, "playback_processes": 0, "active_http_connections": 0}
    result = summarize_resource_lifecycle(before, after)
    assert result["deltas"]["process_rss_mib"] == pytest.approx(1.0)
    assert result["leak_suspected"] is False


def test_cancellation_summary_requires_no_stale_pcm_and_idle_recovery():
    rows = [
        {"status": "measured", "software_stop_s": 0.12, "stale_pcm": False, "pipeline_state": "IDLE", "recovered": True}
        for _ in range(5)
    ]
    result = summarize_cancellation_runs(rows)
    assert result["count"] == 5
    assert result["software_stop"]["median"] == pytest.approx(0.12)
    assert result["stale_pcm_count"] == 0
    assert result["recovery_rate"] == 1.0
    assert result["status"] == "pass"


def test_echo_classifier_uses_active_window_under_high_noise() -> None:
    sample_rate = 16_000
    t = np.arange(int(0.8 * sample_rate), dtype=np.float32) / sample_rate
    reference = np.concatenate(
        [
            np.zeros(int(0.2 * sample_rate), dtype=np.float32),
            (0.35 * np.sin(2 * np.pi * 310 * t)).astype(np.float32),
        ]
    )
    fixture = build_double_talk_fixture(
        reference,
        np.zeros_like(reference),
        sample_rate,
        name="high_noise_echo",
        label="assistant_only",
        offset_s=0.0,
        user_gain=0.0,
        echo_gain=0.4,
        noise_rms=0.006,
        lag_s=0.06,
        seed=123,
        playback_active=True,
    )
    result = evaluate_echo_rejection(
        [fixture],
        sample_rate,
        config=EchoRejectionConfig(
            correlation_threshold=0.9642105263,
            residual_energy_ratio_threshold=0.9,
            max_lag_s=0.066,
            energy_ratio_min=0.1416587676,
            energy_ratio_max=0.5,
        ),
    )
    assert result["rows"][0]["reject"] is True
    assert result["rows"][0]["metrics"]["max_window_correlation"] >= result["rows"][0]["metrics"]["correlation"]


def test_detected_onset_is_not_classified_as_threshold_failure() -> None:
    assert classify_onset_failure({"detected": True})["cause"] == "measured"


def test_audio_restore_comparison_and_free_vram_monitor() -> None:
    state = {
        "speaker": {"target": "sink", "volume_percent": 40.0, "muted": False},
        "microphone": {"target": "source", "volume_percent": 100.0, "muted": False},
        "default_sink": "sink",
        "default_source": "source",
    }
    assert audio_state_matches(state, dict(state)) is True
    changed = dict(state)
    changed["default_sink"] = "other"
    assert audio_state_matches(state, changed) is False
    monitor = ResourceMonitor()
    monitor.started_gpu_free = [900]
    monitor.gpu_free_samples = [[700], [650]]
    assert monitor.gpu_memory_free_min_mib == 650


def test_fault_matrix_is_bounded_and_preserves_recovery_status(tmp_path) -> None:
    result = run_fault_injection_matrix({"paths": {"results_dir": str(tmp_path)}})
    assert result["status"] == "measured"
    assert result["all_faults_measured"] is True
    assert result["next_turn_recovery"] is True
    assert len(result["faults"]) == 6


def test_http_cancel_precedes_playback_release() -> None:
    events: list[str] = []

    class Client:
        def close(self) -> None:
            events.append("http")

    class Playback:
        def cancel(self) -> None:
            events.append("playback")

    engine = VLLMOmniTTSEngine(client_factory=lambda **_: Client())
    engine._active_client = Client()
    engine._active_playback = Playback()
    engine.cancel()

    assert events == ["http", "playback"]


def test_resource_monitor_retains_per_turn_process_resource_series() -> None:
    monitor = ResourceMonitor()
    monitor.started_process_resources = {
        "open_fds": 4,
        "child_processes": 0,
        "playback_processes": 0,
        "active_http_connections": 0,
    }
    monitor.process_resource_samples = [
        {"open_fds": 4, "child_processes": 0, "playback_processes": 0, "active_http_connections": 0},
        {"open_fds": 5, "child_processes": 0, "playback_processes": 0, "active_http_connections": 0},
        {"open_fds": 6, "child_processes": 0, "playback_processes": 0, "active_http_connections": 0},
    ]

    summary = monitor.process_resource_summary

    assert summary["sample_count"] == 3
    assert summary["end"]["open_fds"] == 6
    assert summary["monotonic_growth"]["open_fds"] is True


def test_resource_lifecycle_does_not_call_one_warm_cache_step_monotonic_leak() -> None:
    result = summarize_resource_lifecycle(
        {"open_fds": 4},
        {"open_fds": 43},
        samples=[{"open_fds": 4}, {"open_fds": 43}, {"open_fds": 43}, {"open_fds": 43}],
    )

    assert result["monotonic_growth"]["open_fds"] is False
    assert result["fd_growth_requires_followup"] is True


def test_stability_resource_summary_uses_started_snapshot() -> None:
    rows = [
        {
            "memory": {
                "process_resources": {
                    "started": {"open_fds": 43, "child_processes": 4, "playback_processes": 0, "active_http_connections": 3},
                    "end": {"open_fds": 43, "child_processes": 4, "playback_processes": 0, "active_http_connections": 3},
                    "deltas": {"open_fds": 0, "child_processes": 0, "playback_processes": 0, "active_http_connections": 0},
                    "monotonic_growth": {"open_fds": False, "child_processes": False, "playback_processes": False, "active_http_connections": False},
                    "sample_count": 3,
                }
            }
        }
    ]
    result = _process_resource_series_summary(rows)
    assert result["turn_count"] == 1
    assert result["open_fds"]["start"]["median"] == pytest.approx(43)
    assert result["open_fds"]["end"]["median"] == pytest.approx(43)


def test_live_http_counter_excludes_time_wait(monkeypatch) -> None:
    class FakeError(Exception):
        pass

    class FakeProcess:
        def num_fds(self) -> int:
            return 9

        def children(self, recursive: bool = False) -> list[object]:
            return []

    class FakeItem:
        info = {"name": "python", "cmdline": ["python", "bench"]}

    class Endpoint:
        def __init__(self, port: int) -> None:
            self.port = port

    class Connection:
        def __init__(self, status: str, port: int) -> None:
            self.status = status
            self.laddr = Endpoint(port)
            self.raddr = None

    class FakePsutil:
        Error = FakeError

        @staticmethod
        def Process() -> FakeProcess:
            return FakeProcess()

        @staticmethod
        def process_iter(fields: list[str]) -> list[FakeItem]:
            return [FakeItem()]

        @staticmethod
        def net_connections(kind: str) -> list[Connection]:
            return [Connection("TIME_WAIT", 8091), Connection("ESTABLISHED", 8091)]

    monkeypatch.setattr(telemetry, "psutil", FakePsutil)
    result = telemetry.current_process_resources(http_ports=(8091,))
    assert result["open_fds"] == 9
    assert result["active_http_connections"] == 1
