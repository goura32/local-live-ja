from __future__ import annotations

import numpy as np
import pytest

from local_live.phase6_bench import (
    build_expected_onset_window,
    classify_physical_onset,
    frame_energy_series,
    physical_latency_summary,
    reference_alignment_evidence,
    separate_application_measurement_status,
    split_pcm_chunks,
    summarize_first_sentence_buffering,
)


def test_reference_alignment_recovers_known_physical_offset() -> None:
    rate = 4000
    reference = np.zeros(1200, dtype=np.float32)
    t = np.arange(1000, dtype=np.float32) / rate
    reference[200:] = (0.4 * np.sin(2 * np.pi * 173 * t[:1000]) + 0.15 * np.sin(2 * np.pi * 311 * t[:1000])).astype(np.float32)
    recording = np.zeros(5000, dtype=np.float32)
    reference_start = 2400
    recording[reference_start : reference_start + len(reference)] = reference
    recording += np.random.default_rng(6).normal(0.0, 0.001, size=len(recording)).astype(np.float32)

    result = reference_alignment_evidence(
        reference,
        recording,
        sample_rate=rate,
        recording_rate=rate,
        expected_playback_start_s=0.5,
        reference_onset_s=0.05,
        physical_path_distribution={"low_s": 0.05, "high_s": 0.25},
    )

    assert result["matched"] is True
    assert result["best_correlation"] > 0.95
    assert result["aligned_onset_s"] == pytest.approx(0.65, abs=0.01)
    assert result["lag_s"] == pytest.approx(0.10, abs=0.01)


def test_expected_onset_window_uses_measured_path_distribution() -> None:
    result = build_expected_onset_window(
        0.4,
        0.6,
        {"low_s": 0.2, "high_s": 0.5, "source": "fixture_distribution"},
        margin_s=0.1,
    )
    assert result["start_s"] == pytest.approx(1.1)
    assert result["end_s"] == pytest.approx(1.6)
    assert result["source"] == "fixture_distribution"


def test_physical_onset_classifies_energy_and_reference_evidence() -> None:
    playback = {"playback_active": True, "playback_success": True, "pcm_bytes_written": 100, "process_exit_status": 0}
    microphone = {"recording_stats": {"duration_s": 2.0, "rms": 0.1, "peak": 0.3}}
    expected = {"start_s": 0.5, "end_s": 1.0}
    confirmed = classify_physical_onset(playback, microphone, {"detected": True}, {"matched": True, "aligned_onset_s": 0.7}, expected)
    recovered = classify_physical_onset(playback, microphone, {"detected": False}, {"matched": True, "aligned_onset_s": 0.7}, expected)
    assert confirmed["classification"] == "confirmed"
    assert confirmed["measurement_confirmed"] is True
    assert recovered["classification"] == "correlation_recovered"
    assert recovered["measurement_confirmed"] is True


def test_physical_onset_classifies_negative_and_capture_failures() -> None:
    microphone = {"recording_stats": {"duration_s": 1.0, "rms": 0.01, "peak": 0.1}}
    negative = classify_physical_onset(
        {"playback_active": False}, microphone, {"detected": False}, {"matched": False}, None
    )
    false_positive = classify_physical_onset(
        {"playback_active": False}, microphone, {"detected": True}, {"matched": False}, None
    )
    mic_failure = classify_physical_onset(
        {"playback_active": True, "playback_success": True, "pcm_bytes_written": 100, "process_exit_status": 0},
        {"recording_stats": {"duration_s": 0.2, "rms": 0.0, "peak": 0.0}},
        {"detected": False},
        {"matched": False},
        None,
    )
    playback_failure = classify_physical_onset(
        {"playback_active": True, "playback_success": False}, microphone, {"detected": False}, {"matched": False}, None
    )
    assert negative["classification"] == "no_playback_negative"
    assert false_positive["classification"] == "false_positive"
    assert mic_failure["classification"] == "microphone_capture_failure"
    assert playback_failure["classification"] == "playback_failure"


def test_application_success_is_not_relabelled_as_measurement_failure() -> None:
    result = separate_application_measurement_status(True, "energy_only")
    assert result == {
        "application_status": "success",
        "physical_measurement_status": "energy_only",
        "application_failed_due_to_measurement": False,
    }


def test_physical_latency_summary_separates_confirmed_and_recovered() -> None:
    rows = [
        {"measurement_classification": "confirmed", "physical_measurement_status": "confirmed", "physical_latency_s": 0.2, "application_status": "success"},
        {"measurement_classification": "correlation_recovered", "physical_measurement_status": "correlation_recovered", "physical_latency_s": 0.3, "application_status": "success"},
        {"measurement_classification": "unknown", "physical_measurement_status": "unknown", "physical_latency_s": None, "application_status": "success"},
    ]
    result = physical_latency_summary(rows)
    assert result["confirmed_only"]["median"] == pytest.approx(0.2)
    assert result["confirmed_plus_recovered"]["median"] == pytest.approx(0.25)
    assert result["application_success_rate"] == pytest.approx(1.0)
    assert result["measurement_confirmation_rate"] == pytest.approx(2 / 3)


def test_frame_energy_series_pads_and_retains_partial_final_frame() -> None:
    result = frame_energy_series(np.ones(25, dtype=np.float32), sample_rate=1000, frame_ms=10)
    assert result["frame_count"] == 3
    assert result["partial_final_frame_padded"] is True
    assert len(result["values"]) == 3


def test_split_pcm_chunks_preserves_exact_even_byte_stream() -> None:
    payload = bytes(range(40))
    chunks = split_pcm_chunks(payload, chunk_bytes=7)
    assert b"".join(chunks) == payload
    assert all(len(chunk) % 2 == 0 for chunk in chunks)
    with pytest.raises(ValueError):
        split_pcm_chunks(b"\x00", chunk_bytes=4)


def test_energy_detector_hit_and_miss_are_separate_from_reference_alignment() -> None:
    from local_live.audio_metrics import detect_acoustic_onset

    silence = detect_acoustic_onset(np.zeros(16000, dtype=np.float32), 16000)
    tone = np.zeros(16000, dtype=np.float32)
    tone[4800:] = 0.1
    hit = detect_acoustic_onset(tone, 16000)
    assert silence["detected"] is False
    assert hit["detected"] is True


def test_alignment_outside_expected_window_is_late_not_confirmed() -> None:
    result = classify_physical_onset(
        {"playback_active": True, "playback_success": True, "pcm_bytes_written": 100, "process_exit_status": 0},
        {"recording_stats": {"duration_s": 2.0, "rms": 0.1, "peak": 0.3}},
        {"detected": True},
        {"matched": True, "aligned_onset_s": 1.5},
        {"start_s": 0.5, "end_s": 1.0},
    )
    assert result["classification"] == "late_outside_window"


def test_first_sentence_buffering_keeps_unrecorded_timeout_cause_explicit() -> None:
    result = summarize_first_sentence_buffering(
        [
            {"stages": {"first_sentence_buffering": 0.2}, "llm": {"first_text_chunk": "確認しました。"}},
            {"stages": {"first_sentence_buffering": 1.1}, "llm": {"first_text_chunk": "続きです"}, "outlier": {"is_outlier": True, "dominant_cause": "first_sentence_buffering"}},
        ]
    )
    assert result["bug_found"] is False
    assert result["observed_count"] == 2
    assert result["natural_boundary_count"] == 1
    assert result["outlier_dominant_count"] == 1
    assert result["timeout_fallback"].startswith("not_recorded")
