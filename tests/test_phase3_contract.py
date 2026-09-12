import numpy as np
import pytest

from local_live.audio_metrics import (
    measure_generated_audio_leading_silence,
    trim_leading_silence,
)
from local_live.bench import _distribution, _latency_budget


def test_leading_silence_reports_first_nonzero_and_stable_onset():
    sample_rate = 16000
    audio = np.zeros(sample_rate, dtype=np.float32)
    audio[8] = 1e-7
    audio[3200:] = 0.2
    result = measure_generated_audio_leading_silence(audio, sample_rate)
    assert result["first_nonzero_sample"] == 8
    assert result["first_low_threshold_crossing_s"] == pytest.approx(0.2)
    assert result["stable_speech_onset_s"] == pytest.approx(0.2)
    assert result["leading_silence_duration_s"] == pytest.approx(0.2)
    assert result["peak"] == pytest.approx(0.2)


def test_trim_pre_roll_never_uses_negative_start():
    sample_rate = 1000
    audio = np.arange(20, dtype=np.float32)
    analysis = {"detected": True, "stable_speech_onset_s": 0.01}
    trimmed, metadata = trim_leading_silence(audio, sample_rate, analysis, pre_roll_s=0.15)
    assert metadata["start_sample"] == 0
    assert np.array_equal(trimmed, audio)


def test_trim_keeps_requested_pre_roll_before_stable_onset():
    sample_rate = 1000
    audio = np.arange(1000, dtype=np.float32)
    analysis = {"detected": True, "stable_speech_onset_s": 0.5}
    trimmed, metadata = trim_leading_silence(audio, sample_rate, analysis, pre_roll_s=0.1)
    assert metadata["start_sample"] == 400
    assert metadata["removed_duration_s"] == pytest.approx(0.4)
    assert trimmed[0] == 400


def test_all_silent_audio_is_not_trimmed():
    audio = np.zeros(32, dtype=np.float32)
    analysis = measure_generated_audio_leading_silence(audio, 16000)
    trimmed, metadata = trim_leading_silence(audio, 16000, analysis, pre_roll_s=0.1)
    assert analysis["detected"] is False
    assert analysis["stable_speech_onset_s"] is None
    assert metadata["start_sample"] == 0
    assert np.array_equal(trimmed, audio)


def test_non_silent_from_first_frame_is_not_overtrimmed():
    audio = np.ones(80, dtype=np.float32) * 0.1
    analysis = measure_generated_audio_leading_silence(audio, 16000)
    trimmed, metadata = trim_leading_silence(audio, 16000, analysis, pre_roll_s=0.15)
    assert analysis["stable_speech_onset_s"] == pytest.approx(0.0)
    assert metadata["start_sample"] == 0
    assert len(trimmed) == len(audio)


def test_very_short_audio_and_threshold_boundary_are_finite():
    audio = np.array([0.0, 0.004, 0.004, 0.004], dtype=np.float32)
    result = measure_generated_audio_leading_silence(
        audio,
        1000,
        frame_ms=1,
        min_consecutive_frames=2,
        absolute_floor=0.004,
        noise_window_s=0.001,
    )
    assert result["detected"] is True
    assert result["stable_speech_onset_s"] == pytest.approx(0.001)
    distribution = _distribution([1.0, 2.0, 3.0])
    assert distribution["mean"] == pytest.approx(2.0)
    assert distribution["stddev"] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1))


def test_latency_budget_keeps_medians_and_total_shares():
    budget = _latency_budget(
        [
            {
                "asr_duration_s": 1.0,
                "llm_ttft_s": 0.2,
                "sentence_buffering_s": 0.1,
                "tts_inference_s": 1.2,
                "tts_postprocess_s": 0.01,
                "trimmed_leading_audio_s": 0.1,
                "wav_ready_to_pw_play_s": 0.01,
                "expected_speaker_onset_to_measured_mic_s": 0.5,
            }
        ],
        3.12,
    )
    assert budget["components"]["tts_inference"]["median_s"] == pytest.approx(1.2)
    assert budget["components"]["tts_inference"]["share_of_total"] == pytest.approx(1.2 / 3.12)
