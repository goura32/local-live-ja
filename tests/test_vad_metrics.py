import numpy as np

from local_live.audio_metrics import aligned_correlation, residual_echo_db
from local_live.vad import detect_speech_intervals


def test_energy_vad_finds_cpu_speech_interval():
    sample_rate = 16000
    audio = np.zeros(sample_rate * 2, dtype=np.float32)
    audio[4000:12000] = 0.25
    intervals = detect_speech_intervals(audio, sample_rate, frame_ms=20, min_speech_ms=100)
    assert intervals
    start, end = intervals[0]
    assert 0.20 < start < 0.30
    assert 0.70 < end < 0.80


def test_audio_alignment_reports_correlation_and_residual_reduction():
    rng = np.random.default_rng(7)
    reference = rng.normal(0, 0.2, 8000).astype(np.float32)
    delayed = np.concatenate([np.zeros(300, dtype=np.float32), reference])
    delayed = delayed[: len(reference)]
    assert aligned_correlation(reference, delayed, sample_rate=16000)["correlation"] > 0.9
    off = reference * 0.8
    on = reference * 0.2
    assert residual_echo_db(off, on) > 10.0


def test_audio_alignment_marks_silent_recording_as_unmeasured():
    silent = np.zeros(8000, dtype=np.float32)
    result = aligned_correlation(silent, silent, sample_rate=16000)
    assert result["correlation"] is None
    assert result["lag_samples"] is None
    assert result["lag_seconds"] is None
