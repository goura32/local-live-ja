import numpy as np
import pytest

from local_live.audio import AudioNode, AudioVolumeGuard, PipeWirePlayback, stable_target
from local_live.audio_metrics import clipping_ratio, detect_acoustic_onset
from local_live.bench import classify_cpu_asr_profile_phases, summarize_aec_conditions
from local_live.vad import assistant_only_vad_metrics


def test_stable_audio_target_requires_pactl_name():
    node = AudioNode(69, "USB Audio", "sink", "alsa_output.usb-Generic-00.analog-stereo")
    assert stable_target(node) == "alsa_output.usb-Generic-00.analog-stereo"
    with pytest.raises(ValueError):
        stable_target(AudioNode(69, "USB Audio", "sink"))


def test_volume_guard_restores_state_on_exception():
    state = {"sink_volume": 40, "source_volume": 100, "sink_mute": False, "source_mute": False}
    calls = []

    def runner(command, timeout=10.0):
        calls.append(command)
        if command[:2] == ["pactl", "get-default-sink"]:
            return 0, "sink", ""
        if command[:2] == ["pactl", "get-default-source"]:
            return 0, "source", ""
        if command[:2] == ["pactl", "get-sink-volume"]:
            return 0, f"Volume: front-left: {state['sink_volume'] * 65536 // 100} / {state['sink_volume']}% / 0 dB", ""
        if command[:2] == ["pactl", "get-source-volume"]:
            return 0, f"Volume: front-left: {state['source_volume'] * 65536 // 100} / {state['source_volume']}% / 0 dB", ""
        if command[:2] == ["pactl", "get-sink-mute"]:
            return 0, f"Mute: {'yes' if state['sink_mute'] else 'no'}", ""
        if command[:2] == ["pactl", "get-source-mute"]:
            return 0, f"Mute: {'yes' if state['source_mute'] else 'no'}", ""
        if command[:2] == ["pactl", "set-sink-volume"]:
            state["sink_volume"] = int(command[-1].rstrip("%"))
            return 0, "", ""
        if command[:2] == ["pactl", "set-source-volume"]:
            state["source_volume"] = int(command[-1].rstrip("%"))
            return 0, "", ""
        if command[:2] == ["pactl", "set-sink-mute"]:
            state["sink_mute"] = command[-1] == "yes"
            return 0, "", ""
        if command[:2] == ["pactl", "set-source-mute"]:
            state["source_mute"] = command[-1] == "yes"
            return 0, "", ""
        if command[:2] == ["pactl", "set-default-sink"] or command[:2] == ["pactl", "set-default-source"]:
            return 0, "", ""
        raise AssertionError(command)

    with pytest.raises(RuntimeError):
        with AudioVolumeGuard(
            speaker_target="sink",
            microphone_target="source",
            command_runner=runner,
        ) as guard:
            guard.set_volumes(speaker_percent=25, microphone_percent=50)
            guard.set_mutes(speaker_muted=False, microphone_muted=False)
            raise RuntimeError("synthetic failure")

    assert state == {"sink_volume": 40, "source_volume": 100, "sink_mute": False, "source_mute": False}
    assert any(command[:2] == ["pactl", "set-sink-volume"] for command in calls)
    assert any(command[:2] == ["pactl", "set-source-volume"] for command in calls)


def test_clipping_ratio_uses_inclusive_abs_threshold():
    samples = np.array([0.0, 0.5, 0.98, -0.99, 1.0], dtype=np.float32)
    assert clipping_ratio(samples) == pytest.approx(3 / 5)


def test_acoustic_onset_uses_noise_floor_and_consecutive_frames():
    sample_rate = 16000
    recording = np.zeros(sample_rate, dtype=np.float32)
    recording[6400:] = 0.2
    result = detect_acoustic_onset(
        recording,
        sample_rate,
        search_start_s=0.25,
        noise_window_s=0.2,
        frame_ms=10,
        min_consecutive_frames=2,
    )
    assert result["detected"] is True
    assert 0.39 <= result["onset_s"] <= 0.42
    assert result["noise_floor_rms"] == pytest.approx(0.0)


def test_assistant_only_vad_metrics_are_explicit_false_trigger_metrics():
    sample_rate = 16000
    audio = np.zeros(sample_rate, dtype=np.float32)
    audio[3200:6400] = 0.2
    result = assistant_only_vad_metrics(audio, sample_rate, playback_duration_s=1.0)
    assert result["false_trigger_count"] == 1
    assert result["speech_frames"] >= 8
    assert result["false_trigger_total_duration_s"] > 0.0
    assert result["false_trigger_ratio"] > 0.0


def test_playback_rejects_numeric_target_before_process_start(tmp_path):
    with pytest.raises(ValueError):
        PipeWirePlayback(123).play(tmp_path / "missing.wav")


def test_cpu_profile_labels_cold_and_warm_runs_separately():
    result = classify_cpu_asr_profile_phases(
        [
            {"phase": "first_transcription"},
            {"phase": "warm_transcription"},
            {"phase": "warm_transcription"},
        ]
    )
    assert result["cold_run_count"] == 1
    assert result["warm_run_count"] == 2
    assert result["cold_warm_separated"] is True


def test_aec_condition_aggregation_keeps_rows_and_median():
    rows = [
        {"status": "measured", "speaker_volume_percent": 25, "microphone_volume_percent": 25, "residual_echo_attenuation_db": 1.0},
        {"status": "measured", "speaker_volume_percent": 25, "microphone_volume_percent": 25, "residual_echo_attenuation_db": 3.0},
        {"status": "skipped_safety", "speaker_volume_percent": 100, "microphone_volume_percent": 100},
    ]
    summary = summarize_aec_conditions(rows)
    assert summary["condition_count"] == 3
    assert summary["measured_count"] == 2
    assert summary["attenuation_median_db"] == 2.0
    assert summary["rows"] == rows
