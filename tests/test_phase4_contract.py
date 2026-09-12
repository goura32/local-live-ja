import threading

import numpy as np
import pytest

from local_live.echo_rejection import (
    EchoRejectionConfig,
    analyze_echo_pair,
    calibrate_thresholds,
    classify_echo_candidate,
    evaluate_echo_rejection,
)
from local_live.llm.events import Cancelled, Completion, TextDelta
from local_live.phase4_bench import (
    STABILITY_INPUTS,
    classify_latency_outlier,
    percentile_summary,
)
from local_live.pipeline import LivePipeline


def _delayed(signal: np.ndarray, delay: int, gain: float) -> np.ndarray:
    result = np.zeros(len(signal) + delay, dtype=np.float32)
    result[delay:] = signal * gain
    return result


def _reference(sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(1.2 * sample_rate), dtype=np.float32) / sample_rate
    envelope = np.zeros_like(t)
    envelope[int(0.08 * sample_rate) : int(0.42 * sample_rate)] = 1.0
    envelope[int(0.58 * sample_rate) : int(1.08 * sample_rate)] = 1.0
    return (0.25 * envelope * (np.sin(2 * np.pi * 233 * t) + 0.5 * np.sin(2 * np.pi * 377 * t))).astype(np.float32)


def test_echo_analysis_reports_correlation_lag_and_energy_ratio():
    reference = _reference()
    microphone = _delayed(reference, 480, 0.35)
    metrics = analyze_echo_pair(reference, microphone, 16000, max_lag_s=0.1)
    assert metrics["correlation"] > 0.98
    assert metrics["lag_seconds"] == pytest.approx(0.03, abs=0.002)
    assert metrics["energy_ratio"] == pytest.approx(0.35, abs=0.03)
    assert metrics["window_metrics"]
    assert "residual_energy_ratio" in metrics["window_metrics"][0]


def test_echo_classifier_rejects_reference_explained_signal():
    reference = _reference()
    microphone = _delayed(reference, 320, 0.3)
    config = EchoRejectionConfig(correlation_threshold=0.9, residual_energy_ratio_threshold=0.25, max_lag_s=0.1)
    decision = classify_echo_candidate(reference, microphone, 16000, playback_active=True, config=config)
    assert decision["decision"] == "probable_self_echo"
    assert decision["reject"] is True


def test_echo_classifier_keeps_unexplained_user_like_component():
    reference = _reference()
    echo = _delayed(reference, 320, 0.25)
    user = np.zeros_like(echo)
    t = np.arange(len(user), dtype=np.float32) / 16000
    user[int(0.45 * 16000) : int(0.95 * 16000)] = 0.35 * np.sin(2 * np.pi * 911 * t[int(0.45 * 16000) : int(0.95 * 16000)])
    microphone = echo + user
    config = EchoRejectionConfig(correlation_threshold=0.9, residual_energy_ratio_threshold=0.25, max_lag_s=0.1)
    decision = classify_echo_candidate(reference, microphone, 16000, playback_active=True, config=config)
    assert decision["decision"] == "possible_user_speech"
    assert decision["reject"] is False


def test_echo_classifier_accepts_without_reference_or_when_playback_inactive():
    reference = _reference()
    microphone = reference.copy()
    assert classify_echo_candidate(None, microphone, 16000, playback_active=True)["reject"] is False
    assert classify_echo_candidate(reference, microphone, 16000, playback_active=False)["reject"] is False


def test_echo_calibration_and_dataset_metrics_meet_fixture_targets():
    reference = _reference()
    fixtures = []
    for index in range(8):
        echo = _delayed(reference, 240 + index * 40, 0.2 + index * 0.01)
        fixtures.append({"label": "assistant_only", "name": f"echo_{index}", "reference": reference, "microphone": echo, "playback_active": True})
        user = np.zeros_like(echo)
        start = (0.42 + 0.02 * (index % 3)) * 16000
        end = (0.9 + 0.01 * (index % 3)) * 16000
        t = np.arange(len(user), dtype=np.float32) / 16000
        user[int(start) : int(end)] = 0.4 * np.sin(2 * np.pi * (811 + index * 17) * t[int(start) : int(end)])
        fixtures.append({"label": "synthetic_user_like", "name": f"user_{index}", "reference": reference, "microphone": echo + user, "playback_active": True})
    calibration = calibrate_thresholds(fixtures, 16000)
    result = evaluate_echo_rejection(fixtures, 16000, config=calibration)
    assert result["status"] == "measured"
    assert result["assistant_only"]["false_accept_rate"] <= 0.10
    assert result["synthetic_user_like"]["acceptance_rate"] >= 0.90
    assert result["correlation_distribution"]["count"] == 16
    assert result["energy_ratio_distribution"]["count"] == 16


def test_percentiles_include_p90_p95_p99_and_all_values():
    result = percentile_summary([1.0, 2.0, 3.0, 10.0])
    assert result["count"] == 4
    assert result["values"] == [1.0, 2.0, 3.0, 10.0]
    assert result["p90"] == pytest.approx(7.9)
    assert result["p95"] == pytest.approx(8.95)
    assert result["p99"] == pytest.approx(9.79)
    assert result["max"] == 10.0


def test_latency_outlier_uses_component_medians_not_only_fixed_cutoff():
    row = {"status": "measured", "stages": {"asr": 0.4, "tts_first_pcm": 0.8, "playback": 0.1}}
    medians = {"asr": 0.3, "tts_first_pcm": 0.05, "playback": 0.1}
    result = classify_latency_outlier(row, medians)
    assert result["is_outlier"] is True
    assert result["dominant_cause"] == "tts_first_pcm"
    assert result["stage_flags"]["tts_first_pcm"]["ratio_to_median"] > 10


class TwoChunkLLM:
    requested_model = "fake"

    def stream(self, messages, tools=None, cancel_event=None):
        yield TextDelta(text="最初の文です。")
        yield TextDelta(text="キャンセル後の文です。")
        yield Completion(reason="stop", actual_model="fake")


class CancellingPlayback:
    def __init__(self):
        self.played = []
        self.calls = 0

    def play(self, path, cancel_event=None):
        self.played.append(path)
        self.calls += 1
        if self.calls == 1:
            cancel_event.set()
            return {"path": path, "cancelled": False}
        return {"path": path, "cancelled": True}


class FakeTTS:
    def synthesize(self, text, output_path=None, cancel_event=None):
        return {"path": str(output_path), "text": text}


def test_pipeline_commits_only_spoken_text_on_cancel(tmp_path):
    playback = CancellingPlayback()
    result = LivePipeline(llm=TwoChunkLLM(), tts=FakeTTS(), playback=playback, artifact_dir=tmp_path).respond("入力")
    assert result.cancelled is True
    assert result.spoken_text == "最初の文です。"
    assert result.assistant_text == "最初の文です。"
    assert result.generated_text == "最初の文です。キャンセル後の文です。"
    assert result.state == "IDLE"
    assert [event.kind for event in result.events if event.kind == "spoken_text_committed"] == ["spoken_text_committed"]


def test_stability_fixture_set_has_multiple_japanese_inputs():
    assert len(STABILITY_INPUTS) >= 5
    assert len({item["name"] for item in STABILITY_INPUTS}) == len(STABILITY_INPUTS)
    assert all(item["text"] for item in STABILITY_INPUTS)


class BlockingLLM:
    requested_model = "blocking"

    def __init__(self):
        self.started = threading.Event()

    def stream(self, messages, tools=None, cancel_event=None):
        self.started.set()
        while cancel_event is None or not cancel_event.is_set():
            self.started.wait(0.005)
        yield Cancelled(reason="interrupt")


class CancelHook:
    def __init__(self):
        self.calls = 0

    def cancel(self):
        self.calls += 1


def test_pipeline_cancel_propagates_to_tts_and_playback_and_returns():
    llm = BlockingLLM()
    tts = CancelHook()
    playback = CancelHook()
    pipeline = LivePipeline(llm=llm, tts=tts, playback=playback)
    results = []
    worker = threading.Thread(target=lambda: results.append(pipeline.respond("割り込み")), daemon=True)
    worker.start()
    assert llm.started.wait(timeout=1.0)
    pipeline.cancel()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert results[0].cancelled is True
    assert tts.calls == 1
    assert playback.calls == 1


def test_pipeline_exposes_echo_aware_post_vad_gate():
    reference = np.sin(np.linspace(0, 30, 16000)).astype(np.float32)
    microphone = np.concatenate([np.zeros(320, dtype=np.float32), reference * 0.25])
    pipeline = LivePipeline(llm=BlockingLLM(), tts=CancelHook(), playback=CancelHook(), echo_rejection_config=EchoRejectionConfig(correlation_threshold=0.8, residual_energy_ratio_threshold=0.4, max_lag_s=0.05, energy_ratio_max=0.5))
    echo = pipeline.classify_vad_candidate(reference, microphone, 16000, playback_active=True)
    user = pipeline.classify_vad_candidate(None, microphone, 16000, playback_active=True)
    assert echo["decision"] == "probable_self_echo"
    assert echo["reject"] is True
    assert user["decision"] == "possible_user_speech"
    assert user["reject"] is False
