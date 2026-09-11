import numpy as np
import soundfile as sf

import local_live.tts as tts_module
from local_live.audio import audio_file_stats
from local_live.bench import _summarize_runs, _tool_call_messages
from local_live.llm.events import ToolCall
from local_live.tools import MockToolRegistry
from local_live.tts import Qwen3TTSEngine


def test_ollama_followup_uses_native_object_arguments():
    call = ToolCall(call_id="call-1", name="calculator", arguments={"expression": "17*23"})
    messages, _ = _tool_call_messages(
        [{"role": "user", "content": "計算して"}],
        [call],
        MockToolRegistry(),
        argument_format="ollama",
    )
    assistant_call = messages[-2]["tool_calls"][0]
    assert assistant_call == {
        "function": {"name": "calculator", "arguments": {"expression": "17*23"}}
    }


def test_repeated_run_summary_keeps_values_and_computes_median():
    runs = [
        {"asr_duration_s": 1.0, "e2e_duration_s": 1.0},
        {"asr_duration_s": 100.0, "e2e_duration_s": 100.0},
        {"asr_duration_s": 2.0, "e2e_duration_s": 2.0},
    ]
    summary = _summarize_runs(runs, ["asr_duration_s", "e2e_duration_s"])
    assert summary["asr_duration_s"] == {"values": [1.0, 100.0, 2.0], "median": 2.0}
    assert summary["e2e_duration_s"]["values"] == [1.0, 100.0, 2.0]


def test_tts_result_exposes_generation_and_playback_boundaries(tmp_path, monkeypatch):
    class FakeMonitor:
        gpu_memory_peak_mib = 0
        gpu_memory_delta_peak_mib = 0
        cpu_load_percent = 0.0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeModel:
        def generate_custom_voice(self, **kwargs):
            return [np.zeros(100, dtype=np.float32)], 1000

    monkeypatch.setattr(tts_module, "ResourceMonitor", FakeMonitor)
    engine = Qwen3TTSEngine(device="cpu")
    engine._model = FakeModel()
    engine.resolved_device = "cpu"
    result = engine.synthesize("短い文です。", output_path=tmp_path / "out.wav")

    timing = result["timing_ns"]
    assert timing["inference_start"] <= timing["audio_complete"]
    assert timing["audio_complete"] <= timing["playback_possible"]
    assert result["audio_complete_seconds"] >= result["first_audio_equivalent_seconds"]
    assert result["playback_possible_seconds"] >= result["audio_complete_seconds"]


def test_audio_file_stats_exposes_actual_duration_rms_and_peak(tmp_path):
    path = tmp_path / "capture.wav"
    sf.write(path, np.array([0.0, 0.5, -0.25], dtype=np.float32), 1000)
    stats = audio_file_stats(path)
    assert stats["duration_s"] == 0.003
    assert stats["rms"] > 0.0
    assert stats["peak"] == 0.5
