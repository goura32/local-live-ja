from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from local_live.llm.events import Completion, TextDelta
from local_live.pipeline import LivePipeline
from local_live.session import (
    ConversationHistory,
    FixtureAudioSource,
    SessionController,
    SessionState,
    StreamingVAD,
    VADConfig,
)


class IncrementalLLM:
    requested_model = "fake-qwen"

    def __init__(self, text: str = "最初の文です。次の文も生成中です。") -> None:
        self.text = text
        self.completion_seen = threading.Event()
        self.requests: list[list[dict[str, object]]] = []

    def stream(self, messages, tools=None, cancel_event=None):
        self.requests.append(messages)
        yield TextDelta("最初の文です。")
        if cancel_event is not None and cancel_event.is_set():
            return
        yield TextDelta("次の文も生成中です。")
        self.completion_seen.set()
        yield Completion(reason="stop", actual_model="fake-qwen")


class RecordingTTS:
    streaming = False

    def __init__(self, llm: IncrementalLLM | None = None) -> None:
        self.texts: list[str] = []
        self.started_before_completion = False
        self.llm = llm

    def synthesize(self, text, output_path=None, cancel_event=None):
        self.texts.append(text)
        if self.llm is not None:
            self.started_before_completion = not self.llm.completion_seen.is_set()
        return {"text": text, "path": str(output_path)}


class RecordingPlayback:
    def __init__(self) -> None:
        self.played: list[str] = []
        self.cancelled = False

    def play(self, path, cancel_event=None):
        self.played.append(path)
        return {"path": path, "cancelled": self.cancelled or bool(cancel_event and cancel_event.is_set())}

    def cancel(self):
        self.cancelled = True
        return {"cancelled": True}


class BlockingPlayback(RecordingPlayback):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.released = threading.Event()

    def play(self, path, cancel_event=None):
        self.played.append(path)
        self.started.set()
        while not (cancel_event and cancel_event.is_set()) and not self.released.is_set():
            time.sleep(0.005)
        return {"path": path, "cancelled": bool(cancel_event and cancel_event.is_set())}

    def cancel(self):
        self.cancelled = True
        self.released.set()
        return {"cancelled": True}


class ScriptedASR:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)

    def transcribe_samples(self, samples, *, sample_rate=16000, event_log=None):
        return SimpleNamespace(text=self.texts.pop(0), to_dict=lambda: {"text": self.texts[0] if self.texts else ""})


def speech(duration_s: float = 0.24, sample_rate: int = 16000, amplitude: float = 0.2) -> np.ndarray:
    count = int(duration_s * sample_rate)
    t = np.arange(count, dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(duration_s: float = 0.16, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


def test_history_commits_only_spoken_text_and_trims_pairs() -> None:
    history = ConversationHistory("system", max_turns=2, max_chars=200)
    history.append_user("一")
    history.commit_assistant_spoken("返答一")
    history.append_user("二")
    history.commit_assistant_spoken("返答二")
    history.append_user("三")
    history.commit_assistant_spoken("生成されたが未再生")
    messages = history.messages()
    assert [item["role"] for item in messages] == ["system", "user", "assistant", "user", "assistant"]
    assert "一" not in " ".join(str(item["content"]) for item in messages)
    history.reset()
    assert history.messages() == [{"role": "system", "content": "system"}]


def test_history_truncates_oversize_message_with_evidence() -> None:
    history = ConversationHistory("system", max_turns=2, max_chars=5)
    history.append_user("abcdefgh")
    assert history.messages()[-1] == {"role": "user", "content": "abcde"}
    assert history.to_dict()["truncated_count"] == 1


def test_streaming_vad_emits_start_continuation_end_and_rejects_short() -> None:
    vad = StreamingVAD(VADConfig(min_speech_duration_s=0.12, end_silence_s=0.10, frame_ms=20, threshold=0.02))
    events = vad.process(speech()[:1600])
    events += vad.process(speech()[1600:])
    events += vad.process(silence())
    assert [event.kind for event in events] == ["speech_start", "speech_continuation", "speech_end"]
    assert events[-1].duration_s >= 0.12
    short = StreamingVAD(VADConfig(min_speech_duration_s=0.3, end_silence_s=0.1, frame_ms=20, threshold=0.02))
    short_events = short.process(speech(0.08)) + short.process(silence())
    assert short_events[-1].kind == "too_short"


def test_streaming_vad_does_not_learn_speech_as_noise_over_repeated_turns() -> None:
    vad = StreamingVAD(VADConfig(min_speech_duration_s=0.12, end_silence_s=0.10, frame_ms=20, threshold=0.02))
    starts = 0
    for _ in range(20):
        starts += sum(event.kind == "speech_start" for event in vad.process(speech()))
        vad.process(silence())
    assert starts == 20


def test_incremental_pipeline_starts_tts_before_llm_completion(tmp_path) -> None:
    llm = IncrementalLLM()
    tts = RecordingTTS(llm)
    pipeline = LivePipeline(llm=llm, tts=tts, playback=RecordingPlayback(), artifact_dir=tmp_path)
    result = pipeline.respond("こんにちは")
    assert result.error is None
    assert tts.texts == ["最初の文です。", "次の文も生成中です。"]
    assert tts.started_before_completion is True
    assert result.spoken_text == result.generated_text


def test_session_controller_multiturn_history_and_reset(tmp_path) -> None:
    llm = IncrementalLLM("はい。覚えています。")
    tts = RecordingTTS(llm)
    pipeline = LivePipeline(llm=llm, tts=tts, playback=RecordingPlayback(), artifact_dir=tmp_path)
    source = FixtureAudioSource([speech(), silence(), speech(), silence(), speech(), silence()], delay_s=0.04)
    controller = SessionController(
        source=source,
        asr=ScriptedASR(["合言葉は青い星です", "別の質問です", "さっきの合言葉は？"]),
        pipeline=pipeline,
        config=VADConfig(min_speech_duration_s=0.12, end_silence_s=0.10, frame_ms=20, threshold=0.02),
        artifact_dir=tmp_path,
    )
    summary = controller.run(max_turns=3)
    assert summary["application_success_count"] == 3
    assert summary["state"] == SessionState.IDLE.value
    assert [item["role"] for item in controller.history.messages()] == [
        "system", "user", "assistant", "user", "assistant", "user", "assistant"
    ]
    assert len(llm.requests) == 3
    assert any(item["content"] == "合言葉は青い星です" for item in llm.requests[-1])
    controller.reset()
    assert controller.history.messages() == [controller.history.messages()[0]]


@pytest.mark.integration
def test_phase7_application_acceptance_harness(tmp_path):
    from local_live.app_bench import run_application_bench

    config = {
        "app": {"artifact_dir": str(tmp_path / "artifacts"), "result_dir": str(tmp_path / "results")},
        "chat": {"history_max_turns": 12, "history_max_chars": 8000},
        "tts": {"sentence_max_chars": 48, "sentence_timeout_s": 0.8},
    }
    result = run_application_bench(config, turns=50)
    assert result["data"]["status"] == "unattended_poc_complete"
    assert result["data"]["checks"]


def test_synthetic_application_barge_in_cancels_and_starts_next_turn(tmp_path) -> None:
    llm = IncrementalLLM("応答を再生します。")
    playback = BlockingPlayback()
    pipeline = LivePipeline(llm=llm, tts=RecordingTTS(llm), playback=playback, artifact_dir=tmp_path)
    controller = SessionController(
        source=FixtureAudioSource([]),
        asr=ScriptedASR(["最初の質問", "割り込みの質問"]),
        pipeline=pipeline,
        config=VADConfig(min_speech_duration_s=0.12, end_silence_s=0.10, frame_ms=20, threshold=0.02),
        artifact_dir=tmp_path,
    )
    controller.start()
    controller.submit_utterance(speech())
    assert playback.started.wait(timeout=2.0)
    controller.process_audio_chunk(speech())
    controller.process_audio_chunk(silence())
    controller.wait_for_idle(timeout=3.0)
    assert controller.barge_in_count == 1
    assert controller.cancel_count == 1
    assert controller.state == SessionState.LISTENING
    assert controller.history.spoken_assistant_texts() == ["最初の文です。次の文も生成中です。"]
    controller.stop()
