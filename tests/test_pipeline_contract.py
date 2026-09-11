import threading

from local_live.llm.events import Completion, TextDelta
from local_live.pipeline import Cancellation, LivePipeline


class FakeLLM:
    requested = []

    def stream(self, messages, tools=None, cancel_event=None):
        self.requested.append(messages)
        yield TextDelta(text="短い返答です。")
        if cancel_event and cancel_event.is_set():
            return
        yield Completion(reason="stop", actual_model="fake")


class FakeTTS:
    def __init__(self):
        self.texts = []

    def synthesize(self, text, output_path=None, cancel_event=None):
        self.texts.append(text)
        return {"text": text, "path": str(output_path) if output_path else None}


class FakePlayback:
    def __init__(self):
        self.played = []

    def play(self, path, cancel_event=None):
        self.played.append(path)
        return {"path": path, "cancelled": bool(cancel_event and cancel_event.is_set())}


class SlowLLM:
    def stream(self, messages, tools=None, cancel_event=None):
        yield TextDelta(text="生成中")
        if cancel_event:
            cancel_event.set()
        yield TextDelta(text="ですが続けません。")
        yield Completion(reason="cancelled", actual_model="slow")


def test_live_pipeline_turn_reaches_tts_and_playback(tmp_path):
    tts = FakeTTS()
    playback = FakePlayback()
    pipeline = LivePipeline(llm=FakeLLM(), tts=tts, playback=playback, artifact_dir=tmp_path)
    result = pipeline.respond("ユーザーの発話")
    assert result.cancelled is False
    assert tts.texts == ["短い返答です。"]
    assert len(playback.played) == 1


def test_live_pipeline_cancel_stops_pending_work(tmp_path):
    tts = FakeTTS()
    pipeline = LivePipeline(llm=SlowLLM(), tts=tts, playback=FakePlayback(), artifact_dir=tmp_path)
    result = pipeline.respond("キャンセル試験")
    assert result.cancelled is True
    assert tts.texts == []
    assert result.events[-1].kind == "cancelled"
