from local_live.llm.events import Completion, TextDelta
from local_live.pipeline import Cancellation, LivePipeline


class OneChunkLLM:
    requested_model = "fake"

    def stream(self, messages, tools=None, cancel_event=None):
        yield TextDelta(text="再生中の返答です。")
        yield Completion(reason="stop", actual_model="fake")


class CancellingPlayback:
    def play(self, path, cancel_event=None):
        assert cancel_event is not None
        cancel_event.set()
        return {"path": path, "cancelled": True}


class RecordingTTS:
    def synthesize(self, text, output_path=None, cancel_event=None):
        return {"path": str(output_path), "text": text}


def test_live_pipeline_propagates_playback_cancellation(tmp_path):
    result = LivePipeline(
        llm=OneChunkLLM(),
        tts=RecordingTTS(),
        playback=CancellingPlayback(),
        artifact_dir=tmp_path,
    ).respond("再生キャンセル")
    assert result.cancelled is True
    assert result.events[-1].kind == "cancelled"
    assert result.timing["cancel_completed"] >= result.timing["cancel_requested"]
