import threading
import time

from local_live.llm.events import Completion, TextDelta
from local_live.pipeline import LivePipeline


class BlockingLLM:
    requested_model = "fake"

    def stream(self, messages, tools=None, cancel_event=None):
        yield TextDelta(text="再生を開始します。")
        yield Completion(reason="stop", actual_model="fake")


class BlockingPlayback:
    def __init__(self):
        self.started = threading.Event()

    def play(self, path, cancel_event=None):
        self.started.set()
        assert cancel_event is not None
        cancel_event.wait(timeout=5.0)
        return {"path": path, "cancelled": cancel_event.is_set()}


class NoopTTS:
    def synthesize(self, text, output_path=None, cancel_event=None):
        return {"path": str(output_path), "text": text}


def test_external_cancel_interrupts_active_playback(tmp_path):
    playback = BlockingPlayback()
    pipeline = LivePipeline(llm=BlockingLLM(), tts=NoopTTS(), playback=playback, artifact_dir=tmp_path)
    result_holder = []
    worker = threading.Thread(target=lambda: result_holder.append(pipeline.respond("外部キャンセル")))
    worker.start()
    assert playback.started.wait(timeout=2.0)
    pipeline.cancel()
    worker.join(timeout=3.0)
    assert not worker.is_alive()
    assert result_holder[0].cancelled is True
    assert result_holder[0].timing["cancel_completed"] >= result_holder[0].timing["cancel_requested"]
