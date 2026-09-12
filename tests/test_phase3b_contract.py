import subprocess
import wave

import numpy as np
import pytest

from local_live.audio import PipeWirePCMPlayback
from local_live.tts_backends import build_tts_backend
from local_live.vllm_omni_tts import (
    PCMChunkParser,
    VLLMOmniTTSEngine,
    aggregate_stream_timing,
    decode_pcm16,
)


class FakeStdin:
    def __init__(self):
        self.writes = []
        self.closed = False

    def write(self, value):
        self.writes.append(value)
        return len(value)

    def flush(self):
        return None

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self):
        self.stdin = FakeStdin()
        self.returncode = 0
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return None if not self.terminated and not self.killed else self.returncode

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def communicate(self, timeout=None):
        self.waited = True
        return b"", b""


def test_pcm_chunk_parser_preserves_order_and_reassembles_odd_network_boundaries():
    parser = PCMChunkParser(sample_width=2)
    assert parser.feed(b"\x01") == []
    assert parser.feed(b"\x00\x02") == [b"\x01\x00"]
    assert parser.feed(b"\x00\x03\x00") == [b"\x02\x00", b"\x03\x00"]
    parser.finish()


def test_pcm_chunk_parser_ignores_empty_chunks_and_rejects_truncated_sample():
    parser = PCMChunkParser(sample_width=2)
    assert parser.feed(b"") == []
    parser.feed(b"\x01")
    with pytest.raises(ValueError, match="incomplete PCM sample"):
        parser.finish()


def test_decode_pcm16_handles_empty_and_returns_float32():
    assert decode_pcm16(b"").dtype == np.float32
    values = decode_pcm16(b"\x00\x80\xff\x7f")
    assert values.dtype == np.float32
    assert values[0] == pytest.approx(-1.0)
    assert values[1] > 0.99


def test_decode_pcm16_rejects_odd_byte_count():
    with pytest.raises(ValueError, match="even number of bytes"):
        decode_pcm16(b"\x00")


def test_persistent_playback_queues_one_stream_and_finishes_without_per_chunk_process():
    process = FakeProcess()
    playback = PipeWirePCMPlayback(
        "USB Speaker",
        popen_factory=lambda command, **kwargs: process,
    )
    started = playback.start(sample_rate=24000, channels=1)
    assert started["started_ns"] > 0
    assert playback.queue(b"first") == {"queued": True}
    assert playback.queue(b"second") == {"queued": True}
    result = playback.finish()
    assert result["cancelled"] is False
    assert process.stdin.writes == [b"first", b"second"]
    assert process.stdin.closed is True
    assert process.waited is True


def test_persistent_playback_cancel_terminates_and_rejects_late_pcm():
    process = FakeProcess()
    playback = PipeWirePCMPlayback(
        "USB Speaker",
        popen_factory=lambda command, **kwargs: process,
    )
    playback.start(sample_rate=24000, channels=1)
    result = playback.cancel()
    assert result["cancelled"] is True
    assert process.terminated is True
    with pytest.raises(RuntimeError, match="not active"):
        playback.queue(b"stale")


def test_persistent_playback_rejects_numeric_target_and_reports_spawn_error():
    with pytest.raises(ValueError, match="numeric"):
        PipeWirePCMPlayback(42)

    def fail_spawn(command, **kwargs):
        raise OSError("missing pw-cat")

    playback = PipeWirePCMPlayback("USB Speaker", popen_factory=fail_spawn)
    with pytest.raises(RuntimeError, match="pw-cat unavailable"):
        playback.start(sample_rate=24000, channels=1)


def test_backend_switch_keeps_python_and_builds_vllm_backend_without_importing_server():
    python_backend = build_tts_backend({"tts": {"backend": "python"}})
    assert python_backend.__class__.__name__ == "Qwen3TTSEngine"
    vllm_backend = build_tts_backend(
        {
            "tts": {
                "backend": "vllm_omni",
                "vllm_base_url": "http://127.0.0.1:8091/v1",
                "vllm_streaming": True,
            }
        }
    )
    assert vllm_backend.__class__.__name__ == "VLLMOmniTTSEngine"
    assert vllm_backend.streaming is True


def test_backend_switch_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported TTS backend"):
        build_tts_backend({"tts": {"backend": "unknown"}})


def test_stream_timing_uses_monotonic_boundaries_and_preserves_missing_values():
    timing = aggregate_stream_timing(
        {
            "request_start": 100,
            "first_text_sent": 120,
            "first_audio_chunk_received": 320,
            "first_audio_chunk_queued": 330,
            "playback_stream_started": 325,
            "last_audio_chunk_received": 520,
            "playback_completed": 620,
        },
        audio_duration_s=0.5,
    )
    assert timing["request_to_first_audio_chunk_s"] == pytest.approx(0.00000022)
    assert timing["first_audio_chunk_receive_to_queue_s"] == pytest.approx(0.00000001)
    assert timing["request_to_physical_audio_s"] is None
    assert timing["rtf"] == pytest.approx(0.00000042 / 0.5)


def test_stream_timing_rejects_non_monotonic_boundaries():
    with pytest.raises(ValueError, match="monotonic"):
        aggregate_stream_timing(
            {"request_start": 10, "first_audio_chunk_received": 9},
            audio_duration_s=1.0,
        )


def test_stream_backend_cancellation_does_not_leave_playback_queue():
    process = FakeProcess()
    playback = PipeWirePCMPlayback("USB Speaker", popen_factory=lambda command, **kwargs: process)
    playback.start(sample_rate=24000, channels=1)
    playback.queue(b"audio")
    playback.cancel()
    assert playback.active is False
    assert process.stdin.closed is True


def test_no_real_subprocess_is_spawned_by_contract_fakes(monkeypatch):
    called = []

    def forbidden(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("real process must not be spawned")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    process = FakeProcess()
    playback = PipeWirePCMPlayback("USB Speaker", popen_factory=lambda command, **kwargs: process)
    playback.start(sample_rate=24000, channels=1)
    playback.finish()
    assert called == []


def _wav_bytes() -> bytes:
    pcm = b"\x00\x00\xff\x7f" * 120
    from io import BytesIO

    handle = BytesIO()
    with wave.open(handle, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(pcm)
    return handle.getvalue()


class FakeHTTPResponse:
    status_code = 200
    headers = {"content-type": "audio/wav"}

    def __init__(self, *, content=b"", chunks=None, error=None, stream_error=None):
        self.content = content
        self._chunks = chunks or []
        self._header_error = error
        self._stream_error = stream_error

    def raise_for_status(self):
        if self._header_error:
            raise self._header_error

    def json(self):
        return {"voices": ["Ono_Anna"]}

    def iter_bytes(self):
        yield from self._chunks
        if self._stream_error:
            raise self._stream_error


class FakeStreamContext:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self.response

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeHTTPClient:
    def __init__(self, response):
        self.response = response
        self.payloads = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, json):
        self.payloads.append(json)
        return self.response

    def stream(self, method, url, *, json):
        self.payloads.append(json)
        return FakeStreamContext(self.response)

    def get(self, url):
        return self.response


def test_vllm_non_streaming_request_saves_wav_and_exact_fixed_fields(tmp_path):
    client = FakeHTTPClient(FakeHTTPResponse(content=_wav_bytes()))
    engine = VLLMOmniTTSEngine(
        base_url="http://127.0.0.1:8091/v1",
        model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        speaker="Ono_Anna",
        language="Japanese",
        client_factory=lambda **kwargs: client,
    )
    result = engine.synthesize("はい、確認しました。", output_path=tmp_path / "answer.wav")
    assert result["status"] == "measured"
    assert result["sample_rate"] == 24000
    assert client.payloads[0] == {
        "input": "はい、確認しました。",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "voice": "Ono_Anna",
        "language": "Japanese",
        "task_type": "CustomVoice",
        "response_format": "wav",
    }


def test_vllm_streaming_queues_raw_pcm_once_and_records_first_chunk(tmp_path):
    pcm = b"\x00\x00\xff\x7f" * 120
    response = FakeHTTPResponse(chunks=[b"", pcm[:1], pcm[1:5], pcm[5:]])
    client = FakeHTTPClient(response)
    process = FakeProcess()
    playback = PipeWirePCMPlayback("USB Speaker", popen_factory=lambda command, **kwargs: process)
    engine = VLLMOmniTTSEngine(client_factory=lambda **kwargs: client, streaming=True)
    result = engine.synthesize_stream("はい。", output_path=tmp_path / "stream.wav", playback=playback)
    assert result["status"] == "measured"
    assert result["timing_ns"]["first_audio_chunk_received"] is not None
    assert result["timing_ns"]["first_audio_chunk_queued"] is not None
    assert result["timing_ns"]["playback_stream_started"] is not None
    assert b"".join(process.stdin.writes) == pcm
    assert client.payloads[0]["stream"] is True
    assert client.payloads[0]["stream_format"] == "audio"
    assert client.payloads[0]["response_format"] == "pcm"


def test_vllm_streaming_connection_interruption_cleans_persistent_playback(tmp_path):
    pcm = b"\x00\x00\xff\x7f" * 4
    response = FakeHTTPResponse(chunks=[pcm], stream_error=RuntimeError("connection interrupted"))
    client = FakeHTTPClient(response)
    process = FakeProcess()
    playback = PipeWirePCMPlayback("USB Speaker", popen_factory=lambda command, **kwargs: process)
    engine = VLLMOmniTTSEngine(client_factory=lambda **kwargs: client, streaming=True)
    result = engine.synthesize_stream("確認します。", output_path=tmp_path / "broken.wav", playback=playback)
    assert result["status"] == "error"
    assert "connection interrupted" in result["error"]
    assert playback.active is False
    assert process.terminated is True


def test_vllm_streaming_midstream_cancel_clears_queue(tmp_path):
    class CancelAfterFirst:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 2

    pcm = b"\x00\x00\xff\x7f" * 8
    client = FakeHTTPClient(FakeHTTPResponse(chunks=[pcm, pcm]))
    process = FakeProcess()
    playback = PipeWirePCMPlayback("USB Speaker", popen_factory=lambda command, **kwargs: process)
    engine = VLLMOmniTTSEngine(client_factory=lambda **kwargs: client, streaming=True)
    result = engine.synthesize_stream(
        "設定を確認します。",
        output_path=tmp_path / "cancelled.wav",
        playback=playback,
        cancel_event=CancelAfterFirst(),
    )
    assert result["status"] == "cancelled"
    assert result["cancelled"] is True
    assert playback.active is False
    assert process.terminated is True


def test_vllm_health_reports_server_unavailable_without_guessing_success():
    import httpx

    client = FakeHTTPClient(FakeHTTPResponse(error=httpx.ConnectError("connection refused")))
    engine = VLLMOmniTTSEngine(client_factory=lambda **kwargs: client)
    result = engine.health()
    assert result["status"] == "unavailable"
    assert result["error_type"] == "ConnectError"
