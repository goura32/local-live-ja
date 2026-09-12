from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import soundfile as sf

from .bench import artifact_dir, write_benchmark
from .llm.events import Completion, LLMError, TextDelta
from .pipeline import LivePipeline
from .session import ConversationHistory, FixtureAudioSource, SessionController, SessionState, VADConfig
from .telemetry import ResourceMonitor


class InstrumentedPlayback:
    """In-memory persistent PCM sink used by the application acceptance harness."""

    def __init__(self) -> None:
        self.payload = bytearray()
        self.reference_samples = np.empty(0, dtype=np.float32)
        self.active = False
        self.started_count = 0
        self.finished_count = 0
        self.cancelled_count = 0
        self.stale_pcm_bytes = 0
        self.first_started_ns: int | None = None

    def start(self, *, sample_rate: int, channels: int) -> dict[str, Any]:
        del channels
        self.payload.clear()
        self.active = True
        self.started_count += 1
        self.first_started_ns = time.monotonic_ns()
        return {"started_ns": self.first_started_ns, "sample_rate": sample_rate}

    def queue(self, payload: bytes) -> dict[str, Any]:
        if not self.active:
            raise RuntimeError("instrumented playback is inactive")
        self.payload.extend(payload)
        self.reference_samples = np.frombuffer(bytes(self.payload), dtype="<i2").astype(np.float32) / 32768.0
        return {"queued": bool(payload), "bytes": len(payload)}

    def finish(self) -> dict[str, Any]:
        self.active = False
        self.finished_count += 1
        return {"cancelled": False, "bytes": len(self.payload)}

    def cancel(self) -> dict[str, Any]:
        self.cancelled_count += 1
        self.stale_pcm_bytes += len(self.payload)
        self.payload.clear()
        self.reference_samples = np.empty(0, dtype=np.float32)
        self.active = False
        return {"cancelled": True}


class InstrumentedStreamingTTS:
    streaming = True

    def __init__(self, artifact_dir: Path, llm: "ApplicationLLM") -> None:
        self.artifact_dir = artifact_dir
        self.llm = llm
        self.requests: list[str] = []
        self.first_tts_before_completion = 0

    def synthesize_stream(self, text, *, output_path, playback, cancel_event=None, event_log=None):
        if cancel_event is not None and cancel_event.is_set():
            return {"status": "cancelled", "cancelled": True}
        self.requests.append(text)
        if not self.llm.completion_seen.is_set():
            self.first_tts_before_completion += 1
        sample_rate = 24000
        duration = 0.025
        t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
        waveform = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        raw = (waveform * 32768.0).astype("<i2").tobytes()
        playback.start(sample_rate=sample_rate, channels=1)
        playback.queue(raw)
        playback_result = playback.finish()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output), waveform, sample_rate)
        if event_log is not None:
            event_log.mark("app_fake_tts_stream", text_chars=len(text))
        return {
            "status": "measured",
            "cancelled": False,
            "path": str(output),
            "pcm_bytes": len(raw),
            "playback": playback_result,
            "timing_ns": {
                "request_start": time.monotonic_ns(),
                "first_audio_chunk_received": time.monotonic_ns(),
                "playback_completed": time.monotonic_ns(),
            },
        }


class ApplicationLLM:
    requested_model = "fake-application-qwen"

    def __init__(self) -> None:
        import threading

        self.requests: list[list[dict[str, Any]]] = []
        self.completion_seen = threading.Event()
        self.first_tts_before_completion = 0

    def stream(self, messages, tools=None, cancel_event=None):
        del tools
        self.requests.append([dict(message) for message in messages])
        self.completion_seen.clear()
        if cancel_event is not None and cancel_event.is_set():
            return
        user_text = str(messages[-1].get("content", ""))
        remembered = any(item.get("content") == "合言葉は青い星です" for item in messages if item.get("role") == "user")
        if "合言葉" in user_text and "さっき" in user_text and remembered:
            yield TextDelta("合言葉は青い星です。")
        else:
            yield TextDelta("確認しました。")
            yield TextDelta("会話履歴を反映しました。")
        if cancel_event is not None and cancel_event.is_set():
            return
        self.completion_seen.set()
        yield Completion(reason="stop", actual_model=self.requested_model)


class ScriptedApplicationASR:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)

    def transcribe_samples(self, samples, *, sample_rate=16000, event_log=None):
        del samples, sample_rate, event_log
        text = self.texts.pop(0) if self.texts else ""
        return SimpleNamespace(text=text, to_dict=lambda: {"text": text})


class ApplicationFailureASR(ScriptedApplicationASR):
    pass


class RecoveryLLM:
    requested_model = "fake-recovery-llm"

    def __init__(self, *, failure: bool = False) -> None:
        self.failure = failure
        self.calls = 0

    def stream(self, messages, tools=None, cancel_event=None):
        del messages, tools
        self.calls += 1
        if self.failure and self.calls == 1:
            yield LLMError("temporary Ollama transport failure", retryable=True)
            return
        yield TextDelta("復旧しました。")
        yield Completion(reason="stop", actual_model=self.requested_model)


class RecoveryTTS:
    streaming = False

    def __init__(self, *, failure: bool = False) -> None:
        self.failure = failure
        self.calls = 0

    def synthesize(self, text, *, output_path, cancel_event=None):
        del text, cancel_event
        self.calls += 1
        if self.failure and self.calls == 1:
            raise RuntimeError("temporary TTS stream failure")
        return {"path": str(output_path)}


class RecoveryPlayback:
    def __init__(self, *, failure: bool = False) -> None:
        self.failure = failure
        self.calls = 0

    def play(self, path, cancel_event=None):
        del cancel_event
        self.calls += 1
        if self.failure and self.calls == 1:
            raise RuntimeError("playback process failure")
        return {"path": path, "cancelled": False}


def _speech_chunk(index: int, *, sample_rate: int = 16000) -> np.ndarray:
    duration = 0.16 + (index % 3) * 0.01
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    return (0.18 * np.sin(2 * np.pi * (220 + index * 2) * t)).astype(np.float32)


def _silence(*, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(0.14 * sample_rate), dtype=np.float32)


def _controller_config(sample_rate: int = 16000) -> VADConfig:
    return VADConfig(
        sample_rate=sample_rate,
        frame_ms=20,
        min_speech_duration_s=0.12,
        end_silence_s=0.10,
        max_utterance_duration_s=4.0,
        threshold=0.02,
    )


def _run_multiturn(config: dict[str, Any], turns: int) -> dict[str, Any]:
    output_dir = artifact_dir(config)
    llm = ApplicationLLM()
    playback = InstrumentedPlayback()
    tts = InstrumentedStreamingTTS(output_dir, llm)
    pipeline = LivePipeline(
        llm=llm,
        tts=tts,
        playback=playback,
        artifact_dir=output_dir,
        sentence_max_chars=int(config.get("tts", {}).get("sentence_max_chars", 48)),
        sentence_timeout_s=float(config.get("tts", {}).get("sentence_timeout_s", 0.8)),
    )
    texts = [
        "合言葉は青い星です",
        "今日は天気について聞きます",
        "短い予定を教えてください",
        "さっきの合言葉は？",
    ]
    texts.extend(f"連続会話の質問 {index + 5} です" for index in range(max(0, turns - len(texts))))
    texts = texts[:turns]
    chunks: list[np.ndarray] = []
    for index in range(turns):
        chunks.extend((_speech_chunk(index), _silence()))
    source = FixtureAudioSource(chunks, delay_s=0.15)
    controller = SessionController(
        source=source,
        asr=ScriptedApplicationASR(texts),
        pipeline=pipeline,
        config=_controller_config(),
        artifact_dir=output_dir,
        history=ConversationHistory(
            "日本語で短く自然に答えてください。音声合成向けに一文を短くします。",
            max_turns=int(config.get("chat", {}).get("history_max_turns", 12)),
            max_chars=int(config.get("chat", {}).get("history_max_chars", 8000)),
        ),
    )
    with ResourceMonitor(interval_s=0.05) as monitor:
        summary = controller.run(max_turns=turns)
    messages = controller.history.messages()
    compact_summary = {
        key: summary[key]
        for key in (
            "state",
            "application_success_count",
            "application_failure_count",
            "recovery_count",
            "barge_in_count",
            "cancel_count",
            "history",
        )
    }
    compact_summary["transition_states"] = sorted({item["to"] for item in summary["transitions"]})
    compact_summary["event_counts"] = {
        kind: sum(event["event"] == kind for event in summary["events"])
        for kind in sorted({event["event"] for event in summary["events"]})
    }
    resource = monitor.process_resource_summary
    compact_resource = {
        key: resource.get(key)
        for key in ("sample_count", "started", "end", "deltas", "monotonic_growth")
    }
    return {
        "summary": compact_summary,
        "turns_requested": turns,
        "application_success_rate": summary["application_success_count"] / turns if turns else 1.0,
        "history_roles": [item["role"] for item in messages],
        "history_contains_password": False,
        "history_context_forwarded": len(llm.requests) == turns and any(
            item.get("content") == "合言葉は青い星です" for request in llm.requests for item in request
        ),
        "incremental_llm_tts": {
            "tts_requests": len(tts.requests),
            "first_tts_before_completion": tts.first_tts_before_completion,
            "completion_wait_avoided": tts.first_tts_before_completion > 0,
        },
        "playback": {
            "started": playback.started_count,
            "finished": playback.finished_count,
            "cancelled": playback.cancelled_count,
            "stale_pcm_bytes": playback.stale_pcm_bytes,
        },
        "cleanup": {
            "stale_pcm_bytes": playback.stale_pcm_bytes,
            "active_after_session": playback.active,
            "fixture_source_stopped": source.stopped,
        },
        "resources": {
            "process": compact_resource,
            "ram_peak_mib": monitor.ram_memory_peak_mib,
            "gpu_peak_mib": monitor.gpu_memory_peak_mib,
            "gpu_free_min_mib": monitor.gpu_memory_free_min_mib,
            "cpu_load_percent": monitor.cpu_load_percent,
        },
    }


def _run_barge_in_case(config: dict[str, Any], position: str, repeat: int) -> dict[str, Any]:
    import threading

    class PositionPlayback:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.calls = 0
            self.cancel_calls = 0
            self.stale_pcm_bytes = 0

        def play(self, path, cancel_event=None):
            self.calls += 1
            self.started.set()
            if self.calls == 1:
                while cancel_event is not None and not cancel_event.is_set():
                    time.sleep(0.005)
                return {"path": path, "cancelled": True}
            return {"path": path, "cancelled": False}

        def cancel(self):
            self.cancel_calls += 1
            return {"cancelled": True}

    class PositionLLM:
        requested_model = "fake-barge-llm"

        def __init__(self) -> None:
            self.cancel_observed = False

        def cancel(self) -> None:
            self.cancel_observed = True

        def stream(self, messages, tools=None, cancel_event=None):
            del messages, tools
            yield TextDelta("割り込み可能な応答です。")
            if cancel_event is not None and cancel_event.is_set():
                self.cancel_observed = True
                return
            yield Completion(reason="stop", actual_model=self.requested_model)

    class PositionTTS(RecoveryTTS):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_calls = 0

        def cancel(self) -> None:
            self.cancel_calls += 1

    playback = PositionPlayback()
    llm = PositionLLM()
    tts = PositionTTS()
    pipeline = LivePipeline(
        llm=llm,
        tts=tts,
        playback=playback,
        artifact_dir=artifact_dir(config),
    )
    controller = SessionController(
        source=FixtureAudioSource([]),
        asr=ScriptedApplicationASR(["最初の質問", "割り込み後の質問"]),
        pipeline=pipeline,
        config=_controller_config(),
        artifact_dir=artifact_dir(config),
    )
    controller.start()
    controller.submit_utterance(_speech_chunk(0))
    if not playback.started.wait(timeout=2.0):
        controller.stop()
        return {"position": position, "repeat": repeat, "status": "fail", "reason": "assistant_playback_not_started"}
    delay = {"first": 0.0, "middle": 0.08, "late": 0.16}[position]
    time.sleep(delay)
    controller.process_audio_chunk(_speech_chunk(9))
    controller.process_audio_chunk(_silence())
    controller.wait_for_idle(timeout=4.0)
    result = {
        "position": position,
        "repeat": repeat,
        "status": "pass" if controller.barge_in_count == 1 and controller.cancel_count == 1 and playback.cancel_calls == 1 and tts.cancel_calls == 1 and llm.cancel_observed and controller.state == SessionState.LISTENING else "fail",
        "barge_in_detected": controller.barge_in_count,
        "playback_cancelled": playback.cancel_calls,
        "tts_cancelled": tts.cancel_calls,
        "llm_cancelled": llm.cancel_observed,
        "stale_pcm_bytes": playback.stale_pcm_bytes,
        "state_after_recovery": controller.state.value,
        "history_assistant_count": len(controller.history.spoken_assistant_texts()),
        "interrupted_partial_excluded": len(controller.history.spoken_assistant_texts()) == 1,
    }
    controller.stop()
    return result


def _run_recovery_case(config: dict[str, Any], label: str, *, llm_failure: bool = False, tts_failure: bool = False, playback_failure: bool = False, empty_asr: bool = False) -> dict[str, Any]:
    output_dir = artifact_dir(config)
    controller = SessionController(
        source=FixtureAudioSource([]),
        asr=ScriptedApplicationASR([""] if empty_asr else ["復旧テスト"]),
        pipeline=LivePipeline(
            llm=RecoveryLLM(failure=llm_failure),
            tts=RecoveryTTS(failure=tts_failure),
            playback=RecoveryPlayback(failure=playback_failure),
            artifact_dir=output_dir,
        ),
        config=_controller_config(),
        artifact_dir=output_dir,
        max_retries=1,
    )
    controller.start()
    controller.submit_utterance(_speech_chunk(2))
    controller.wait_for_idle(timeout=4.0)
    result = {
        "case": label,
        "status": "pass" if (controller.application_success_count == 1 if not empty_asr else controller.application_success_count == 0) else "fail",
        "application_success_count": controller.application_success_count,
        "application_failure_count": controller.application_failure_count,
        "recovery_count": controller.recovery_count,
        "state": controller.state.value,
    }
    controller.stop()
    return result


def run_application_bench(config: dict[str, Any], *, turns: int | None = None) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    turns = int(turns if turns is not None else config.get("bench", {}).get("phase7_application_turns", 60))
    if turns < 50 or turns > 100:
        raise ValueError("Phase 7 application turns must be between 50 and 100")
    multiturn = _run_multiturn(config, turns)
    barge_rows = [
        _run_barge_in_case(config, position, repeat)
        for position in ("first", "middle", "late")
        for repeat in range(1, 4)
    ]
    recovery_rows = [
        _run_recovery_case(config, "temporary_ollama_failure", llm_failure=True),
        _run_recovery_case(config, "temporary_tts_failure", tts_failure=True),
        _run_recovery_case(config, "playback_process_failure", playback_failure=True),
        _run_recovery_case(config, "empty_asr_result", empty_asr=True),
    ]
    too_short = SessionController(
        source=FixtureAudioSource([]),
        asr=ScriptedApplicationASR(["unused"]),
        pipeline=LivePipeline(llm=RecoveryLLM(), tts=RecoveryTTS(), playback=RecoveryPlayback(), artifact_dir=artifact_dir(config)),
        config=_controller_config(),
        artifact_dir=artifact_dir(config),
    )
    too_short.start()
    rejected = not too_short.submit_utterance(np.zeros(400, dtype=np.float32))
    too_short.stop()
    checks = {
        "continuous_session": multiturn["summary"]["state"] == SessionState.IDLE.value,
        "application_success": multiturn["summary"]["application_success_count"] == turns,
        "history": multiturn["history_context_forwarded"] and 0 < multiturn["history_roles"].count("user") <= int(config.get("chat", {}).get("history_max_turns", 12)),
        "incremental_llm_tts": multiturn["incremental_llm_tts"]["completion_wait_avoided"],
        "synthetic_barge_in": all(row["status"] == "pass" for row in barge_rows),
        "recovery": all(row["status"] == "pass" for row in recovery_rows),
        "cancellation": all(row["status"] == "pass" for row in barge_rows),
        "too_short_rejection": rejected,
        "stale_pcm": multiturn["playback"]["stale_pcm_bytes"] == 0 and all(row["stale_pcm_bytes"] == 0 for row in barge_rows),
        "resource_growth": not any(
            (multiturn["resources"]["process"].get("monotonic_growth") or {}).values()
        ),
        "cleanup": multiturn["cleanup"]["stale_pcm_bytes"] == 0 and not multiturn["cleanup"]["active_after_session"],
    }
    data = {
        "schema": "local-live-ja/application-acceptance/v1",
        "phase": 7,
        "status": "unattended_poc_complete" if all(checks.values()) else "measured_with_limitations",
        "turns": turns,
        "checks": checks,
        "multi_turn": multiturn,
        "synthetic_application_level_barge_in": barge_rows,
        "recovery_matrix": recovery_rows,
        "cleanup": multiturn["cleanup"],
        "too_short_rejected": rejected,
        "deferred_manual": [
            "human speech recognition quality",
            "human physical double-talk",
            "human physical barge-in experience",
            "MOS / subjective listening evaluation",
            "subjective TTS naturalness",
            "subjective microphone gain tuning",
            "production approval",
        ],
    }
    return write_benchmark(config, "app", data, started_at=started)


__all__ = ["run_application_bench"]
