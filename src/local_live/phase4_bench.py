from __future__ import annotations

import json
import os
import statistics
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

from .asr import WhisperASR
from .audio import AudioVolumeGuard, EchoCancelSession, PipeWireInventory, PipeWirePCMPlayback, RawCaptureSession, audio_file_stats, record_fixed, stable_target
from .audio_metrics import resample_mono
from .bench import _collect_first_sentence_chunk, _live_messages, artifact_dir, write_benchmark
from .config import nested
from .echo_rejection import echo_config_from_mapping, evaluate_echo_rejection
from .llm.events import Cancelled, Completion, TextDelta
from .llm.ollama import OllamaLLM
from .telemetry import EventLog, ResourceMonitor, current_gpu_memory, nvidia_smi
from .tts import Qwen3TTSEngine
from .vad import detect_speech_intervals
from .vllm_bench import (
    _make_vllm_client,
    _physical_measurement,
    _server_memory_note,
    _server_env,
    _server_version,
    _targets,
    _vllm_model,
    _vllm_python,
)
from .vllm_server import VLLMOmniServer
from .pipeline import LivePipeline

# Fixed synthetic user corpus. The text is generated to WAV before the resident
# server starts, so fixture generation cannot contaminate per-turn latency.
STABILITY_INPUTS: list[dict[str, str]] = [
    {"name": "simple_question", "text": "準備はできましたか。"},
    {"name": "number", "text": "数字の二十四を確認してください。"},
    {"name": "technical", "text": "PipeWireとCUDAの状態を教えてください。"},
    {"name": "yes_no", "text": "このまま続けても大丈夫ですか。"},
    {"name": "short_explanation", "text": "結果を短く説明してください。"},
    {"name": "two_sentence", "text": "測定結果を確認しました。次の手順を教えてください。"},
    {"name": "latency", "text": "レイテンシーは安定していますか。"},
    {"name": "next_step", "text": "次のテストへ進めますか。"},
]


def percentile_summary(values: list[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    ordered = sorted(numeric)

    def percentile(level: float) -> float | None:
        if not ordered:
            return None
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * level
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "values": numeric,
        "count": len(numeric),
        "median": statistics.median(numeric) if numeric else None,
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
        "mean": statistics.mean(numeric) if numeric else None,
        "stddev": statistics.stdev(numeric) if len(numeric) > 1 else (0.0 if numeric else None),
    }


def classify_latency_outlier(
    row: Mapping[str, Any],
    component_medians: Mapping[str, float],
    *,
    ratio_threshold: float = 2.0,
    minimum_delta_s: float = 0.05,
) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for name, median in component_medians.items():
        value = (row.get("stages") or {}).get(name)
        if not isinstance(value, (int, float)) or not isinstance(median, (int, float)) or median <= 0:
            continue
        ratio = float(value) / float(median)
        flags[name] = {
            "value_s": float(value),
            "median_s": float(median),
            "ratio_to_median": ratio,
            "flagged": ratio >= ratio_threshold and float(value) - float(median) >= minimum_delta_s,
        }
    flagged = {name: value for name, value in flags.items() if value["flagged"]}
    dominant = max(flagged, key=lambda name: flagged[name]["ratio_to_median"]) if flagged else None
    if row.get("status") == "blocked" and row.get("blocked_reason") == "physical_audio_not_detected":
        dominant = "acoustic_onset_detection"
    return {"is_outlier": bool(flagged) or dominant == "acoustic_onset_detection", "dominant_cause": dominant, "stage_flags": flags}


def _process_memory_mib() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        return None


def _system_memory_mib() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().used / (1024 * 1024))
    except Exception:
        return None


def _service_process_memory() -> dict[str, Any]:
    services: list[dict[str, Any]] = []
    try:
        import psutil

        for process in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                command = " ".join(process.info.get("cmdline") or [])
                name = f"{process.info.get('name') or ''} {command}".casefold()
                if "ollama" not in name and "vllm" not in name:
                    continue
                label = "ollama" if "ollama" in name else "vllm"
                memory_info = process.info.get("memory_info")
                rss = memory_info.rss if memory_info else 0
                services.append(
                    {
                        "pid": int(process.info["pid"]),
                        "service": label,
                        "rss_mib": float(rss / (1024 * 1024)),
                        "command": command[:300],
                    }
                )
            except (psutil.Error, OSError):
                continue
    except Exception:
        pass
    gpu_by_pid: dict[int, int] = {}
    for item in nvidia_smi("compute_apps.pid,compute_apps.used_memory"):
        try:
            gpu_by_pid[int(item.get("compute_apps.pid", ""))] = int(float(item.get("compute_apps.used_memory", "0")))
        except (TypeError, ValueError):
            continue
    for item in services:
        item["gpu_memory_mib"] = gpu_by_pid.get(item["pid"])
    grouped: dict[str, dict[str, Any]] = {}
    for service in ("vllm", "ollama"):
        rows = [item for item in services if item["service"] == service]
        gpu_values = [int(item["gpu_memory_mib"]) for item in rows if isinstance(item.get("gpu_memory_mib"), int)]
        grouped[service] = {
            "rss_mib": sum(float(item["rss_mib"]) for item in rows),
            "gpu_memory_mib": sum(gpu_values) if gpu_values else None,
            "processes": rows,
        }
    return grouped


def _delta(timing: Mapping[str, Any], end: str, start: str) -> float | None:
    end_ns = timing.get(end)
    start_ns = timing.get(start)
    if isinstance(end_ns, int) and isinstance(start_ns, int) and end_ns >= start_ns:
        return (end_ns - start_ns) / 1e9
    return None


def _fixture_path(config: dict[str, Any], item: Mapping[str, str]) -> Path:
    return artifact_dir(config) / f"phase4_input_{item['name']}.wav"


def _ensure_stability_fixtures(config: dict[str, Any]) -> list[dict[str, Any]]:
    engine = Qwen3TTSEngine(
        model=nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        speaker=nested(config, "tts", "speaker", default="Ono_Anna"),
        language=nested(config, "tts", "language", default="Japanese"),
        device="auto",
        max_new_tokens=int(nested(config, "tts", "max_new_tokens", default=2048)),
        generation_kwargs=dict(nested(config, "tts", "generation_kwargs", default={}) or {}),
    )
    fixtures: list[dict[str, Any]] = []
    try:
        for item in STABILITY_INPUTS:
            path = _fixture_path(config, item)
            if not path.exists():
                engine.synthesize(item["text"], output_path=path)
            info = sf.info(str(path))
            fixtures.append({**item, "path": str(path), "sample_rate": int(info.samplerate), "duration_s": float(info.duration)})
    finally:
        engine.unload()
    return fixtures


def _run_health(engine: Any, provider: OllamaLLM) -> dict[str, Any]:
    return {"vllm": engine.health(), "ollama": provider.probe()}


class _InterruptFixtureLLM:
    requested_model = "phase4-interrupt-fixture"

    def stream(self, messages: Any, tools: Any = None, cancel_event: Any = None):
        yield TextDelta(text="短い返答を再生します。")
        yield Completion(reason="stop", actual_model=self.requested_model)


class _InterruptPlayback:
    """Hold after the first physical queue write until cancellation arrives."""

    streaming = True

    def __init__(self, target: str) -> None:
        self.inner = PipeWirePCMPlayback(target)
        self.first_queue = threading.Event()
        self.released = threading.Event()
        self.queue_count = 0
        self.queue_after_cancel = 0

    @property
    def active(self) -> bool:
        return self.inner.active

    @property
    def last_queued_ns(self) -> int | None:
        return self.inner.last_queued_ns

    def start(self, *, sample_rate: int, channels: int) -> dict[str, Any]:
        return self.inner.start(sample_rate=sample_rate, channels=channels)

    def queue(self, payload: bytes) -> dict[str, Any]:
        if self.released.is_set():
            self.queue_after_cancel += 1
        result = self.inner.queue(payload)
        self.queue_count += int(bool(result.get("queued")))
        if not self.first_queue.is_set():
            self.first_queue.set()
            self.released.wait(timeout=5.0)
        return result

    def finish(self) -> dict[str, Any]:
        self.released.set()
        return self.inner.finish()

    def cancel(self) -> dict[str, Any]:
        self.released.set()
        return self.inner.cancel()


class _BlockingCancellationLLM:
    requested_model = "phase4-llm-cancel-fixture"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancel_seen = False

    def stream(self, messages: Any, tools: Any = None, cancel_event: Any = None):
        self.started.set()
        while cancel_event is None or not cancel_event.is_set():
            time.sleep(0.005)
        self.cancel_seen = True
        yield Cancelled(reason="interrupt")


class _CancellationProbeOwner:
    streaming = False

    def __init__(self) -> None:
        self.cancel_count = 0

    def cancel(self) -> dict[str, Any]:
        self.cancel_count += 1
        return {"cancelled": True}


def _run_llm_cancel_probe() -> dict[str, Any]:
    llm = _BlockingCancellationLLM()
    tts = _CancellationProbeOwner()
    playback = _CancellationProbeOwner()
    pipeline = LivePipeline(llm=llm, tts=tts, playback=playback)
    result_holder: list[Any] = []
    worker_started_ns = time.monotonic_ns()
    worker = threading.Thread(target=lambda: result_holder.append(pipeline.respond("LLM中断")), daemon=True)
    worker.start()
    if not llm.started.wait(timeout=2.0):
        return {"status": "blocked", "reason": "LLM cancellation probe did not start"}
    request_ns = time.monotonic_ns()
    pipeline.cancel()
    worker.join(timeout=2.0)
    completed_ns = time.monotonic_ns()
    result = result_holder[0] if result_holder else None
    return {
        "status": "measured" if result is not None and not worker.is_alive() else "blocked",
        "cancel_request_to_return_s": (completed_ns - request_ns) / 1e9,
        "llm_stream_cancelled": llm.cancel_seen,
        "pipeline_cancelled": bool(result and result.cancelled),
        "state": result.state if result else None,
        "tts_cancel_hook_calls": tts.cancel_count,
        "playback_cancel_hook_calls": playback.cancel_count,
        "worker_started_to_return_s": (completed_ns - worker_started_ns) / 1e9,
    }


def run_interruption_regression(
    config: dict[str, Any],
    *,
    engine: Any,
    playback_target: str,
    capture_target: str,
) -> dict[str, Any]:
    """Interrupt a real streaming turn after first queued PCM and verify cleanup."""
    capture = RawCaptureSession(artifact_dir(config) / "phase4_interruption_raw.wav", target=capture_target, sample_rate=16000)
    playback = _InterruptPlayback(playback_target)
    pipeline = LivePipeline(
        llm=_InterruptFixtureLLM(),
        tts=engine,
        playback=playback,
        artifact_dir=artifact_dir(config),
        sentence_max_chars=48,
        sentence_timeout_s=0.8,
        echo_rejection_config=echo_config_from_mapping(config.get("echo_rejection")),
    )
    result_holder: list[Any] = []
    worker = threading.Thread(target=lambda: result_holder.append(pipeline.respond("中断テスト")), name="phase4-interruption", daemon=True)
    data: dict[str, Any] = {"status": "blocked", "expected": {"persistent_pcm_stopped": True, "stale_pcm_rejected": True, "pipeline_cancelled": True, "llm_stream_cancelled": True}}
    try:
        llm_cancel_probe = _run_llm_cancel_probe()
        capture.start()
        worker.start()
        if not playback.first_queue.wait(timeout=float(nested(config, "bench", "interruption_wait_s", default=30.0))):
            raise RuntimeError("interruption fixture did not reach first PCM queue")
        request_ns = time.monotonic_ns()
        pipeline.cancel()
        worker.join(timeout=10.0)
        stop_ns = time.monotonic_ns()
        if worker.is_alive():
            raise RuntimeError("pipeline did not return after programmatic cancellation")
        capture_result = capture.stop(tail_s=0.3)
        result = result_holder[0] if result_holder else None
        physical_stop_s: float | None = None
        physical_intervals: list[tuple[float, float]] = []
        try:
            recorded, recorded_rate = sf.read(str(capture.output_path), always_2d=False)
            physical_intervals = detect_speech_intervals(np.asarray(recorded), int(recorded_rate))
            request_offset_s = (request_ns - (capture.started_ns or request_ns)) / 1e9
            post_interrupt = [end for start, end in physical_intervals if end >= request_offset_s and start <= request_offset_s + 1.0]
            if post_interrupt:
                physical_stop_s = max(post_interrupt) - request_offset_s
        except Exception:
            physical_intervals = []
        data.update(
            {
                "status": "measured",
                "interrupt_request_ns": request_ns,
                "playback_process_stopped_ns": stop_ns if not playback.active else None,
                "software_playback_stop_s": (stop_ns - request_ns) / 1e9 if not playback.active else None,
                "persistent_pcm_active_after_interrupt": playback.active,
                "queued_pcm_discarded": playback.queue_after_cancel == 0,
                "queued_pcm_count_before_cancel": playback.queue_count,
                "pipeline": {"cancelled": bool(result and result.cancelled), "state": result.state if result else None, "error": result.error if result else "missing result", "spoken_text": result.spoken_text if result else ""},
                "vllm_http_stream_cancelled": bool(getattr(engine, "_active_http_cancelled", False) or (result and getattr(result, "http_stream_cancelled", False))),
                "llm_cancel_probe": llm_cancel_probe,
                "capture": capture_result,
                "physical_stop_s": physical_stop_s,
                "physical_stop_intervals_s": physical_intervals,
                "physical_stop_method": "last VAD interval after interrupt; null means no separable post-interrupt interval",
            }
        )
    except Exception as exc:
        data.update({"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)})
        try:
            playback.cancel()
        except Exception:
            pass
        if worker.is_alive():
            worker.join(timeout=2.0)
        try:
            if capture._result is None:
                capture.stop(tail_s=0.1)
        except Exception:
            pass
    return data


def _stage_values(row: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in (row.get("stages") or {}).items() if isinstance(value, (int, float))}


def _run_stability_turn(
    config: dict[str, Any],
    *,
    engine: Any,
    asr: WhisperASR,
    provider: OllamaLLM,
    fixture: Mapping[str, Any],
    turn_index: int | str,
    playback_target: str,
    capture_target: str,
    phase: str = "continuous",
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    run_label = str(turn_index)
    log = EventLog()
    capture = RawCaptureSession(
        artifact_dir(config) / f"phase4_{phase}_turn_{run_label}_raw_mic.wav",
        target=capture_target,
        sample_rate=16000,
    )
    playback: PipeWirePCMPlayback | None = None
    monitor = ResourceMonitor(interval_s=0.25)
    memory_start = {
        "process_rss_mib": _process_memory_mib(),
        "system_used_mib": _system_memory_mib(),
        "services": _service_process_memory(),
    }
    timing: dict[str, Any] = {"turn_start": started_ns}
    row: dict[str, Any] = {
        "status": "error",
        "phase": phase,
        "turn_index": turn_index,
        "input_fixture": {key: fixture[key] for key in ("name", "text", "path", "sample_rate", "duration_s")},
        "retry": {"count": 0, "performed": False, "reason": None},
        "error": None,
        "error_type": None,
        "cancellation": {"requested": False, "state": "not_requested"},
    }
    try:
        with monitor:
            capture.start()
            synthetic_end_ns = time.monotonic_ns()
            timing["synthetic_user_end"] = synthetic_end_ns
            log.mark_at("synthetic_user_end", synthetic_end_ns, fixture=fixture["name"])
            health_start = _run_health(engine, provider)
            user_asr = asr.transcribe(fixture["path"], event_log=log)
            asr_start = next((item["monotonic_ns"] for item in log.events if item["event"] == "asr_start"), None)
            asr_end = next((item["monotonic_ns"] for item in reversed(log.events) if item["event"] == "asr_final"), None)
            timing.update({"asr_start": asr_start, "asr_end": asr_end})
            llm_start_ns = time.monotonic_ns()
            timing["llm_start"] = llm_start_ns
            log.mark_at("llm_start", llm_start_ns)
            turn = _collect_first_sentence_chunk(
                provider,
                _live_messages(user_asr.text),
                max_chars=int(nested(config, "tts", "sentence_max_chars", default=48)),
                timeout_s=float(nested(config, "tts", "sentence_timeout_s", default=0.8)),
            )
            if turn.get("first_token_ns") is not None:
                timing["llm_first_token"] = turn["first_token_ns"]
                log.mark_at("llm_first_token", turn["first_token_ns"])
            if turn.get("first_chunk_ns") is not None:
                timing["first_text_chunk_ready"] = turn["first_chunk_ns"]
                log.mark_at("first_text_chunk_ready", turn["first_chunk_ns"], text_chars=len(turn.get("first_chunk") or ""))
            timing["llm_end"] = turn.get("ended_ns")
            if turn.get("error"):
                raise RuntimeError(str(turn["error"].message))
            chunk = str(turn.get("first_chunk") or "")
            if not chunk:
                raise RuntimeError("LLM returned no first sentence chunk")
            tts_start_ns = time.monotonic_ns()
            timing["tts_request_start"] = tts_start_ns
            log.mark_at("tts_request_start", tts_start_ns, text_chars=len(chunk))
            output = artifact_dir(config) / f"phase4_{phase}_turn_{run_label}_assistant.wav"
            playback = PipeWirePCMPlayback(playback_target)
            tts_result = engine.synthesize_stream(chunk, output_path=output, playback=playback, event_log=log)
            if tts_result.get("status") != "measured":
                raise RuntimeError(str(tts_result.get("error") or "vLLM streaming TTS failed"))
            timing.update(tts_result.get("timing_ns", {}))
            physical = _physical_measurement(
                capture=capture,
                playback={"timing_ns": dict(tts_result.get("timing_ns", {})), "path": str(output)},
                reference_path=output,
                capture_target=capture_target,
            )
            timing.update(physical.get("timing_ns", {}))
            timing["turn_end"] = time.monotonic_ns()
            stages = {
                "asr": _delta(timing, "asr_end", "synthetic_user_end"),
                "asr_duration": _delta(timing, "asr_end", "asr_start"),
                "llm_ttft": _delta(timing, "llm_first_token", "llm_start"),
                "llm_total": _delta(timing, "llm_end", "llm_start"),
                "first_sentence_buffering": _delta(timing, "first_text_chunk_ready", "llm_first_token"),
                "tts_request_to_first_pcm": _delta(timing, "first_audio_chunk_received", "tts_request_start"),
                "tts_request_to_actual": _delta(timing, "first_actual_speech_pcm", "tts_request_start"),
                "first_pcm_to_actual": _delta(timing, "first_actual_speech_pcm", "first_audio_chunk_received"),
                "first_pcm_to_playback": _delta(timing, "playback_stream_started", "first_audio_chunk_received"),
                "playback": _delta(timing, "playback_completed", "playback_stream_started"),
                "acoustic_onset": _delta(timing, "physical_audio_detected", "playback_stream_started"),
                "total_turn": _delta(timing, "physical_audio_detected", "synthetic_user_end"),
            }
            row.update(
                {
                    "status": "measured" if stages["total_turn"] is not None else "blocked",
                    "blocked_reason": None if stages["total_turn"] is not None else "physical_audio_not_detected",
                    "timing_ns": timing,
                    "stages": {key: value for key, value in stages.items() if value is not None},
                    "user_asr": user_asr.to_dict(),
                    "llm": {
                        "first_text_chunk": chunk,
                        "ttft_s": turn.get("ttft_s"),
                        "total_s": stages["llm_total"],
                        "until_first_chunk_s": turn.get("until_first_chunk_s"),
                        "actual_model": getattr(provider, "last_actual_model", None),
                    },
                    "tts": tts_result,
                    "physical": physical,
                    "health": {"start": health_start, "end": _run_health(engine, provider)},
                    "cancellation": {"requested": False, "state": "not_requested"},
                }
            )
    except Exception as exc:
        if playback is not None:
            try:
                playback.cancel()
            except Exception:
                pass
        try:
            if capture._result is None:
                capture.stop(tail_s=0.1)
        except Exception:
            pass
        row.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc), "timing_ns": timing})
    finally:
        memory_end = {
            "process_rss_mib": _process_memory_mib(),
            "system_used_mib": _system_memory_mib(),
            "services": _service_process_memory(),
        }
        row["memory"] = {
            "start": memory_start,
            "end": memory_end,
            "process_rss_peak_mib": monitor.ram_memory_peak_mib,
            "gpu_memory_start_mib": monitor.started_gpu,
            "gpu_memory_peak_mib": monitor.gpu_memory_peak_mib,
            "gpu_memory_free_min_mib": monitor.gpu_memory_free_min_mib,
            "gpu_memory_end_mib": current_gpu_memory(),
            "cpu_load_percent": monitor.cpu_load_percent,
        }
        row["peak_vram_mib"] = monitor.gpu_memory_peak_mib
        row["gpu_memory_free_min_mib"] = monitor.gpu_memory_free_min_mib
        row["ram_mib"] = monitor.ram_memory_peak_mib
        row["cpu_percent"] = monitor.cpu_load_percent
        row["turn_elapsed_s"] = (time.monotonic_ns() - started_ns) / 1e9
    return row


def _mode_stage_medians(rows: list[dict[str, Any]]) -> dict[str, float]:
    measured = [row for row in rows if row.get("status") == "measured"]
    names = sorted({name for row in measured for name in _stage_values(row)})
    return {
        name: float(statistics.median([_stage_values(row)[name] for row in measured if name in _stage_values(row)]))
        for name in names
        if any(name in _stage_values(row) for row in measured)
    }


def _window_summary(rows: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    selected = [row for row in rows if isinstance(row.get("turn_index"), int) and start <= row["turn_index"] <= end]
    measured = [row for row in selected if row.get("status") == "measured"]
    def vals(key: str) -> list[float]:
        values: list[float] = []
        for row in measured:
            value = (row.get("stages") or {}).get(key) if key in {"total_turn", "tts_request_to_actual"} else row.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values
    return {
        "turn_range": [start, end],
        "attempt_count": len(selected),
        "measured_count": len(measured),
        "ttfa_s": percentile_summary(vals("tts_request_to_actual")),
        "physical_first_audio_s": percentile_summary(vals("total_turn")),
        "vram_peak_mib": percentile_summary(vals("peak_vram_mib")),
        "vram_free_min_mib": percentile_summary(vals("gpu_memory_free_min_mib")),
        "ram_peak_mib": percentile_summary(vals("ram_mib")),
    }


def _phase3b_outlier_reference(config: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(nested(config, "app", "result_dir", default="results")) / "bench_live_latency.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8")).get("data", {})
        rows = [row for row in data.get("runs", []) if isinstance(row.get("speech_end_to_first_physical_audio_s"), (int, float))]
        row = max(rows, key=lambda item: float(item["speech_end_to_first_physical_audio_s"]))
        medians: dict[str, float] = {}
        for name in ("asr", "llm_ttft", "first_chunk_buffering", "tts_request_to_first_pcm", "first_pcm_to_queue", "stream_to_physical"):
            values = [float(item["stages"][name]) for item in rows if isinstance(item.get("stages", {}).get(name), (int, float))]
            if values:
                medians[name] = float(statistics.median(values))
        classification = classify_latency_outlier(row, medians)
        return {
            "source": str(path),
            "latency_s": row["speech_end_to_first_physical_audio_s"],
            "stage_values": row.get("stages"),
            "component_medians": medians,
            "classification": classification,
            "interpretation": "The dominant relative stage is retained as the primary hypothesis; acoustic onset remains a measured downstream contributor.",
        }
    except Exception as exc:
        return {"source": str(path), "status": "unavailable", "error_type": type(exc).__name__}


def _stability_status(data: dict[str, Any]) -> str:
    summary = data.get("summary") or {}
    dist = summary.get("physical_first_audio_s") or {}
    onset_rate = summary.get("physical_onset_detection_success_rate")
    if summary.get("measured_turn_count", 0) < 20:
        return "blocked"
    if (
        summary.get("measured_turn_count", 0) >= 50
        and isinstance(dist.get("median"), (int, float))
        and dist["median"] < 2.0
        and isinstance(dist.get("p95"), (int, float))
        and dist["p95"] < 2.5
        and isinstance(onset_rate, (int, float))
        and onset_rate >= 0.95
        and not (summary.get("memory_leak") or {}).get("suspected")
    ):
        return "pass"
    return "measured_with_limitations"


def run_stability_bench(config: dict[str, Any]) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    target_turns = max(50, int(nested(config, "bench", "stability_target_turns", default=50)))
    minimum_turns = max(20, int(nested(config, "bench", "stability_minimum_turns", default=20)))
    data: dict[str, Any] = {
        "status": "blocked",
        "benchmark": "stability",
        "configuration": {
            "asr": {"model": nested(config, "asr", "model", default="large-v3-turbo"), "device": "cuda", "compute_type": nested(config, "asr", "gpu_default_compute_type", default="int8_float16")},
            "llm": {"model": "qwen3.5:9b-q4_K_M", "provider": "ollama"},
            "tts": {"model": _vllm_model(config), "speaker": nested(config, "tts", "speaker", default="Ono_Anna"), "language": nested(config, "tts", "language", default="Japanese"), "server": "vLLM-Omni 0.28.0 / vLLM 0.28.0", "streaming": True},
            "target_turns": target_turns,
            "minimum_turns": minimum_turns,
            "aec": "not used in synthetic latency path; Phase 2 echo-only reference retained separately",
        },
        "turns": [],
        "restart_runs": [],
        "warm_state_windows": [],
        "server": None,
        "server_restart": None,
        "microphone_readiness": None,
        "summary": None,
        "interruption": None,
        "phase3b_outlier_reference": _phase3b_outlier_reference(config),
        "memory": {"before": _server_memory_note()},
        "volume_snapshot": None,
        "volume_restore_error": None,
    }
    server: VLLMOmniServer | None = None
    asr: WhisperASR | None = None
    volume_guard: AudioVolumeGuard | None = None
    fixtures: list[dict[str, Any]] = []
    try:
        fixtures = _ensure_stability_fixtures(config)
        inventory, playback_target, capture_target = _targets(config)
        if not playback_target or not capture_target:
            raise RuntimeError("USB speaker and microphone are required for stability benchmark")
        data["targets"] = {"inventory": inventory, "speaker": playback_target, "microphone": capture_target}
        data["microphone_readiness"] = run_microphone_readiness(config, inventory=inventory, playback_target=playback_target, capture_target=capture_target)
        volume_guard = AudioVolumeGuard(speaker_target=playback_target, microphone_target=capture_target)
        volume_guard.__enter__()
        data["volume_snapshot"] = asdict(volume_guard.snapshot) if volume_guard.snapshot else None
        volume_guard.set_mutes(speaker_muted=False, microphone_muted=False)
        server = VLLMOmniServer(
            python_bin=_vllm_python(config),
            model=_vllm_model(config),
            host="127.0.0.1",
            port=8091,
            deploy_config=nested(config, "tts", "vllm_deploy_config", default=None),
            log_path=artifact_dir(config) / "phase4_vllm_server.log",
            gpu_memory_utilization=nested(config, "tts", "vllm_gpu_memory_utilization", default=None),
            extra_env=_server_env(config),
        )
        server_info = server.start(timeout_s=float(nested(config, "tts", "vllm_server_start_timeout_s", default=900.0)))
        server_info["versions"] = _server_version(_vllm_python(config))
        server_info["official_repository_commit"] = "bc0c9f4b45c45c59aa2f92471842e8c18ae403ca"
        server_info["official_stable_tag"] = "v0.28.0"
        data["server"] = server_info
        data["memory"]["server_ready"] = _server_memory_note()
        asr = WhisperASR(
            model=nested(config, "asr", "model", default="large-v3-turbo"),
            device="cuda",
            compute_type=str(nested(config, "asr", "gpu_default_compute_type", default="int8_float16")),
            language="ja",
            beam_size=int(nested(config, "asr", "beam_size", default=5)),
        )
        asr.load()
        data["memory"]["after_whisper_load"] = _server_memory_note()
        provider = OllamaLLM(
            base_url=str(nested(config, "llm", "local_base_url", default="http://127.0.0.1:11434")),
            model="qwen3.5:9b-q4_K_M",
            max_tokens=int(nested(config, "llm", "max_tokens", default=96)),
            num_ctx=int(nested(config, "llm", "num_ctx", default=8192)),
            temperature=float(nested(config, "llm", "temperature", default=0.2)),
            timeout_s=float(nested(config, "llm", "stability_timeout_s", default=60.0)),
        )
        data["ollama_health"] = provider.probe()
        engine = _make_vllm_client(config, streaming=True, initial=nested(config, "tts", "vllm_initial_codec_chunk_frames", default=None))
        for turn_index in range(1, target_turns + 1):
            row = _run_stability_turn(
                config,
                engine=engine,
                asr=asr,
                provider=provider,
                fixture=fixtures[(turn_index - 1) % len(fixtures)],
                turn_index=turn_index,
                playback_target=str(playback_target),
                capture_target=str(capture_target),
            )
            data["turns"].append(row)
        measured = [row for row in data["turns"] if row.get("status") == "measured"]
        medians = _mode_stage_medians(data["turns"])
        for row in data["turns"]:
            row["outlier"] = classify_latency_outlier(row, medians)
        all_total = [float(row["stages"]["total_turn"]) for row in measured if isinstance(row.get("stages", {}).get("total_turn"), (int, float))]
        physical_detected = sum(row.get("physical", {}).get("physical_audio_detected", False) for row in data["turns"])
        error_count = sum(row.get("status") == "error" for row in data["turns"])
        blocked_count = sum(row.get("status") == "blocked" for row in data["turns"])
        first_window = _window_summary(data["turns"], 1, min(10, target_turns))
        windows = [first_window]
        for start in range(11, target_turns + 1, 10):
            windows.append(_window_summary(data["turns"], start, min(target_turns, start + 9)))
        first_vram = first_window["vram_peak_mib"].get("median")
        last_vram = windows[-1]["vram_peak_mib"].get("median")
        first_ram = first_window["ram_peak_mib"].get("median")
        last_ram = windows[-1]["ram_peak_mib"].get("median")
        memory_leak = {
            "suspected": bool(
                isinstance(first_vram, (int, float)) and isinstance(last_vram, (int, float)) and last_vram - first_vram > 512
            ) or bool(isinstance(first_ram, (int, float)) and isinstance(last_ram, (int, float)) and last_ram - first_ram > 128),
            "vram_first_to_last_delta_mib": last_vram - first_vram if isinstance(first_vram, (int, float)) and isinstance(last_vram, (int, float)) else None,
            "ram_first_to_last_delta_mib": last_ram - first_ram if isinstance(first_ram, (int, float)) and isinstance(last_ram, (int, float)) else None,
            "method": "first versus last 10-turn median; no per-turn GC/cache clear",
        }
        data["warm_state_windows"] = windows
        data["summary"] = {
            "target_turn_count": target_turns,
            "measured_turn_count": len(measured),
            "attempt_count": len(data["turns"]),
            "failed_turn_count": error_count,
            "blocked_turn_count": blocked_count,
            "failure_rate": (error_count + blocked_count) / len(data["turns"]) if data["turns"] else None,
            "physical_onset_detection_success_rate": physical_detected / len(data["turns"]) if data["turns"] else None,
            "physical_onset_detection_failure_rate": 1.0 - physical_detected / len(data["turns"]) if data["turns"] else None,
            "physical_first_audio_s": percentile_summary(all_total),
            "stage_medians_s": medians,
            "gpu_memory_baseline_mib": next((max(row.get("memory", {}).get("gpu_memory_start_mib") or []) for row in data["turns"] if row.get("memory", {}).get("gpu_memory_start_mib")), None),
            "gpu_memory_peak_mib": max((row.get("peak_vram_mib") for row in data["turns"] if isinstance(row.get("peak_vram_mib"), (int, float))), default=None),
            "gpu_memory_free_min_mib": min((row.get("gpu_memory_free_min_mib") for row in data["turns"] if isinstance(row.get("gpu_memory_free_min_mib"), (int, float))), default=None),
            "vram_warning_below_500_mib": any(isinstance(row.get("gpu_memory_free_min_mib"), (int, float)) and row["gpu_memory_free_min_mib"] < 500 for row in data["turns"]),
            "outlier_count": sum(row.get("outlier", {}).get("is_outlier", False) for row in data["turns"]),
            "outlier_cause_counts": {
                cause: sum(row.get("outlier", {}).get("dominant_cause") == cause for row in data["turns"])
                for cause in sorted({row.get("outlier", {}).get("dominant_cause") for row in data["turns"] if row.get("outlier", {}).get("dominant_cause")})
            },
            "dominant_outlier_cause": max(
                (row.get("outlier", {}).get("dominant_cause") for row in data["turns"] if row.get("outlier", {}).get("dominant_cause")),
                key=lambda cause: sum(row.get("outlier", {}).get("dominant_cause") == cause for row in data["turns"]),
                default=None,
            ),
            "memory_leak": memory_leak,
            "goals": {"median_lt_2s": bool(all_total and statistics.median(all_total) < 2.0), "p95_lt_2_5s": bool(percentile_summary(all_total).get("p95") is not None and percentile_summary(all_total)["p95"] < 2.5), "onset_success_ge_95pct": physical_detected / len(data["turns"]) >= 0.95 if data["turns"] else False},
        }
        data["memory"]["during_continuous"] = _server_memory_note()
        data["interruption"] = run_interruption_regression(
            config,
            engine=engine,
            playback_target=str(playback_target),
            capture_target=str(capture_target),
        )
        if len(measured) >= minimum_turns:
            if asr is not None:
                asr.unload()
                asr = None
            if server is not None:
                stop_info = server.stop()
                restart_started = time.monotonic_ns()
                restart_server = VLLMOmniServer(
                    python_bin=_vllm_python(config), model=_vllm_model(config), host="127.0.0.1", port=8091,
                    deploy_config=nested(config, "tts", "vllm_deploy_config", default=None),
                    log_path=artifact_dir(config) / "phase4_vllm_restart_server.log", extra_env=_server_env(config),
                )
                restart_info = restart_server.start(timeout_s=float(nested(config, "tts", "vllm_server_start_timeout_s", default=900.0)))
                restart_info["restart_stop"] = stop_info
                restart_info["restart_elapsed_s"] = (time.monotonic_ns() - restart_started) / 1e9
                data["server_restart"] = restart_info
                server = restart_server
                asr = WhisperASR(model=nested(config, "asr", "model", default="large-v3-turbo"), device="cuda", compute_type=str(nested(config, "asr", "gpu_default_compute_type", default="int8_float16")), language="ja", beam_size=int(nested(config, "asr", "beam_size", default=5)))
                asr.load()
                restart_engine = _make_vllm_client(config, streaming=True, initial=nested(config, "tts", "vllm_initial_codec_chunk_frames", default=None))
                restart_turns = max(3, int(nested(config, "bench", "stability_restart_turns", default=3)))
                for index in range(1, restart_turns + 1):
                    data["restart_runs"].append(_run_stability_turn(config, engine=restart_engine, asr=asr, provider=provider, fixture=fixtures[(index - 1) % len(fixtures)], turn_index=index, phase="restart", playback_target=str(playback_target), capture_target=str(capture_target)))
                data["server_restart"]["requested_runs"] = restart_turns
                data["server_restart"]["completed_runs"] = sum(row.get("status") == "measured" for row in data["restart_runs"])
        data["component_status"] = {
            "continuous_stability": _stability_status(data),
            "server_restart": "pass" if data.get("server_restart", {}).get("completed_runs") == data.get("server_restart", {}).get("requested_runs", max(3, int(nested(config, "bench", "stability_restart_turns", default=3)))) else "measured_with_limitations",
            "microphone_readiness": data.get("microphone_readiness", {}).get("status", "blocked"),
        }
        data["status"] = "measured" if len(measured) >= minimum_turns else "partial"
    except Exception as exc:
        data["status"] = "blocked"
        data["error_type"] = type(exc).__name__
        data["error"] = str(exc)
    finally:
        if asr is not None:
            asr.unload()
        if server is not None:
            data.setdefault("server", {})
            data["server"]["stop"] = server.stop()
            data["memory"]["after_server_stop"] = _server_memory_note()
        if volume_guard is not None:
            try:
                volume_guard.__exit__(None, None, None)
            except Exception as exc:
                data["volume_restore_error"] = f"{type(exc).__name__}: {exc}"
            else:
                data["volume_restore_error"] = volume_guard.restore_error
    return write_benchmark(config, "stability", data, started_at=started)


def run_interruption_bench(config: dict[str, Any]) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    server: VLLMOmniServer | None = None
    guard: AudioVolumeGuard | None = None
    data: dict[str, Any] = {"status": "blocked", "benchmark": "interruption", "server": None, "interruption": None}
    try:
        _inventory, speaker, mic = _targets(config)
        if not speaker or not mic:
            raise RuntimeError("stable USB playback and capture targets are required")
        guard = AudioVolumeGuard(speaker_target=speaker, microphone_target=mic)
        guard.__enter__()
        guard.set_mutes(speaker_muted=False, microphone_muted=False)
        server = VLLMOmniServer(
            python_bin=_vllm_python(config),
            model=_vllm_model(config),
            host="127.0.0.1",
            port=8091,
            deploy_config=nested(config, "tts", "vllm_deploy_config", default=None),
            log_path=artifact_dir(config) / "phase4_interruption_vllm_server.log",
            extra_env=_server_env(config),
        )
        data["server"] = server.start(timeout_s=float(nested(config, "tts", "vllm_server_start_timeout_s", default=900.0)))
        engine = _make_vllm_client(config, streaming=True, initial=nested(config, "tts", "vllm_initial_codec_chunk_frames", default=None))
        data["interruption"] = run_interruption_regression(config, engine=engine, playback_target=str(speaker), capture_target=str(mic))
        data["status"] = data["interruption"].get("status", "blocked")
    except Exception as exc:
        data.update({"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if server is not None:
            data.setdefault("server", {})
            data["server"]["stop"] = server.stop()
            data["memory_after_server_stop"] = _server_memory_note()
        if guard is not None:
            try:
                guard.__exit__(None, None, None)
            except Exception as exc:
                data["volume_restore_error"] = f"{type(exc).__name__}: {exc}"
            else:
                data["volume_restore_error"] = guard.restore_error
    return write_benchmark(config, "interruption", data, started_at=started)


def run_microphone_readiness(
    config: dict[str, Any],
    *,
    inventory: dict[str, Any] | None = None,
    playback_target: str | None = None,
    capture_target: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "status": "blocked",
        "content_judgement": "not performed; no human speech required",
        "raw_usb_microphone": None,
        "aec_source": None,
        "vad": None,
        "asr": None,
    }
    guard: AudioVolumeGuard | None = None
    aec: EchoCancelSession | None = None
    asr: WhisperASR | None = None
    try:
        inv = PipeWireInventory.discover()
        mic = inv.usb_microphone()
        speaker = inv.usb_speaker()
        raw_target = capture_target or (stable_target(mic) if mic else None)
        speaker_name = playback_target or (stable_target(speaker) if speaker else None)
        if not raw_target or not speaker_name:
            raise RuntimeError("stable USB microphone and speaker are required")
        aec_sink_master = next(
            (stable_target(node) for node in inv.sinks if "gostream" in f"{node.name} {node.target}".casefold()),
            speaker_name,
        )
        guard = AudioVolumeGuard(speaker_target=speaker_name, microphone_target=raw_target)
        guard.__enter__()
        guard.set_mutes(speaker_muted=False, microphone_muted=False)
        raw_path = artifact_dir(config) / "phase4_mic_readiness_raw.wav"
        raw_record = record_fixed(raw_path, target=raw_target, duration_s=float(nested(config, "bench", "mic_readiness_duration_s", default=1.0)), sample_rate=16000, channels=1)
        raw_stats = audio_file_stats(raw_path)
        raw_audio, raw_rate = sf.read(str(raw_path), always_2d=False)
        data["raw_usb_microphone"] = {"target": raw_target, "record": raw_record, "stats": raw_stats}
        data["vad"] = {"raw": {"intervals": detect_speech_intervals(np.asarray(raw_audio), int(raw_rate)), "speech_ratio": sum(end - start for start, end in detect_speech_intervals(np.asarray(raw_audio), int(raw_rate))) / max(float(raw_stats["duration_s"]), 1e-9)}}
        aec = EchoCancelSession(
            sink_name=nested(config, "pipewire", "echo_cancel_sink", default="Local Live Echo Cancellation Sink"),
            source_name=nested(config, "pipewire", "echo_cancel_source", default="Local Live Echo Cancellation Source"),
            capture_name=nested(config, "pipewire", "echo_cancel_capture", default="Local Live Echo Cancellation Capture"),
            playback_name=nested(config, "pipewire", "echo_cancel_playback", default="Local Live Echo Cancellation Playback"),
            latency=nested(config, "pipewire", "node_latency", default="1024/48000"),
            sink_master=aec_sink_master,
            source_master=raw_target,
        )
        aec_info = aec.load()
        aec_target = aec.source_target
        if not aec_target:
            raise RuntimeError("AEC source has no stable target")
        aec_path = artifact_dir(config) / "phase4_mic_readiness_aec.wav"
        aec_record = record_fixed(aec_path, target=aec_target, duration_s=float(nested(config, "bench", "mic_readiness_duration_s", default=1.0)), sample_rate=16000, channels=1)
        aec_stats = audio_file_stats(aec_path)
        aec_audio, aec_rate = sf.read(str(aec_path), always_2d=False)
        data["aec_source"] = {"target": aec_target, "sink_master": aec_sink_master, "loader": aec_info, "record": aec_record, "stats": aec_stats}
        aec_intervals = detect_speech_intervals(np.asarray(aec_audio), int(aec_rate))
        data["vad"]["aec"] = {"intervals": aec_intervals, "speech_ratio": sum(end - start for start, end in aec_intervals) / max(float(aec_stats["duration_s"]), 1e-9)}
        asr = WhisperASR(model=nested(config, "asr", "model", default="large-v3-turbo"), device="cuda", compute_type=str(nested(config, "asr", "gpu_default_compute_type", default="int8_float16")), language="ja", beam_size=int(nested(config, "asr", "beam_size", default=5)))
        asr.load()
        asr_result = asr.transcribe(aec_path)
        data["asr"] = {"status": "loaded_and_transcribed", "result": asr_result.to_dict(), "content_used": False}
        data["status"] = "measured"
    except Exception as exc:
        data["error_type"] = type(exc).__name__
        data["error"] = str(exc)
    finally:
        if asr is not None:
            asr.unload()
        if aec is not None:
            aec.unload()
        if guard is not None:
            try:
                guard.__exit__(None, None, None)
            except Exception as exc:
                data["volume_restore_error"] = f"{type(exc).__name__}: {exc}"
            else:
                data["volume_restore_error"] = guard.restore_error
    return data


def run_mic_readiness_bench(config: dict[str, Any]) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return write_benchmark(config, "mic_readiness", run_microphone_readiness(config), started_at=started)


def _echo_reference(config: dict[str, Any]) -> tuple[Path, np.ndarray, int]:
    path = artifact_dir(config) / "phase4_echo_assistant_reference.wav"
    if not path.exists():
        engine = Qwen3TTSEngine(
            model=nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            speaker=nested(config, "tts", "speaker", default="Ono_Anna"),
            language=nested(config, "tts", "language", default="Japanese"),
            device="auto",
            max_new_tokens=int(nested(config, "tts", "max_new_tokens", default=2048)),
            generation_kwargs=dict(nested(config, "tts", "generation_kwargs", default={}) or {}),
        )
        try:
            engine.synthesize("はい、確認しました。", output_path=path)
        finally:
            engine.unload()
    audio, rate = sf.read(str(path), always_2d=False)
    return path, resample_mono(np.asarray(audio, dtype=np.float32), int(rate), 16000), 16000


def _delayed(signal: np.ndarray, delay_samples: int, gain: float) -> np.ndarray:
    result = np.zeros(len(signal) + delay_samples, dtype=np.float32)
    result[delay_samples:] = signal * gain
    return result


def _echo_fixtures(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_path, reference, sample_rate = _echo_reference(config)
    rng = np.random.default_rng(phase4_seed := 20260912)
    fixtures: list[dict[str, Any]] = []
    artifact_paths: dict[str, Any] = {"reference": str(reference_path), "fixtures": {}}
    for index in range(10):
        delay = int(round((0.02 + index * 0.004) * sample_rate))
        gain = 0.20 + index * 0.012
        echo = _delayed(reference, delay, gain)
        noise = rng.normal(0.0, 0.0015 + index * 0.0001, size=len(echo)).astype(np.float32)
        assistant = echo + noise
        name = f"assistant_only_{index:02d}"
        mic_path = artifact_dir(config) / f"phase4_echo_{name}.wav"
        sf.write(str(mic_path), assistant, sample_rate)
        fixtures.append({"label": "assistant_only", "name": name, "reference": reference, "microphone": assistant, "playback_active": True})
        artifact_paths["fixtures"][name] = str(mic_path)
        user = np.zeros_like(echo)
        start = int((0.35 + 0.025 * (index % 4)) * sample_rate)
        end = min(len(user), int((0.85 + 0.02 * (index % 4)) * sample_rate))
        t = np.arange(max(0, end - start), dtype=np.float32) / sample_rate
        user[start:end] = (0.35 + index * 0.01) * (np.sin(2 * np.pi * (731 + index * 19) * t) + 0.25 * np.sin(2 * np.pi * (1091 + index * 11) * t))
        combined = echo + user + noise
        name = f"synthetic_user_like_{index:02d}"
        mic_path = artifact_dir(config) / f"phase4_echo_{name}.wav"
        sf.write(str(mic_path), combined, sample_rate)
        fixtures.append({"label": "synthetic_user_like", "name": name, "reference": reference, "microphone": combined, "playback_active": True})
        artifact_paths["fixtures"][name] = str(mic_path)
    return fixtures, artifact_paths


def run_echo_rejection_bench(config: dict[str, Any]) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data: dict[str, Any] = {
        "status": "blocked",
        "benchmark": "echo_rejection",
        "method": "post-VAD reference correlation + short-window residual energy + measured lag",
        "dataset": {"assistant_only": "assistant reference with no injected user signal", "synthetic_user_like": "reference residual echo plus independent injected signal; not a double-talk success test"},
        "threshold_selection": "grid search over measured fixture distributions; no production threshold claim",
    }
    try:
        fixtures, paths = _echo_fixtures(config)
        result = evaluate_echo_rejection(fixtures, 16000)
        result.setdefault("dataset", {})
        result["artifact_paths"] = paths
        result["dataset"]["fixture_count"] = len(fixtures)
        result["targets"] = {"assistant_only_false_accept_le_10pct": result["assistant_only"].get("false_accept_rate") is not None and result["assistant_only"]["false_accept_rate"] <= 0.10, "synthetic_user_like_acceptance_ge_90pct": result["synthetic_user_like"].get("acceptance_rate") is not None and result["synthetic_user_like"]["acceptance_rate"] >= 0.90}
        result["component_status"] = "pass" if all(result["targets"].values()) else "measured_with_limitations"
        result["mute_baseline"] = {
            "assistant_only_vad_positives": result["assistant_only"].get("vad_positives"),
            "false_accept_count": 0,
            "false_accept_rate": 0.0,
            "adopted": False,
            "reason": "disabling VAD during assistant playback would remove future barge-in candidates",
        }
        result["adopted"] = {"decision": "echo-aware", "reason": "mute-during-playback reference is not adopted because it removes future barge-in candidates"}
        data.update(result)
        data["status"] = "measured"
    except Exception as exc:
        data["error_type"] = type(exc).__name__
        data["error"] = str(exc)
    return write_benchmark(config, "echo_rejection", data, started_at=started)
