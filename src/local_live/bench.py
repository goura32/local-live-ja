from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from .asr import WhisperASR
from .audio import EchoCancelSession, PipeWireInventory, compare_aec_recordings, play_and_record
from .config import load_config, nested
from .llm.events import Cancelled, Completion, LLMError, TextDelta, ToolCall
from .llm.ollama import OllamaLLM
from .llm.openrouter import OpenRouterLLM
from .sentence_chunker import SentenceChunker
from .telemetry import EventLog, environment_snapshot, nvidia_smi, write_json
from .tools import MockToolRegistry
from .tts import Qwen3TTSEngine
from .vad import detect_speech_intervals, trim_to_speech


SYNTHETIC_ASR_TEXT = (
    "こんにちは。今日は通常会話の音声認識を確認します。数字は一、二、三と12345です。"
    "GPU、CUDA、Docker、Ollama、Python 3.12を使い、2026年9月12日土曜日の午後3時30分にテストします。"
    "RTX 5070 TiとUSB microphoneの状態も確認します。"
)
TTS_BENCH_TEXT = "こんにちは。Live音声の応答速度を測定しています。短い日本語で返します。"
E2E_SYSTEM_PROMPT = "日本語で一文だけ、40文字以内の自然な返答をしてください。考え中の説明は出さないでください。"


def benchmark_path(config: dict[str, Any], name: str) -> Path:
    return Path(nested(config, "app", "result_dir", default="results")) / f"bench_{name}.json"


def artifact_dir(config: dict[str, Any]) -> Path:
    path = Path(nested(config, "app", "artifact_dir", default="results/artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_benchmark(config: dict[str, Any], name: str, data: dict[str, Any], *, started_at: str | None = None) -> dict[str, Any]:
    result = {
        "schema": f"local-live-ja/bench-{name}/v1",
        "benchmark": name,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_snapshot(),
        "data": data,
    }
    write_json(benchmark_path(config, name), result)
    return result


def ensure_synthetic_audio(config: dict[str, Any], *, force: bool = False) -> tuple[Path, dict[str, Any]]:
    path = artifact_dir(config) / "synthetic_asr_regression.wav"
    if path.exists() and not force:
        try:
            info = sf.info(str(path))
            return path, {
                "text": SYNTHETIC_ASR_TEXT,
                "path": str(path),
                "source": "reused_existing_generated_audio",
                "sample_rate": info.samplerate,
                "duration_s": info.duration,
            }
        except Exception:
            path.unlink(missing_ok=True)
    engine = Qwen3TTSEngine(
        model=nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        speaker=nested(config, "tts", "speaker", default="Ono_Anna"),
        language=nested(config, "tts", "language", default="Japanese"),
        device="auto",
        max_new_tokens=int(nested(config, "tts", "max_new_tokens", default=2048)),
    )
    result = engine.synthesize(SYNTHETIC_ASR_TEXT, output_path=path)
    engine.unload()
    return path, {"text": SYNTHETIC_ASR_TEXT, "path": str(path), "source": "qwen3_tts", "tts": result}


def run_asr_bench(config: dict[str, Any], *, audio_path: str | None = None, force_audio: bool = False, skip_cpu: bool = False) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    expected_text: str | None = None
    generated: dict[str, Any] | None = None
    if audio_path:
        path = Path(audio_path)
    else:
        path, generated = ensure_synthetic_audio(config, force=force_audio)
        expected_text = SYNTHETIC_ASR_TEXT
    modes: list[tuple[str, str, str]] = []
    if _cuda_available():
        modes.extend([("gpu_float16", "cuda", "float16"), ("gpu_int8_float16", "cuda", "int8_float16")])
    else:
        modes.extend([("gpu_float16", "cuda", "float16"), ("gpu_int8_float16", "cuda", "int8_float16")])
    if not skip_cpu:
        modes.append(("cpu_int8", "cpu", "int8"))
    rows: dict[str, Any] = {}
    for name, device, compute_type in modes:
        asr = WhisperASR(
            model=nested(config, "asr", "model", default="large-v3-turbo"),
            device=device,
            compute_type=compute_type,
            language=nested(config, "asr", "language", default="ja"),
            beam_size=int(nested(config, "asr", "beam_size", default=5)),
        )
        try:
            result = asr.transcribe(path)
            row = result.to_dict()
            row["status"] = "measured"
            row["cer"] = _cer(expected_text, result.text)
        except Exception as exc:
            row = {"status": "error", "model": asr.model_name, "device": device, "compute_type": compute_type, "error_type": type(exc).__name__, "error": str(exc)}
        finally:
            asr.unload()
        rows[name] = row
    data = {
        "metric_name": "synthetic ASR regression",
        "input_audio": str(path),
        "same_wav_for_all_modes": True,
        "reference_text": expected_text,
        "synthetic_generation": generated,
        "modes": rows,
        "comparison": {"gpu_compute_types": ["float16", "int8_float16"], "cpu_compute_type": "int8"},
        "limitations": ["This is TTS-generated audio, not a human-speech accuracy evaluation."],
    }
    return write_benchmark(config, "asr", data, started_at=started)


def run_tts_bench(config: dict[str, Any], *, skip_cpu: bool = False) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = TTS_BENCH_TEXT
    rows: dict[str, Any] = {}
    gpu_available = _cuda_available()
    if gpu_available:
        gpu = Qwen3TTSEngine(
            model=nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            speaker=nested(config, "tts", "speaker", default="Ono_Anna"),
            language=nested(config, "tts", "language", default="Japanese"),
            device="cuda:0",
            max_new_tokens=int(nested(config, "tts", "max_new_tokens", default=2048)),
        )
        try:
            preload_started = time.monotonic_ns()
            gpu.load()
            preload_seconds = (time.monotonic_ns() - preload_started) / 1e9
            rows["gpu"] = gpu.synthesize(text, output_path=artifact_dir(config) / "tts_bench_gpu.wav")
            rows["gpu"]["preload_seconds"] = preload_seconds
        except Exception as exc:
            rows["gpu"] = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
        finally:
            gpu.unload()
    else:
        rows["gpu"] = {"status": "unavailable", "reason": "CUDA not available"}

    if not skip_cpu and bool(nested(config, "tts", "cpu_reference", default=True)):
        cpu = Qwen3TTSEngine(
            model=nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            speaker=nested(config, "tts", "speaker", default="Ono_Anna"),
            language=nested(config, "tts", "language", default="Japanese"),
            device="cpu",
            max_new_tokens=int(nested(config, "tts", "max_new_tokens", default=2048)),
        )
        try:
            preload_started = time.monotonic_ns()
            cpu.load()
            preload_seconds = (time.monotonic_ns() - preload_started) / 1e9
            rows["cpu_reference"] = cpu.synthesize(text, output_path=artifact_dir(config) / "tts_bench_cpu.wav")
            rows["cpu_reference"]["preload_seconds"] = preload_seconds
        except Exception as exc:
            rows["cpu_reference"] = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
        finally:
            cpu.unload()
    else:
        rows["cpu_reference"] = {"status": "skipped"}
    return write_benchmark(
        config,
        "tts",
        {
            "text": text,
            "model": nested(config, "tts", "model"),
            "speaker": nested(config, "tts", "speaker"),
            "language": "Japanese",
            "streaming_supported_by_official_python_api": False,
            "first_audio_metric_definition": "request start to complete waveform returned; this is first-audio-equivalent, not online packet streaming",
            "runs": rows,
        },
        started_at=started,
    )


def _provider_pair(config: dict[str, Any]) -> dict[str, Any]:
    local = OllamaLLM(
        base_url=nested(config, "llm", "local_base_url", default="http://127.0.0.1:11434"),
        model=nested(config, "llm", "local_model", default="auto"),
        num_ctx=int(nested(config, "llm", "num_ctx", default=8192)),
        max_tokens=int(nested(config, "llm", "max_tokens", default=96)),
        temperature=float(nested(config, "llm", "temperature", default=0.2)),
    )
    openrouter = OpenRouterLLM(
        base_url=nested(config, "llm", "openrouter_base_url", default="https://openrouter.ai/api/v1"),
        model=nested(config, "llm", "openrouter_model", default="openrouter/free"),
        credential_path=nested(config, "credentials", "openrouter_file", default="~/.config/credstore/openrouter.key"),
        num_ctx=int(nested(config, "llm", "num_ctx", default=8192)),
        max_tokens=int(nested(config, "llm", "openrouter_max_tokens", default=nested(config, "llm", "max_tokens", default=96))),
        temperature=float(nested(config, "llm", "temperature", default=0.2)),
    )
    return {"local": local, "openrouter": openrouter}


def _collect_turn(
    provider: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    cancel_event: Any = None,
    cancel_after_first: bool = False,
) -> dict[str, Any]:
    if cancel_after_first and cancel_event is None:
        cancel_event = threading.Event()
    started_ns = time.monotonic_ns()
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    events: list[str] = []
    first_token_ns: int | None = None
    completion: Completion | None = None
    error: LLMError | None = None
    cancelled = False
    for event in provider.stream(messages, tools=tools, cancel_event=cancel_event):
        events.append(event.kind)
        if isinstance(event, TextDelta):
            if first_token_ns is None:
                first_token_ns = time.monotonic_ns()
            text_parts.append(event.text)
            if cancel_after_first and cancel_event is not None:
                cancel_event.set()
        elif isinstance(event, ToolCall):
            calls.append(event)
        elif isinstance(event, Completion):
            completion = event
        elif isinstance(event, Cancelled):
            cancelled = True
        elif isinstance(event, LLMError):
            error = event
    ended_ns = time.monotonic_ns()
    return {
        "text": "".join(text_parts),
        "tool_calls": calls,
        "events": events,
        "completion": completion,
        "error": error,
        "cancelled": cancelled,
        "started_ns": started_ns,
        "first_token_ns": first_token_ns,
        "ended_ns": ended_ns,
        "ttft_s": (first_token_ns - started_ns) / 1e9 if first_token_ns else None,
    }


def _public_turn(turn: dict[str, Any]) -> dict[str, Any]:
    completion = turn.get("completion")
    error = turn.get("error")
    return {
        "text": turn["text"],
        "events": turn["events"],
        "tool_calls": [{"call_id": call.call_id, "name": call.name, "arguments": call.arguments} for call in turn["tool_calls"]],
        "completion": asdict(completion) if completion else None,
        "error": {"message": error.message, "status_code": error.status_code, "retryable": error.retryable} if error else None,
        "cancelled": turn["cancelled"],
        "ttft_s": turn["ttft_s"],
        "elapsed_s": (turn["ended_ns"] - turn["started_ns"]) / 1e9,
        "timing_ns": {
            "llm_start": turn["started_ns"],
            "llm_first_token": turn["first_token_ns"],
            "llm_end": turn["ended_ns"],
        },
    }


def _tool_call_messages(messages: list[dict[str, Any]], calls: list[ToolCall], registry: MockToolRegistry) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    updated = list(messages)
    updated.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": call.call_id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
                for call in calls
            ],
        }
    )
    results = []
    for call in calls:
        content = registry.call(call.name, call.arguments)
        result = {"role": "tool", "tool_call_id": call.call_id, "name": call.name, "content": content}
        updated.append(result)
        results.append({"call_id": call.call_id, "name": call.name, "content": content})
    return updated, results


def _run_tool_probe(provider: Any, registry: MockToolRegistry) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "日本語で短く答える。必要なら提供されたmock toolを使う。"},
        {"role": "user", "content": "まずcalculatorで17*23を計算し、次にfixed_test_dataのstatusを取得して、その結果を日本語で一文にまとめてください。"},
    ]
    rounds: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    for round_index in range(3):
        turn = _collect_turn(provider, messages, tools=registry.definitions())
        public = _public_turn(turn)
        public["round"] = round_index + 1
        rounds.append(public)
        if turn["error"] or turn["cancelled"] or not turn["tool_calls"]:
            break
        messages, current_results = _tool_call_messages(messages, turn["tool_calls"], registry)
        tool_results.extend(current_results)
    call_count = sum(len(row["tool_calls"]) for row in rounds)
    final_text = next((row["text"] for row in reversed(rounds) if row["text"]), "")
    return {
        "status": "success" if call_count and final_text else ("unsupported_or_failed" if not call_count else "failed"),
        "tool_call_count": call_count,
        "tool_round_count": len(rounds),
        "tool_results": tool_results,
        "rounds": rounds,
        "final_text": final_text,
    }


def _run_cancel_probe(provider: Any) -> dict[str, Any]:
    event = threading.Event()
    turn = _collect_turn(
        provider,
        [{"role": "system", "content": E2E_SYSTEM_PROMPT}, {"role": "user", "content": "長めに説明してください。"}],
        cancel_event=event,
        cancel_after_first=True,
    )
    # The first delta is the earliest safe external cancellation point. A
    # provider may finish in one packet; that outcome is recorded, not forced.
    return {"pre_cancel": False, "midstream_cancel_observed": turn["cancelled"], "turn": _public_turn(turn)}


def run_llm_bench(config: dict[str, Any]) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    registry = MockToolRegistry()
    results: dict[str, Any] = {}
    for provider_name, provider in _provider_pair(config).items():
        normal = _collect_turn(
            provider,
            [{"role": "system", "content": E2E_SYSTEM_PROMPT}, {"role": "user", "content": "今日は元気ですか。"}],
        )
        provider_result: dict[str, Any] = {"requested_model": provider.requested_model, "normal_stream": _public_turn(normal)}
        provider_result["actual_model"] = provider.last_actual_model or (normal["completion"].actual_model if normal["completion"] else None)
        try:
            provider_result["tool_calling"] = _run_tool_probe(provider, registry)
        except Exception as exc:
            provider_result["tool_calling"] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        try:
            provider_result["cancel"] = _run_cancel_probe(provider)
        except Exception as exc:
            provider_result["cancel"] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        results[provider_name] = provider_result
    return write_benchmark(
        config,
        "llm",
        {
            "requested_openrouter_model": nested(config, "llm", "openrouter_model", default="openrouter/free"),
            "context_target": nested(config, "llm", "num_ctx", default=8192),
            "providers": results,
            "tool_scope": "PoC-only calculator, fixed_test_data, deterministic_time; no destructive service calls.",
        },
        started_at=started,
    )


def run_e2e_bench(config: dict[str, Any], *, force_audio: bool = False, skip_openrouter: bool = False) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        user_audio, generated = ensure_synthetic_audio(config, force=force_audio)
    except Exception as exc:
        return write_benchmark(config, "e2e", {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc), "configurations": {}}, started_at=started)
    tts = Qwen3TTSEngine(
        model=nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        speaker=nested(config, "tts", "speaker", default="Ono_Anna"),
        language="Japanese",
        device="cuda:0" if _cuda_available() else "auto",
        max_new_tokens=int(nested(config, "tts", "max_new_tokens", default=2048)),
    )
    providers = _provider_pair(config)
    cases = {
        "A_gpu_asr_local_gpu_tts": ("cuda", "float16", "local"),
        "B_cpu_asr_local_gpu_tts": ("cpu", "int8", "local"),
        "C_gpu_asr_openrouter_gpu_tts": ("cuda", "float16", "openrouter"),
        "D_cpu_asr_openrouter_gpu_tts": ("cpu", "int8", "openrouter"),
    }
    asr_cache: dict[tuple[str, str], WhisperASR] = {}
    outputs: dict[str, Any] = {}
    try:
        for case_name, (device, compute_type, provider_name) in cases.items():
            if skip_openrouter and provider_name == "openrouter":
                outputs[case_name] = {"status": "skipped"}
                continue
            if device == "cuda" and not _cuda_available():
                outputs[case_name] = {"status": "unavailable", "reason": "CUDA not available"}
                continue
            asr = asr_cache.setdefault(
                (device, compute_type),
                WhisperASR(
                    model=nested(config, "asr", "model", default="large-v3-turbo"),
                    device=device,
                    compute_type=compute_type,
                    language="ja",
                    beam_size=int(nested(config, "asr", "beam_size", default=5)),
                ),
            )
            event_log = EventLog()
            started_ns = time.monotonic_ns()
            event_log.mark("vad_start")
            try:
                audio, sample_rate = sf.read(str(user_audio), always_2d=False)
                intervals = detect_speech_intervals(np.asarray(audio), sample_rate)
                speech_audio = trim_to_speech(np.asarray(audio), sample_rate, intervals)
                speech_path = artifact_dir(config) / f"{case_name}_user_trim.wav"
                sf.write(str(speech_path), speech_audio, sample_rate)
                event_log.mark("vad_end", intervals=intervals)
                user_asr = asr.transcribe(speech_path, event_log=event_log)
                event_log.mark("llm_start")
                turn = _collect_turn(
                    providers[provider_name],
                    [{"role": "system", "content": E2E_SYSTEM_PROMPT}, {"role": "user", "content": user_asr.text}],
                )
                event_log.mark("llm_first_token", ttft_s=turn["ttft_s"]) if turn["first_token_ns"] else None
                event_log.mark("llm_end")
                if turn["error"] or not turn["text"]:
                    outputs[case_name] = {
                        "status": "error",
                        "provider": provider_name,
                        "requested_model": providers[provider_name].requested_model,
                        "actual_model": providers[provider_name].last_actual_model or (turn["completion"].actual_model if turn["completion"] else None),
                        "user_asr": user_asr.to_dict(),
                        "llm": _public_turn(turn),
                        "events": event_log.events,
                    }
                    continue
                chunker = SentenceChunker(
                    max_chars=int(nested(config, "tts", "sentence_max_chars", default=48)),
                    timeout_s=float(nested(config, "tts", "sentence_timeout_s", default=0.8)),
                )
                chunks = chunker.push(turn["text"])
                chunks.extend(chunker.flush())
                assistant_paths: list[str] = []
                tts_rows: list[dict[str, Any]] = []
                for index, chunk in enumerate(chunks):
                    event_log.mark("tts_request", text_chars=len(chunk))
                    tts_row = tts.synthesize(chunk, output_path=artifact_dir(config) / f"{case_name}_assistant_{index}.wav", event_log=event_log)
                    tts_rows.append(tts_row)
                    assistant_paths.append(tts_row["path"])
                output_asr_rows = [asr.transcribe(path, event_log=event_log).to_dict() for path in assistant_paths]
                event_log.mark("playback_start", path=assistant_paths[0] if assistant_paths else None)
                # E2E confirmation uses the generated file; physical playback is
                # independently validated in bench aec to avoid conflating tests.
                event_log.mark("playback_end", path=assistant_paths[-1] if assistant_paths else None)
                ended_ns = time.monotonic_ns()
                outputs[case_name] = {
                    "status": "measured",
                    "provider": provider_name,
                    "requested_model": providers[provider_name].requested_model,
                    "actual_model": providers[provider_name].last_actual_model or (turn["completion"].actual_model if turn["completion"] else None),
                    "user_audio": str(user_audio),
                    "user_asr": user_asr.to_dict(),
                    "llm": _public_turn(turn),
                    "tts": tts_rows,
                    "assistant_asr": output_asr_rows,
                    "assistant_cer": [_cer(row["text"], output_asr_rows[index]["text"]) for index, row in enumerate(tts_rows)],
                    "e2e_latency_s": (ended_ns - started_ns) / 1e9,
                    "events": event_log.events,
                }
            except Exception as exc:
                outputs[case_name] = {"status": "error", "error_type": type(exc).__name__, "error": str(exc), "events": event_log.events}
    finally:
        for asr in asr_cache.values():
            asr.unload()
        tts.unload()
    return write_benchmark(config, "e2e", {"status": "measured", "synthetic_user_audio": generated, "configurations": outputs}, started_at=started)


def run_aec_bench(config: dict[str, Any], *, force_audio: bool = False) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        reference, generated = ensure_tts_reference(config, force=force_audio)
    except Exception as exc:
        return write_benchmark(config, "aec", {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)}, started_at=started)
    inventory = PipeWireInventory.discover()
    mic = inventory.usb_microphone()
    speaker = inventory.usb_speaker()
    data: dict[str, Any] = {
        "status": "blocked",
        "reference_audio": str(reference),
        "reference_generation": generated,
        "physical_path": "TTS -> Echo Cancellation Sink -> USB speaker -> room -> USB microphone -> Echo Cancellation Source",
        "inventory_before": inventory.to_dict(),
        "aec_library": nested(config, "pipewire", "aec_library", default="aec/libspa-aec-webrtc"),
        "off": None,
        "on": None,
        "comparison": None,
    }
    if not mic or not speaker:
        data["reason"] = "USB microphone or USB speaker was not detected"
        return write_benchmark(config, "aec", data, started_at=started)
    off_path = artifact_dir(config) / "aec_off.wav"
    on_path = artifact_dir(config) / "aec_on.wav"
    try:
        data["off"] = play_and_record(reference, off_path, playback_target=speaker.node_id, capture_target=mic.node_id)
        session = EchoCancelSession(
            sink_name=nested(config, "pipewire", "echo_cancel_sink", default="Local Live Echo Cancellation Sink"),
            source_name=nested(config, "pipewire", "echo_cancel_source", default="Local Live Echo Cancellation Source"),
            capture_name=nested(config, "pipewire", "echo_cancel_capture", default="Local Live Echo Cancellation Capture"),
            playback_name=nested(config, "pipewire", "echo_cancel_playback", default="Local Live Echo Cancellation Playback"),
            latency=nested(config, "pipewire", "node_latency", default="1024/48000"),
            sink_master=speaker.node_id,
            source_master=mic.node_id,
        )
        with session:
            after = PipeWireInventory.discover()
            named_sink = next((node for node in after.sinks if node.node_id == session.sink_node_id), None)
            named_source = next((node for node in after.sources if node.node_id == session.source_node_id), None)
            if not named_sink or not named_source:
                raise RuntimeError("echo-cancel source/sink unavailable after module load")
            data["aec_module"] = {"module_id": session.module_id, "named_sink": named_sink.__dict__, "named_source": named_source.__dict__}
            data["aec_module"].update({"sink_master": speaker.node_id, "source_master": mic.node_id})
            data["on"] = play_and_record(reference, on_path, playback_target=session.sink_node_id, capture_target=session.source_node_id)
        data["comparison"] = compare_aec_recordings(reference, off_path, on_path)
        if not data["comparison"].get("measurement_valid"):
            data["status"] = "blocked"
            data["reason"] = "physical capture was silent; AEC attenuation and ASR rerecognition are unmeasured"
            data["asr_self_rerecognition"] = {"status": "unavailable", "reason": "off/on recording RMS was at or below the signal floor"}
            return write_benchmark(config, "aec", data, started_at=started)
        try:
            asr_device = "cuda" if _cuda_available() else "cpu"
            asr_compute_type = "float16" if asr_device == "cuda" else "int8"
            asr = WhisperASR(
                model=nested(config, "asr", "model", default="large-v3-turbo"),
                device=asr_device,
                compute_type=asr_compute_type,
                language="ja",
                beam_size=int(nested(config, "asr", "beam_size", default=5)),
            )
            rerecognition: dict[str, Any] = {}
            for label, recording_path in (("off", off_path), ("on", on_path)):
                transcription = asr.transcribe(recording_path)
                score = _cer(TTS_BENCH_TEXT, transcription.text)
                rerecognition[label] = {
                    "transcript": transcription.text,
                    "cer_vs_reference_text": score,
                    "recognition_score_1_minus_cer_clamped": max(0.0, 1.0 - score) if score is not None else None,
                    "asr": transcription.to_dict(),
                }
            data["asr_self_rerecognition"] = rerecognition
        except Exception as exc:
            data["asr_self_rerecognition"] = {"status": "unavailable", "error_type": type(exc).__name__, "error": str(exc)}
        finally:
            try:
                asr.unload()
            except UnboundLocalError:
                pass
        data["status"] = "measured"
    except Exception as exc:
        data["status"] = "blocked" if isinstance(exc, (RuntimeError, OSError)) else "error"
        data["error_type"] = type(exc).__name__
        data["error"] = str(exc)
        if data["status"] == "blocked":
            data["reason"] = "physical PipeWire capture/playback route could not be completed"
    return write_benchmark(config, "aec", data, started_at=started)


def ensure_tts_reference(config: dict[str, Any], *, force: bool = False) -> tuple[Path, dict[str, Any]]:
    path = artifact_dir(config) / "aec_reference.wav"
    if path.exists() and not force:
        info = sf.info(str(path))
        return path, {"source": "reused_existing_generated_audio", "text": TTS_BENCH_TEXT, "duration_s": info.duration}
    engine = Qwen3TTSEngine(
        model=nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        speaker=nested(config, "tts", "speaker", default="Ono_Anna"),
        language="Japanese",
        device="cuda:0" if _cuda_available() else "auto",
    )
    row = engine.synthesize(TTS_BENCH_TEXT, output_path=path)
    engine.unload()
    return path, {"source": "qwen3_tts", "text": TTS_BENCH_TEXT, "tts": row}


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cer(reference: str | None, hypothesis: str) -> float | None:
    if reference is None:
        return None
    from .normalize import cer

    return cer(reference, hypothesis)
