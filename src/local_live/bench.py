from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from .asr import WhisperASR
from .audio import EchoCancelSession, PipeWireInventory, audio_file_stats, compare_aec_recordings, play_and_record
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
E2E_MEDIAN_FIELDS = [
    "asr_duration_s",
    "llm_ttft_s",
    "llm_total_duration_s",
    "tts_duration_s",
    "playback_roundtrip_duration_s",
    "total_e2e_duration_s",
    "peak_vram_mib",
    "input_read_s",
    "vad_duration_s",
    "assistant_asr_duration_s",
]
TTS_LATENCY_CHUNKS = {
    "short": "はい、元気です。",
    "medium": "今日は日本語の音声応答を確認しています。",
    "long": "これは日本語のLive音声対話における文単位chunkの初動時間と再生準備時間を測定するための長めのテスト文です。",
}
TTS_LATENCY_MEDIAN_FIELDS = [
    "elapsed_seconds",
    "model_load_seconds",
    "inference_elapsed_seconds",
    "first_audio_equivalent_seconds",
    "warm_first_audio_equivalent_seconds",
    "audio_complete_seconds",
    "playback_possible_seconds",
    "rtf",
    "gpu_memory_peak_mib",
    "gpu_memory_delta_peak_mib",
]


def benchmark_path(config: dict[str, Any], name: str) -> Path:
    return Path(nested(config, "app", "result_dir", default="results")) / f"bench_{name}.json"


def artifact_dir(config: dict[str, Any]) -> Path:
    path = Path(nested(config, "app", "artifact_dir", default="results/artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _measurement_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{time.monotonic_ns()}"


def write_benchmark(config: dict[str, Any], name: str, data: dict[str, Any], *, started_at: str | None = None) -> dict[str, Any]:
    result = {
        "schema": f"local-live-ja/bench-{name}/v2",
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


def _tts_attempt(
    engine: Qwen3TTSEngine,
    *,
    text: str,
    output_path: Path,
    chunk: str,
    phase: str,
    run_number: int,
) -> dict[str, Any]:
    row: dict[str, Any]
    try:
        row = engine.synthesize(text, output_path=output_path)
        row["status"] = "measured"
    except Exception as exc:
        row = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    row.update({"chunk": chunk, "phase": phase, "run_number": run_number, "text_chars": len(text)})
    return row


def _measure_tts_latency_matrix(config: dict[str, Any], engine: Qwen3TTSEngine, *, label: str) -> dict[str, Any]:
    warm_repeats = max(3, int(nested(config, "bench", "tts_warm_repeats", default=3)))
    run_id = _measurement_id(f"tts_{label}")
    cold_text = TTS_LATENCY_CHUNKS["short"]
    cold = _tts_attempt(
        engine,
        text=cold_text,
        output_path=artifact_dir(config) / f"{run_id}_cold_short.wav",
        chunk="short",
        phase="cold_start",
        run_number=0,
    )
    warm_start: dict[str, Any] = {}
    for chunk, text in TTS_LATENCY_CHUNKS.items():
        runs = [
            _tts_attempt(
                engine,
                text=text,
                output_path=artifact_dir(config) / f"{run_id}_warm_{chunk}_{run_number}.wav",
                chunk=chunk,
                phase="warm_start",
                run_number=run_number,
            )
            for run_number in range(1, warm_repeats + 1)
        ]
        warm_start[chunk] = {
            "text": text,
            "text_chars": len(text),
            "model_resident_for_all_runs": True,
            "runs": runs,
            "median": _summarize_runs(runs, TTS_LATENCY_MEDIAN_FIELDS),
        }
    all_warm_ok = all(
        all(row.get("status") == "measured" for row in chunk_data["runs"])
        for chunk_data in warm_start.values()
    )
    return {
        "status": "measured" if cold.get("status") == "measured" and all_warm_ok else "partial",
        "device": engine.resolved_device or engine.resolve_device(),
        "cold_start": cold,
        "warm_start": warm_start,
        "warm_repeat_target": warm_repeats,
        "cold_definition": "first synthesize on a newly constructed engine; model load is included",
        "warm_definition": "subsequent synthesize calls on the same resident engine; model_load_seconds should be zero",
    }


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
            matrix = _measure_tts_latency_matrix(config, gpu, label="gpu")
            rows["gpu"] = {
                "status": matrix["status"],
                "model": gpu.model_name,
                "speaker": gpu.speaker,
                "language": gpu.language,
                "device": matrix["device"],
                "latency_matrix": matrix,
                "cold_start": matrix["cold_start"],
                "warm_start": matrix["warm_start"],
            }
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
            row = _tts_attempt(
                cpu,
                text=text,
                output_path=artifact_dir(config) / f"{_measurement_id('tts_cpu')}.wav",
                chunk="reference",
                phase="cold_reference",
                run_number=0,
            )
            rows["cpu_reference"] = row
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
            "language": nested(config, "tts", "language", default="Japanese"),
            "streaming_supported_by_official_python_api": False,
            "first_audio_metric_definition": "request start to complete waveform returned; this is first-audio-equivalent, not online packet streaming",
            "latency_matrix_definition": "GPU cold first call followed by resident-model warm calls for short/medium/long Japanese chunks",
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
        "error": (
            {
                "message": error.message,
                "status_code": error.status_code,
                "retryable": error.retryable,
                "details": error.details,
            }
            if error
            else None
        ),
        "cancelled": turn["cancelled"],
        "ttft_s": turn["ttft_s"],
        "elapsed_s": (turn["ended_ns"] - turn["started_ns"]) / 1e9,
        "timing_ns": {
            "llm_start": turn["started_ns"],
            "llm_first_token": turn["first_token_ns"],
            "llm_end": turn["ended_ns"],
        },
    }


def _summarize_runs(runs: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    """Return all observed values and medians without dropping outliers."""
    summary: dict[str, Any] = {}
    for field in fields:
        values = [
            row[field]
            for row in runs
            if isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)
        ]
        summary[field] = {"values": values, "median": statistics.median(values) if values else None}
    return summary


def _tool_call_messages(
    messages: list[dict[str, Any]],
    calls: list[ToolCall],
    registry: MockToolRegistry,
    *,
    argument_format: str = "openai",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if argument_format not in {"openai", "ollama"}:
        raise ValueError(f"unsupported tool argument format: {argument_format}")
    updated = list(messages)
    if argument_format == "ollama":
        tool_calls = [{"function": {"name": call.name, "arguments": call.arguments}} for call in calls]
        updated.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
    else:
        updated.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                    }
                    for call in calls
                ],
            }
        )
    results = []
    for call in calls:
        content = registry.call(call.name, call.arguments)
        result = (
            {"role": "tool", "content": content}
            if argument_format == "ollama"
            else {"role": "tool", "tool_call_id": call.call_id, "name": call.name, "content": content}
        )
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
        messages, current_results = _tool_call_messages(
            messages,
            turn["tool_calls"],
            registry,
            argument_format="ollama" if isinstance(provider, OllamaLLM) else "openai",
        )
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


def _actual_model(provider: Any, turn: dict[str, Any]) -> str | None:
    completion = turn.get("completion")
    return getattr(provider, "last_actual_model", None) or (completion.actual_model if completion else None)


def _numeric_max(rows: list[dict[str, Any]], field: str) -> int | float | None:
    values = [row[field] for row in rows if isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)]
    return max(values) if values else None


def _run_e2e_case(
    config: dict[str, Any],
    *,
    case_name: str,
    run_id: str,
    run_number: int,
    warmup: bool,
    user_audio: Path,
    provider: Any,
    asr: WhisperASR,
    tts: Qwen3TTSEngine,
) -> dict[str, Any]:
    """Run one complete logical E2E turn and retain all stage boundaries."""
    provider.last_actual_model = None
    event_log = EventLog()
    started_ns = time.monotonic_ns()
    event_log.mark("e2e_start", run_number=run_number, warmup=warmup)
    user_asr: Any = None
    turn: dict[str, Any] | None = None
    tts_rows: list[dict[str, Any]] = []
    output_asr_rows: list[dict[str, Any]] = []
    intervals: list[tuple[float, float]] = []
    read_start_ns: int | None = None
    read_end_ns: int | None = None
    vad_start_ns: int | None = None
    vad_end_ns: int | None = None
    asr_start_ns: int | None = None
    asr_end_ns: int | None = None
    llm_start_ns: int | None = None
    llm_end_ns: int | None = None
    assistant_asr_start_ns: int | None = None
    assistant_asr_end_ns: int | None = None

    def finalize(status: str, *, error: str | None = None) -> dict[str, Any]:
        ended_ns = time.monotonic_ns()
        all_rows = ([] if user_asr is None else [user_asr.to_dict()]) + tts_rows + output_asr_rows
        stage_durations: dict[str, Any] = {
            "input_read_s": (read_end_ns - read_start_ns) / 1e9 if read_start_ns and read_end_ns else None,
            "vad_duration_s": (vad_end_ns - vad_start_ns) / 1e9 if vad_start_ns and vad_end_ns else None,
            "asr_duration_s": user_asr.elapsed_seconds if user_asr is not None else ((asr_end_ns - asr_start_ns) / 1e9 if asr_start_ns and asr_end_ns else None),
            "llm_ttft_s": turn["ttft_s"] if turn else None,
            "llm_total_duration_s": ((turn["ended_ns"] - turn["started_ns"]) / 1e9) if turn else ((llm_end_ns - llm_start_ns) / 1e9 if llm_start_ns and llm_end_ns else None),
            "tts_duration_s": sum(float(row["elapsed_seconds"]) for row in tts_rows) if tts_rows else None,
            "assistant_asr_duration_s": sum(float(row["elapsed_seconds"]) for row in output_asr_rows) if output_asr_rows else None,
            "playback_roundtrip_duration_s": None,
            "total_e2e_duration_s": (ended_ns - started_ns) / 1e9,
        }
        if tts_rows:
            first_request = tts_rows[0].get("timing_ns", {}).get("request_start")
            last_possible = tts_rows[-1].get("timing_ns", {}).get("playback_possible")
            stage_durations["playback_possible_duration_s"] = (last_possible - first_request) / 1e9 if first_request and last_possible else None
        else:
            stage_durations["playback_possible_duration_s"] = None
        roundtrip: list[dict[str, Any]] = []
        for index, tts_row in enumerate(tts_rows):
            asr_row = output_asr_rows[index] if index < len(output_asr_rows) else None
            roundtrip.append(
                {
                    "spoken_text": tts_row["text"],
                    "asr_text": asr_row["text"] if asr_row else None,
                    "cer": _cer(tts_row["text"], asr_row["text"]) if asr_row else None,
                }
            )
        result: dict[str, Any] = {
            "status": status,
            "case_name": case_name,
            "run_number": run_number,
            "warmup": warmup,
            "provider": getattr(provider, "name", None),
            "requested_model": getattr(provider, "requested_model", None),
            "actual_model": _actual_model(provider, turn or {}),
            "user_audio": str(user_audio),
            "intervals": intervals,
            "user_asr": user_asr.to_dict() if user_asr is not None else None,
            "llm": _public_turn(turn) if turn else None,
            "tts": tts_rows,
            "assistant_asr": output_asr_rows,
            "assistant_roundtrip": roundtrip,
            "assistant_cer": [item["cer"] for item in roundtrip],
            "stage_durations": stage_durations,
            "asr_duration_s": stage_durations["asr_duration_s"],
            "llm_ttft_s": stage_durations["llm_ttft_s"],
            "llm_total_duration_s": stage_durations["llm_total_duration_s"],
            "tts_duration_s": stage_durations["tts_duration_s"],
            "playback_roundtrip_duration_s": stage_durations["playback_roundtrip_duration_s"],
            "total_e2e_duration_s": stage_durations["total_e2e_duration_s"],
            "input_read_s": stage_durations["input_read_s"],
            "vad_duration_s": stage_durations["vad_duration_s"],
            "assistant_asr_duration_s": stage_durations["assistant_asr_duration_s"],
            "playback_possible_duration_s": stage_durations["playback_possible_duration_s"],
            "peak_vram_mib": _numeric_max(all_rows, "gpu_memory_peak_mib"),
            "playback": {
                "status": "not_measured",
                "duration_s": None,
                "reason": "E2E confirms generated WAVs; physical USB playback/roundtrip is measured only by bench aec",
            },
            "timing_ns": {
                "e2e_start": started_ns,
                "input_read_start": read_start_ns,
                "input_read_end": read_end_ns,
                "vad_start": vad_start_ns,
                "vad_end": vad_end_ns,
                "asr_start": asr_start_ns,
                "asr_end": asr_end_ns,
                "llm_start": llm_start_ns,
                "llm_end": llm_end_ns,
                "assistant_asr_start": assistant_asr_start_ns,
                "assistant_asr_end": assistant_asr_end_ns,
                "e2e_end": ended_ns,
            },
            "events": event_log.events,
        }
        if error:
            result["error_type"] = error.split(":", 1)[0]
            result["error"] = error
        return result

    try:
        read_start_ns = time.monotonic_ns()
        audio, sample_rate = sf.read(str(user_audio), always_2d=False)
        read_end_ns = time.monotonic_ns()
        event_log.mark("input_read_end", duration_s=(read_end_ns - read_start_ns) / 1e9)
        vad_start_ns = time.monotonic_ns()
        event_log.mark("vad_start")
        intervals = detect_speech_intervals(np.asarray(audio), sample_rate)
        speech_audio = trim_to_speech(np.asarray(audio), sample_rate, intervals)
        speech_path = artifact_dir(config) / f"{run_id}_{case_name}_run{run_number}_user_trim.wav"
        sf.write(str(speech_path), speech_audio, sample_rate)
        vad_end_ns = time.monotonic_ns()
        event_log.mark("vad_end", intervals=intervals, duration_s=(vad_end_ns - vad_start_ns) / 1e9)
        asr_start_ns = time.monotonic_ns()
        user_asr = asr.transcribe(speech_path, event_log=event_log)
        asr_end_ns = time.monotonic_ns()
        event_log.mark("user_asr_end", duration_s=user_asr.elapsed_seconds)
        llm_start_ns = time.monotonic_ns()
        event_log.mark("llm_start")
        turn = _collect_turn(
            provider,
            [{"role": "system", "content": E2E_SYSTEM_PROMPT}, {"role": "user", "content": user_asr.text}],
        )
        llm_end_ns = time.monotonic_ns()
        if turn["first_token_ns"]:
            event_log.mark("llm_first_token", ttft_s=turn["ttft_s"])
        event_log.mark("llm_end", duration_s=(llm_end_ns - llm_start_ns) / 1e9, actual_model=_actual_model(provider, turn))
        if turn["error"] or not turn["text"]:
            return finalize("error", error=turn["error"].message if turn["error"] else "LLM returned no visible text")
        chunker = SentenceChunker(
            max_chars=int(nested(config, "tts", "sentence_max_chars", default=48)),
            timeout_s=float(nested(config, "tts", "sentence_timeout_s", default=0.8)),
        )
        chunks = chunker.push(turn["text"])
        chunks.extend(chunker.flush())
        for index, chunk in enumerate(chunks):
            event_log.mark("tts_request", text_chars=len(chunk))
            tts_rows.append(
                tts.synthesize(
                    chunk,
                    output_path=artifact_dir(config) / f"{run_id}_{case_name}_run{run_number}_assistant_{index}.wav",
                    event_log=event_log,
                )
            )
        assistant_paths = [row["path"] for row in tts_rows]
        assistant_asr_start_ns = time.monotonic_ns()
        output_asr_rows = [asr.transcribe(path, event_log=event_log).to_dict() for path in assistant_paths]
        assistant_asr_end_ns = time.monotonic_ns()
        event_log.mark("assistant_asr_end", duration_s=(assistant_asr_end_ns - assistant_asr_start_ns) / 1e9)
        event_log.mark("playback_start", path=assistant_paths[0] if assistant_paths else None, physical=False)
        event_log.mark("playback_end", path=assistant_paths[-1] if assistant_paths else None, physical=False)
        return finalize("measured")
    except Exception as exc:
        return finalize("error", error=f"{type(exc).__name__}: {exc}")


def _annotate_e2e_outliers(runs: list[dict[str, Any]], median: dict[str, Any]) -> None:
    """Annotate, never remove, runs that are over three times their median."""
    fields = ["input_read_s", "vad_duration_s", "asr_duration_s", "llm_total_duration_s", "tts_duration_s", "assistant_asr_duration_s"]
    for run in runs:
        flags: list[str] = []
        values = {field: run.get("stage_durations", {}).get(field) for field in fields}
        for field in fields:
            value = values[field]
            med = (median.get(field) or {}).get("median")
            if isinstance(value, (int, float)) and isinstance(med, (int, float)) and med > 0 and value > max(3.0 * med, med + 1.0):
                flags.append(field)
        run["outlier_flags"] = flags
        numeric = [(field, value) for field, value in values.items() if isinstance(value, (int, float))]
        run["dominant_stage"] = max(numeric, key=lambda item: item[1])[0] if numeric else None


def run_e2e_bench(config: dict[str, Any], *, force_audio: bool = False, skip_openrouter: bool = False) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        user_audio, generated = ensure_synthetic_audio(config, force=force_audio)
    except Exception as exc:
        return write_benchmark(config, "e2e", {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc), "configurations": {}}, started_at=started)
    providers = _provider_pair(config)
    gpu_compute_type = str(nested(config, "asr", "gpu_default_compute_type", default="float16"))
    cases = {
        "A_gpu_asr_local_gpu_tts": ("cuda", gpu_compute_type, "local"),
        "B_cpu_asr_local_gpu_tts": ("cpu", "int8", "local"),
        "C_gpu_asr_openrouter_gpu_tts": ("cuda", gpu_compute_type, "openrouter"),
        "D_cpu_asr_openrouter_gpu_tts": ("cpu", "int8", "openrouter"),
    }
    repeats = max(3, int(nested(config, "bench", "e2e_repeats", default=3)))
    max_attempts = max(repeats, int(nested(config, "bench", "e2e_max_attempts", default=repeats * 3)))
    run_id = _measurement_id("e2e")
    outputs: dict[str, Any] = {}
    for case_name, (device, compute_type, provider_name) in cases.items():
        if skip_openrouter and provider_name == "openrouter":
            outputs[case_name] = {"status": "skipped", "repeat_target": repeats}
            continue
        if device == "cuda" and not _cuda_available():
            outputs[case_name] = {"status": "unavailable", "repeat_target": repeats, "reason": "CUDA not available"}
            continue
        asr = WhisperASR(
            model=nested(config, "asr", "model", default="large-v3-turbo"),
            device=device,
            compute_type=compute_type,
            language="ja",
            beam_size=int(nested(config, "asr", "beam_size", default=5)),
        )
        tts = Qwen3TTSEngine(
            model=nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            speaker=nested(config, "tts", "speaker", default="Ono_Anna"),
            language=nested(config, "tts", "language", default="Japanese"),
            device="cuda:0" if _cuda_available() else "auto",
            max_new_tokens=int(nested(config, "tts", "max_new_tokens", default=2048)),
        )
        try:
            warmup = _run_e2e_case(
                config,
                case_name=case_name,
                run_id=run_id,
                run_number=0,
                warmup=True,
                user_audio=user_audio,
                provider=providers[provider_name],
                asr=asr,
                tts=tts,
            )
            warmup_recovery = None
            if warmup["status"] != "measured":
                warmup_recovery = _tts_attempt(
                    tts,
                    text="音声モデルのwarm-upです。",
                    output_path=artifact_dir(config) / f"{run_id}_{case_name}_warmup_tts.wav",
                    chunk="warmup_recovery",
                    phase="warmup_recovery",
                    run_number=0,
                )
            runs: list[dict[str, Any]] = []
            while len([run for run in runs if run["status"] == "measured"]) < repeats and len(runs) < max_attempts:
                run_number = len(runs) + 1
                runs.append(
                    _run_e2e_case(
                        config,
                        case_name=case_name,
                        run_id=run_id,
                        run_number=run_number,
                        warmup=False,
                        user_audio=user_audio,
                        provider=providers[provider_name],
                        asr=asr,
                        tts=tts,
                    )
                )
            measured = [run for run in runs if run["status"] == "measured"]
            median = _summarize_runs(measured, E2E_MEDIAN_FIELDS)
            _annotate_e2e_outliers(runs, median)
            outputs[case_name] = {
                "status": "measured" if len(measured) == repeats else "partial",
                "provider": provider_name,
                "requested_model": providers[provider_name].requested_model,
                "actual_models_by_run": [run.get("actual_model") for run in runs],
                "compute_type": compute_type,
                "repeat_target": repeats,
                "max_attempts": max_attempts,
                "attempt_count": len(runs),
                "measured_run_count": len(measured),
                "warmup": warmup,
                "warmup_recovery": warmup_recovery,
                "runs": runs,
                "median": median,
                "outlier_policy": "retain every run; annotate a stage when value exceeds max(3x median, median+1s)",
            }
        finally:
            asr.unload()
            tts.unload()
    requested = [value for value in outputs.values() if value.get("status") not in {"skipped", "unavailable"}]
    if requested and all(value.get("status") == "measured" for value in requested):
        status = "measured"
    elif any(value.get("status") in {"measured", "partial"} for value in requested):
        status = "partial"
    else:
        status = "blocked"
    return write_benchmark(
        config,
        "e2e",
        {
            "status": status,
            "repeat_target": repeats,
            "synthetic_user_audio": generated,
            "gpu_compute_type": gpu_compute_type,
            "physical_playback_measured_separately_by": "bench aec",
            "configurations": outputs,
        },
        started_at=started,
    )


def run_aec_bench(config: dict[str, Any], *, force_audio: bool = False) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        reference, generated = ensure_tts_reference(config, force=force_audio)
    except Exception as exc:
        return write_benchmark(config, "aec", {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)}, started_at=started)
    run_id = _measurement_id("aec")
    signal_floor = float(nested(config, "bench", "audio_signal_floor_rms", default=1e-6))
    inventory = PipeWireInventory.discover()
    mic = inventory.usb_microphone()
    speaker = inventory.usb_speaker()
    data: dict[str, Any] = {
        "status": "blocked",
        "run_id": run_id,
        "reference_audio": str(reference),
        "reference_generation": generated,
        "physical_path": "TTS -> Echo Cancellation Sink -> USB speaker -> room -> USB microphone -> Echo Cancellation Source",
        "inventory_before": inventory.to_dict(),
        "aec_library": nested(config, "pipewire", "aec_library", default="aec/libspa-aec-webrtc"),
        "signal_floor_rms": signal_floor,
        "recording_files_are_unique_per_run": True,
        "node_ids_are_runtime_only": True,
        "raw_capture": None,
        "raw_capture_validation": None,
        "off": None,
        "on": None,
        "comparison": None,
        "aec_attempted": False,
        "asr_self_rerecognition": None,
    }
    if not mic or not speaker:
        data["reason"] = "USB microphone or USB speaker was not detected"
        return write_benchmark(config, "aec", data, started_at=started)
    asr: WhisperASR | None = None
    try:
        raw_path = artifact_dir(config) / f"{run_id}_raw.wav"
        raw = play_and_record(reference, raw_path, playback_target=speaker.node_id, capture_target=mic.node_id)
        raw_stats = raw.get("recording_stats") or audio_file_stats(raw_path)
        raw["recording_stats"] = raw_stats
        data["raw_capture"] = raw
        data["raw_capture_validation"] = {
            "status": "pass" if raw_stats["rms"] > signal_floor and raw_stats["peak"] > signal_floor else "blocked",
            "duration_s": raw_stats["duration_s"],
            "rms": raw_stats["rms"],
            "peak": raw_stats["peak"],
            "signal_floor_rms": signal_floor,
            "playback_target": speaker.__dict__,
            "capture_target": mic.__dict__,
        }
        if raw_stats["rms"] <= signal_floor or raw_stats["peak"] <= signal_floor:
            data["reason"] = "raw USB capture is silent or below signal floor; AEC was not attempted"
            data["asr_self_rerecognition"] = {"status": "blocked", "reason": "raw capture did not contain a measurable signal"}
            return write_benchmark(config, "aec", data, started_at=started)

        # Re-resolve both masters before the AEC-off run; no node ID is read from config.
        off_inventory = PipeWireInventory.discover()
        off_mic = off_inventory.usb_microphone()
        off_speaker = off_inventory.usb_speaker()
        if not off_mic or not off_speaker:
            data["reason"] = "USB targets disappeared before AEC-off run"
            return write_benchmark(config, "aec", data, started_at=started)
        off_path = artifact_dir(config) / f"{run_id}_off.wav"
        data["off"] = play_and_record(reference, off_path, playback_target=off_speaker.node_id, capture_target=off_mic.node_id)
        off_stats = data["off"].get("recording_stats") or audio_file_stats(off_path)
        data["off"]["recording_stats"] = off_stats
        if off_stats["rms"] <= signal_floor or off_stats["peak"] <= signal_floor:
            data["reason"] = "AEC-off capture fell below signal floor; AEC-on was not attempted"
            data["aec_attempted"] = False
            return write_benchmark(config, "aec", data, started_at=started)

        # Resolve current USB masters again for module loading.
        aec_inventory = PipeWireInventory.discover()
        aec_mic = aec_inventory.usb_microphone()
        aec_speaker = aec_inventory.usb_speaker()
        if not aec_mic or not aec_speaker:
            data["reason"] = "USB targets disappeared before AEC module load"
            return write_benchmark(config, "aec", data, started_at=started)
        session = EchoCancelSession(
            sink_name=nested(config, "pipewire", "echo_cancel_sink", default="Local Live Echo Cancellation Sink"),
            source_name=nested(config, "pipewire", "echo_cancel_source", default="Local Live Echo Cancellation Source"),
            capture_name=nested(config, "pipewire", "echo_cancel_capture", default="Local Live Echo Cancellation Capture"),
            playback_name=nested(config, "pipewire", "echo_cancel_playback", default="Local Live Echo Cancellation Playback"),
            latency=nested(config, "pipewire", "node_latency", default="1024/48000"),
            sink_master=aec_speaker.node_id,
            source_master=aec_mic.node_id,
        )
        module_info = session.load()
        data["aec_attempted"] = True
        try:
            after = PipeWireInventory.discover()
            named_sink = next((node for node in after.sinks if node.node_id == session.sink_node_id), None)
            named_source = next((node for node in after.sources if node.node_id == session.source_node_id), None)
            if not named_sink or not named_source:
                raise RuntimeError("echo-cancel source/sink unavailable after module load")
            data["aec_module"] = module_info
            data["aec_module"].update(
                {
                    "named_sink": named_sink.__dict__,
                    "named_source": named_source.__dict__,
                    "runtime_master_resolution": {
                        "speaker": aec_speaker.__dict__,
                        "microphone": aec_mic.__dict__,
                    },
                }
            )
            on_path = artifact_dir(config) / f"{run_id}_on.wav"
            data["on"] = play_and_record(
                reference,
                on_path,
                playback_target=session.sink_node_id,
                capture_target=session.source_node_id,
            )
        finally:
            session.unload()
        on_stats = data["on"].get("recording_stats") or audio_file_stats(data["on"]["path"])
        data["on"]["recording_stats"] = on_stats
        data["comparison"] = compare_aec_recordings(reference, off_path, data["on"]["path"])
        if not data["comparison"].get("measurement_valid"):
            data["status"] = "blocked"
            data["reason"] = "AEC-off or AEC-on capture was below signal floor; attenuation and ASR rerecognition are unmeasured"
            data["asr_self_rerecognition"] = {"status": "unavailable", "reason": "off/on recording RMS was at or below the signal floor"}
            return write_benchmark(config, "aec", data, started_at=started)
        try:
            asr_device = "cuda" if _cuda_available() else "cpu"
            asr_compute_type = str(nested(config, "asr", "gpu_default_compute_type", default="float16")) if asr_device == "cuda" else "int8"
            asr = WhisperASR(
                model=nested(config, "asr", "model", default="large-v3-turbo"),
                device=asr_device,
                compute_type=asr_compute_type,
                language="ja",
                beam_size=int(nested(config, "asr", "beam_size", default=5)),
            )
            rerecognition: dict[str, Any] = {}
            for label, recording_path in (("off", off_path), ("on", data["on"]["path"])):
                transcription = asr.transcribe(recording_path)
                score = _cer(TTS_BENCH_TEXT, transcription.text)
                rerecognition[label] = {
                    "transcript": transcription.text,
                    "cer_vs_spoken_reference_text": score,
                    "recognition_score_1_minus_cer_clamped": max(0.0, 1.0 - score) if score is not None else None,
                    "asr": transcription.to_dict(),
                }
            data["asr_self_rerecognition"] = rerecognition
        except Exception as exc:
            data["asr_self_rerecognition"] = {"status": "unavailable", "error_type": type(exc).__name__, "error": str(exc)}
        data["status"] = "measured"
    except Exception as exc:
        data["status"] = "blocked" if isinstance(exc, (RuntimeError, OSError)) else "error"
        data["error_type"] = type(exc).__name__
        data["error"] = str(exc)
        if data["status"] == "blocked":
            data["reason"] = "physical PipeWire capture/playback route could not be completed"
    finally:
        if asr is not None:
            asr.unload()
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
