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
from .audio import (
    AudioVolumeGuard,
    EchoCancelSession,
    PipeWireInventory,
    RawCaptureSession,
    audio_file_stats,
    compare_aec_recordings,
    playback_on_active_capture,
    play_and_record,
    stable_target,
)
from .audio_metrics import clipping_ratio, detect_acoustic_onset, noise_floor_rms, resample_mono
from .config import load_config, nested
from .llm.events import Cancelled, Completion, LLMError, TextDelta, ToolCall
from .llm.ollama import OllamaLLM
from .llm.openrouter import OpenRouterLLM
from .sentence_chunker import SentenceChunker
from .telemetry import EventLog, environment_snapshot, nvidia_smi, write_json
from .tools import MockToolRegistry
from .tts import Qwen3TTSEngine
from .vad import assistant_only_vad_metrics, detect_speech_intervals, trim_to_speech


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
    "chars_5": "元気です。",
    "chars_8": "はい、元気です。",
    "chars_12": "今日は日本語で話します。",
    "chars_20": "今日は日本語の音声応答を確認しています。",
    "chars_30": "日本語のLive音声対話で最初の応答時間を測定しています。",
    "chars_50": "これは日本語のLive音声対話で、文単位chunkの最初の音声がどれだけ早く準備できるかを確認するための長めの測定文です。",
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

AEC_BASELINE_ATTENUATION_DB = 1.6591379179219612
AEC_VOLUME_AXIS = [25, 50, 75, 100]
LIVE_CHUNKING_CANDIDATES = [
    {"max_chars": 48, "timeout_s": 0.8, "label": "baseline_48_0.8"},
    {"max_chars": 32, "timeout_s": 0.5, "label": "bounded_32_0.5"},
    {"max_chars": 24, "timeout_s": 0.5, "label": "bounded_24_0.5"},
    {"max_chars": 16, "timeout_s": 0.3, "label": "aggressive_16_0.3"},
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
    if not skip_cpu:
        data["cpu_resident_profile"] = run_cpu_asr_profile(config, path)
    return write_benchmark(config, "asr", data, started_at=started)


def _cpu_profile_attempt(
    asr: WhisperASR,
    model: Any,
    audio_path: Path,
    *,
    phase: str,
    run_number: int,
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    decode_start_ns = time.monotonic_ns()
    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
    samples = resample_mono(np.asarray(audio), int(sample_rate), 16000)
    decode_end_ns = time.monotonic_ns()
    vad_start_ns = time.monotonic_ns()
    intervals = detect_speech_intervals(samples, 16000)
    speech_samples = trim_to_speech(samples, 16000, intervals)
    vad_end_ns = time.monotonic_ns()
    inference_start_ns = time.monotonic_ns()
    result = asr.transcribe_samples(speech_samples, sample_rate=16000, model=model)
    inference_end_ns = time.monotonic_ns()
    return {
        "status": "measured",
        "phase": phase,
        "run_number": run_number,
        "audio_path": str(audio_path),
        "source_sample_rate": int(sample_rate),
        "audio_duration_s": len(samples) / 16000.0,
        "speech_audio_duration_s": len(speech_samples) / 16000.0,
        "vad_intervals": intervals,
        "file_decode_resample_s": (decode_end_ns - decode_start_ns) / 1e9,
        "vad_s": (vad_end_ns - vad_start_ns) / 1e9,
        "inference_s": (inference_end_ns - inference_start_ns) / 1e9,
        "faster_whisper_elapsed_s": result.elapsed_seconds,
        "total_s": (inference_end_ns - started_ns) / 1e9,
        "rtf_on_speech_audio": result.rtf,
        "text": result.text,
        "asr": result.to_dict(),
        "timing_ns": {
            "attempt_start": started_ns,
            "decode_start": decode_start_ns,
            "decode_end": decode_end_ns,
            "vad_start": vad_start_ns,
            "vad_end": vad_end_ns,
            "inference_start": inference_start_ns,
            "inference_end": inference_end_ns,
        },
    }


def classify_cpu_asr_profile_phases(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep the first resident transcription separate from warm repetitions."""
    cold = [row for row in rows if row.get("phase") == "first_transcription"]
    warm = [row for row in rows if row.get("phase") == "warm_transcription"]
    return {
        "cold_phase": "first_transcription",
        "warm_phase": "warm_transcription",
        "cold_run_count": len(cold),
        "warm_run_count": len(warm),
        "cold_warm_separated": len(cold) == 1 and bool(warm) and not any(row.get("phase") == "first_transcription" for row in warm),
    }


def run_cpu_asr_profile(config: dict[str, Any], audio_path: Path) -> dict[str, Any]:
    """Split CPU ASR cold load, decode/resample, VAD, and warm inference."""
    warm_repeats = max(5, int(nested(config, "bench", "cpu_asr_warm_repeats", default=5)))
    asr = WhisperASR(
        model=nested(config, "asr", "model", default="large-v3-turbo"),
        device="cpu",
        compute_type=str(nested(config, "asr", "cpu_compute_type", default="int8")),
        language=nested(config, "asr", "language", default="ja"),
        beam_size=int(nested(config, "asr", "beam_size", default=5)),
    )
    load_start_ns = time.monotonic_ns()
    try:
        model = asr.load()
        load_end_ns = time.monotonic_ns()
        first = _cpu_profile_attempt(asr, model, audio_path, phase="first_transcription", run_number=0)
        warm_runs = [
            _cpu_profile_attempt(asr, model, audio_path, phase="warm_transcription", run_number=index)
            for index in range(1, warm_repeats + 1)
        ]
        all_runs = [first, *warm_runs]
        phase_classification = classify_cpu_asr_profile_phases(all_runs)
        e2e_reference: dict[str, Any] | None = None
        e2e_path = benchmark_path(config, "e2e")
        try:
            e2e_payload = json.loads(e2e_path.read_text(encoding="utf-8"))
            e2e_reference = (
                e2e_payload.get("data", {})
                .get("configurations", {})
                .get("B_cpu_asr_local_gpu_tts", {})
                .get("median", {})
            )
        except (OSError, ValueError, TypeError):
            e2e_reference = None
        return {
            "status": "measured",
            "device": "cpu",
            "compute_type": asr.compute_type,
            "model": asr.model_name,
            "resident_model_for_all_runs": True,
            "cpu_thread_setting": "faster-whisper/CTranslate2 default; no sweep",
            "model_load_s": (load_end_ns - load_start_ns) / 1e9,
            "first_transcription": first,
            "warm_transcriptions": warm_runs,
            "warm_repeat_target": warm_repeats,
            "phase_classification": phase_classification,
            "warm_summary": _summarize_runs(
                warm_runs,
                ["file_decode_resample_s", "vad_s", "inference_s", "faster_whisper_elapsed_s", "total_s", "rtf_on_speech_audio"],
            ),
            "all_run_summary": _summarize_runs(
                all_runs,
                ["file_decode_resample_s", "vad_s", "inference_s", "faster_whisper_elapsed_s", "total_s", "rtf_on_speech_audio"],
            ),
            "e2e_cpu_stage_reference": {
                "source": str(e2e_path),
                "case": "B_cpu_asr_local_gpu_tts",
                "median": e2e_reference,
                "explanation": "E2E CPU ASR is dominated by transcription over the VAD-trimmed utterance; this profile separates that inference from file decode/resample and VAD.",
            },
        }
    except Exception as exc:
        return {
            "status": "error",
            "device": "cpu",
            "compute_type": asr.compute_type,
            "model": asr.model_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_load_s": (time.monotonic_ns() - load_start_ns) / 1e9,
        }
    finally:
        asr.unload()


def _collect_first_sentence_chunk(
    provider: Any,
    messages: list[dict[str, Any]],
    *,
    max_chars: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Collect provider deltas until the first bounded sentence chunk exists."""
    started_ns = time.monotonic_ns()
    chunker = SentenceChunker(max_chars=max_chars, timeout_s=timeout_s)
    text_parts: list[str] = []
    events: list[str] = []
    first_token_ns: int | None = None
    first_chunk_ns: int | None = None
    first_chunk: str | None = None
    error: LLMError | None = None
    cancelled = False
    iterator = iter(provider.stream(messages))
    try:
        for event in iterator:
            events.append(event.kind)
            if isinstance(event, TextDelta):
                if first_token_ns is None:
                    first_token_ns = time.monotonic_ns()
                text_parts.append(event.text)
                chunks = chunker.push(event.text)
                if chunks and first_chunk is None:
                    first_chunk = chunks[0]
                    first_chunk_ns = time.monotonic_ns()
                    break
            elif isinstance(event, Cancelled):
                cancelled = True
                break
            elif isinstance(event, LLMError):
                error = event
                break
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    if first_chunk is None and text_parts and not error and not cancelled:
        flushed = chunker.flush()
        if flushed:
            first_chunk = flushed[0]
            first_chunk_ns = time.monotonic_ns()
    ended_ns = time.monotonic_ns()
    return {
        "text": "".join(text_parts),
        "first_chunk": first_chunk,
        "events": events,
        "error": error,
        "cancelled": cancelled,
        "started_ns": started_ns,
        "first_token_ns": first_token_ns,
        "first_chunk_ns": first_chunk_ns,
        "ended_ns": ended_ns,
        "ttft_s": (first_token_ns - started_ns) / 1e9 if first_token_ns else None,
        "until_first_chunk_s": (first_chunk_ns - started_ns) / 1e9 if first_chunk_ns else None,
        "sentence_buffering_s": (first_chunk_ns - first_token_ns) / 1e9 if first_chunk_ns and first_token_ns else None,
        "error_detail": {
            "message": error.message,
            "status_code": error.status_code,
            "details": error.details,
        }
        if error
        else None,
    }


def _mark_timing_event(log: EventLog, name: str, timestamp_ns: int | None, **payload: Any) -> None:
    if timestamp_ns is not None:
        log.mark_at(name, timestamp_ns, **payload)


def _physical_playback(
    reference_path: Path,
    recording_path: Path,
    *,
    playback_target: str,
    capture_target: str,
    lead_s: float = 0.4,
    tail_s: float = 0.5,
    active_capture: RawCaptureSession | None = None,
) -> dict[str, Any]:
    if active_capture is None:
        playback = play_and_record(
            reference_path,
            recording_path,
            playback_target=playback_target,
            capture_target=capture_target,
            lead_s=lead_s,
            tail_s=tail_s,
            sample_rate=16000,
        )
    else:
        if active_capture.output_path != recording_path:
            raise ValueError("active capture output must match recording_path")
        playback = playback_on_active_capture(reference_path, playback_target=playback_target, capture=active_capture)
        recording = active_capture.stop(tail_s=tail_s)
        playback["recording_target"] = capture_target
        playback["record_returncode"] = recording.get("record_returncode")
        playback["record_stdout"] = recording.get("record_stdout")
        playback["record_stderr"] = recording.get("record_stderr")
        playback["recording_stats"] = recording.get("recording_stats")
        playback["timing_ns"].update(recording.get("timing_ns", {}))
    recording, recording_rate = sf.read(str(recording_path), always_2d=False)
    reference, reference_rate = sf.read(str(reference_path), always_2d=False)
    record_start_ns = playback["timing_ns"].get("record_start")
    playback_start_ns = playback["timing_ns"].get("pw_play_start")
    playback_offset_s = (
        (playback_start_ns - record_start_ns) / 1e9
        if record_start_ns is not None and playback_start_ns is not None
        else lead_s
    )
    noise_window_s = max(0.1, min(1.0, playback_offset_s - 0.05))
    onset = detect_acoustic_onset(
        np.asarray(recording),
        int(recording_rate),
        search_start_s=max(0.1, playback_offset_s - 0.05),
        noise_window_s=noise_window_s,
        reference=np.asarray(reference),
        reference_rate=int(reference_rate),
    )
    playback["acoustic_onset"] = onset
    playback["physical_audio_detected"] = bool(onset.get("detected"))
    playback["timing_ns"]["physical_audio_detected"] = (
        playback["timing_ns"].get("record_start") + int(round(float(onset["onset_s"]) * 1e9))
        if onset.get("detected") and playback["timing_ns"].get("record_start") is not None
        else None
    )
    return playback


def _live_messages(user_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": E2E_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def _live_latency_attempt(
    config: dict[str, Any],
    *,
    run_id: str,
    run_number: int,
    user_audio: Path,
    provider: Any,
    asr: WhisperASR,
    tts: Qwen3TTSEngine,
    playback_target: str,
    capture_target: str,
    max_chars: int,
    timeout_s: float,
) -> dict[str, Any]:
    log = EventLog()
    started_ns = time.monotonic_ns()
    recording_path = artifact_dir(config) / f"{run_id}_run{run_number}_raw_mic.wav"
    capture = RawCaptureSession(recording_path, target=capture_target, sample_rate=16000)
    capture_started = False
    synthetic_user_end_ns: int | None = None
    user_asr: Any = None
    turn: dict[str, Any] | None = None
    tts_row: dict[str, Any] | None = None
    physical: dict[str, Any] | None = None
    try:
        capture.start()
        capture_started = True
        synthetic_user_end_ns = time.monotonic_ns()
        log.mark_at("synthetic_user_end", synthetic_user_end_ns, source="direct_input_wav_handoff")
        user_asr = asr.transcribe(user_audio, event_log=log)
        llm_start_ns = time.monotonic_ns()
        log.mark_at("llm_start", llm_start_ns)
        turn = _collect_first_sentence_chunk(
            provider,
            _live_messages(user_asr.text),
            max_chars=max_chars,
            timeout_s=timeout_s,
        )
        _mark_timing_event(log, "llm_first_token", turn.get("first_token_ns"), ttft_s=turn.get("ttft_s"))
        _mark_timing_event(log, "first_sentence_chunk_ready", turn.get("first_chunk_ns"), text_chars=len(turn.get("first_chunk") or ""))
        log.mark_at("llm_end", turn["ended_ns"], first_chunk=True, actual_model=getattr(provider, "last_actual_model", None))
        if turn.get("error"):
            raise RuntimeError(turn["error"].message)
        if turn.get("cancelled") or not turn.get("first_chunk"):
            raise RuntimeError("LLM returned no first sentence chunk")
        chunk = str(turn["first_chunk"])
        tts_start_ns = time.monotonic_ns()
        log.mark_at("tts_start", tts_start_ns, text_chars=len(chunk))
        tts_row = tts.synthesize(
            chunk,
            output_path=artifact_dir(config) / f"{run_id}_run{run_number}_assistant_first.wav",
            event_log=log,
        )
        tts_timing = tts_row.get("timing_ns", {})
        _mark_timing_event(log, "tts_audio_ready", tts_timing.get("playback_possible"), path=tts_row.get("path"))
        physical = _physical_playback(
            Path(tts_row["path"]),
            recording_path,
            playback_target=playback_target,
            capture_target=capture_target,
            lead_s=0.0,
            active_capture=capture,
        )
        playback_timing = physical.get("timing_ns", {})
        _mark_timing_event(log, "pw_play_start", playback_timing.get("pw_play_start"), target=playback_target)
        _mark_timing_event(
            log,
            "physical_audio_detected",
            playback_timing.get("physical_audio_detected"),
            onset_s=physical.get("acoustic_onset", {}).get("onset_s"),
        )
        _mark_timing_event(log, "playback_end", playback_timing.get("playback_end"), path=tts_row.get("path"))
        physical_ns = playback_timing.get("physical_audio_detected")
        pw_play_ns = playback_timing.get("pw_play_start")
        playback_end_ns = playback_timing.get("playback_end")
        first_token_ns = turn.get("first_token_ns")
        first_chunk_ns = turn.get("first_chunk_ns")
        tts_ready_ns = tts_timing.get("playback_possible")
        stages = {
            "asr_duration_s": user_asr.elapsed_seconds,
            "llm_ttft_s": turn.get("ttft_s"),
            "llm_until_first_chunk_s": turn.get("until_first_chunk_s"),
            "sentence_buffering_s": turn.get("sentence_buffering_s"),
            "tts_duration_s": tts_row.get("elapsed_seconds"),
            "wav_ready_to_pw_play_s": (pw_play_ns - tts_ready_ns) / 1e9 if pw_play_ns and tts_ready_ns else None,
            "pw_play_to_acoustic_onset_s": (physical_ns - pw_play_ns) / 1e9 if physical_ns and pw_play_ns else None,
            "playback_duration_s": (playback_end_ns - pw_play_ns) / 1e9 if playback_end_ns and pw_play_ns else None,
            "speech_end_to_first_physical_audio_s": (physical_ns - synthetic_user_end_ns) / 1e9 if physical_ns else None,
        }
        dominant_candidates = [
            (name, value)
            for name, value in stages.items()
            if name not in {"llm_ttft_s", "llm_until_first_chunk_s", "speech_end_to_first_physical_audio_s", "playback_duration_s"}
            and isinstance(value, (int, float))
        ]
        return {
            "status": "measured" if physical.get("physical_audio_detected") else "blocked",
            "run_number": run_number,
            "warmup": False,
            "user_audio": str(user_audio),
            "user_asr": user_asr.to_dict(),
            "llm": {
                "text_until_first_chunk": turn.get("text"),
                "first_chunk": chunk,
                "ttft_s": turn.get("ttft_s"),
                "until_first_chunk_s": turn.get("until_first_chunk_s"),
                "actual_model": getattr(provider, "last_actual_model", None),
                "events": turn.get("events", []),
                "error": turn.get("error_detail"),
            },
            "tts": tts_row,
            "playback": physical,
            "stages": stages,
            "speech_end_to_first_physical_audio_s": stages["speech_end_to_first_physical_audio_s"],
            "dominant_stage": max(dominant_candidates, key=lambda item: item[1])[0] if dominant_candidates else None,
            "timing_ns": {
                "synthetic_user_end": synthetic_user_end_ns,
                "asr_start": next((item["monotonic_ns"] for item in log.events if item["event"] == "asr_start"), None),
                "asr_final": next((item["monotonic_ns"] for item in reversed(log.events) if item["event"] == "asr_final"), None),
                "llm_start": llm_start_ns,
                "llm_first_token": first_token_ns,
                "first_sentence_chunk_ready": first_chunk_ns,
                "tts_start": tts_start_ns,
                "tts_audio_ready": tts_ready_ns,
                "pw_play_start": pw_play_ns,
                "physical_audio_detected": physical_ns,
                "playback_end": playback_end_ns,
            },
            "events": log.events,
        }
    except Exception as exc:
        return {
            "status": "error",
            "run_number": run_number,
            "warmup": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "user_asr": user_asr.to_dict() if user_asr is not None else None,
            "llm": {"error": turn.get("error_detail") if turn else None},
            "tts": tts_row,
            "playback": physical,
            "speech_end_to_first_physical_audio_s": None,
            "timing_ns": {"synthetic_user_end": synthetic_user_end_ns, "attempt_start": started_ns},
            "events": log.events,
        }
    finally:
        if capture_started:
            try:
                capture.stop(tail_s=0.0)
            except Exception:
                pass


def _chunking_candidate_measurement(
    config: dict[str, Any],
    *,
    run_id: str,
    candidate: dict[str, Any],
    representative_text: str,
    tts: Qwen3TTSEngine,
    playback_target: str,
    capture_target: str,
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    chunker = SentenceChunker(max_chars=int(candidate["max_chars"]), timeout_s=float(candidate["timeout_s"]))
    chunks = chunker.push(representative_text)
    chunks.extend(chunker.flush())
    first_chunk_ready_ns = time.monotonic_ns()
    if not chunks:
        return {**candidate, "status": "blocked", "reason": "representative output produced no chunk"}
    first_chunk = chunks[0]
    log = EventLog()
    tts_start_ns = time.monotonic_ns()
    log.mark_at("first_sentence_chunk_ready", first_chunk_ready_ns, text_chars=len(first_chunk))
    log.mark_at("tts_start", tts_start_ns, text_chars=len(first_chunk))
    try:
        tts_row = tts.synthesize(
            first_chunk,
            output_path=artifact_dir(config) / f"{run_id}_{candidate['label']}_assistant.wav",
            event_log=log,
        )
        physical = _physical_playback(
            Path(tts_row["path"]),
            artifact_dir(config) / f"{run_id}_{candidate['label']}_raw.wav",
            playback_target=playback_target,
            capture_target=capture_target,
        )
        tts_ready_ns = tts_row.get("timing_ns", {}).get("playback_possible")
        pw_play_ns = physical.get("timing_ns", {}).get("pw_play_start")
        physical_ns = physical.get("timing_ns", {}).get("physical_audio_detected")
        log.mark_at("tts_audio_ready", tts_ready_ns or time.monotonic_ns(), path=tts_row.get("path"))
        _mark_timing_event(log, "pw_play_start", pw_play_ns, target=playback_target)
        _mark_timing_event(log, "physical_audio_detected", physical_ns, onset_s=physical.get("acoustic_onset", {}).get("onset_s"))
        return {
            **candidate,
            "status": "measured" if physical.get("physical_audio_detected") else "blocked",
            "representative_text": representative_text,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "first_chunk": first_chunk,
            "first_chunk_chars": len(first_chunk),
            "fragmentation_proxy": {
                "short_chunk_count_lt_8": sum(len(chunk) < 8 for chunk in chunks),
                "classification": "possible_overfragmentation" if len(chunks) > 2 else "no_obvious_overfragmentation",
            },
            "tts": tts_row,
            "playback": physical,
            "first_chunk_ready_s": (first_chunk_ready_ns - started_ns) / 1e9,
            "tts_ready_s": (tts_ready_ns - started_ns) / 1e9 if tts_ready_ns else None,
            "physical_onset_s": (physical_ns - started_ns) / 1e9 if physical_ns else None,
            "wav_ready_to_pw_play_s": (pw_play_ns - tts_ready_ns) / 1e9 if pw_play_ns and tts_ready_ns else None,
            "pw_play_to_acoustic_onset_s": (physical_ns - pw_play_ns) / 1e9 if physical_ns and pw_play_ns else None,
            "events": log.events,
        }
    except Exception as exc:
        return {**candidate, "status": "error", "error_type": type(exc).__name__, "error": str(exc)}


def run_live_latency_bench(config: dict[str, Any]) -> dict[str, Any]:
    """Measure direct synthetic speech-end to first physical assistant audio."""
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_id = _measurement_id("live_latency")
    repeats = max(5, int(nested(config, "bench", "live_latency_repeats", default=5)))
    max_attempts = max(repeats, int(nested(config, "bench", "live_latency_max_attempts", default=8)))
    data: dict[str, Any] = {
        "status": "blocked",
        "run_id": run_id,
        "metric_name": "synthetic_user_end_to_first_physical_assistant_audio",
        "metric_definition": "Direct synthetic input WAV handoff to first sustained above-noise-floor sample in a raw USB microphone recording of the assistant USB speaker output; not a human speech latency measurement.",
        "event_definition": "synthetic_user_end is the monotonic handoff immediately before direct WAV ASR; physical_audio_detected is derived from raw microphone acoustic onset, not a sleep.",
        "user_audio": None,
        "raw_microphone_side_channel": True,
        "raw_microphone_not_aec_source": True,
        "speaker_target_is_stable_name": True,
        "microphone_target_is_stable_name": True,
        "volume_snapshot": None,
        "volume_restore_error": None,
        "warmup": None,
        "runs": [],
        "summary": None,
        "chunking_comparison": [],
    }
    asr: WhisperASR | None = None
    tts: Qwen3TTSEngine | None = None
    volume_guard: AudioVolumeGuard | None = None
    try:
        user_audio, generated = ensure_synthetic_audio(config)
        data["user_audio"] = generated
        inventory = PipeWireInventory.discover()
        mic = inventory.usb_microphone()
        speaker = inventory.usb_speaker()
        playback_target = stable_target(speaker)
        capture_target = stable_target(mic)
        data["inventory"] = inventory.to_dict()
        data["targets"] = {"speaker": speaker.__dict__, "microphone": mic.__dict__}
        asr_device = "cuda" if _cuda_available() else "cpu"
        asr = WhisperASR(
            model=nested(config, "asr", "model", default="large-v3-turbo"),
            device=asr_device,
            compute_type=str(nested(config, "asr", "gpu_default_compute_type", default="int8_float16")) if asr_device == "cuda" else "int8",
            language=nested(config, "asr", "language", default="ja"),
            beam_size=int(nested(config, "asr", "beam_size", default=5)),
        )
        tts = Qwen3TTSEngine(
            model=nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            speaker=nested(config, "tts", "speaker", default="Ono_Anna"),
            language=nested(config, "tts", "language", default="Japanese"),
            device="cuda:0" if _cuda_available() else "auto",
            max_new_tokens=int(nested(config, "tts", "max_new_tokens", default=2048)),
        )
        provider = _provider_pair(config)["local"]
        with AudioVolumeGuard(speaker_target=playback_target, microphone_target=capture_target) as volume:
            volume_guard = volume
            data["volume_snapshot"] = volume.snapshot.to_dict() if volume.snapshot else None
            volume.set_mutes(speaker_muted=False, microphone_muted=False)
            warmup: dict[str, Any] = {"status": "measured"}
            try:
                warmup_asr = asr.transcribe(user_audio)
                warmup_turn = _collect_turn(provider, _live_messages(warmup_asr.text))
                representative_text = warmup_turn["text"] or "はい、元気です。"
                warmup_chunker = SentenceChunker(
                    max_chars=int(nested(config, "tts", "sentence_max_chars", default=48)),
                    timeout_s=float(nested(config, "tts", "sentence_timeout_s", default=0.8)),
                )
                warmup_chunks = warmup_chunker.push(representative_text)
                warmup_chunks.extend(warmup_chunker.flush())
                warmup_tts = tts.synthesize(
                    warmup_chunks[0] if warmup_chunks else "はい、元気です。",
                    output_path=artifact_dir(config) / f"{run_id}_warmup.wav",
                )
                warmup.update(
                    {
                        "user_asr": warmup_asr.to_dict(),
                        "llm": _public_turn(warmup_turn),
                        "representative_text": representative_text,
                        "chunks": warmup_chunks,
                        "tts": warmup_tts,
                    }
                )
            except Exception as exc:
                warmup = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
                representative_text = "はい、元気です。"
            data["warmup"] = warmup
            data["representative_output"] = representative_text
            max_chars = int(nested(config, "tts", "sentence_max_chars", default=48))
            timeout_s = float(nested(config, "tts", "sentence_timeout_s", default=0.8))
            while len([row for row in data["runs"] if row.get("status") == "measured"]) < repeats and len(data["runs"]) < max_attempts:
                run_number = len(data["runs"]) + 1
                data["runs"].append(
                    _live_latency_attempt(
                        config,
                        run_id=run_id,
                        run_number=run_number,
                        user_audio=user_audio,
                        provider=provider,
                        asr=asr,
                        tts=tts,
                        playback_target=playback_target,
                        capture_target=capture_target,
                        max_chars=max_chars,
                        timeout_s=timeout_s,
                    )
                )
            measured = [row for row in data["runs"] if row.get("status") == "measured"]
            latency_values = [row["speech_end_to_first_physical_audio_s"] for row in measured if isinstance(row.get("speech_end_to_first_physical_audio_s"), (int, float))]
            stage_fields = [
                "asr_duration_s",
                "llm_ttft_s",
                "sentence_buffering_s",
                "tts_duration_s",
                "wav_ready_to_pw_play_s",
                "pw_play_to_acoustic_onset_s",
            ]
            stage_medians = _summarize_runs([row["stages"] for row in measured], stage_fields)
            dominant_stage = max(
                ((field, value.get("median")) for field, value in stage_medians.items() if isinstance(value.get("median"), (int, float))),
                key=lambda item: item[1],
                default=(None, None),
            )[0]
            data["summary"] = {
                "attempt_count": len(data["runs"]),
                "measured_run_count": len(measured),
                "repeat_target": repeats,
                "max_attempts": max_attempts,
                "speech_end_to_first_physical_audio_s": _distribution(latency_values),
                "stage_distributions": {
                    field: _distribution([row["stages"][field] for row in measured if isinstance(row.get("stages", {}).get(field), (int, float))])
                    for field in stage_fields
                },
                "stage_medians": stage_medians,
                "dominant_stage": dominant_stage,
                "outlier_policy": "retain every attempt; no latency value is deleted",
            }
            data["chunking_comparison"] = [
                _chunking_candidate_measurement(
                    config,
                    run_id=run_id,
                    candidate=candidate,
                    representative_text=representative_text,
                    tts=tts,
                    playback_target=playback_target,
                    capture_target=capture_target,
                )
                for candidate in LIVE_CHUNKING_CANDIDATES
            ]
            data["status"] = "measured" if len(measured) >= repeats else "partial"
            data["measurement_parameters"] = {
                "baseline_max_chars": max_chars,
                "baseline_timeout_s": timeout_s,
                "lead_s": 0.4,
                "tail_s": 0.5,
                "onset_gate": "10 ms RMS frames, leading noise-floor median/MAD, threshold max(4x floor, floor+6x MAD, 0.004), 2 consecutive frames",
            }
        data["volume_restore_error"] = volume_guard.restore_error if volume_guard is not None else None
    except Exception as exc:
        data["status"] = "blocked"
        data["error_type"] = type(exc).__name__
        data["error"] = str(exc)
    finally:
        if volume_guard is not None:
            data["volume_restore_error"] = volume_guard.restore_error
        if asr is not None:
            asr.unload()
        if tts is not None:
            tts.unload()
    return write_benchmark(config, "live_latency", data, started_at=started)


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
    warm_repeats = max(5, int(nested(config, "bench", "tts_warm_repeats", default=5)))
    run_id = _measurement_id(f"tts_{label}")
    cold_text = TTS_LATENCY_CHUNKS["chars_8"]
    cold = _tts_attempt(
        engine,
        text=cold_text,
        output_path=artifact_dir(config) / f"{run_id}_cold_chars_8.wav",
        chunk="chars_8",
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
        "cold_definition": "first chars_8 synthesize on a newly constructed engine; model load is included",
        "warm_definition": "subsequent synthesize calls on the same resident engine for chars_5/chars_8/chars_12/chars_20/chars_30/chars_50; model_load_seconds should be zero",
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
            "latency_matrix_definition": "GPU cold chars_8 first call followed by resident-model warm calls for six Japanese chunk lengths (5, 8, 12, 20, about 30, about 50 characters)",
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


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: list[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    return {
        "values": numeric,
        "count": len(numeric),
        "median": statistics.median(numeric) if numeric else None,
        "p95_equivalent": _percentile(numeric, 0.95),
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
    }


def summarize_aec_conditions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate matrix rows without hiding skipped conditions or outliers."""
    measured = [row for row in rows if row.get("status") == "measured"]
    attenuations = [
        float(row["residual_echo_attenuation_db"])
        for row in measured
        if isinstance(row.get("residual_echo_attenuation_db"), (int, float))
    ]
    return {
        "condition_count": len(rows),
        "measured_count": len(measured),
        "skipped_safety_count": sum(row.get("status") == "skipped_safety" for row in rows),
        "blocked_count": sum(row.get("status") in {"blocked", "error"} for row in rows),
        "attenuation_median_db": statistics.median(attenuations) if attenuations else None,
        "attenuation_distribution_db": _distribution(attenuations),
        "rows": rows,
    }


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
        raw = play_and_record(reference, raw_path, playback_target=stable_target(speaker), capture_target=stable_target(mic))
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
        data["off"] = play_and_record(reference, off_path, playback_target=stable_target(off_speaker), capture_target=stable_target(off_mic))
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
            sink_master=stable_target(aec_speaker),
            source_master=stable_target(aec_mic),
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
                playback_target=session.sink_target,
                capture_target=session.source_target,
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


def _aec_clip_detected(row: dict[str, Any], *, ratio_threshold: float, peak_threshold: float) -> bool:
    for label in ("raw", "off", "on"):
        stats = (row.get(label) or {}).get("recording_stats") or {}
        ratio = stats.get("clipping_ratio")
        peak = stats.get("peak")
        if isinstance(ratio, (int, float)) and ratio > ratio_threshold:
            return True
        if isinstance(peak, (int, float)) and peak >= peak_threshold:
            return True
    return False


def _aec_row_correlation(comparison: dict[str, Any] | None, label: str) -> float | None:
    value = (comparison or {}).get(f"reference_recording_correlation_{label}") or {}
    correlation = value.get("correlation")
    return float(correlation) if isinstance(correlation, (int, float)) else None


def _aec_condition_measurement(
    config: dict[str, Any],
    *,
    reference: Path,
    run_id: str,
    condition_id: str,
    speaker_percent: int,
    microphone_percent: int,
    volume: AudioVolumeGuard,
    asr: WhisperASR | None,
    signal_floor: float,
) -> dict[str, Any]:
    """Measure one echo-only condition in raw, OFF, ON order."""
    row: dict[str, Any] = {
        "status": "blocked",
        "condition_id": condition_id,
        "speaker_volume_percent": speaker_percent,
        "microphone_volume_percent": microphone_percent,
        "reference_audio": str(reference),
        "sequence": ["raw", "aec_off", "aec_on"],
        "ambient_noise_floor_rms": None,
        "raw_rms": None,
        "raw_peak": None,
        "clipping_ratio_raw": None,
        "aec_off_rms": None,
        "aec_off_peak": None,
        "clipping_ratio_off": None,
        "aec_on_rms": None,
        "aec_on_peak": None,
        "clipping_ratio_on": None,
        "residual_echo_attenuation_db": None,
        "reference_correlation_off": None,
        "reference_correlation_on": None,
        "correlation_lag": {"off": None, "on": None},
        "whisper_transcript_off": None,
        "whisper_transcript_on": None,
        "whisper_cer_off": None,
        "whisper_cer_on": None,
        "assistant_only_vad_off": None,
        "assistant_only_vad_on": None,
        "aec_module": None,
        "targets": None,
    }
    volume.set_volumes(speaker_percent=speaker_percent, microphone_percent=microphone_percent)
    volume.set_mutes(speaker_muted=False, microphone_muted=False)
    inventory = PipeWireInventory.discover()
    mic = inventory.usb_microphone()
    speaker = inventory.usb_speaker()
    playback_target = stable_target(speaker)
    capture_target = stable_target(mic)
    row["targets"] = {"speaker": speaker.__dict__, "microphone": mic.__dict__}
    raw_path = artifact_dir(config) / f"{run_id}_{condition_id}_raw.wav"
    off_path = artifact_dir(config) / f"{run_id}_{condition_id}_off.wav"
    on_path = artifact_dir(config) / f"{run_id}_{condition_id}_on.wav"
    raw = play_and_record(reference, raw_path, playback_target=playback_target, capture_target=capture_target)
    row["raw"] = raw
    raw_stats = raw["recording_stats"]
    row["ambient_noise_floor_rms"] = noise_floor_rms(
        np.asarray(sf.read(str(raw_path), always_2d=False)[0]),
        int(raw_stats["sample_rate"]),
        window_s=0.25,
    )
    row["raw_rms"] = raw_stats["rms"]
    row["raw_peak"] = raw_stats["peak"]
    row["clipping_ratio_raw"] = raw_stats.get("clipping_ratio")
    off = play_and_record(reference, off_path, playback_target=playback_target, capture_target=capture_target)
    row["off"] = off
    off_stats = off["recording_stats"]
    row["aec_off_rms"] = off_stats["rms"]
    row["aec_off_peak"] = off_stats["peak"]
    row["clipping_ratio_off"] = off_stats.get("clipping_ratio")
    off_inventory = PipeWireInventory.discover()
    off_mic = off_inventory.usb_microphone()
    off_speaker = off_inventory.usb_speaker()
    off_target = stable_target(off_speaker)
    off_capture = stable_target(off_mic)
    session = EchoCancelSession(
        sink_name=nested(config, "pipewire", "echo_cancel_sink", default="Local Live Echo Cancellation Sink"),
        source_name=nested(config, "pipewire", "echo_cancel_source", default="Local Live Echo Cancellation Source"),
        capture_name=nested(config, "pipewire", "echo_cancel_capture", default="Local Live Echo Cancellation Capture"),
        playback_name=nested(config, "pipewire", "echo_cancel_playback", default="Local Live Echo Cancellation Playback"),
        latency=nested(config, "pipewire", "node_latency", default="1024/48000"),
        sink_master=off_target,
        source_master=off_capture,
    )
    try:
        module_info = session.load()
        row["aec_module"] = module_info
        on = play_and_record(
            reference,
            on_path,
            playback_target=session.sink_target,
            capture_target=session.source_target,
        )
        row["on"] = on
    finally:
        session.unload()
    on_stats = row["on"]["recording_stats"]
    row["aec_on_rms"] = on_stats["rms"]
    row["aec_on_peak"] = on_stats["peak"]
    row["clipping_ratio_on"] = on_stats.get("clipping_ratio")
    comparison = compare_aec_recordings(reference, off_path, on_path)
    row["comparison"] = comparison
    row["residual_echo_attenuation_db"] = comparison.get("residual_echo_attenuation_db")
    row["reference_correlation_off"] = _aec_row_correlation(comparison, "off")
    row["reference_correlation_on"] = _aec_row_correlation(comparison, "on")
    row["correlation_lag"] = {
        "off": (comparison.get("reference_recording_correlation_off") or {}).get("lag_seconds"),
        "on": (comparison.get("reference_recording_correlation_on") or {}).get("lag_seconds"),
    }
    reference_info = sf.info(str(reference))
    for label, recording_path in (("off", off_path), ("on", on_path)):
        recording, sample_rate = sf.read(str(recording_path), always_2d=False)
        row[f"assistant_only_vad_{label}"] = assistant_only_vad_metrics(
            np.asarray(recording),
            int(sample_rate),
            playback_duration_s=float(reference_info.duration),
        )
    if asr is not None:
        for label, recording_path in (("off", off_path), ("on", on_path)):
            transcription = asr.transcribe(recording_path)
            row[f"whisper_transcript_{label}"] = transcription.text
            row[f"whisper_cer_{label}"] = _cer(TTS_BENCH_TEXT, transcription.text)
            row.setdefault("whisper_asr", {})[label] = transcription.to_dict()
    row["status"] = "measured"
    row["signal_floor_rms"] = signal_floor
    return row


def _aec_quality_key(row: dict[str, Any]) -> tuple[float, ...]:
    off_vad = (row.get("assistant_only_vad_off") or {}).get("false_trigger_ratio")
    on_vad = (row.get("assistant_only_vad_on") or {}).get("false_trigger_ratio")
    off_cer = row.get("whisper_cer_off")
    on_cer = row.get("whisper_cer_on")
    off_corr = row.get("reference_correlation_off")
    on_corr = row.get("reference_correlation_on")
    attenuation = row.get("residual_echo_attenuation_db")
    vad_improvement = float(off_vad - on_vad) if isinstance(off_vad, (int, float)) and isinstance(on_vad, (int, float)) else -1_000_000.0
    cer_improvement = float(off_cer - on_cer) if isinstance(off_cer, (int, float)) and isinstance(on_cer, (int, float)) else -1_000_000.0
    correlation_improvement = float(off_corr - on_corr) if isinstance(off_corr, (int, float)) and isinstance(on_corr, (int, float)) else -1_000_000.0
    return (
        1.0 if not row.get("clipping_detected") else 0.0,
        vad_improvement,
        cer_improvement,
        float(attenuation) if isinstance(attenuation, (int, float)) else -1_000_000.0,
        correlation_improvement,
        float(row.get("microphone_volume_percent", 0)),
        float(row.get("speaker_volume_percent", 0)),
    )


def _aec_best_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if row.get("status") == "measured"]
    attenuations = [row["residual_echo_attenuation_db"] for row in measured if isinstance(row.get("residual_echo_attenuation_db"), (int, float))]
    vad_ratios = [(row.get("assistant_only_vad_on") or {}).get("false_trigger_ratio") for row in measured]
    vad_ratios = [float(value) for value in vad_ratios if isinstance(value, (int, float))]
    on_cers = [row.get("whisper_cer_on") for row in measured]
    on_cers = [float(value) for value in on_cers if isinstance(value, (int, float))]
    return {
        "repeat_count": len(rows),
        "measured_count": len(measured),
        "attenuation_db": _distribution([float(value) for value in attenuations]),
        "vad_false_trigger_ratio": _distribution(vad_ratios),
        "vad_false_trigger_rate": sum((row.get("assistant_only_vad_on") or {}).get("false_trigger_count", 0) > 0 for row in measured) / len(measured) if measured else None,
        "whisper_self_rerecognition_rate": sum(bool((row.get("whisper_transcript_on") or "").strip()) for row in measured) / len(measured) if measured else None,
        "whisper_cer_on": _distribution(on_cers),
        "clipping_detected": any(bool(row.get("clipping_detected")) for row in rows),
        "clipping_ratios": {
            label: _distribution([float((row.get(label) or 0.0)) for row in rows if isinstance(row.get(label), (int, float))])
            for label in ("clipping_ratio_raw", "clipping_ratio_off", "clipping_ratio_on")
        },
        "rows": rows,
    }


def run_aec_matrix_bench(config: dict[str, Any], *, force_audio: bool = False) -> dict[str, Any]:
    """Run the serial speaker-volume × microphone-gain echo-only matrix."""
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_id = _measurement_id("aec_matrix")
    signal_floor = float(nested(config, "bench", "audio_signal_floor_rms", default=1e-6))
    ratio_threshold = float(nested(config, "bench", "aec_matrix_clipping_ratio_threshold", default=0.001))
    peak_threshold = float(nested(config, "bench", "aec_matrix_peak_threshold", default=0.98))
    repeats = max(3, int(nested(config, "bench", "aec_matrix_repeats", default=3)))
    data: dict[str, Any] = {
        "status": "blocked",
        "run_id": run_id,
        "metric_name": "echo-only operating envelope",
        "reference_audio": None,
        "matrix": {"speaker_volume_percent": AEC_VOLUME_AXIS, "microphone_volume_percent": AEC_VOLUME_AXIS},
        "sequence": "each condition: raw capture, AEC OFF, AEC ON; all conditions serial",
        "volume_policy": "pactl set-sink-volume/set-source-volume, 0-100% only; snapshot and finally restore speaker/microphone volume, mute, and defaults",
        "clipping_definition": {"threshold": 0.98, "ratio_gate": ratio_threshold, "peak_gate": peak_threshold, "formula": "count(abs(sample) >= 0.98) / sample_count"},
        "rows": [],
        "matrix_summary": None,
        "best_candidates": [],
        "best_condition_remeasurements": {},
        "baseline_attenuation_db": AEC_BASELINE_ATTENUATION_DB,
        "volume_snapshot": None,
        "volume_restore_error": None,
    }
    asr: WhisperASR | None = None
    volume_guard: AudioVolumeGuard | None = None
    try:
        reference, generated = ensure_tts_reference(config, force=force_audio)
        data["reference_audio"] = str(reference)
        data["reference_generation"] = generated
        inventory = PipeWireInventory.discover()
        mic = inventory.usb_microphone()
        speaker = inventory.usb_speaker()
        speaker_target = stable_target(speaker)
        microphone_target = stable_target(mic)
        data["inventory_before"] = inventory.to_dict()
        asr_device = "cuda" if _cuda_available() else "cpu"
        asr = WhisperASR(
            model=nested(config, "asr", "model", default="large-v3-turbo"),
            device=asr_device,
            compute_type=str(nested(config, "asr", "gpu_default_compute_type", default="int8_float16")) if asr_device == "cuda" else "int8",
            language=nested(config, "asr", "language", default="ja"),
            beam_size=int(nested(config, "asr", "beam_size", default=5)),
        )
        speaker_100_skip = False
        microphone_100_skip = False
        with AudioVolumeGuard(speaker_target=speaker_target, microphone_target=microphone_target) as volume:
            volume_guard = volume
            data["volume_snapshot"] = volume.snapshot.to_dict() if volume.snapshot else None
            volume.set_mutes(speaker_muted=False, microphone_muted=False)
            for speaker_percent in AEC_VOLUME_AXIS:
                for microphone_percent in AEC_VOLUME_AXIS:
                    condition_id = f"sp{speaker_percent}_mic{microphone_percent}"
                    if (speaker_percent == 100 and speaker_100_skip) or (microphone_percent == 100 and microphone_100_skip):
                        data["rows"].append(
                            {
                                "status": "skipped_safety",
                                "condition_id": condition_id,
                                "speaker_volume_percent": speaker_percent,
                                "microphone_volume_percent": microphone_percent,
                                "reason": "75% gate detected clipping on this volume/gain axis; 100% condition was not attempted",
                            }
                        )
                        continue
                    try:
                        row = _aec_condition_measurement(
                            config,
                            reference=reference,
                            run_id=run_id,
                            condition_id=condition_id,
                            speaker_percent=speaker_percent,
                            microphone_percent=microphone_percent,
                            volume=volume,
                            asr=asr,
                            signal_floor=signal_floor,
                        )
                    except Exception as exc:
                        row = {
                            "status": "blocked",
                            "condition_id": condition_id,
                            "speaker_volume_percent": speaker_percent,
                            "microphone_volume_percent": microphone_percent,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    row["clipping_detected"] = _aec_clip_detected(row, ratio_threshold=ratio_threshold, peak_threshold=peak_threshold)
                    data["rows"].append(row)
                    if speaker_percent == 75 and row["clipping_detected"]:
                        speaker_100_skip = True
                    if microphone_percent == 75 and row["clipping_detected"]:
                        microphone_100_skip = True
            data["matrix_summary"] = summarize_aec_conditions(data["rows"])
            measured_matrix = [row for row in data["rows"] if row.get("status") == "measured" and not row.get("clipping_detected")]
            candidates = sorted(measured_matrix, key=_aec_quality_key, reverse=True)[:3]
            data["best_candidates"] = [
                {
                    "condition_id": row["condition_id"],
                    "speaker_volume_percent": row["speaker_volume_percent"],
                    "microphone_volume_percent": row["microphone_volume_percent"],
                    "selection_key": list(_aec_quality_key(row)),
                    "single_run": {
                        "attenuation_db": row.get("residual_echo_attenuation_db"),
                        "vad_false_trigger_ratio_on": (row.get("assistant_only_vad_on") or {}).get("false_trigger_ratio"),
                        "whisper_cer_on": row.get("whisper_cer_on"),
                        "clipping_detected": row.get("clipping_detected"),
                    },
                }
                for row in candidates
            ]
            for candidate in candidates:
                key = candidate["condition_id"]
                rerun_rows: list[dict[str, Any]] = []
                for repeat in range(1, repeats + 1):
                    condition_id = f"{key}_repeat{repeat}"
                    try:
                        rerun = _aec_condition_measurement(
                            config,
                            reference=reference,
                            run_id=run_id,
                            condition_id=condition_id,
                            speaker_percent=int(candidate["speaker_volume_percent"]),
                            microphone_percent=int(candidate["microphone_volume_percent"]),
                            volume=volume,
                            asr=asr,
                            signal_floor=signal_floor,
                        )
                    except Exception as exc:
                        rerun = {"status": "blocked", "condition_id": condition_id, "error_type": type(exc).__name__, "error": str(exc)}
                    rerun["clipping_detected"] = _aec_clip_detected(rerun, ratio_threshold=ratio_threshold, peak_threshold=peak_threshold)
                    rerun_rows.append(rerun)
                data["best_condition_remeasurements"][key] = {
                    "speaker_volume_percent": candidate["speaker_volume_percent"],
                    "microphone_volume_percent": candidate["microphone_volume_percent"],
                    "summary": _aec_best_summary(rerun_rows),
                }
            data["best_echo_only_operating_envelope"] = {
                "interpretation": "range observed in the selected repeat-tested conditions; not an absolute optimum and not a human-speech gain recommendation",
                "selected_conditions": [row["condition_id"] for row in candidates],
                "speaker_volume_percent_range": [min((row["speaker_volume_percent"] for row in candidates), default=None), max((row["speaker_volume_percent"] for row in candidates), default=None)],
                "microphone_volume_percent_range": [min((row["microphone_volume_percent"] for row in candidates), default=None), max((row["microphone_volume_percent"] for row in candidates), default=None)],
            }
            data["status"] = "measured" if data["matrix_summary"]["measured_count"] else "partial"
        data["volume_restore_error"] = volume_guard.restore_error if volume_guard is not None else None
    except Exception as exc:
        data["status"] = "blocked"
        data["error_type"] = type(exc).__name__
        data["error"] = str(exc)
    finally:
        if volume_guard is not None:
            data["volume_restore_error"] = volume_guard.restore_error
        if asr is not None:
            asr.unload()
    for key, summary in data.get("best_condition_remeasurements", {}).items():
        attenuation_median = (summary.get("summary") or {}).get("attenuation_db", {}).get("median")
        summary["comparison_to_baseline"] = {
            "baseline_db": AEC_BASELINE_ATTENUATION_DB,
            "median_minus_baseline_db": attenuation_median - AEC_BASELINE_ATTENUATION_DB if isinstance(attenuation_median, (int, float)) else None,
            "clearly_improved": bool(isinstance(attenuation_median, (int, float)) and attenuation_median >= AEC_BASELINE_ATTENUATION_DB + 0.5),
        }
    return write_benchmark(config, "aec_matrix", data, started_at=started)


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
