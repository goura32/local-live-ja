from __future__ import annotations

import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .asr import WhisperASR
from .audio import (
    AudioVolumeGuard,
    PipeWireInventory,
    PipeWirePCMPlayback,
    RawCaptureSession,
    playback_on_active_capture,
    stable_target,
)
from .audio_metrics import detect_acoustic_onset, measure_generated_audio_leading_silence, trim_leading_silence
from .bench import (
    E2E_SYSTEM_PROMPT,
    _cer,
    _collect_first_sentence_chunk,
    _distribution,
    _live_messages,
    artifact_dir,
    ensure_synthetic_audio,
    nested,
    write_benchmark,
)
from .llm.ollama import OllamaLLM
from .telemetry import EventLog, ResourceMonitor, current_gpu_memory, write_json
from .tts_backends import build_tts_backend
from .vllm_omni_tts import VLLMOmniTTSEngine, aggregate_stream_timing, audio_metrics_from_pcm
from .vllm_server import VLLMOmniServer

SERVING_RESPONSES: dict[str, str] = {
    "short_ack": "はい、確認しました。",
    "short_sentence": "準備が完了しました。",
    "numeric": "確認した件数は二十四件です。",
    "technical": "PipeWireのAEC設定を確認します。",
    "two_sentences": "設定を確認しました。次に測定を開始します。",
}


class _MemoryPCMPlayback:
    """Test/benchmark sink that preserves stream timing without opening audio hardware."""

    def __init__(self) -> None:
        self.payload = bytearray()
        self.started_ns: int | None = None
        self.last_queued_ns: int | None = None
        self.finished = False
        self.cancelled = False

    def start(self, *, sample_rate: int, channels: int) -> dict[str, Any]:
        self.started_ns = time.monotonic_ns()
        return {"started_ns": self.started_ns, "sample_rate": sample_rate, "channels": channels}

    def queue(self, payload: bytes) -> dict[str, Any]:
        if self.cancelled or self.finished:
            raise RuntimeError("memory PCM playback is inactive")
        if payload:
            self.payload.extend(payload)
            self.last_queued_ns = time.monotonic_ns()
        return {"queued": bool(payload)}

    def finish(self) -> dict[str, Any]:
        self.finished = True
        return {"cancelled": False, "bytes": len(self.payload)}

    def cancel(self) -> dict[str, Any]:
        self.cancelled = True
        self.finished = True
        self.payload.clear()
        return {"cancelled": True}


def _vllm_python(config: dict[str, Any]) -> Path:
    configured = nested(
        config,
        "tts",
        "vllm_python",
        default="/home/ws1/.venvs/local-live-vllm-omni-0.28.0/bin/python",
    )
    return Path(str(configured))


def _vllm_model(config: dict[str, Any]) -> str:
    return str(nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"))


def _make_vllm_client(config: dict[str, Any], *, streaming: bool, initial: int | None = None) -> VLLMOmniTTSEngine:
    return VLLMOmniTTSEngine(
        base_url=str(nested(config, "tts", "vllm_base_url", default="http://127.0.0.1:8091/v1")),
        model=_vllm_model(config),
        speaker=str(nested(config, "tts", "speaker", default="Ono_Anna")),
        language=str(nested(config, "tts", "language", default="Japanese")),
        timeout_s=float(nested(config, "tts", "vllm_timeout_s", default=300.0)),
        initial_codec_chunk_frames=initial,
        streaming=streaming,
    )


def _make_python_engine(config: dict[str, Any]) -> Any:
    return build_tts_backend(config, backend_override="python", streaming_override=False)


def _trim_source_wav(config: dict[str, Any], source: Path, destination: Path) -> dict[str, Any]:
    samples, rate = sf.read(str(source), always_2d=False)
    array = np.asarray(samples, dtype=np.float32)
    analysis = measure_generated_audio_leading_silence(array, int(rate))
    trimmed, metadata = trim_leading_silence(
        array,
        int(rate),
        analysis,
        pre_roll_s=float(nested(config, "tts", "safe_trim_pre_roll_s", default=0.05)),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), trimmed, int(rate))
    metadata["source_path"] = str(source)
    metadata["destination_path"] = str(destination)
    metadata["source_analysis"] = analysis
    return metadata


def _physical_measurement(
    *,
    capture: RawCaptureSession,
    playback: dict[str, Any],
    reference_path: Path,
    capture_target: str,
    tail_s: float = 0.5,
) -> dict[str, Any]:
    recording = capture.stop(tail_s=tail_s)
    recorded, recorded_rate = sf.read(str(capture.output_path), always_2d=False)
    reference, reference_rate = sf.read(str(reference_path), always_2d=False)
    timing = playback.setdefault("timing_ns", {})
    record_start_ns = timing.get("record_start", capture.started_ns)
    playback_start_ns = timing.get("pw_play_start") or timing.get("playback_stream_started")
    offset_s = (
        (playback_start_ns - record_start_ns) / 1e9
        if record_start_ns is not None and playback_start_ns is not None
        else 0.0
    )
    search_start_s = max(0.1, offset_s - 0.05)
    onset = detect_acoustic_onset(
        np.asarray(recorded),
        int(recorded_rate),
        search_start_s=search_start_s,
        noise_window_s=max(0.1, min(1.0, search_start_s)),
        reference=np.asarray(reference),
        reference_rate=int(reference_rate),
    )
    physical_ns = (
        record_start_ns + int(round(float(onset["onset_s"]) * 1e9))
        if onset.get("detected") and record_start_ns is not None
        else None
    )
    timing["record_start"] = record_start_ns
    timing["physical_audio_detected"] = physical_ns
    playback["recording_target"] = capture_target
    playback["record_returncode"] = recording.get("record_returncode")
    playback["record_stderr"] = recording.get("record_stderr")
    playback["recording_stats"] = recording.get("recording_stats")
    playback["acoustic_onset"] = onset
    playback["physical_audio_detected"] = bool(onset.get("detected"))
    return playback


def _source_metadata(path: Path) -> dict[str, Any]:
    samples, rate = sf.read(str(path), always_2d=False)
    array = np.asarray(samples, dtype=np.float32)
    analysis = measure_generated_audio_leading_silence(array, int(rate))
    return {
        "path": str(path),
        "sample_rate": int(rate),
        "frames": int(array.shape[0]) if array.ndim else int(array.size),
        "audio_duration_s": float(array.shape[0] / rate) if array.ndim else float(array.size / rate),
        "rms": float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0,
        "peak": float(np.max(np.abs(array))) if array.size else 0.0,
        "clipping_ratio": float(np.mean(np.abs(array) >= 0.98)) if array.size else 0.0,
        "generated_audio_analysis": analysis,
    }


def _annotate_row(
    *,
    mode: str,
    text_name: str,
    text: str,
    request_start_ns: int,
    result: dict[str, Any],
    source_path: Path,
    playback_path: Path | None,
    physical: dict[str, Any] | None,
    trim_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    timing = dict(result.get("timing_ns", {}))
    timing.setdefault("request_start", request_start_ns)
    if physical is not None:
        timing.update(physical.get("timing_ns", {}))
    source = _source_metadata(source_path)
    source_onset = source["generated_audio_analysis"].get("stable_speech_onset_s")
    first_chunk_ns = timing.get("first_audio_chunk_received")
    actual_speech_ns = (
        first_chunk_ns + int(round(float(source_onset) * 1e9))
        if first_chunk_ns is not None and isinstance(source_onset, (int, float))
        else None
    )
    timing["first_actual_speech_pcm"] = actual_speech_ns
    audio_duration = float(result.get("audio_duration_s") or source["audio_duration_s"])
    derived = aggregate_stream_timing(timing, audio_duration_s=audio_duration)
    physical_ns = timing.get("physical_audio_detected")
    physical_latency = (
        (physical_ns - request_start_ns) / 1e9 if physical_ns is not None else None
    )
    row = {
        "status": result.get("status", "error"),
        "measurement_kind": "serving_standalone",
        "mode": mode,
        "response_name": text_name,
        "text": text,
        "first_chunk_text": text,
        "source_audio": source,
        "tts": result,
        "path": str(playback_path or source_path),
        "trim": trim_metadata,
        "physical": physical,
        "timing_ns": timing,
        "timing": derived,
        "request_to_first_audio_chunk_s": (
            (timing["first_audio_chunk_received"] - request_start_ns) / 1e9
            if timing.get("first_audio_chunk_received") is not None
            else None
        ),
        "first_actual_speech_pcm_latency_s": (
            (actual_speech_ns - request_start_ns) / 1e9 if actual_speech_ns is not None else None
        ),
        "physical_first_audio_s": physical_latency,
        "audio_duration_s": audio_duration,
        "rtf": derived.get("rtf"),
        "quality": None,
    }
    return row


def _run_serving_once(
    config: dict[str, Any],
    *,
    engine: Any,
    mode: str,
    text_name: str,
    text: str,
    run_number: int,
    physical_enabled: bool,
    playback_target: str | None,
    capture_target: str | None,
    asr: WhisperASR | None = None,
) -> dict[str, Any]:
    run_id = f"serving_{mode}_{text_name}_{run_number}_{time.monotonic_ns()}"
    output = artifact_dir(config) / f"{run_id}.wav"
    request_start_ns = time.monotonic_ns()
    capture: RawCaptureSession | None = None
    capture_started = False
    physical: dict[str, Any] | None = None
    trim_metadata: dict[str, Any] | None = None
    try:
        if physical_enabled:
            if not playback_target or not capture_target:
                raise RuntimeError("physical serving benchmark requires stable speaker and microphone targets")
            capture_path = artifact_dir(config) / f"{run_id}_raw_mic.wav"
            capture = RawCaptureSession(capture_path, target=capture_target, sample_rate=16000)
            capture.start()
            capture_started = True
            request_start_ns = time.monotonic_ns()
        if mode.startswith("vllm_omni_streaming"):
            playback: Any = (
                PipeWirePCMPlayback(playback_target)
                if physical_enabled and playback_target
                else _MemoryPCMPlayback()
            )
            result = engine.synthesize_stream(text, output_path=output, playback=playback)
            playback_path = output if result.get("status") == "measured" else None
            if physical_enabled and result.get("status") == "measured":
                assert capture is not None
                physical = _physical_measurement(
                    capture=capture,
                    playback={"timing_ns": result.get("timing_ns", {})},
                    reference_path=output,
                    capture_target=str(capture_target),
                )
                result["timing_ns"].update(physical.get("timing_ns", {}))
        else:
            result = engine.synthesize(text, output_path=output)
            playback_path = output
            source_path = output
            tts_success = result.get("status") == "measured" or (
                mode == "python" and isinstance(result.get("path"), str)
            )
            if tts_success and mode == "python":
                trimmed = artifact_dir(config) / f"{run_id}_trimmed.wav"
                trim_metadata = _trim_source_wav(config, output, trimmed)
                playback_path = trimmed
            if physical_enabled and tts_success:
                assert capture is not None
                playback = playback_on_active_capture(
                    playback_path,
                    playback_target=str(playback_target),
                    capture=capture,
                )
                physical = _physical_measurement(
                    capture=capture,
                    playback=playback,
                    reference_path=playback_path,
                    capture_target=str(capture_target),
                )
                result["timing_ns"].update(physical.get("timing_ns", {}))
        if capture_started and physical is None and capture is not None:
            capture.stop(tail_s=0.1)
        if mode == "python":
            python_timing = result.setdefault("timing_ns", {})
            python_timing.setdefault(
                "first_audio_chunk_received",
                python_timing.get("tts_waveform_ready") or python_timing.get("generation_complete"),
            )
            python_timing.setdefault(
                "last_audio_chunk_received",
                python_timing.get("wav_ready") or python_timing.get("audio_complete"),
            )
        is_measured = result.get("status") == "measured" or (
            mode == "python" and isinstance(result.get("path"), str)
        )
        if not is_measured:
            return {
                "status": result.get("status", "error"),
                "mode": mode,
                "response_name": text_name,
                "text": text,
                "run_number": run_number,
                "tts": result,
                "error": result.get("error"),
            }
        result.setdefault("status", "measured")
        row = _annotate_row(
            mode=mode,
            text_name=text_name,
            text=text,
            request_start_ns=request_start_ns,
            result=result,
            source_path=output,
            playback_path=playback_path,
            physical=physical,
            trim_metadata=trim_metadata,
        )
        row["run_number"] = run_number
        return row
    except Exception as exc:
        if capture_started and capture is not None and capture._result is None:
            try:
                capture.stop(tail_s=0.1)
            except Exception:
                pass
        return {
            "status": "error",
            "mode": mode,
            "response_name": text_name,
            "text": text,
            "run_number": run_number,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _mode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if row.get("status") == "measured"]
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in measured if isinstance(row.get(key), (int, float))]
    return {
        "attempt_count": len(rows),
        "measured_count": len(measured),
        "request_to_first_audio_chunk_s": _distribution(values("request_to_first_audio_chunk_s")),
        "first_actual_speech_pcm_latency_s": _distribution(values("first_actual_speech_pcm_latency_s")),
        "physical_first_audio_s": _distribution(values("physical_first_audio_s")),
        "audio_duration_s": _distribution(values("audio_duration_s")),
        "rtf": _distribution(values("rtf")),
        "outlier_policy": "retain every attempt; no latency or CER outlier is deleted",
    }


def _fill_roundtrip_quality(rows_by_mode: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    paths: list[tuple[str, str, dict[str, Any]]] = []
    for mode, rows in rows_by_mode.items():
        seen: set[str] = set()
        for row in rows:
            path = row.get("source_audio", {}).get("path")
            name = row.get("response_name")
            if row.get("status") == "measured" and isinstance(path, str) and name not in seen:
                seen.add(str(name))
                paths.append((mode, str(name), row))
    if not paths:
        return {"status": "not_measured", "reason": "no successful audio rows"}
    asr: WhisperASR | None = None
    try:
        asr = WhisperASR(
            model=nested(config, "asr", "model", default="large-v3-turbo"),
            device="cuda" if _cuda_available() else "cpu",
            compute_type=str(nested(config, "asr", "gpu_default_compute_type", default="int8_float16")),
            language="ja",
            beam_size=int(nested(config, "asr", "beam_size", default=5)),
        )
        quality_rows: list[dict[str, Any]] = []
        for mode, name, row in paths:
            transcript = asr.transcribe(row["source_audio"]["path"])
            quality = {
                "mode": mode,
                "response_name": name,
                "reference_text": row["text"],
                "transcript": transcript.text,
                "cer": _cer(row["text"], transcript.text),
                "audio_duration_s": row["source_audio"]["audio_duration_s"],
                "rms": row["source_audio"]["rms"],
                "peak": row["source_audio"]["peak"],
                "clipping_ratio": row["source_audio"]["clipping_ratio"],
            }
            row["quality"] = quality
            quality_rows.append(quality)
        return {"status": "measured", "rows": quality_rows}
    except Exception as exc:
        return {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        if asr is not None:
            asr.unload()


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _server_version(python_bin: Path) -> dict[str, Any]:
    command = [
        str(python_bin),
        "-c",
        "import importlib.metadata as m; print(m.version('vllm')); print(m.version('vllm-omni'))",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        lines = completed.stdout.strip().splitlines()
        return {
            "vllm": lines[0] if len(lines) > 0 else None,
            "vllm_omni": lines[1] if len(lines) > 1 else None,
            "exit_code": completed.returncode,
        }
    except Exception as exc:
        return {"vllm": None, "vllm_omni": None, "error_type": type(exc).__name__}


def _targets(config: dict[str, Any]) -> tuple[dict[str, Any], str | None, str | None]:
    inventory = PipeWireInventory.discover()
    mic = inventory.usb_microphone()
    speaker = inventory.usb_speaker()
    return (
        inventory.to_dict(),
        stable_target(speaker) if speaker else None,
        stable_target(mic) if mic else None,
    )


def _server_memory_note() -> dict[str, Any]:
    return {"gpu_memory_mib": current_gpu_memory(), "process_snapshot": _process_snapshot()}


def _server_env(config: dict[str, Any]) -> dict[str, str]:
    if bool(nested(config, "tts", "vllm_use_flashinfer_sampler", default=False)):
        return {}
    return {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def _process_snapshot() -> list[str]:
    try:
        completed = subprocess.run(["pgrep", "-af", "ollama|vllm"], capture_output=True, text=True, timeout=10, check=False)
        return [line for line in completed.stdout.splitlines() if line]
    except Exception:
        return []


def _load_python_reference(config: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    result_path = Path(nested(config, "app", "result_dir", default="results")) / "bench_live_latency.json"
    if not result_path.is_file():
        return None, None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        data = payload.get("data", {})
        if data.get("backend") == "vllm_omni":
            reference_path = result_path.with_name("bench_live_latency_python.json")
            if reference_path.is_file():
                reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
                return str(reference_path), reference_payload.get("data", {})
            return None, None
        reference_path = result_path.with_name("bench_live_latency_python.json")
        if not reference_path.exists():
            write_json(reference_path, payload)
        return str(reference_path), data
    except Exception:
        return None, None


def run_tts_serving_bench(
    config: dict[str, Any],
    *,
    skip_python: bool = False,
    skip_physical: bool = False,
    initial_codec_chunk_frames: list[int] | None = None,
) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    repeats = max(5, int(nested(config, "bench", "tts_serving_repeats", default=5)))
    physical_enabled = not skip_physical
    data: dict[str, Any] = {
        "status": "blocked",
        "benchmark": "tts_serving",
        "model": _vllm_model(config),
        "speaker": str(nested(config, "tts", "speaker", default="Ono_Anna")),
        "language": str(nested(config, "tts", "language", default="Japanese")),
        "sample_rate_hz": 24000,
        "repeat_target_per_text_per_mode": repeats,
        "physical_enabled": physical_enabled,
        "responses": SERVING_RESPONSES,
        "modes": {},
        "initial_codec_chunk_frames_comparison": [],
        "async_chunk": {
            "official_default": True,
            "source": "vllm_omni/deploy/qwen3_tts.yaml at bc0c9f4b45c45c59aa2f92471842e8c18ae403ca",
            "non_async_comparison": "not_run; default async mode benchmarked first",
        },
        "server": None,
        "quality": None,
        "targets": None,
        "volume_snapshot": None,
        "volume_restore_error": None,
        "memory": {"before_server": _server_memory_note()},
    }
    reference_path, reference_data = _load_python_reference(config)
    if reference_path:
        data["python_baseline_reference"] = {"path": reference_path, "data": reference_data}
    volume_guard: AudioVolumeGuard | None = None
    server: VLLMOmniServer | None = None
    python_engine: Any = None
    try:
        inventory, playback_target, capture_target = _targets(config)
        data["targets"] = {"inventory": inventory, "speaker": playback_target, "microphone": capture_target}
        if physical_enabled:
            if not playback_target or not capture_target:
                raise RuntimeError("USB speaker and microphone are required for physical serving benchmark")
            volume_guard = AudioVolumeGuard(speaker_target=playback_target, microphone_target=capture_target)
            volume_guard.__enter__()
            data["volume_snapshot"] = volume_guard.snapshot.to_dict() if volume_guard.snapshot else None
            volume_guard.set_mutes(speaker_muted=False, microphone_muted=False)
        if not skip_python:
            python_engine = _make_python_engine(config)
            python_rows: list[dict[str, Any]] = []
            for name, text in SERVING_RESPONSES.items():
                for run_number in range(1, repeats + 1):
                    python_rows.append(
                        _run_serving_once(
                            config,
                            engine=python_engine,
                            mode="python",
                            text_name=name,
                            text=text,
                            run_number=run_number,
                            physical_enabled=physical_enabled,
                            playback_target=playback_target,
                            capture_target=capture_target,
                        )
                    )
            data["modes"]["python"] = {
                "summary": _mode_summary(python_rows),
                "rows": python_rows,
                "engine": "official Qwen3-TTS Python API",
            }
            python_engine.unload()
            python_engine = None
        python_bin = _vllm_python(config)
        server = VLLMOmniServer(
            python_bin=python_bin,
            model=_vllm_model(config),
            host="127.0.0.1",
            port=8091,
            deploy_config=nested(config, "tts", "vllm_deploy_config", default=None),
            log_path=artifact_dir(config) / "vllm_omni_server.log",
            gpu_memory_utilization=nested(config, "tts", "vllm_gpu_memory_utilization", default=None),
            extra_env=_server_env(config),
        )
        server_info = server.start(timeout_s=float(nested(config, "tts", "vllm_server_start_timeout_s", default=900.0)))
        server_info["versions"] = _server_version(python_bin)
        server_info["official_repository_commit"] = "bc0c9f4b45c45c59aa2f92471842e8c18ae403ca"
        server_info["official_stable_tag"] = "v0.28.0"
        server_info["memory_at_ready"] = _server_memory_note()
        data["server"] = server_info
        for mode, streaming in (("vllm_omni_non_streaming", False), ("vllm_omni_streaming", True)):
            engine = _make_vllm_client(config, streaming=streaming, initial=None)
            rows: list[dict[str, Any]] = []
            for name, text in SERVING_RESPONSES.items():
                for run_number in range(1, repeats + 1):
                    rows.append(
                        _run_serving_once(
                            config,
                            engine=engine,
                            mode=mode,
                            text_name=name,
                            text=text,
                            run_number=run_number,
                            physical_enabled=physical_enabled,
                            playback_target=playback_target,
                            capture_target=capture_target,
                        )
                    )
            data["modes"][mode] = {
                "summary": _mode_summary(rows),
                "rows": rows,
                "engine": "vLLM-Omni OpenAI-compatible speech API",
                "request_initial_codec_chunk_frames": None,
            }
        candidates = initial_codec_chunk_frames or [None, 1, 2, 4]
        for candidate in candidates:
            engine = _make_vllm_client(config, streaming=True, initial=candidate)
            candidate_rows: list[dict[str, Any]] = []
            for run_number in range(1, 4):
                candidate_rows.append(
                    _run_serving_once(
                        config,
                        engine=engine,
                        mode="vllm_omni_streaming_initial_codec_probe",
                        text_name="short_ack",
                        text=SERVING_RESPONSES["short_ack"],
                        run_number=run_number,
                        physical_enabled=False,
                        playback_target=None,
                        capture_target=None,
                    )
                )
            data["initial_codec_chunk_frames_comparison"].append(
                {
                    "initial_codec_chunk_frames": candidate,
                    "summary": _mode_summary(candidate_rows),
                    "rows": candidate_rows,
                }
            )
        data["memory"]["after_serving_requests"] = _server_memory_note()
        data["status"] = "measured"
    except Exception as exc:
        data["status"] = "blocked"
        data["error_type"] = type(exc).__name__
        data["error"] = str(exc)
    finally:
        if python_engine is not None:
            python_engine.unload()
        if server is not None:
            data.setdefault("server", {})
            data["server"]["stop"] = server.stop()
            data["memory"]["after_server_stop"] = _server_memory_note()
        if volume_guard is not None:
            volume_guard.__exit__(None, None, None)
            data["volume_restore_error"] = volume_guard.restore_error
    if data.get("modes"):
        data["quality"] = _fill_roundtrip_quality(
            {mode: value.get("rows", []) for mode, value in data["modes"].items()},
            config,
        )
        data["initial_codec_chunk_frames_quality"] = _fill_roundtrip_quality(
            {
                f"initial_codec_chunk_frames_{entry.get('initial_codec_chunk_frames')}": entry.get("rows", [])
                for entry in data.get("initial_codec_chunk_frames_comparison", [])
            },
            config,
        )
    return write_benchmark(config, "tts_serving", data, started_at=started)


def _vllm_live_attempt(
    config: dict[str, Any],
    *,
    engine: VLLMOmniTTSEngine,
    asr: WhisperASR,
    provider: OllamaLLM,
    user_audio: Path,
    run_number: int,
    playback_target: str,
    capture_target: str,
) -> dict[str, Any]:
    log = EventLog()
    started_ns = time.monotonic_ns()
    recording_path = artifact_dir(config) / f"vllm_live_run{run_number}_raw_mic.wav"
    capture = RawCaptureSession(recording_path, target=capture_target, sample_rate=16000)
    capture.start()
    synthetic_end_ns = time.monotonic_ns()
    log.mark_at("synthetic_user_end", synthetic_end_ns, source="direct_input_wav_handoff")
    try:
        user_asr = asr.transcribe(user_audio, event_log=log)
        llm_start_ns = time.monotonic_ns()
        log.mark_at("llm_start", llm_start_ns)
        turn = _collect_first_sentence_chunk(
            provider,
            _live_messages(user_asr.text),
            max_chars=int(nested(config, "tts", "sentence_max_chars", default=48)),
            timeout_s=float(nested(config, "tts", "sentence_timeout_s", default=0.8)),
        )
        first_token_ns = turn.get("first_token_ns")
        first_chunk_ns = turn.get("first_chunk_ns")
        if first_token_ns is not None:
            log.mark_at("llm_first_token", first_token_ns)
        if first_chunk_ns is not None:
            log.mark_at("first_sentence_chunk_ready", first_chunk_ns, text_chars=len(turn.get("first_chunk") or ""))
        if turn.get("error"):
            raise RuntimeError(str(turn["error"].message))
        chunk = str(turn.get("first_chunk") or "")
        if not chunk:
            raise RuntimeError("LLM returned no first sentence chunk")
        tts_start_ns = time.monotonic_ns()
        log.mark_at("tts_start", tts_start_ns, text_chars=len(chunk))
        output = artifact_dir(config) / f"vllm_live_run{run_number}_assistant.wav"
        playback = PipeWirePCMPlayback(playback_target)
        result = engine.synthesize_stream(chunk, output_path=output, playback=playback, event_log=log)
        if result.get("status") != "measured":
            raise RuntimeError(str(result.get("error") or "vLLM streaming TTS failed"))
        playback_result = {"timing_ns": result.get("timing_ns", {}), "path": str(output)}
        physical = _physical_measurement(
            capture=capture,
            playback=playback_result,
            reference_path=output,
            capture_target=capture_target,
        )
        timing = dict(result.get("timing_ns", {}))
        timing.update(physical.get("timing_ns", {}))
        timing.update(
            {
                "synthetic_user_end": synthetic_end_ns,
                "asr_start": next((item["monotonic_ns"] for item in log.events if item["event"] == "asr_start"), None),
                "asr_final": next((item["monotonic_ns"] for item in reversed(log.events) if item["event"] == "asr_final"), None),
                "llm_start": llm_start_ns,
                "llm_first_token": first_token_ns,
                "first_sentence_chunk_ready": first_chunk_ns,
                "tts_start": tts_start_ns,
            }
        )
        source = _source_metadata(output)
        physical_ns = timing.get("physical_audio_detected")
        latency = (physical_ns - synthetic_end_ns) / 1e9 if physical_ns is not None else None
        budget = {
            "asr": _delta_seconds(timing, "asr_final", "synthetic_user_end"),
            "llm_ttft": _delta_seconds(timing, "llm_first_token", "llm_start"),
            "first_chunk_buffering": _delta_seconds(timing, "first_sentence_chunk_ready", "llm_first_token"),
            "tts_request_to_first_pcm": _delta_seconds(timing, "first_audio_chunk_received", "tts_start"),
            "first_pcm_to_queue": _delta_seconds(timing, "first_audio_chunk_queued", "first_audio_chunk_received"),
            "stream_to_physical": _delta_seconds(timing, "physical_audio_detected", "playback_stream_started"),
        }
        return {
            "status": "measured" if latency is not None else "blocked",
            "measurement_kind": "primary_playback_path",
            "backend": "vllm_omni",
            "streaming": True,
            "run_number": run_number,
            "synthetic_user_end_ns": synthetic_end_ns,
            "user_asr": user_asr.to_dict(),
            "llm": {
                "first_chunk": chunk,
                "first_chunk_chars": len(chunk),
                "ttft_s": turn.get("ttft_s"),
                "until_first_chunk_s": turn.get("until_first_chunk_s"),
                "actual_model": getattr(provider, "last_actual_model", None),
            },
            "tts": result,
            "source_audio": source,
            "physical": physical,
            "timing_ns": timing,
            "stages": budget,
            "speech_end_to_first_physical_audio_s": latency,
            "blocked_reason": None if latency is not None else "physical_audio_not_detected",
            "events": log.events,
        }
    except Exception as exc:
        try:
            if capture._result is None:
                capture.stop(tail_s=0.1)
        except Exception:
            pass
        return {
            "status": "error",
            "backend": "vllm_omni",
            "streaming": True,
            "run_number": run_number,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "events": log.events,
        }


def _delta_seconds(timing: dict[str, Any], end: str, start: str) -> float | None:
    e = timing.get(end)
    s = timing.get(start)
    return (e - s) / 1e9 if isinstance(e, int) and isinstance(s, int) and e >= s else None


def _live_budget(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if row.get("status") == "measured"]
    total_values = [float(row["speech_end_to_first_physical_audio_s"]) for row in measured]
    components: dict[str, dict[str, Any]] = {}
    names = ["asr", "llm_ttft", "first_chunk_buffering", "tts_request_to_first_pcm", "first_pcm_to_queue", "stream_to_physical"]
    for name in names:
        values = [float(row["stages"][name]) for row in measured if isinstance(row.get("stages", {}).get(name), (int, float))]
        shares = [
            float(row["stages"][name]) / total
            for row, total in zip(measured, total_values)
            if isinstance(row.get("stages", {}).get(name), (int, float))
        ]
        components[name] = {"seconds": _distribution(values), "share_of_total": _distribution(shares)}
    return {
        "valid_run_count": len(measured),
        "denominator": "each row speech_end_to_first_physical_audio_s; shares calculated per run before distribution",
        "components": components,
        "total": _distribution(total_values),
    }


def run_vllm_live_latency_bench(config: dict[str, Any], *, streaming: bool = True) -> dict[str, Any]:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    repeats = max(10, int(nested(config, "bench", "live_latency_repeats", default=10)))
    data: dict[str, Any] = {
        "status": "blocked",
        "backend": "vllm_omni",
        "streaming": streaming,
        "model": _vllm_model(config),
        "speaker": str(nested(config, "tts", "speaker", default="Ono_Anna")),
        "language": str(nested(config, "tts", "language", default="Japanese")),
        "sample_rate_hz": 24000,
        "repeat_target": repeats,
        "comparison_scope": "vllm_omni_streaming_vs_existing_python_reference",
        "runs": [],
        "summary": None,
        "latency_budget": None,
        "server": None,
        "volume_snapshot": None,
        "volume_restore_error": None,
        "memory": {"before_server": _server_memory_note()},
    }
    reference_path, reference_data = _load_python_reference(config)
    data["python_baseline_reference"] = {"path": reference_path, "data": reference_data}
    server: VLLMOmniServer | None = None
    asr: WhisperASR | None = None
    volume_guard: AudioVolumeGuard | None = None
    try:
        inventory, playback_target, capture_target = _targets(config)
        if not playback_target or not capture_target:
            raise RuntimeError("USB speaker and microphone are required for vLLM live-latency benchmark")
        data["targets"] = {"inventory": inventory, "speaker": playback_target, "microphone": capture_target}
        volume_guard = AudioVolumeGuard(speaker_target=playback_target, microphone_target=capture_target)
        volume_guard.__enter__()
        data["volume_snapshot"] = volume_guard.snapshot.to_dict() if volume_guard.snapshot else None
        volume_guard.set_mutes(speaker_muted=False, microphone_muted=False)
        server = VLLMOmniServer(
            python_bin=_vllm_python(config),
            model=_vllm_model(config),
            host="127.0.0.1",
            port=8091,
            deploy_config=nested(config, "tts", "vllm_deploy_config", default=None),
            log_path=artifact_dir(config) / "vllm_omni_live_server.log",
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
        data["memory"]["after_whisper_load"] = _server_memory_note()
        provider = OllamaLLM(
            base_url=nested(config, "llm", "local_base_url", default="http://127.0.0.1:11434"),
            model=nested(config, "llm", "local_model", default="auto"),
            max_tokens=int(nested(config, "llm", "max_tokens", default=96)),
            num_ctx=int(nested(config, "llm", "num_ctx", default=8192)),
            temperature=float(nested(config, "llm", "temperature", default=0.2)),
        )
        data["ollama_probe"] = provider.probe()
        data["memory"]["after_ollama_probe"] = _server_memory_note()
        user_audio, _ = ensure_synthetic_audio(config)
        engine = _make_vllm_client(
            config,
            streaming=streaming,
            initial=nested(config, "tts", "vllm_initial_codec_chunk_frames", default=None),
        )
        max_attempts = max(
            repeats,
            int(nested(config, "bench", "live_latency_max_attempts", default=repeats)),
        )
        for run_number in range(1, max_attempts + 1):
            row = _vllm_live_attempt(
                config,
                engine=engine,
                asr=asr,
                provider=provider,
                user_audio=user_audio,
                run_number=run_number,
                playback_target=str(playback_target),
                capture_target=str(capture_target),
            )
            data["runs"].append(row)
            if sum(item.get("status") == "measured" for item in data["runs"]) >= repeats:
                break
        measured = [row for row in data["runs"] if row.get("status") == "measured"]
        values = [float(row["speech_end_to_first_physical_audio_s"]) for row in measured]
        data["summary"] = {
            "measured_run_count": len(measured),
            "repeat_target": repeats,
            "attempt_count": len(data["runs"]),
            "max_attempts": max_attempts,
            "speech_end_to_first_physical_audio_s": _distribution(values),
            "request_to_first_audio_chunk_s": _distribution(
                [
                    (row["timing_ns"]["first_audio_chunk_received"] - row["timing_ns"]["tts_start"]) / 1e9
                    for row in measured
                    if row.get("timing_ns", {}).get("first_audio_chunk_received") is not None
                ]
            ),
            "first_actual_speech_pcm_latency_s": _distribution(
                [
                    float(row["first_actual_speech_pcm_latency_s"])
                    for row in measured
                    if isinstance(row.get("first_actual_speech_pcm_latency_s"), (int, float))
                ]
            ),
            "outlier_policy": "retain every attempt; no latency value is deleted",
        }
        data["latency_budget"] = _live_budget(data["runs"])
        data["status"] = "measured" if len(measured) >= repeats else "partial"
        data["memory"]["live_pipeline"] = _server_memory_note()
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
            volume_guard.__exit__(None, None, None)
            data["volume_restore_error"] = volume_guard.restore_error
    return write_benchmark(config, "live_latency", data, started_at=started)
