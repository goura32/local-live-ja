from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .asr import WhisperASR
from .audio import EchoCancelSession, PipeWireInventory, PipeWirePlayback, record_fixed
from .bench import run_aec_bench, run_asr_bench, run_e2e_bench, run_llm_bench, run_tts_bench
from .config import load_config, nested
from .doctor import run_doctor
from .llm.ollama import OllamaLLM
from .llm.openrouter import OpenRouterLLM
from .pipeline import Cancellation, LivePipeline
from .telemetry import EventLog, write_json
from .tts import Qwen3TTSEngine
from .vad import detect_speech_intervals, trim_to_speech


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-live", description="Japanese low-latency live voice PoC")
    parser.add_argument("--config", default="config/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor")
    bench = subparsers.add_parser("bench")
    bench_sub = bench.add_subparsers(dest="bench_name", required=True)
    asr = bench_sub.add_parser("asr")
    asr.add_argument("--audio")
    asr.add_argument("--force-audio", action="store_true")
    asr.add_argument("--skip-cpu", action="store_true")
    tts = bench_sub.add_parser("tts")
    tts.add_argument("--skip-cpu", action="store_true")
    bench_sub.add_parser("llm")
    e2e = bench_sub.add_parser("e2e")
    e2e.add_argument("--force-audio", action="store_true")
    e2e.add_argument("--skip-openrouter", action="store_true")
    aec = bench_sub.add_parser("aec")
    aec.add_argument("--force-audio", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--input-wav")
    run.add_argument("--provider", choices=["local", "openrouter"], default="local")
    run.add_argument("--duration", type=float, default=8.0, help="record duration when --input-wav is omitted")
    run.add_argument("--asr-device", choices=["auto", "cuda", "cpu"], default="auto")
    run.add_argument("--compute-type", default=None)
    run.add_argument("--no-aec", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "doctor":
        result = run_doctor(config, result_path=Path(nested(config, "app", "result_dir", default="results")) / "doctor.json")
        _print_status("doctor", result.get("status"), Path(nested(config, "app", "result_dir", default="results")) / "doctor.json")
        return 0 if result.get("status") != "fail" else 1
    if args.command == "bench":
        result = _run_bench(args, config)
        path = Path(nested(config, "app", "result_dir", default="results")) / f"bench_{args.bench_name}.json"
        _print_status(f"bench {args.bench_name}", result.get("data", {}).get("status", "written"), path)
        return 0
    if args.command == "run":
        return _run_live(args, config)
    return 2


def _run_bench(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if args.bench_name == "asr":
        return run_asr_bench(config, audio_path=args.audio, force_audio=args.force_audio, skip_cpu=args.skip_cpu)
    if args.bench_name == "tts":
        return run_tts_bench(config, skip_cpu=args.skip_cpu)
    if args.bench_name == "llm":
        return run_llm_bench(config)
    if args.bench_name == "e2e":
        return run_e2e_bench(config, force_audio=args.force_audio, skip_openrouter=args.skip_openrouter)
    if args.bench_name == "aec":
        return run_aec_bench(config, force_audio=args.force_audio)
    raise ValueError(args.bench_name)


def _run_live(args: argparse.Namespace, config: dict[str, Any]) -> int:
    artifact = Path(nested(config, "app", "artifact_dir", default="results/artifacts"))
    artifact.mkdir(parents=True, exist_ok=True)
    log = EventLog()
    aec_session: EchoCancelSession | None = None
    try:
        inventory = PipeWireInventory.discover()
        mic = inventory.usb_microphone()
        speaker = inventory.usb_speaker()
        if not args.input_wav:
            if not mic:
                print(json.dumps({"status": "blocked", "reason": "USB microphone not detected"}, ensure_ascii=False))
                return 1
            if not args.no_aec:
                aec_session = _make_aec_session(
                    config,
                    sink_master=speaker.node_id if speaker else None,
                    source_master=mic.node_id if mic else None,
                )
                aec_session.load()
                after = PipeWireInventory.discover()
                mic = next((node for node in after.sources if node.node_id == aec_session.source_node_id), mic)
                speaker = next((node for node in after.sinks if node.node_id == aec_session.sink_node_id), speaker)
            record_target = mic.node_id if mic else None
            log.mark("vad_start")
            record_path = artifact / "live_input.wav"
            record_fixed(record_path, target=record_target, duration_s=args.duration, sample_rate=16000, channels=1)
            audio, sample_rate = sf.read(str(record_path), always_2d=False)
            intervals = detect_speech_intervals(np.asarray(audio), sample_rate)
            log.mark("vad_end", intervals=intervals)
            trimmed = trim_to_speech(np.asarray(audio), sample_rate, intervals)
            input_path = artifact / "live_input_vad.wav"
            sf.write(str(input_path), trimmed, sample_rate)
        else:
            input_path = Path(args.input_wav)
            audio, sample_rate = sf.read(str(input_path), always_2d=False)
            log.mark("vad_start")
            intervals = detect_speech_intervals(np.asarray(audio), sample_rate)
            log.mark("vad_end", intervals=intervals)

        if args.asr_device == "auto":
            asr_device = "cuda" if _cuda_available() else "cpu"
        else:
            asr_device = args.asr_device
        compute_type = args.compute_type or (
            str(nested(config, "asr", "gpu_default_compute_type", default="int8_float16"))
            if asr_device == "cuda"
            else "int8"
        )
        asr = WhisperASR(
            model=nested(config, "asr", "model", default="large-v3-turbo"),
            device=asr_device,
            compute_type=compute_type,
            language="ja",
            beam_size=int(nested(config, "asr", "beam_size", default=5)),
        )
        user_asr = asr.transcribe(input_path, event_log=log)
        providers = _make_providers(config)
        provider = providers[args.provider]
        tts = Qwen3TTSEngine(
            model=nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            speaker=nested(config, "tts", "speaker", default="Ono_Anna"),
            language="Japanese",
            device="auto",
            max_new_tokens=int(nested(config, "tts", "max_new_tokens", default=2048)),
        )
        playback_target = speaker.node_id if speaker else None
        pipeline = LivePipeline(
            llm=provider,
            tts=tts,
            playback=PipeWirePlayback(playback_target),
            artifact_dir=artifact,
            sentence_max_chars=int(nested(config, "tts", "sentence_max_chars", default=48)),
            sentence_timeout_s=float(nested(config, "tts", "sentence_timeout_s", default=0.8)),
        )
        cancellation = Cancellation()
        signal.signal(signal.SIGINT, lambda _signum, _frame: cancellation.request())
        result = pipeline.respond(user_asr.text, cancellation)
        payload = {
            "status": "cancelled" if result.cancelled else ("error" if result.error else "completed"),
            "input": str(input_path),
            "user_asr": user_asr.to_dict(),
            "assistant_text": result.assistant_text,
            "audio_paths": result.audio_paths,
            "error": result.error,
            "events": log.events + [event.__dict__ for event in result.events],
            "timing": result.timing,
        }
        output_path = Path(nested(config, "app", "result_dir", default="results")) / "run_latest.json"
        write_json(output_path, payload)
        print(json.dumps({"status": payload["status"], "assistant_text": payload["assistant_text"], "result": str(output_path)}, ensure_ascii=False))
        return 0 if payload["status"] in {"completed", "cancelled"} else 1
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 1
    finally:
        if aec_session:
            aec_session.unload()


def _make_aec_session(
    config: dict[str, Any],
    *,
    sink_master: int | str | None = None,
    source_master: int | str | None = None,
) -> EchoCancelSession:
    return EchoCancelSession(
        sink_name=nested(config, "pipewire", "echo_cancel_sink", default="Local Live Echo Cancellation Sink"),
        source_name=nested(config, "pipewire", "echo_cancel_source", default="Local Live Echo Cancellation Source"),
        capture_name=nested(config, "pipewire", "echo_cancel_capture", default="Local Live Echo Cancellation Capture"),
        playback_name=nested(config, "pipewire", "echo_cancel_playback", default="Local Live Echo Cancellation Playback"),
        latency=nested(config, "pipewire", "node_latency", default="1024/48000"),
        sink_master=sink_master,
        source_master=source_master,
    )


def _make_providers(config: dict[str, Any]) -> dict[str, Any]:
    common = {
        "num_ctx": int(nested(config, "llm", "num_ctx", default=8192)),
        "temperature": float(nested(config, "llm", "temperature", default=0.2)),
    }
    return {
        "local": OllamaLLM(
            base_url=nested(config, "llm", "local_base_url", default="http://127.0.0.1:11434"),
            model=nested(config, "llm", "local_model", default="auto"),
            max_tokens=int(nested(config, "llm", "max_tokens", default=96)),
            **common,
        ),
        "openrouter": OpenRouterLLM(
            base_url=nested(config, "llm", "openrouter_base_url", default="https://openrouter.ai/api/v1"),
            model=nested(config, "llm", "openrouter_model", default="openrouter/free"),
            credential_path=nested(config, "credentials", "openrouter_file", default="~/.config/credstore/openrouter.key"),
            max_tokens=int(nested(config, "llm", "openrouter_max_tokens", default=nested(config, "llm", "max_tokens", default=96))),
            **common,
        ),
    }


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _print_status(label: str, status: Any, path: Path) -> None:
    print(json.dumps({"command": label, "status": status, "result": str(path)}, ensure_ascii=False))
