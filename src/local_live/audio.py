from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio_metrics import aligned_correlation, residual_echo_db, signal_rms


@dataclass(frozen=True)
class AudioNode:
    node_id: int
    name: str
    kind: str


@dataclass
class PipeWireInventory:
    sinks: list[AudioNode]
    sources: list[AudioNode]
    devices: list[str]
    raw_status: str = ""

    @classmethod
    def discover(cls) -> "PipeWireInventory":
        status = _run(["wpctl", "status"])[1]
        sinks = _parse_nodes(status, "Sinks:")
        sources = _parse_nodes(status, "Sources:")
        devices = []
        for line in status.splitlines():
            if "Devices:" in line:
                continue
            match = re.search(r"\s+\d+\.\s+(.+?)\s+\[alsa\]", line)
            if match:
                devices.append(match.group(1).strip())
        # ALSA enumeration is useful when WirePlumber abbreviates the device.
        for command in (["arecord", "-l"], ["aplay", "-l"]):
            output = _run(command)[1]
            devices.extend(
                line.strip()
                for line in output.splitlines()
                if "USB Audio" in line or "GoStream" in line
            )
        return cls(sinks=sinks, sources=sources, devices=sorted(set(devices)), raw_status=status)

    def usb_microphone(self) -> AudioNode | None:
        return _prefer_usb(self.sources)

    def usb_speaker(self) -> AudioNode | None:
        return _prefer_usb(self.sinks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sinks": [node.__dict__ for node in self.sinks],
            "sources": [node.__dict__ for node in self.sources],
            "devices": self.devices,
            "usb_microphone": self.usb_microphone().__dict__ if self.usb_microphone() else None,
            "usb_speaker": self.usb_speaker().__dict__ if self.usb_speaker() else None,
        }


def _prefer_usb(nodes: list[AudioNode]) -> AudioNode | None:
    if not nodes:
        return None
    explicit_usb = [node for node in nodes if "usb" in node.name.casefold()]
    if explicit_usb:
        return explicit_usb[0]
    gostream = [node for node in nodes if "gostream" in node.name.casefold()]
    return gostream[0] if gostream else nodes[0]


def _parse_nodes(status: str, section_name: str) -> list[AudioNode]:
    lines = status.splitlines()
    section_index = next((i for i, line in enumerate(lines) if section_name in line), None)
    if section_index is None:
        return []
    result: list[AudioNode] = []
    for line in lines[section_index + 1 :]:
        if "endpoint" in line.casefold():
            break
        if re.search(r"(?:Sinks|Sources|Streams|Video|Settings)\s*$", line):
            break
        match = re.search(r"(?:[│├└]\s*)?\*?\s*(\d+)\.\s*(.+?)\s*$", line)
        if not match:
            continue
        name = re.sub(r"\s+\[vol:.*?\]\s*$", "", match.group(2)).strip()
        if name:
            result.append(AudioNode(int(match.group(1)), name, section_name[:-1].lower().rstrip("s")))
    return result


def build_echo_cancel_args(
    *,
    sink_name: str,
    source_name: str,
    capture_name: str,
    playback_name: str,
    latency: str = "1024/48000",
) -> str:
    return """{\n  library.name = aec/libspa-aec-webrtc\n  node.latency = %s\n  capture.props = { node.name = \"%s\" }\n  source.props = { node.name = \"%s\" }\n  sink.props = { node.name = \"%s\" }\n  playback.props = { node.name = \"%s\" }\n}""" % (
        latency,
        capture_name,
        source_name,
        sink_name,
        playback_name,
    )


def _pulse_identifier(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return identifier or "local_live_aec"


def build_pulse_echo_cancel_args(
    *,
    sink_name: str,
    source_name: str,
    sink_master: int | str | None = None,
    source_master: int | str | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """Build arguments for PipeWire's Pulse-compatible echo-cancel module."""
    options = [
        f"source_name={_pulse_identifier(source_name)}",
        f"sink_name={_pulse_identifier(sink_name)}",
    ]
    if source_master is not None:
        options.append(f"source_master={source_master}")
    if sink_master is not None:
        options.append(f"sink_master={sink_master}")
    options.extend(["aec_method=webrtc", f"rate={sample_rate}", f"channels={channels}"])
    return " ".join(options)


class EchoCancelSession:
    def __init__(
        self,
        *,
        sink_name: str,
        source_name: str,
        capture_name: str,
        playback_name: str,
        latency: str = "1024/48000",
        sink_master: int | str | None = None,
        source_master: int | str | None = None,
    ) -> None:
        self.sink_name = sink_name
        self.source_name = source_name
        self.capture_name = capture_name
        self.playback_name = playback_name
        self.latency = latency
        self.sink_master = sink_master
        self.source_master = source_master
        self.module_id: str | None = None
        self.sink_node_id: int | None = None
        self.source_node_id: int | None = None

    def load(self) -> dict[str, Any]:
        args = build_pulse_echo_cancel_args(
            sink_name=self.sink_name,
            source_name=self.source_name,
            sink_master=self.sink_master,
            source_master=self.source_master,
        )
        before = PipeWireInventory.discover()
        code, stdout, stderr = _run(["pactl", "load-module", "module-echo-cancel", args], timeout=20.0)
        if code != 0:
            raise RuntimeError(f"PipeWire echo-cancel module load failed: {stderr or 'unknown error'}")
        self.module_id = stdout.strip() or None
        deadline = time.monotonic() + 5.0
        inventory = before
        before_sink_ids = {node.node_id for node in before.sinks}
        before_source_ids = {node.node_id for node in before.sources}
        sink: AudioNode | None = None
        source: AudioNode | None = None
        while time.monotonic() < deadline:
            inventory = PipeWireInventory.discover()
            new_sinks = [node for node in inventory.sinks if node.node_id not in before_sink_ids]
            new_sources = [node for node in inventory.sources if node.node_id not in before_source_ids]
            sink = next((node for node in new_sinks if "echo-cancel" in node.name.casefold()), new_sinks[0] if new_sinks else None)
            source = next((node for node in new_sources if "echo-cancel" in node.name.casefold()), new_sources[0] if new_sources else None)
            if sink and source:
                break
            time.sleep(0.1)
        if not sink or not source:
            self.unload()
            raise RuntimeError("echo-cancel module loaded but new source/sink did not appear")
        self.sink_node_id = sink.node_id
        self.source_node_id = source.node_id
        return {
            "module_id": self.module_id,
            "module_loader": "pactl module-echo-cancel",
            "inventory": inventory.to_dict(),
            "args": args,
            "sink": sink.__dict__,
            "source": source.__dict__,
        }

    def unload(self) -> None:
        if self.module_id:
            _run(["pactl", "unload-module", self.module_id], timeout=10.0)
            self.module_id = None
        self.sink_node_id = None
        self.source_node_id = None

    def __enter__(self) -> "EchoCancelSession":
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.unload()


class PipeWirePlayback:
    """Cancellable playback process for the live path."""

    def __init__(self, target: int | str | None = None) -> None:
        self.target = target
        self._process: subprocess.Popen[str] | None = None

    def play(self, audio_path: str | Path, *, cancel_event: Any = None) -> dict[str, Any]:
        command = ["pw-play"]
        if self.target is not None:
            command += ["--target", str(self.target)]
        command.append(str(audio_path))
        try:
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as exc:
            raise RuntimeError(f"pw-play unavailable: {type(exc).__name__}") from exc
        while self._process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
                return {"path": str(audio_path), "cancelled": True}
            time.sleep(0.02)
        stdout, stderr = self._process.communicate()
        if self._process.returncode != 0:
            raise RuntimeError(f"pw-play failed: {stderr.strip() or 'unknown error'}")
        return {"path": str(audio_path), "cancelled": False, "stdout": stdout.strip()}


def _module_id(stdout: str) -> str | None:
    matches = re.findall(r"(?:module|id|object)\D*(\d+)", stdout, flags=re.IGNORECASE)
    return matches[-1] if matches else None


def _run(command: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", type(exc).__name__


def record_fixed(
    output_path: str | Path,
    *,
    target: int | str | None,
    duration_s: float,
    sample_rate: int = 16000,
    channels: int = 1,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["pw-record"]
    if target is not None:
        command += ["--target", str(target)]
    command += ["--rate", str(sample_rate), "--channels", str(channels), "--format", "s16", str(output)]
    started = time.monotonic_ns()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        raise RuntimeError(f"pw-record unavailable: {type(exc).__name__}") from exc
    try:
        time.sleep(max(0.0, duration_s))
    finally:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
    if process.returncode not in (0, -15, -signal_number("SIGTERM")) and not output.exists():
        raise RuntimeError(f"pw-record failed: {stderr.strip() or 'unknown error'}")
    if not output.exists() or output.stat().st_size <= 44:
        raise RuntimeError(
            f"pw-record produced an empty file (returncode={process.returncode}): {stderr.strip() or 'unknown error'}"
        )
    return {"path": str(output), "duration_s": duration_s, "started_ns": started, "stdout": stdout.strip()}


def play_and_record(
    audio_path: str | Path,
    recording_path: str | Path,
    *,
    playback_target: int | str | None,
    capture_target: int | str | None,
    lead_s: float = 0.4,
    tail_s: float = 0.5,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    info = sf.info(str(audio_path))
    record_duration = float(info.duration) + lead_s + tail_s
    output = Path(recording_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    record_command = ["pw-record"]
    if capture_target is not None:
        record_command += ["--target", str(capture_target)]
    record_command += ["--rate", str(sample_rate), "--channels", "1", "--format", "s16", str(output)]
    play_command = ["pw-play"]
    if playback_target is not None:
        play_command += ["--target", str(playback_target)]
    play_command.append(str(audio_path))
    recorder = subprocess.Popen(record_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(lead_s)
    started = time.monotonic_ns()
    player = subprocess.run(play_command, capture_output=True, text=True, timeout=max(10.0, info.duration + 10.0), check=False)
    remaining = max(0.0, record_duration - lead_s - (time.monotonic_ns() - started) / 1e9)
    if remaining:
        time.sleep(remaining)
    recorder.terminate()
    try:
        rec_stdout, rec_stderr = recorder.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        recorder.kill()
        rec_stdout, rec_stderr = recorder.communicate()
    if player.returncode != 0:
        raise RuntimeError(f"pw-play failed: {player.stderr.strip() or 'unknown error'}")
    if not output.exists() or output.stat().st_size <= 44:
        raise RuntimeError(
            f"pw-record produced an empty file (returncode={recorder.returncode}): {rec_stderr.strip() or 'unknown error'}"
        )
    return {
        "path": str(output),
        "reference": str(audio_path),
        "duration_s": record_duration,
        "playback_target": playback_target,
        "capture_target": capture_target,
        "record_returncode": recorder.returncode,
        "record_stdout": rec_stdout.strip(),
        "record_stderr": rec_stderr.strip(),
    }


def compare_aec_recordings(
    reference_path: str | Path,
    off_path: str | Path,
    on_path: str | Path,
) -> dict[str, Any]:
    reference, reference_rate = sf.read(str(reference_path), always_2d=False)
    off, off_rate = sf.read(str(off_path), always_2d=False)
    on, on_rate = sf.read(str(on_path), always_2d=False)
    off_corr = aligned_correlation(reference, off, sample_rate=reference_rate, recording_rate=off_rate)
    on_corr = aligned_correlation(reference, on, sample_rate=reference_rate, recording_rate=on_rate)
    off_rms = signal_rms(off)
    on_rms = signal_rms(on)
    return {
        "reference_recording_correlation_off": off_corr,
        "reference_recording_correlation_on": on_corr,
        "residual_echo_attenuation_db": residual_echo_db(off, on),
        "recording_rms": {"off": off_rms, "on": on_rms},
        "measurement_valid": off_rms > 1e-6 and on_rms > 1e-6,
        "recording_sample_rates": {"reference": reference_rate, "off": off_rate, "on": on_rate},
    }


def _signal_number(name: str) -> int:
    import signal

    return int(getattr(signal, name, 15))
