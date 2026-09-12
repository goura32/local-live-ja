from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from collections.abc import Callable
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
    target: str | None = None


@dataclass
class PipeWireInventory:
    sinks: list[AudioNode]
    sources: list[AudioNode]
    devices: list[str]
    raw_status: str = ""

    @classmethod
    def discover(cls) -> "PipeWireInventory":
        status = _run(["wpctl", "status"])[1]
        wpctl_sinks = _parse_nodes(status, "Sinks:")
        wpctl_sources = _parse_nodes(status, "Sources:")
        pactl_sinks = _parse_pactl_nodes(_run(["pactl", "list", "sinks"])[1], "sink")
        pactl_sources = _parse_pactl_nodes(_run(["pactl", "list", "sources"])[1], "source")
        sinks = pactl_sinks or wpctl_sinks
        sources = pactl_sources or wpctl_sources
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
    explicit_usb = [
        node for node in nodes
        if "usb" in f"{node.name} {node.target or ''}".casefold()
        and "gostream" not in f"{node.name} {node.target or ''}".casefold()
    ]
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


def _parse_pactl_nodes(output: str, kind: str) -> list[AudioNode]:
    """Parse pactl objects and retain stable PipeWire/Pulse node names as targets."""
    header = "Sink" if kind == "sink" else "Source"
    result: list[AudioNode] = []
    current_id: int | None = None
    current_target: str | None = None
    current_description: str | None = None

    def flush() -> None:
        nonlocal current_id, current_target, current_description
        if current_id is not None and current_target and current_description:
            if kind != "source" or not current_target.endswith(".monitor"):
                result.append(AudioNode(current_id, current_description, kind, current_target))
        current_id = None
        current_target = None
        current_description = None

    for line in output.splitlines():
        m = re.match(rf"^{header} #(\d+)$", line.strip())
        if m:
            flush()
            current_id = int(m.group(1))
            continue
        stripped = line.strip()
        if stripped.startswith("Name:"):
            current_target = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Description:"):
            current_description = stripped.split(":", 1)[1].strip()
    flush()
    return result


def stable_target(node: AudioNode | None) -> str:
    """Return a persistent Pulse/PipeWire name, never a runtime object ID."""
    if node is None or not node.target or not node.target.strip():
        raise ValueError("audio node has no stable target name")
    target = node.target.strip()
    if target.isdecimal():
        raise ValueError("numeric PipeWire node IDs are not stable audio targets")
    return target


@dataclass(frozen=True)
class PulseVolumeState:
    target: str
    volume_percent: float
    muted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "volume_percent": self.volume_percent,
            "muted": self.muted,
        }


@dataclass(frozen=True)
class AudioStateSnapshot:
    speaker: PulseVolumeState
    microphone: PulseVolumeState
    default_sink: str
    default_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker.to_dict(),
            "microphone": self.microphone.to_dict(),
            "default_sink": self.default_sink,
            "default_source": self.default_source,
        }


def _require_pactl_result(
    runner: Callable[..., tuple[int, str, str]], command: list[str]
) -> str:
    code, stdout, stderr = runner(command, timeout=10.0)
    if code != 0:
        raise RuntimeError(f"pactl command failed: {command[1] if len(command) > 1 else 'unknown'}")
    return stdout.strip()


def _parse_pulse_percent(output: str) -> float:
    match = re.search(r"/\s*([0-9]+(?:\.[0-9]+)?)%", output)
    if not match:
        raise ValueError("pactl volume output did not contain a percentage")
    return float(match.group(1))


def _parse_pulse_mute(output: str) -> bool:
    lowered = output.casefold()
    if re.search(r"\b(?:yes|true)\b", lowered) or "はい" in output:
        return True
    if re.search(r"\b(?:no|false)\b", lowered) or "いいえ" in output:
        return False
    raise ValueError("pactl mute output did not contain a boolean")


def _query_pulse_volume_state(
    target: str,
    *,
    kind: str,
    runner: Callable[..., tuple[int, str, str]],
) -> PulseVolumeState:
    if kind not in {"sink", "source"}:
        raise ValueError(f"unsupported Pulse kind: {kind}")
    volume = _require_pactl_result(runner, ["pactl", f"get-{kind}-volume", target])
    mute = _require_pactl_result(runner, ["pactl", f"get-{kind}-mute", target])
    return PulseVolumeState(target, _parse_pulse_percent(volume), _parse_pulse_mute(mute))


class AudioVolumeGuard:
    """Snapshot and restore Pulse volume/mute/default state around measurements."""

    def __init__(
        self,
        *,
        speaker_target: str,
        microphone_target: str,
        command_runner: Callable[..., tuple[int, str, str]] | None = None,
    ) -> None:
        if not speaker_target or not microphone_target:
            raise ValueError("speaker and microphone targets are required")
        self.speaker_target = speaker_target
        self.microphone_target = microphone_target
        self._runner = command_runner or _run
        self.snapshot: AudioStateSnapshot | None = None
        self.restore_error: str | None = None
        self._restored = False

    def __enter__(self) -> "AudioVolumeGuard":
        self.snapshot = AudioStateSnapshot(
            speaker=_query_pulse_volume_state(self.speaker_target, kind="sink", runner=self._runner),
            microphone=_query_pulse_volume_state(self.microphone_target, kind="source", runner=self._runner),
            default_sink=_require_pactl_result(self._runner, ["pactl", "get-default-sink"]),
            default_source=_require_pactl_result(self._runner, ["pactl", "get-default-source"]),
        )
        if not self.snapshot.default_sink or not self.snapshot.default_source:
            raise RuntimeError("pactl did not return default sink/source")
        return self

    @staticmethod
    def _validate_percent(value: int | float) -> int:
        if isinstance(value, bool) or value < 0 or value > 100:
            raise ValueError("volume must be between 0 and 100 percent")
        return int(value)

    def set_volumes(self, *, speaker_percent: int | float, microphone_percent: int | float) -> None:
        speaker = self._validate_percent(speaker_percent)
        microphone = self._validate_percent(microphone_percent)
        self._run_checked(["pactl", "set-sink-volume", self.speaker_target, f"{speaker}%"])
        self._run_checked(["pactl", "set-source-volume", self.microphone_target, f"{microphone}%"])

    def set_mutes(self, *, speaker_muted: bool, microphone_muted: bool) -> None:
        self._run_checked(["pactl", "set-sink-mute", self.speaker_target, "yes" if speaker_muted else "no"])
        self._run_checked(["pactl", "set-source-mute", self.microphone_target, "yes" if microphone_muted else "no"])

    def _run_checked(self, command: list[str]) -> None:
        code, _stdout, _stderr = self._runner(command, timeout=10.0)
        if code != 0:
            raise RuntimeError(f"pactl command failed: {command[1]}")

    def restore(self) -> None:
        if self.snapshot is None or self._restored:
            return
        errors: list[str] = []
        snapshot = self.snapshot
        commands = [
            ["pactl", "set-sink-volume", snapshot.speaker.target, f"{snapshot.speaker.volume_percent:g}%"],
            ["pactl", "set-source-volume", snapshot.microphone.target, f"{snapshot.microphone.volume_percent:g}%"],
            ["pactl", "set-sink-mute", snapshot.speaker.target, "yes" if snapshot.speaker.muted else "no"],
            ["pactl", "set-source-mute", snapshot.microphone.target, "yes" if snapshot.microphone.muted else "no"],
            ["pactl", "set-default-sink", snapshot.default_sink],
            ["pactl", "set-default-source", snapshot.default_source],
        ]
        for command in commands:
            try:
                self._run_checked(command)
            except Exception as exc:
                errors.append(f"{command[1]}:{type(exc).__name__}")
        self._restored = True
        if errors:
            self.restore_error = ", ".join(errors)
            raise RuntimeError(f"audio state restore failed: {self.restore_error}")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            self.restore()
        except Exception:
            if exc_type is None:
                raise
        return False


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
        self.sink_target: str | None = None
        self.source_target: str | None = None

    def load(self) -> dict[str, Any]:
        if isinstance(self.sink_master, int) or isinstance(self.source_master, int):
            raise ValueError("AEC master targets must be stable names, not numeric node IDs")
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
        try:
            self.sink_target = stable_target(sink)
            self.source_target = stable_target(source)
        except ValueError:
            self.unload()
            raise
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
        self.sink_target = None
        self.source_target = None

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
        if self.target is not None and (not isinstance(self.target, str) or self.target.isdecimal()):
            raise ValueError("numeric PipeWire node IDs are not stable playback targets")
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


class PipeWirePCMPlayback:
    """Persistent raw PCM playback stream for incremental TTS audio."""

    def __init__(
        self,
        target: str,
        *,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        if not isinstance(target, str) or not target or target.isdecimal():
            raise ValueError("numeric PipeWire node IDs are not stable playback targets")
        self.target = target
        self._popen_factory = popen_factory
        self._process: subprocess.Popen[bytes] | None = None
        self.started_ns: int | None = None
        self.last_queued_ns: int | None = None

    @property
    def active(self) -> bool:
        return self._process is not None

    def start(self, *, sample_rate: int, channels: int) -> dict[str, Any]:
        if self._process is not None:
            raise RuntimeError("PCM playback stream already active")
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        command = [
            "pw-cat",
            "--playback",
            "--target",
            self.target,
            "--rate",
            str(sample_rate),
            "--channels",
            str(channels),
            "--format",
            "s16",
            "-",
        ]
        try:
            self._process = self._popen_factory(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise RuntimeError(f"pw-cat unavailable: {type(exc).__name__}") from exc
        process = self._process
        if process is None or process.stdin is None:
            self._process = None
            raise RuntimeError("pw-cat stdin was not created")
        self.started_ns = time.monotonic_ns()
        return {"started_ns": self.started_ns, "target": self.target, "command": command}

    def queue(self, payload: bytes) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("PCM playback stream is not active")
        if not payload:
            return {"queued": False}
        if process.poll() is not None:
            self._process = None
            raise RuntimeError("PCM playback stream exited before queue")
        try:
            process.stdin.write(payload)
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._process = None
            raise RuntimeError("PCM playback stream closed") from exc
        self.last_queued_ns = time.monotonic_ns()
        return {"queued": True}

    def finish(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("PCM playback stream is not active")
        try:
            process.stdin.close()
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        finally:
            self._process = None
        stdout_pipe = getattr(process, "stdout", None)
        stderr_pipe = getattr(process, "stderr", None)
        stdout = stdout_pipe.read() if stdout_pipe is not None else b""
        stderr = stderr_pipe.read() if stderr_pipe is not None else b""
        if process.returncode not in (0, None):
            error = stderr.decode(errors="replace").strip() if isinstance(stderr, bytes) else str(stderr).strip()
            raise RuntimeError(f"pw-cat failed: {error or 'unknown error'}")
        return {"cancelled": False, "stdout": stdout.decode(errors="replace").strip() if isinstance(stdout, bytes) else str(stdout).strip()}

    def cancel(self) -> dict[str, Any]:
        process = self._process
        if process is None:
            return {"cancelled": True, "already_inactive": True}
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        self._process = None
        return {"cancelled": True}


def _module_id(stdout: str) -> str | None:
    matches = re.findall(r"(?:module|id|object)\D*(\d+)", stdout, flags=re.IGNORECASE)
    return matches[-1] if matches else None


def _run(command: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    try:
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        env["LANG"] = "C"
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False, env=env)
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", type(exc).__name__


def record_fixed(
    output_path: str | Path,
    *,
    target: str | None,
    duration_s: float,
    sample_rate: int = 16000,
    channels: int = 1,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if target is not None and (not isinstance(target, str) or target.isdecimal()):
        raise ValueError("numeric PipeWire node IDs are not stable record targets")
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
    if process.returncode not in (0, -15, -_signal_number("SIGTERM")) and not output.exists():
        raise RuntimeError(f"pw-record failed: {stderr.strip() or 'unknown error'}")
    if not output.exists() or output.stat().st_size <= 44:
        raise RuntimeError(
            f"pw-record produced an empty file (returncode={process.returncode}): {stderr.strip() or 'unknown error'}"
        )
    return {"path": str(output), "duration_s": duration_s, "started_ns": started, "stdout": stdout.strip()}


def audio_file_stats(audio_path: str | Path) -> dict[str, Any]:
    """Return measured WAV duration and signal levels for capture validation."""
    path = Path(audio_path)
    info = sf.info(str(path))
    samples, sample_rate = sf.read(str(path), always_2d=False)
    array = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    from .audio_metrics import clipping_ratio

    return {
        "path": str(path),
        "sample_rate": int(sample_rate),
        "frames": int(info.frames),
        "duration_s": float(info.duration),
        "rms": signal_rms(array),
        "peak": peak,
        "clipping_ratio": clipping_ratio(array),
    }


class RawCaptureSession:
    """Keep a stable-name raw capture alive across a complete pipeline turn."""

    def __init__(self, output_path: str | Path, *, target: str, sample_rate: int = 16000) -> None:
        if not isinstance(target, str) or target.isdecimal():
            raise ValueError("numeric PipeWire node IDs are not stable capture targets")
        self.output_path = Path(output_path)
        self.target = target
        self.sample_rate = sample_rate
        self.process: subprocess.Popen[str] | None = None
        self.started_ns: int | None = None
        self._result: dict[str, Any] | None = None

    def start(self) -> dict[str, Any]:
        if self.process is not None:
            raise RuntimeError("raw capture already started")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "pw-record",
            "--target",
            self.target,
            "--rate",
            str(self.sample_rate),
            "--channels",
            "1",
            "--format",
            "s16",
            str(self.output_path),
        ]
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as exc:
            raise RuntimeError(f"pw-record unavailable: {type(exc).__name__}") from exc
        self.started_ns = time.monotonic_ns()
        return {"path": str(self.output_path), "target": self.target, "started_ns": self.started_ns}

    def stop(self, *, tail_s: float = 0.5) -> dict[str, Any]:
        if self._result is not None:
            return self._result
        process = self.process
        if process is None:
            raise RuntimeError("raw capture was not started")
        if tail_s > 0:
            time.sleep(tail_s)
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        ended_ns = time.monotonic_ns()
        if not self.output_path.exists() or self.output_path.stat().st_size <= 44:
            raise RuntimeError(
                f"pw-record produced an empty file (returncode={process.returncode}): {stderr.strip() or 'unknown error'}"
            )
        self._result = {
            "path": str(self.output_path),
            "target": self.target,
            "record_returncode": process.returncode,
            "record_stdout": stdout.strip(),
            "record_stderr": stderr.strip(),
            "timing_ns": {"record_start": self.started_ns, "record_end": ended_ns},
            "recording_stats": audio_file_stats(self.output_path),
        }
        return self._result


def playback_on_active_capture(audio_path: str | Path, *, playback_target: str, capture: RawCaptureSession) -> dict[str, Any]:
    """Play immediately on a capture already recording the physical side-channel."""
    if not isinstance(playback_target, str) or playback_target.isdecimal():
        raise ValueError("numeric PipeWire node IDs are not stable playback targets")
    if capture.process is None or capture.started_ns is None:
        raise RuntimeError("active raw capture must be started before playback")
    info = sf.info(str(audio_path))
    command = ["pw-play", "--target", playback_target, str(audio_path)]
    try:
        player = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        raise RuntimeError(f"pw-play unavailable: {type(exc).__name__}") from exc
    playback_started_ns = time.monotonic_ns()
    try:
        stdout, stderr = player.communicate(timeout=max(10.0, info.duration + 10.0))
    except subprocess.TimeoutExpired:
        player.kill()
        stdout, stderr = player.communicate()
        raise RuntimeError("pw-play timed out")
    playback_ended_ns = time.monotonic_ns()
    if player.returncode != 0:
        raise RuntimeError(f"pw-play failed: {stderr.strip() or 'unknown error'}")
    return {
        "path": str(audio_path),
        "playback_target": playback_target,
        "playback_stdout": stdout.strip(),
        "playback_stderr": stderr.strip(),
        "timing_ns": {
            "record_start": capture.started_ns,
            "pw_play_start": playback_started_ns,
            "playback_end": playback_ended_ns,
        },
        "duration_s": float(info.duration),
    }


def play_and_record(
    audio_path: str | Path,
    recording_path: str | Path,
    *,
    playback_target: str | None,
    capture_target: str | None,
    lead_s: float = 0.4,
    tail_s: float = 0.5,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    for label, target in (("playback", playback_target), ("capture", capture_target)):
        if target is not None and (not isinstance(target, str) or target.isdecimal()):
            raise ValueError(f"numeric PipeWire node IDs are not stable {label} targets")
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
    recorder: subprocess.Popen[str] | None = None
    player: subprocess.Popen[str] | None = None
    record_started_ns: int | None = None
    playback_started_ns: int | None = None
    playback_ended_ns: int | None = None
    record_ended_ns: int | None = None
    rec_stdout = ""
    rec_stderr = ""
    player_stdout = ""
    player_stderr = ""
    try:
        recorder = subprocess.Popen(record_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        record_started_ns = time.monotonic_ns()
        time.sleep(lead_s)
        player = subprocess.Popen(play_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        playback_started_ns = time.monotonic_ns()
        try:
            player_stdout, player_stderr = player.communicate(timeout=max(10.0, info.duration + 10.0))
        except subprocess.TimeoutExpired:
            player.kill()
            player_stdout, player_stderr = player.communicate()
            raise RuntimeError("pw-play timed out")
        playback_ended_ns = time.monotonic_ns()
        remaining = max(0.0, record_duration - lead_s - (playback_ended_ns - playback_started_ns) / 1e9)
        if remaining:
            time.sleep(remaining)
    finally:
        if recorder is not None:
            recorder.terminate()
            try:
                rec_stdout, rec_stderr = recorder.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                recorder.kill()
                rec_stdout, rec_stderr = recorder.communicate()
            record_ended_ns = time.monotonic_ns()
    if player is None or player.returncode != 0:
        raise RuntimeError(f"pw-play failed: {player_stderr.strip() or 'unknown error'}")
    if not output.exists() or output.stat().st_size <= 44:
        return_code = recorder.returncode if recorder is not None else None
        raise RuntimeError(
            f"pw-record produced an empty file (returncode={return_code}): {rec_stderr.strip() or 'unknown error'}"
        )
    return {
        "path": str(output),
        "reference": str(audio_path),
        "duration_s": record_duration,
        "playback_target": playback_target,
        "capture_target": capture_target,
        "record_returncode": recorder.returncode if recorder is not None else None,
        "record_stdout": rec_stdout.strip(),
        "record_stderr": rec_stderr.strip(),
        "playback_stdout": player_stdout.strip(),
        "playback_stderr": player_stderr.strip(),
        "timing_ns": {
            "record_start": record_started_ns,
            "pw_play_start": playback_started_ns,
            "playback_end": playback_ended_ns,
            "record_end": record_ended_ns,
        },
        "recording_stats": audio_file_stats(output),
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
