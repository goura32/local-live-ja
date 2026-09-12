from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

try:
    import psutil
except ImportError:  # pragma: no cover - optional on minimal doctor hosts
    psutil = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_run(command: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", type(exc).__name__


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return platform.processor() or None
    return platform.processor() or None


def nvidia_smi(query: str) -> list[dict[str, str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    code, stdout, _ = safe_run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], timeout=15)
    if code != 0:
        return []
    fields = [field.strip() for field in query.split(",")]
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values)))
    return rows


def environment_snapshot() -> dict[str, Any]:
    gpu_rows = nvidia_smi("name,driver_version,memory.total,compute_cap")
    cuda_version = None
    try:
        import torch

        cuda_version = torch.version.cuda
    except Exception:
        pass
    return {
        "captured_at": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "cpu": {"model": _cpu_model(), "logical_cpus": psutil.cpu_count() if psutil else None},
        "gpu": gpu_rows,
        "cuda_runtime_from_torch": cuda_version,
        "packages": package_versions(
            ["local-live-ja", "faster-whisper", "qwen-tts", "torch", "numpy", "soundfile", "httpx", "psutil", "PyYAML"]
        ),
    }


def current_gpu_memory() -> list[int]:
    values = nvidia_smi("memory.used")
    result: list[int] = []
    for row in values:
        match = re.search(r"\d+", row.get("memory.used", ""))
        if match:
            result.append(int(match.group(0)))
    return result


@dataclass
class ResourceMonitor:
    interval_s: float = 0.1
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    cpu_samples: list[float] = field(default_factory=list, init=False)
    gpu_samples: list[list[int]] = field(default_factory=list, init=False)
    started_gpu: list[int] = field(default_factory=list, init=False)

    def __enter__(self) -> "ResourceMonitor":
        if psutil:
            psutil.cpu_percent(interval=None)
        self.started_gpu = current_gpu_memory()
        self._thread = threading.Thread(target=self._sample, name="local-live-resource-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _sample(self) -> None:
        while not self._stop.is_set():
            if psutil:
                self.cpu_samples.append(float(psutil.cpu_percent(interval=None)))
            self.gpu_samples.append(current_gpu_memory())
            self._stop.wait(self.interval_s)

    @property
    def cpu_load_percent(self) -> float | None:
        return sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else None

    @property
    def gpu_memory_peak_mib(self) -> int | None:
        values = [value for sample in self.gpu_samples for value in sample]
        return max(values) if values else (max(self.started_gpu) if self.started_gpu else None)

    @property
    def gpu_memory_delta_peak_mib(self) -> int | None:
        peak = self.gpu_memory_peak_mib
        if peak is None:
            return None
        baseline = max(self.started_gpu) if self.started_gpu else 0
        return max(0, peak - baseline)


@dataclass
class EventLog:
    events: list[dict[str, Any]] = field(default_factory=list)

    def mark(self, name: str, **payload: Any) -> dict[str, Any]:
        return self.mark_at(name, time.monotonic_ns(), **payload)

    def mark_at(self, name: str, monotonic_ns: int, **payload: Any) -> dict[str, Any]:
        item = {"event": name, "monotonic_ns": monotonic_ns, **payload}
        self.events.append(item)
        return item

    def as_dict(self) -> dict[str, Any]:
        return {"events": self.events}


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, output)
