from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .telemetry import current_gpu_memory
from .vllm_omni_tts import VLLMOmniTTSEngine


class VLLMOmniServer:
    """Own a local vLLM-Omni server process and never stop an external one."""

    def __init__(
        self,
        *,
        python_bin: str | Path,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8091,
        deploy_config: str | Path | None = None,
        log_path: str | Path = "results/artifacts/vllm_omni_server.log",
        gpu_memory_utilization: float | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.python_bin = Path(python_bin)
        self.model = model
        self.host = host
        self.port = port
        self.deploy_config = Path(deploy_config) if deploy_config else None
        self.log_path = Path(log_path)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.extra_env = dict(extra_env or {})
        self.process: subprocess.Popen[str] | None = None
        self.started_ns: int | None = None
        self.ready_ns: int | None = None
        self.owned = False
        self.health: dict[str, Any] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _resolved_deploy_config(self) -> Path:
        if self.deploy_config is not None:
            if not self.deploy_config.is_file():
                raise FileNotFoundError(f"vLLM-Omni deploy config not found: {self.deploy_config}")
            return self.deploy_config
        command = [
            str(self.python_bin),
            "-c",
            "import importlib.resources; print(importlib.resources.files('vllm_omni') / 'deploy' / 'qwen3_tts.yaml')",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        if completed.returncode != 0:
            raise RuntimeError("unable to resolve installed vLLM-Omni deploy config")
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        path = Path(next((line for line in reversed(lines) if line.endswith("qwen3_tts.yaml")), ""))
        if not path.is_file():
            raise FileNotFoundError(f"resolved vLLM-Omni deploy config not found: {path}")
        return path

    def command(self) -> list[str]:
        command = [
            str(self.python_bin),
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--deploy-config",
            str(self._resolved_deploy_config()),
            "--omni",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--trust-remote-code",
        ]
        if self.gpu_memory_utilization is not None:
            command += ["--gpu-memory-utilization", str(self.gpu_memory_utilization)]
        return command

    def start(self, *, timeout_s: float = 900.0, poll_s: float = 2.0) -> dict[str, Any]:
        client = VLLMOmniTTSEngine(base_url=self.base_url, model=self.model)
        existing = client.health()
        if existing.get("status") == "ready":
            raise RuntimeError(f"vLLM-Omni port already has a ready service: {self.base_url}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self.log_path.open("w", encoding="utf-8")
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env.update(self.extra_env)
            self.process = subprocess.Popen(
                self.command(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        except Exception:
            log_handle.close()
            raise
        self.started_ns = time.monotonic_ns()
        self.owned = True
        deadline = time.monotonic() + timeout_s
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"vLLM-Omni server exited before readiness (returncode={self.process.returncode})")
                health = client.health()
                if health.get("status") == "ready":
                    self.health = health
                    self.ready_ns = time.monotonic_ns()
                    return self.to_dict()
                time.sleep(poll_s)
            raise TimeoutError(f"vLLM-Omni server was not ready within {timeout_s:.1f}s")
        except Exception:
            self.stop()
            raise
        finally:
            log_handle.close()

    def stop(self) -> dict[str, Any]:
        process = self.process
        if process is None or not self.owned:
            return {"stopped": False, "owned": False}
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        result = {"stopped": True, "owned": True, "returncode": process.returncode}
        self.process = None
        self.owned = False
        return result

    def to_dict(self) -> dict[str, Any]:
        start_to_ready = (
            (self.ready_ns - self.started_ns) / 1e9
            if self.ready_ns is not None and self.started_ns is not None
            else None
        )
        model_load_stage_seconds = self._model_load_stage_seconds()
        return {
            "model": self.model,
            "base_url": self.base_url,
            "host": self.host,
            "port": self.port,
            "python": str(self.python_bin),
            "deploy_config": str(self._resolved_deploy_config()),
            "command": self.command(),
            "owned": self.owned,
            "server_start_ns": self.started_ns,
            "server_ready_ns": self.ready_ns,
            "server_start_to_ready_s": start_to_ready,
            "model_load_s": sum(model_load_stage_seconds) if model_load_stage_seconds else None,
            "model_load_stage_seconds": model_load_stage_seconds,
            "model_load_measurement": "sum of stage log values; independent server-side API timestamp unavailable",
            "health": self.health,
            "gpu_memory_at_ready_mib": current_gpu_memory(),
            "gpu_memory_utilization_override": self.gpu_memory_utilization,
            "environment_overrides": self.extra_env,
            "stop": None,
        }

    def _model_load_stage_seconds(self) -> list[float]:
        if not self.log_path.is_file():
            return []
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        values = re.findall(r"Model loading took [^\n]*? and ([0-9]+(?:\.[0-9]+)?) seconds", text)
        return [float(value) for value in values]
