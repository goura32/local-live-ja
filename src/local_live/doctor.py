from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .audio import PipeWireInventory
from .config import load_config, load_openrouter_key, nested
from .llm.ollama import OllamaLLM
from .llm.openrouter import OpenRouterLLM
from .telemetry import environment_snapshot, nvidia_smi, safe_run, write_json


PACKAGE_IMPORTS = {
    "numpy": "numpy",
    "soundfile": "soundfile",
    "httpx": "httpx",
    "faster-whisper": "faster_whisper",
    "qwen-tts": "qwen_tts",
    "torch": "torch",
    "sounddevice": "sounddevice",
    "webrtcvad": "webrtcvad",
}


def run_doctor(config: dict[str, Any], *, result_path: str | Path = "results/doctor.json") -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "local-live-ja/doctor/v1",
        "environment": environment_snapshot(),
        "checks": {},
        "warnings": [],
        "failures": [],
    }
    result["checks"]["gpu_cuda"] = _gpu_check()
    result["checks"]["python_dependencies"] = _dependency_check()
    result["checks"]["ollama"] = _ollama_check(config)
    result["checks"]["pipewire_usb_audio"] = _pipewire_check()
    result["checks"]["aec_module"] = _aec_module_check()
    result["checks"]["openrouter_credential"] = _openrouter_check(config)
    result["checks"]["git_github"] = _git_github_check()

    for name, check in result["checks"].items():
        status = check.get("status")
        if status == "fail":
            result["failures"].append(name)
        elif status == "warn":
            result["warnings"].append(name)
    result["status"] = "fail" if result["failures"] else ("warn" if result["warnings"] else "pass")
    write_json(result_path, result)
    return result


def _gpu_check() -> dict[str, Any]:
    rows = nvidia_smi("name,driver_version,memory.total,compute_cap")
    torch_info: dict[str, Any] = {"installed": False, "cuda_available": None, "torch_cuda": None}
    try:
        import torch

        torch_info = {
            "installed": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except ImportError:
        pass
    except Exception as exc:
        torch_info = {"installed": True, "cuda_available": False, "error_type": type(exc).__name__}
    status = "pass" if rows and (not torch_info["installed"] or torch_info.get("cuda_available")) else "warn"
    return {"status": status, "nvidia_smi": rows, "torch": torch_info, "nvcc": shutil.which("nvcc")}


def _dependency_check() -> dict[str, Any]:
    found = {name: bool(importlib.util.find_spec(module)) for name, module in PACKAGE_IMPORTS.items()}
    required = {name: found[name] for name in ("numpy", "soundfile", "httpx")}
    status = "pass" if all(required.values()) else "fail"
    return {"status": status, "found": found, "required_for_doctor": required}


def _ollama_check(config: dict[str, Any]) -> dict[str, Any]:
    base_url = nested(config, "llm", "local_base_url", default="http://127.0.0.1:11434")
    probe = OllamaLLM(base_url=base_url).probe()
    probe["status"] = "pass" if probe.get("reachable") else "warn"
    probe["binary"] = shutil.which("ollama")
    return probe


def _pipewire_check() -> dict[str, Any]:
    inventory = PipeWireInventory.discover()
    commands = {name: shutil.which(name) for name in ("pw-cli", "pw-dump", "wpctl", "pw-record", "pw-play", "arecord", "aplay")}
    mic = inventory.usb_microphone()
    speaker = inventory.usb_speaker()
    status = "pass" if mic and speaker else "warn"
    return {"status": status, "commands": commands, "inventory": inventory.to_dict()}


def _aec_module_check() -> dict[str, Any]:
    candidates = [
        "/usr/lib/x86_64-linux-gnu/pipewire-0.3/libpipewire-module-echo-cancel.so",
        "/usr/lib/aarch64-linux-gnu/pipewire-0.3/libpipewire-module-echo-cancel.so",
    ]
    module_paths = [path for path in candidates if Path(path).exists()]
    code, stdout, stderr = safe_run(["pactl", "list", "modules"], timeout=10.0)
    return {
        "status": "pass" if module_paths else "fail",
        "module_paths": module_paths,
        "loaded_in_current_server": "echo-cancel" in (stdout + stderr).casefold(),
        "pipewire_module_name": "libpipewire-module-echo-cancel",
        "aec_library": "aec/libspa-aec-webrtc",
        "pactl_probe_returncode": code,
    }


def _openrouter_check(config: dict[str, Any]) -> dict[str, Any]:
    credential_path = nested(config, "credentials", "openrouter_file", default="~/.config/credstore/openrouter.key")
    credential = load_openrouter_key(credential_path)
    file_exists = bool(credential.path and Path(credential.path).exists())
    result: dict[str, Any] = {
        "status": "pass" if credential.value else "warn",
        "credential_present": bool(credential.value),
        "credential_source": credential.reason if credential.reason in {"env", "ok"} else None,
        "file_exists": file_exists,
        "permission_ok": credential.permission_ok,
        "reason": credential.reason,
        "authentication": None,
    }
    if credential.value:
        probe = OpenRouterLLM(
            base_url=nested(config, "llm", "openrouter_base_url", default="https://openrouter.ai/api/v1"),
            model=nested(config, "llm", "openrouter_model", default="openrouter/free"),
            credential_path=credential_path,
        ).probe()
        result["authentication"] = {
            "authenticated": probe.get("authenticated"),
            "status_code": probe.get("status_code"),
            "error_type": probe.get("error_type"),
        }
        if not probe.get("authenticated"):
            # External rate limits/auth outages do not fail the local PoC.
            result["status"] = "warn"
    return result


def _git_github_check() -> dict[str, Any]:
    git_code, git_out, _ = safe_run(["git", "--version"])
    gh_code, gh_out, gh_err = safe_run(["gh", "auth", "status"], timeout=15.0)
    root_code, root_out, _ = safe_run(["git", "rev-parse", "--show-toplevel"])
    gh_authenticated = _github_auth_is_valid(gh_code, gh_out, gh_err)
    return {
        "status": "pass" if git_code == 0 and gh_authenticated else "warn",
        "git_version": git_out if git_code == 0 else None,
        "git_repo_root": root_out if root_code == 0 else None,
        "gh_installed": shutil.which("gh") is not None,
        "gh_auth_exit_code": gh_code,
        "github_authentication": gh_authenticated,
    }


def _github_auth_is_valid(exit_code: int, stdout: str, stderr: str) -> bool:
    if exit_code != 0:
        return False
    output = f"{stdout}\n{stderr}".casefold()
    return not any(marker in output for marker in ("failed to log in", "invalid", "not logged in", "expired"))
