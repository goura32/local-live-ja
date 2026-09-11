from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Credential:
    value: str | None
    reason: str
    path: str | None = None
    permission_ok: bool | None = None


def load_openrouter_key(path: str | Path | None = None) -> Credential:
    """Load a key without ever logging or printing its value."""
    env_value = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_value:
        return Credential(value=env_value, reason="env", permission_ok=None)

    key_path = Path(path or "~/.config/credstore/openrouter.key").expanduser()
    try:
        info = key_path.stat()
    except FileNotFoundError:
        return Credential(value=None, reason="missing", path=str(key_path), permission_ok=None)
    except OSError:
        return Credential(value=None, reason="unreadable", path=str(key_path), permission_ok=False)

    permission_ok = not bool(info.st_mode & (stat.S_IRWXG | stat.S_IRWXO))
    if not permission_ok:
        return Credential(value=None, reason="insecure_permissions", path=str(key_path), permission_ok=False)
    if not stat.S_ISREG(info.st_mode):
        return Credential(value=None, reason="not_regular_file", path=str(key_path), permission_ok=False)
    try:
        value = key_path.read_text(encoding="utf-8").strip()
    except OSError:
        return Credential(value=None, reason="unreadable", path=str(key_path), permission_ok=True)
    if not value:
        return Credential(value=None, reason="empty", path=str(key_path), permission_ok=True)
    return Credential(value=value, reason="ok", path=str(key_path), permission_ok=True)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or "config/default.yaml").expanduser()
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    loaded.setdefault("_path", str(config_path))
    return loaded


def nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value
