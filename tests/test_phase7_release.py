from __future__ import annotations

from pathlib import Path

from local_live.cli import build_parser
from local_live.config import load_config
from local_live.vllm_bench import _vllm_python


ROOT = Path(__file__).resolve().parents[1]


def test_chat_cli_exposes_continuous_options() -> None:
    args = build_parser().parse_args([
        "chat",
        "--provider",
        "local",
        "--tts-backend",
        "vllm_omni",
        "--no-aec",
        "--max-turns",
        "2",
        "--start-vllm",
    ])
    assert args.command == "chat"
    assert args.tts_backend == "vllm_omni"
    assert args.start_vllm is True


def test_public_configs_are_portable_and_live_uses_streaming() -> None:
    default = load_config(ROOT / "config/default.yaml")
    live = load_config(ROOT / "config/live.yaml")
    assert default["tts"]["vllm_python"] == "auto"
    assert live["tts"]["backend"] == "vllm_omni"
    assert live["tts"]["vllm_streaming"] is True
    for path in (ROOT / "config/default.yaml", ROOT / "config/live.yaml"):
        assert "/home/" not in path.read_text(encoding="utf-8")
        assert "/mnt/" not in path.read_text(encoding="utf-8")


def test_vllm_python_auto_is_resolved_without_user_path() -> None:
    resolved = _vllm_python(load_config(ROOT / "config/live.yaml"))
    assert resolved.name == "python"
    assert resolved.is_file()
