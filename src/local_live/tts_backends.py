from __future__ import annotations

from typing import Any

from .config import nested
from .tts import Qwen3TTSEngine
from .vllm_omni_tts import VLLMOmniTTSEngine


def build_tts_backend(config: dict[str, Any], *, backend_override: str | None = None, streaming_override: bool | None = None) -> Any:
    """Build the selected TTS client without importing vLLM server code."""
    backend = backend_override or str(nested(config, "tts", "backend", default="python"))
    model = str(nested(config, "tts", "model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"))
    speaker = str(nested(config, "tts", "speaker", default="Ono_Anna"))
    language = str(nested(config, "tts", "language", default="Japanese"))
    if backend == "python":
        return Qwen3TTSEngine(
            model=model,
            speaker=speaker,
            language=language,
            device=str(nested(config, "tts", "device", default="auto")),
            max_new_tokens=int(nested(config, "tts", "max_new_tokens", default=2048)),
            generation_kwargs=dict(nested(config, "tts", "generation_kwargs", default={}) or {}),
        )
    if backend == "vllm_omni":
        initial = nested(config, "tts", "vllm_initial_codec_chunk_frames", default=None)
        return VLLMOmniTTSEngine(
            base_url=str(nested(config, "tts", "vllm_base_url", default="http://127.0.0.1:8091/v1")),
            model=model,
            speaker=speaker,
            language=language,
            timeout_s=float(nested(config, "tts", "vllm_timeout_s", default=300.0)),
            initial_codec_chunk_frames=int(initial) if initial is not None else None,
            streaming=bool(
                nested(config, "tts", "vllm_streaming", default=False)
                if streaming_override is None
                else streaming_override
            ),
        )
    raise ValueError(f"unsupported TTS backend: {backend}")
