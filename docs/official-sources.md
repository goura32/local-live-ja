# Official references and boundary notes

- Qwen3-TTS upstream: https://github.com/QwenLM/Qwen3-TTS
  - The README lists `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, Japanese support, and the `Ono_Anna` Japanese native speaker.
  - The implementation uses `Qwen3TTSModel.from_pretrained(..., device_map=..., dtype=...)` and `generate_custom_voice`; it does not implement a custom streaming engine.
- faster-whisper upstream: https://github.com/SYSTRAN/faster-whisper
  - The implementation uses `large-v3-turbo`, Japanese `language="ja"`, GPU `float16`/`int8_float16`, and CPU `int8`.
- PipeWire Echo Cancel: https://docs.pipewire.org/page_module_echo_cancel.html
  - The native module creates virtual echo-cancel source/sink and capture/playback streams; the configured engine is `aec/libspa-aec-webrtc`.
  - On this PipeWire 1.0.5 host, the working runtime entry point is the Pulse-compatible `module-echo-cancel` loaded with `pactl`, `aec_method=webrtc`, and detected USB master nodes. This wrapper invokes the installed `libpipewire-module-echo-cancel` implementation.
- PipeWire Pulse echo-cancel man page: `man 7 pipewire-pulse-module-echo-cancel`
  - Documents the runtime `source_master`, `sink_master`, `source_name`, `sink_name`, `aec_method`, `rate`, and `channels` options used by the PoC.
- Ollama chat API: https://docs.ollama.com/api/chat
  - The implementation uses NDJSON streaming, `tools`, and `options.num_ctx=8192`.
- OpenRouter API: https://openrouter.ai/docs/api/reference/overview
  - The implementation uses normalized OpenAI-compatible SSE streaming, tool definitions, and records the response `model` when supplied.

The existing `goura32/local-japanese-tts-benchmark` was inspected for reproducibility and license-boundary practices. No code dependency or copy is used here; Qwen3-TTS is called through its current upstream package/API.
