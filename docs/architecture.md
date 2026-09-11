# Architecture

## Vertical path

`PipeWire capture -> CPU energy VAD -> utterance-final faster-whisper -> LLM provider -> sentence chunker -> Qwen3-TTS -> PipeWire playback`

The first PoC deliberately closes an utterance after VAD rather than implementing incremental Whisper decoding. LLM text is streamed, but Qwen3-TTS is called once per bounded sentence chunk. The official Qwen3-TTS Python API is treated as non-online for this PoC; `first_audio_equivalent_seconds` is therefore request-to-complete-waveform, not a packet-stream latency claim.

## Providers

`LLMProvider` is a small streaming contract. `OllamaLLM` talks to the existing local `/api/chat` service and chooses an observed Qwen3.5 9B tag from `/api/tags` when configured as `auto`. `OpenRouterLLM` talks to `/api/v1/chat/completions` with the fixed `openrouter/free` model. Both emit text deltas, tool calls, completion, cancellation, and error events. The application executes only deterministic PoC mock tools and sends tool results back to the provider.

## Audio and AEC

PipeWire's installed `libpipewire-module-echo-cancel` is loaded temporarily through the host's Pulse-compatible `module-echo-cancel` API with `aec_method=webrtc`. The session resolves the physical USB sink/source to stable PipeWire/Pulse `node.name` targets (rather than transient numeric IDs), uses those as `sink_master`/`source_master`, and uses the generated Echo-Cancel Sink/Source names for playback/capture. AEC-off and AEC-on use the same generated reference WAV and physical USB targets. AEC metrics include aligned reference/recording correlation and RMS residual attenuation; silent or invalid captures are blocked rather than reported as zero performance.

## Measurement

Every benchmark JSON contains a UTC timestamp, hostname, CPU, GPU/driver, CUDA/Python/package versions, and measured/null metrics. Event timestamps use `time.monotonic_ns()`. GPU memory is sampled with `nvidia-smi`; CPU load uses psutil. Missing infrastructure is represented as `null`, `unavailable`, `blocked`, or `unsupported_or_failed`, never as fabricated performance.
