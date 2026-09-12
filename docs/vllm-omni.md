# vLLM-Omni Phase 3B reference

Retrieved 2026-09-12 (JST) before the Phase 3B implementation.

## Official sources and pinned versions

- Repository: https://github.com/vllm-project/vllm-omni
- Repository `main` checked at commit `bc0c9f4b45c45c59aa2f92471842e8c18ae403ca`.
- Installed release used for the experiment: `vllm-omni==0.28.0`, repository tag commit `eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c`.
- Companion package: `vllm==0.28.0`.
- Stable API documentation: https://docs.vllm.ai/projects/vllm-omni/en/stable/serving/speech_api
- Pinned API source: https://raw.githubusercontent.com/vllm-project/vllm-omni/bc0c9f4b45c45c59aa2f92471842e8c18ae403ca/docs/serving/speech_api.md
- Pinned Qwen recipe source: https://raw.githubusercontent.com/vllm-project/vllm-omni/v0.28.0/recipes/Qwen/Qwen3-TTS.md
- Pinned deploy source: https://raw.githubusercontent.com/vllm-project/vllm-omni/v0.28.0/vllm_omni/deploy/qwen3_tts.yaml
- Pinned streaming client source: https://raw.githubusercontent.com/vllm-project/vllm-omni/bc0c9f4b45c45c59aa2f92471842e8c18ae403ca/examples/online_serving/text_to_speech/qwen3_tts/streaming_speech_client.py

The release was installed in the isolated environment
`/home/ws1/.venvs/local-live-vllm-omni-0.28.0`. The repository's existing
`.venv` was not modified. The resolver selected Python 3.12, Torch
`2.13.0+cu130`, Transformers `5.14.1`, and Triton `3.7.1` for the isolated vLLM environment.
The requested `main` SHA is a later `v0.29.0rc1-40` development checkout; its
installation documentation requires a matching vLLM 0.29.x line. The measured
runtime therefore used the coherent stable v0.28.0/vLLM 0.28.0 pair, while the
current main checkout was used for the pre-implementation specification review.

## Confirmed Qwen3-TTS serving contract

The official Qwen recipe documents Qwen3-TTS CustomVoice and states that
smaller `0.6B` CustomVoice variants are available. Phase 3B fixes the actual
request to:

- model: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- task: `CustomVoice`
- voice/speaker: `Ono_Anna`
- language: `Japanese`
- output baseline: 24 kHz mono signed 16-bit PCM/WAV

The server command from the recipe is the OpenAI-compatible speech server:

```text
vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
  --omni --host 127.0.0.1 --port 8091 --trust-remote-code
```

The Phase 3B runner resolves the bundled deploy file from the isolated
installation and records the exact path and command in the result. It does
not expose the server outside localhost.

Non-streaming requests use `POST /v1/audio/speech` and request a complete WAV.
Raw HTTP audio streaming uses the same endpoint with:

```json
{
  "stream": true,
  "stream_format": "audio",
  "response_format": "pcm",
  "speed": 1.0,
  "sample_rate": 24000
}
```

The official recipe's raw playback example specifies 24 kHz, signed 16-bit,
mono PCM. The client therefore preserves byte ordering across network chunk
boundaries and sends PCM to one persistent `pw-cat` playback process; it does
not start a new player for each network chunk. Raw HTTP streaming has no WAV
header, JSON envelope, application-level done marker, or sample-rate field;
HTTP EOF is the completion boundary, so the client sends/records `speed=1.0`
and `sample_rate=24000` explicitly.

## Chunking and WebSocket behavior

The pinned `qwen3_tts.yaml` has:

- `async_chunk: true`.
- Shared-memory connector `codec_streaming: true`.
- `initial_codec_chunk_frames: 1`.
- steady-state `codec_chunk_frames: 25`.
- stage 0 and stage 1 on device `0`.
- `gpu_memory_utilization: 0.3` per stage.

The server request field `initial_codec_chunk_frames` is benchmarked as
omitted/default, `1`, `2`, and `4` in a small probe. The default deploy config
is the first condition; this is not a broad parameter sweep.

The official WebSocket endpoint is `/v1/audio/speech/stream`. The protocol
accepts `session.config`, incremental `input.text`, `input.done`, and
`session.close`. Audio is sentence-scoped: `stream_audio=false` returns one
binary frame per sentence, while `stream_audio=true` emits one or more PCM
chunks. `input.done` flushes the buffered utterance and keeps the connection
open. This means it is not the same as arbitrary token-by-token audio
synthesis. Phase 3B prioritizes the documented HTTP raw PCM path; the
WebSocket incremental-text path is recorded but not made a prerequisite.
The inspected handler does not emit server-to-client `text.delta` events; text
must be generated upstream and sent by the client.

The official HTTP API does not expose a server-side request-received
monotonic timestamp. Phase 3B records response headers as the first
server-visible client boundary and leaves `server_request_received` null
rather than inventing a timestamp.

## Server lifecycle and limitations

The server is started on `127.0.0.1:8091`, readiness is checked with
`GET /v1/audio/voices`, and the model is treated as resident after readiness.
The public API does not provide an independent model-load timestamp. The runner
therefore parses the two stage `Model loading took ... seconds` log entries when
available and records their sum as a log-derived estimate; results always also
record:

- server process start
- server readiness
- combined start-to-ready duration
- `model_load_s` plus the stage values, or `null` when those log entries are unavailable

The server is stopped by the owner after the benchmark. A pre-existing ready
service on port 8091 is treated as a collision and is never terminated.

The Phase 3B measurements also record GPU memory before server start, at
readiness, during requests, after a Whisper-load attempt, and after server
stop. Existing Ollama is observed but not stopped or reconfigured. Any GPU
OOM or unavailable physical target is recorded as `blocked`/`error`; it is not
replaced with a successful-looking value.

The first smoke start used the official FlashInfer sampler default and loaded
the model weights, but failed before readiness because this host has no
`nvcc` and no `/usr/local/cuda`; FlashInfer attempted a JIT build. The log
does not show GPU OOM. The measured fallback starts only the isolated server
with:

```text
VLLM_USE_FLASHINFER_SAMPLER=0
```

This is a host-compatibility workaround, not a model, speaker, or sampling
quality change. The default-failure log is retained outside Git under
`results/artifacts/`.
