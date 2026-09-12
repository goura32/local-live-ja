# bench/

The executable benchmarks live in `src/local_live/bench.py` and are exposed through the `local-live bench ...` CLI. This directory is the requested benchmark entry-point area; generated audio and raw logs remain under the ignored `results/artifacts/` and `results/raw/` paths.

The current measurement contract is:

- `bench tts`: one GPU cold chars_8 run, then the same resident model with Japanese chunks of 5/8/12/20/about 30/about 50 characters and five warm runs per size. It also measures 5 representative responses with 0/50/100/150/200/250 ms safe-trim variants, Whisper CER/prefix regression, and official sampled-vs-cap/deterministic generation policies. `first_audio_equivalent` includes cold model load; `warm_first_audio_equivalent` starts at inference.
- `bench e2e`: A/B/C/D serially, one full warm-up per configuration, three successful measured runs targeted, and up to nine attempts when a free-router response has no visible text. Failed attempts remain in `results/bench_e2e.json`; medians use successful measured rows only.
- Each E2E row contains ASR duration, LLM TTFT/total, TTS duration, physical playback roundtrip (null when not measured), total E2E duration, peak VRAM, stage timings, actual model, and assistant spoken-text roundtrip CER.
- `bench aec`: resolves current PipeWire nodes at runtime, performs raw USB capture first, and does not load AEC when RMS/peak is below the configured signal floor. OFF/ON recordings use unique run-specific filenames.
- `bench live-latency`: measures synthetic WAV handoff to first physical assistant audio. The raw USB microphone recorder starts before the handoff and detects the acoustic onset with a noise-floor gate; it does not use a fixed sleep or the AEC source. Phase 3A additionally stores TTS inference/waveform/WAV-ready boundaries, generated-WAV onset, safe-trim metadata, a flat latency budget, ten measured runs, and representative first-chunk comparisons in `results/bench_live_latency.json`.
- `bench playback-path`: plays one low-level deterministic probe WAV five times and subtracts its known in-WAV signal start, isolating the physical PipeWire/USB/room/microphone path from TTS generation and generated-WAV leading silence in `results/bench_playback_path.json`.
- `bench aec-matrix`: serially measures the 25/50/75/100% speaker-volume × microphone-source-volume matrix as raw/OFF/ON, with clipping safety gates, assistant-only VAD false-trigger metrics, Whisper self-rerecognition, and three-repeat retests of up to three candidates. Pulse volume, mute, and default sink/source are restored in a finally-equivalent guard.
- `bench asr`: includes a CPU resident-model profile with one separately labeled first transcription and five warm transcriptions. Decode/resample, VAD, and faster-whisper inference have separate timings.
- `bench tts-serving`: compares the fixed Qwen3-TTS 0.6B CustomVoice model through the official Python API, vLLM-Omni non-streaming HTTP, and vLLM-Omni HTTP raw-PCM streaming. It starts the isolated vLLM-Omni server on `127.0.0.1:8091`, waits for `/v1/audio/voices`, runs five Japanese responses × five warm rows per mode, records TTFA/first actual PCM/physical onset/RTF/VRAM/CER, probes `initial_codec_chunk_frames` omitted/1/2/4, and stops the owned server in a finally-equivalent path. Streaming writes one continuous PCM stream to persistent `pw-cat`; it never starts a player per chunk.
- `bench live-latency --backend vllm_omni --streaming`: runs the final ten-measured-run synthetic-user → GPU Whisper → local Ollama → natural first sentence → vLLM-Omni raw PCM → persistent playback path. Up to `live_latency_max_attempts` attempts are retained when raw microphone onset is blocked.
- `bench stability`: generates eight short Japanese input fixtures, runs 50 resident-server turns, records every stage/resource/health/cancellation field, classifies component-median outliers, compares 10-turn warm-state windows, performs server restart plus three turns, and runs the real streaming interruption regression. Detailed rows are in `results/bench_stability.json`.
- `bench echo-rejection`: evaluates 10 assistant-only and 10 synthetic user-like post-VAD fixtures using calibrated reference correlation, lag, energy ratio, and residual energy. It does not disable VAD during playback and is not a physical double-talk test.
- `bench interruption`: starts the fixed resident vLLM server, interrupts one real streaming turn after first PCM, and records playback/HTTP/LLM cancellation cleanup in `results/bench_interruption.json`.
- `bench mic-readiness`: resolves raw USB/AEC targets, records short RMS/peak/clipping/VAD metrics, loads the fixed GPU ASR, and restores audio settings in `results/bench_mic_readiness.json`.

After running the benchmarks, refresh the compact machine-readable rollup with:

```text
cd ~/projects/local-live-ja
.venv/bin/python bench/summarize_results.py
```

Benchmark JSON uses `local-live-ja/bench-*/v2`; Phase 3B serving and live result payloads include official source commit/version, server readiness/model-load timing, sampler fallback override, async/default observations, raw PCM timing, and server/memory cleanup. Old baseline and retry results are retained under `results/history/` rather than overwritten without a record. Generated WAV/PCM, raw capture, model weights/cache, credentials, and server logs remain ignored.
