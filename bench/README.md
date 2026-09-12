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

After running the benchmarks, refresh the compact machine-readable rollup with:

```text
cd ~/projects/local-live-ja
.venv/bin/python bench/summarize_results.py
```

Benchmark JSON uses `local-live-ja/bench-*/v2`; phase-specific live-latency and AEC-matrix fields are versioned within their result payloads. Old baseline and retry results are retained under `results/history/` rather than overwritten without a record.
