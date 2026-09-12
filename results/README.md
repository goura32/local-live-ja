# Results directory

Committed files are machine-readable benchmark summaries such as `doctor.json`, `bench_*.json`, `summary.json`, and the final pytest verification. Generated WAV files, full raw logs, model weights, caches, credentials, and `.venv` are ignored.

Benchmark JSON is v2. TTS and E2E results retain cold/warm boundaries, individual attempts, medians, actual provider model, and explicit null/blocked values. E2E failed attempts are retained instead of being silently dropped. AEC results resolve PipeWire node names/properties per run; transient node IDs in a result are runtime observations, not persistent configuration.

`results/history/` contains the baseline v1 JSON from commit `869809d969af98d094afddf753e16c25960bc555`, the pre-fix Ollama tool-calling diagnostic, and earlier serial E2E attempts. These files preserve the old 245-second E2E outlier, uncertain AEC attempts, and free-router failures for comparison.

The phase-2 final baseline is commit `a694324c53c5ab12455ad1806d232ba1f9eaf807` (phase-2 started from `39610a58e1cf16dc49ffa08b3f22f3cb34cb6c0e`). The pre-phase-3 `bench_tts.json`, `bench_live_latency.json`, and `summary.json` are retained under `results/history/phase2_baseline_a694324/` before phase-3 replacement. `bench_live_latency.json` also keeps the pre-active-capture run where a measurement-only lead was present, explicitly separated from the active-capture result.

`bench_live_latency.json` uses a raw USB microphone side-channel and reports synthetic handoff-to-acoustic
onset, not human speech latency. `bench_aec_matrix.json` reports an echo-only operating envelope; it must
not be interpreted as the absolute optimum microphone gain for human speech. `bench_playback_path.json`
measures the pure physical playback path with a known in-WAV onset, excluding TTS inference and generated-WAV
leading silence. Volume changes are bounded to
0–100% and guarded by snapshot/restore of volume, mute, and default sink/source state.

Paths in summaries point to local artifacts and are not distribution assets. Credential values are never written to results, logs, Markdown, or Git.

Phase 3B adds `bench_tts_serving.json` for the three fixed-model serving modes and
`bench_live_latency.json` for the final vLLM-Omni streaming E2E. The latter keeps
the Phase 3A Python baseline in `bench_live_latency_python.json` before replacing
the live-latency payload. Server readiness/model-load timing, raw PCM chunk timing,
initial-codec probes, quality/CER, VRAM snapshots, and cancellation/cleanup status
are committed as metadata. Generated WAV/PCM, raw microphone captures, vLLM server
logs, model weights, caches, credentials, and isolated virtual environments are not.
