# Results directory

Committed files are machine-readable benchmark summaries such as `doctor.json`, `bench_*.json`, `summary.json`, and the final pytest verification. Generated WAV files, full raw logs, model weights, caches, credentials, and `.venv` are ignored.

Benchmark JSON is v2. TTS and E2E results retain cold/warm boundaries, individual attempts, medians, actual provider model, and explicit null/blocked values. E2E failed attempts are retained instead of being silently dropped. AEC results resolve PipeWire node names/properties per run; transient node IDs in a result are runtime observations, not persistent configuration.

`results/history/` contains the baseline v1 JSON from commit `869809d969af98d094afddf753e16c25960bc555`, the pre-fix Ollama tool-calling diagnostic, and earlier serial E2E attempts. These files preserve the old 245-second E2E outlier, uncertain AEC attempts, and free-router failures for comparison.

Paths in summaries point to local artifacts and are not distribution assets. Credential values are never written to results, logs, Markdown, or Git.
