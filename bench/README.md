# bench/

The executable benchmarks live in `src/local_live/bench.py` and are exposed through the `local-live bench ...` CLI. This directory is the requested benchmark entry-point area; generated audio and raw logs remain under the ignored `results/artifacts/` and `results/raw/` paths.

The current measurement contract is:

- `bench tts`: one GPU cold short run, then the same resident model with short/medium/long Japanese chunks and three warm runs per size. `first_audio_equivalent` includes cold model load; `warm_first_audio_equivalent` starts at inference.
- `bench e2e`: A/B/C/D serially, one full warm-up per configuration, three successful measured runs targeted, and up to nine attempts when a free-router response has no visible text. Failed attempts remain in `results/bench_e2e.json`; medians use successful measured rows only.
- Each E2E row contains ASR duration, LLM TTFT/total, TTS duration, physical playback roundtrip (null when not measured), total E2E duration, peak VRAM, stage timings, actual model, and assistant spoken-text roundtrip CER.
- `bench aec`: resolves current PipeWire nodes at runtime, performs raw USB capture first, and does not load AEC when RMS/peak is below the configured signal floor. OFF/ON recordings use unique run-specific filenames.

After running the benchmarks, refresh the compact machine-readable rollup with:

```text
cd ~/projects/local-live-ja
.venv/bin/python bench/summarize_results.py
```

Benchmark JSON uses `local-live-ja/bench-*/v2`. Old baseline and retry results are retained under `results/history/` rather than overwritten without a record.
