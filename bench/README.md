# bench/

The executable benchmark implementations live in `src/local_live/bench.py` and are exposed through the `local-live bench ...` CLI. This directory is kept as the requested benchmark entry-point area; generated audio and raw logs remain under the ignored `results/artifacts/` and `results/raw/` paths.

After running the benchmarks, refresh the compact machine-readable rollup with:

```text
uv run python bench/summarize_results.py
```
