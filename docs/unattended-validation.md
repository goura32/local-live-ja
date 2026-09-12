# Phase 5 unattended validation

This phase is fully unattended. It does not ask a person to speak, read a script, join physical double-talk, tune gain, listen subjectively, or approve production readiness. Human-only checks are recorded as `deferred_manual`.

## Fixed path

- ASR: `faster-whisper large-v3-turbo`, CUDA, `int8_float16`
- LLM: Ollama `qwen3.5:9b-q4_K_M`
- TTS: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, `Ono_Anna`, Japanese
- serving: `vllm-omni==0.28.0` + `vllm==0.28.0`
- transport: vLLM-Omni HTTP raw PCM streaming with persistent playback
- AEC: existing PipeWire WebRTC echo-cancel path; no replacement AEC

## Evidence-producing checks

`local-live bench unattended` runs the bounded checks and writes:

- `results/bench_echo_rejection.json`: 40 synthetic conditions, 20 assistant-only and 20 synthetic user-like, with offset, user/echo level, noise, lag, score distributions, signed threshold margins, and calibration/validation split.
- `results/bench_stability.json`: 100-turn continuous fixture run when the fixed path is available. It retains every measured, blocked, and failed turn, component timings, outlier classification, warm-state windows, restart runs, and 0.25-second per-turn process-resource samples (FDs, child/playback processes, live HTTP connections).
- `results/bench_interruption.json`: first-PCM, middle-playback, and end-playback cancellation cases, five repeats per timing, queue cleanup, HTTP cancellation state, spoken-text commit state, and next-turn recovery.
- `results/bench_unattended.json`: aggregate Phase 5 record including physical onset replay diagnosis, fault probes, resource lifecycle, VRAM margin, and `deferred_manual` items.
- `results/summary.json`: compact aggregate; detailed turn rows remain in the benchmark JSON files.

The echo check applies the Phase 4 threshold first. Any alternative threshold is calibrated using only the calibration partition and is reported separately on validation; the validation partition is never used to choose that alternative.

## Physical onset diagnosis

One fixed assistant reference is replayed at least 30 times without human participation. Each row retains playback return state, raw microphone RMS/peak, recording duration, expected and detected onset, onset timing error, reference cross-correlation, device availability, PipeWire health, threshold sensitivity, and a cause classification. Threshold sensitivity uses bounded 3x/4x/5x noise multipliers and does not change the production detector solely to increase success rate.

## Fault and cleanup policy

Fault probes are process-level or mock-only: HTTP stream disconnect, premature playback exit, Ollama request failure, invalid PCM, unavailable server, and unavailable microphone target. They do not unbind hardware or alter OS configuration. Every server, playback, capture, HTTP client, and volume guard is cleaned up in normal, exception, and cancellation paths. Results retain failed or blocked probes rather than converting them to success.

## Interpretation limits

Synthetic user-like acceptance demonstrates only that the independent fixture contains signal not explained by the assistant reference. It is not evidence of physical human double-talk, human barge-in, MOS, or production readiness. Those items stay `deferred_manual` and are not counted as unattended blockers.
