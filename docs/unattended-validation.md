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

## Phase 6 physical onset hardening

`local-live bench physical-onset` runs the fixed reference 100 times, five independent Japanese fixtures 10 times each, and 30 no-playback negative captures. It uses the actual PCM byte stream queued to persistent `pw-cat` as the alignment reference; HTTP chunk boundaries and generated WAV paths are not treated as authoritative. Each positive row stores playback evidence (first PCM, first actual speech offset, bytes/write count, process alive/exit, completion), raw microphone evidence (duration, RMS/peak/clipping, short-frame energy series, adaptive noise floor), and reference alignment evidence (best lag, correlation, confidence margin, expected window).

Detector A is the existing adaptive short-frame RMS detector. Detector B is a separate multi-window normalized-correlation detector. Final classifications distinguish `confirmed`, `correlation_recovered`, `energy_only`, `microphone_capture_failure`, `playback_failure`, `late_outside_window`, `no_physical_match`, `false_positive`, `no_playback_negative`, and `unknown`. `application_status` is recorded independently from `physical_measurement_status`, so a measurement miss is not counted as an application failure.

The expected onset window is derived from the checked-in physical path distribution (`results/bench_playback_path.json`) plus a bounded margin. Recording tail is derived from the same distribution and playback duration. Phase 6 does not change the Phase 4 echo threshold, ASR/LLM/TTS models, AEC, volume, or serving versions.

`local-live bench phase6` runs the physical protocol and the final 100-turn stability/restart protocol. The final run reached `unattended_validation_complete`: fixed replay 100/100 confirmed, fixtures 50/50 confirmed, negative false-positive rate 0/30, and stability application success 100/100 with physical confirmation 100/100. The detailed result is in `results/bench_unattended.json`; `results/turn57_diagnosis.json` records what the old turn 57 did not capture. The old fixed-name turn 57 WAV was reused by the later stability run, so the diagnosis file explicitly marks the original binary as not preserved.


Fault probes are process-level or mock-only: HTTP stream disconnect, premature playback exit, Ollama request failure, invalid PCM, unavailable server, and unavailable microphone target. They do not unbind hardware or alter OS configuration. Every server, playback, capture, HTTP client, and volume guard is cleaned up in normal, exception, and cancellation paths. Results retain failed or blocked probes rather than converting them to success.

## Interpretation limits

Synthetic user-like acceptance demonstrates only that the independent fixture contains signal not explained by the assistant reference. It is not evidence of physical human double-talk, human barge-in, MOS, or production readiness. Those items stay `deferred_manual` and are not counted as unattended blockers.

## Phase 7 application acceptance

Phase 7は新しい研究benchmarkではなく、実利用経路の最終受入試験です。`local-live chat`と同じ`SessionController` / `LivePipeline`を使い、`FixtureAudioSource`とinstrumented playbackだけを注入します。60 scripted turns、bounded conversation history、stream中のSentenceChunker→TTS、persistent capture state、first/middle/late各3回のsynthetic application-level barge-in、bounded recovery、cancel、cleanupを実行し、`results/bench_app.json`へcompact evidenceを保存します。

この結果の`unattended_poc_complete`は、実人声やphysical human barge-inを意味しません。Phase 7で無人検証を終了し、人間依存項目は`docs/limitations.md`の`deferred_manual`に限定します。
