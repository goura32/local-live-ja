Phase 4: self-echo rejection

Scope

This phase adds a post-VAD echo-aware decision layer. PipeWire's WebRTC echo-cancel module remains the AEC implementation; this layer does not retune or replace AEC.

Decision path

AEC output -> VAD positive -> assistant playback reference match -> probable_self_echo or possible_user_speech

For each short window, the evaluator records:

- normalized reference correlation
- best correlation lag
- microphone/reference energy ratio
- residual energy ratio after the best-fit reference is removed

A candidate is rejected as probable self-echo only when playback is active, a reference is present, the measured lag is within the calibrated range, correlation is high, the residual energy is low, and the energy ratio is within the measured assistant-only envelope. Missing reference or inactive playback is never rejected by this layer.

The live boundary exposes the same decision through `LivePipeline.classify_vad_candidate(...)`; callers invoke it only after VAD is positive. The measured candidate is wired from `config/default.yaml` under `echo_rejection`, while the benchmark recalibrates and records thresholds in `results/bench_echo_rejection.json`.

Dataset

`results/bench_echo_rejection.json` contains 20 deterministic fixtures:

- 10 assistant-only fixtures: delayed/scaled assistant reference plus low noise, with no user component.
- 10 synthetic-user-like fixtures: the same residual echo plus an independent injected signal.

The second class is a regression fixture for “do not reject unexplained additional energy”; it is not a physical double-talk success result and is not a production guarantee. Fixture WAVs are under `results/artifacts/` and are intentionally not tracked.

Threshold selection

Thresholds are selected by a small grid search over the measured fixture distributions, rather than being fixed before measurement. The selected values and per-row distributions are stored in the result JSON. They must be recalibrated when microphone/speaker placement, gain, room, sample rate, or VAD policy changes.

Current measured fixture result

- assistant-only VAD-positive fixtures: 10
- assistant-only echo rejects: 10
- assistant-only false accepts: 0 / 10 (0.0%)
- synthetic user-like accepted fixtures: 10 / 10 (100.0%)
- synthetic user-like false rejects: 0 / 10 (0.0%)
- calibrated correlation threshold: 0.9642
- calibrated residual-energy threshold: 0.9
- calibrated maximum lag: 66 ms

The mute-during-assistant-playback baseline would trivially produce zero assistant-only positives, but it is not adopted because it disables future barge-in candidates. The candidate implementation is the echo-aware path.

Limitations

These fixtures validate signal separation and regression behavior only. They do not establish physical room acoustics, human double-talk performance, or a production false-reject/false-accept guarantee. The next real-microphone phase must remeasure the distributions with the actual AEC output and preserve the same no-reference/inactive-playback safety behavior.
