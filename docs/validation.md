# 検証項目

Phase番号順の開発日誌ではなく、公開PoCの実装単位で整理します。状態は次の意味です。

- `verified`: 自動契約またはmock/integrationで成立を確認
- `measured`: 実機・実サービスの測定結果をartifactへ保存
- `measured_with_limitations`: 測定は成立したが解釈上の制約がある
- `verified_synthetically`: scripted fixtureでapplication code pathを確認
- `deferred_manual`: 人間の主観・実人声が必要で今回の受入範囲外

| カテゴリ | 状態 | 方法 / 根拠 |
|---|---|---|
| ASR | measured | `large-v3-turbo`、GPU `int8_float16`、Phase 1/6の同一fixture回帰 |
| LLM | measured | Ollama `qwen3.5:9b-q4_K_M` stream、tool path、cancellation契約 |
| TTS | measured | official Python fallbackとvLLM-Omni `0.28.0` streaming |
| physical playback | measured | raw PCM、persistent `pw-cat`、Phase 6 physical onset |
| AEC | measured_with_limitations | WebRTC AEC module、audio state restore、physical matrix |
| echo rejection | verified_synthetically | assistant-only rejectとsynthetic user-like accept fixture |
| latency | measured_with_limitations | Phase 3B/6のstage latency、outlier保持 |
| cancellation | verified | playback stop、queued PCM discard、TTS/LLM cancel、recovery契約 |
| synthetic barge-in | verified_synthetically | first/middle/lateの3位置を各3回、各application state recovery |
| multi-turn | verified_synthetically | 60 scripted turns、system/user/assistant order、history forwarding/trim |
| incremental LLM→TTS | verified_synthetically | `TextDelta`受信中にfirst sentence TTS開始をtimestamp/eventで確認 |
| persistent capture state machine | verified_synthetically | callback source、StreamingVAD、start/continuation/end、hangover、max duration |
| fault recovery | verified_synthetically | temporary Ollama/TTS/playback failure、empty ASR、too-short、cancel |
| stability | measured | Phase 6 100-turn実機、Phase 7 60-turn application harness |
| resource leak | measured | FD、child/playback process、HTTP connection monotonic series |
| measurement reliability | measured_with_limitations | actual playback PCM reference、dual detector、alignment/negative controls |
| graceful shutdown | measured / verified | Phase 6 owned server/audio restoreとPhase 7 injected cleanup |
| public security audit | verified | reachable Git history、known-pattern/entropy/large-file scan |
| CI | verified | hardware-free pytest、compileall、uv build、diff check |

## 受入判定

Phase 7 application resultは次を満たした場合だけ`unattended_poc_complete`とします。

- continuous session state machine
- multi-turn bounded history
- incremental LLM→TTS
- vLLM streamingをLive default candidateとして利用
- synthetic application-level barge-in
- bounded cancellation/recovery
- 50〜100 turn application harness
- stale PCM、resource growth、process/port leakなし
- test/build/audit/fresh clone smoke

この判定は`production_ready`を意味しません。

## deferred_manual

今回の最終結果に残す人間依存項目は次だけです。

- human speech recognition quality
- human physical double-talk
- human physical barge-in experience
- MOS / subjective listening evaluation
- subjective TTS naturalness
- subjective microphone gain tuning
- production approval
