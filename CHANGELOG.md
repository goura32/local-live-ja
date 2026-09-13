# 変更履歴

## 0.1.1 — Final unattended PoC release

- Phase 7 session recovery hardening
- public release metadata completion
- CI / release consistency fixes

## 0.1.0 — Initial Phase 7 application snapshot

この版は、local-live-jaのPoCとしての一区切りです。GPT-Live-1との互換性・同等性を意味しません。

- `local-live chat`による連続microphone captureとSessionControllerを追加
- bounded VAD、multi-turn conversation history、incremental LLM→TTSを統合
- vLLM-Omni raw PCM streaming + persistent PipeWire playbackをLive推奨経路に設定
- WebRTC AEC、reference-aware echo rejection、synthetic application-level barge-inを統合
- `local-live bench app`で60 scripted turns、recovery、cancel、cleanup、resource checksを検証
- hardware-free GitHub Actions CI、portable config、公開前secret audit、Apache-2.0 LICENSEを追加

### 未評価

人間音声、physical human double-talk/barge-in、MOS、主観的音質、production approvalはこのPoCの無人検証範囲外です。詳細は[`docs/limitations.md`](docs/limitations.md)を参照してください。
