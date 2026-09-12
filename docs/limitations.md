# 制約と非保証

local-live-jaは公開可能なPoCですが、本番運用を保証する製品ではありません。`unattended_poc_complete`は「人間の主観・実人声・physical double-talkを除き、自動化可能なPoC受入範囲を完了した」という意味です。

## 音声品質

- 人間音声による認識品質は評価していません。
- physical human double-talkは評価していません。
- physical human barge-inの体験・成功率は評価していません。
- MOS、subjective listening evaluation、subjective TTS naturalnessは実施していません。
- microphone gainやdevice placementのmanual tuningは行っていません。
- synthetic user-like fixtureのpassは、実部屋・実話者・実double-talkの保証ではありません。

## 環境

- Linux / PipeWire中心の実装です。
- NVIDIA GPUを前提とした実測で、CPU fallbackは速度・VRAM特性が異なります。
- 固定モデルを同時常駐する構成はVRAM余裕が小さい場合があります。OOMを保証しません。
- audio driver、PipeWire graph、USB device name、sample-rateは環境依存です。
- OpenRouter/freeはrouterが選ぶactual model、reasoning、visible output、TTFTがrequestごとに変動します。

## モデルと用途

- model outputの内容・安全性・正確性を保証しません。
- medical、legal、financial、安全critical用途へ使わないでください。
- production approvalは取得していません。
- `local-live run`はone-shot診断であり、`chat`の連続sessionとは目的が異なります。
- official Python Qwen3-TTSはfallbackで、vLLM-Omni経路と同一の初動特性ではありません。

## 無人検証の境界

Phase 6ではphysical playback、AEC、echo rejection、cancellation、resource cleanupを測定しました。Phase 7では同じapplication pathをscripted sourceで60 turn、barge-in、recovery、history、incremental TTSまで検証しました。ただし、次はPoCの無人検証範囲外です。

```text
human speech recognition quality
human physical double-talk
human physical barge-in experience
MOS / subjective listening evaluation
subjective TTS naturalness
subjective microphone gain tuning
production approval
```
