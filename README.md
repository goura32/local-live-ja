# local-live-ja

## 概要

local-live-jaは、**GPT-Live-1そのものではありません**。GPT-Live-1のような低遅延・割り込み可能な音声会話体験を、ローカルモデル中心で再構成する日本語音声会話PoCです。OpenAI製品との互換性・同等性・品質保証を主張しません。

Phase 7で、固定済みのASR / LLM / TTS / AEC / echo rejection / cancellation / physical playback部品を、実利用向けの連続会話アプリへ統合しました。

## 主な特徴

- `local-live chat` による連続microphone captureとCPU VAD
- speech start / continuation / end、hangover、短すぎる発話の除外、最大発話時間制限
- `system → user → assistant` のbounded multi-turn conversation history
- LLM `TextDelta` → natural Japanese sentence chunk → TTSのincremental経路
- Live推奨経路: Qwen3-TTS vLLM-Omni `0.28.0` HTTP raw PCM streaming + persistent `pw-cat`
- official Python Qwen3-TTS backend fallback
- WebRTC AECとreference-aware echo rejectionをVAD後に適用
- assistant再生中もcaptureを止めず、synthetic application-level barge-inでcancel/recovery
- `local-live run` はone-shot診断・デバッグ用として維持
- 詳細telemetryはJSON artifactへ保存し、通常のchat stdoutを汚さない

## デモ動作イメージ

```text
continuous microphone
  → WebRTC AEC / echo rejection
  → persistent VAD
  → utterance finalization
  → faster-whisper
  → conversation history + Ollama streaming
  → SentenceChunker
  → vLLM-Omni TTS raw PCM streaming
  → persistent PipeWire playback
```

## 検証済み範囲

最新の無人受入結果は `overall_status: unattended_poc_complete` です。これは人間の主観評価を含まない、automation可能なPoC範囲の判定です。

- Phase 6 physical onset: fixed replay `100/100`、独立fixture `50/50`、negative control `30/30`、false positive `0/30`
- Phase 6 continuous stability: application `100/100`、physical measurement confirmation `100/100`
- Phase 6 resource: FD / child process / playback process / active HTTP monotonic growth `0`
- Phase 7 application acceptance: 60 scripted turns、application failure `0`
- Phase 7 multi-turn history、incremental LLM→TTS、VAD、synthetic barge-in 3位置、bounded recoveryを自動検証
- 最終pytest、compileall、`uv build`、`git diff --check`を実行

詳しい根拠は [`docs/results.md`](docs/results.md)、検証項目の一覧は [`docs/validation.md`](docs/validation.md)を参照してください。

## アーキテクチャ

```text
RealMicrophoneSource / FixtureAudioSource
  → SessionController + StreamingVAD
  → AEC後の候補判定 / echo rejection
  → WhisperASR
  → ConversationHistory
  → LivePipeline
      ├─ Ollama LLM TextDelta stream
      ├─ incremental SentenceChunker
      ├─ vLLM-Omni HTTP raw PCM TTS
      └─ persistent PipeWire PCM playback
             └─ cancel / queued PCM discard / spoken_text commit
```

設計図は [`docs/architecture.md`](docs/architecture.md) にあります。

## 必要環境

必須条件と実測環境は [`docs/installation.md`](docs/installation.md) に分けて記載しています。概略はLinux、Python 3.12以上、`uv`、NVIDIA GPU、PipeWire、Ollama、音声入力/出力デバイスです。vLLM-Omniは専用venvへ分離します。

## インストール

```bash
git clone https://github.com/goura32/local-live-ja.git
cd local-live-ja
uv sync --extra dev --extra voice
```

次にOllama model、Hugging Face model cache、vLLM-Omni専用venvを準備します。credentialはconfigへ書きません。詳細は [`docs/installation.md`](docs/installation.md) を参照してください。

## 初期設定

- 診断・既存benchmark: `config/default.yaml`
- 実利用推奨: `config/live.yaml`
- vLLM Pythonの自動解決: `LOCAL_LIVE_VLLM_PYTHON` または `~/.venvs/local-live-vllm-omni-0.28.0/bin/python`
- OpenRouterを使う場合もkey本体は保存・表示せず、既定のcredential fileまたは環境から実行プロセスへだけ渡します

## 使い方

```bash
# 環境と依存関係を確認
uv run local-live doctor

# 実利用の連続会話。デフォルトでconfig/live.yamlを選ぶ
uv run local-live chat

# 明示的にprofileを指定
uv run local-live --config config/live.yaml chat

# one-shot診断（固定時間録音または--input-wav）
uv run local-live run
uv run local-live run --input-wav path/to/input.wav
```

`chat`は`Ctrl-C`またはSIGTERMで停止します。`--provider local|openrouter`、`--tts-backend vllm_omni|python`、`--no-aec`、`--start-vllm`、`--max-turns N`を指定できます。`--max-turns`は自動smoke test向けです。サービス停止、audio settings復元、owned vLLM停止は終了処理で行います。

日常利用者向けの詳細は [`docs/usage.md`](docs/usage.md) を参照してください。

## ベンチマークと受入試験

```bash
uv run local-live bench app
uv run local-live bench physical-onset
uv run local-live bench unattended
uv run python bench/summarize_results.py
```

`bench app`は研究用の新しい物理benchmarkではなく、同じ`SessionController` / `LivePipeline`を使ったPhase 7 application acceptanceです。hardware不要のsynthetic fixtureを使います。Phase 6の物理測定は既存結果を回帰として保持します。

## 既知の制約

- 本番用途、医療用途、安全critical用途向けではありません。
- Linux / PipeWire中心で、NVIDIA GPUを前提に実測しています。
- GPU memory余裕が小さい固定構成があります。
- OpenRouter/freeはactual model・出力・TTFTがrequestごとに変動します。
- model output、ASR品質、AEC品質、TTS自然性を保証しません。
- 人間の発話品質、physical double-talk、physical barge-in体験、MOSは評価していません。

完全な一覧は [`docs/limitations.md`](docs/limitations.md) にあります。

## 無人検証範囲と人間による未評価項目

自動化可能な実装・統合・resource cleanup・error recovery・公開前auditを検証し、主観が必要な項目は`deferred_manual`として結果へ明示しています。これは未実装という意味ではなく、このPoCの無人受入範囲外という意味です。

## ドキュメント一覧

- [`docs/architecture.md`](docs/architecture.md): 連続会話の構成図と責務
- [`docs/installation.md`](docs/installation.md): 第三者向けLinux導入
- [`docs/usage.md`](docs/usage.md): 日常利用とCLI
- [`docs/validation.md`](docs/validation.md): 検証カテゴリと状態
- [`docs/results.md`](docs/results.md): 最新結果とPhase 1〜7の測定履歴
- [`docs/limitations.md`](docs/limitations.md): 公開時の制約
- [`docs/vllm-omni.md`](docs/vllm-omni.md): vLLM-Omni固定仕様
- [`docs/echo-rejection.md`](docs/echo-rejection.md): AEC後echo rejection
- [`docs/physical-onset.md`](docs/physical-onset.md): physical onset measurement
- [`docs/unattended-validation.md`](docs/unattended-validation.md): 無人検証protocol
- [`docs/official-sources.md`](docs/official-sources.md): 参照公式source

## ライセンス

Apache License 2.0。詳細は [`LICENSE`](LICENSE) を参照してください。
