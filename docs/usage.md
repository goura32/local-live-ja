# 使い方

## doctor

```bash
uv run local-live doctor
```

GPU、Python package、Ollama endpoint、PipeWire inventory、audio targetなどを確認し、`results/doctor.json`へ保存します。hardwareがないCIでは失敗を隠さず、hardware checkの状態を記録します。

## 連続会話: `local-live chat`

```bash
uv run local-live chat
```

引数なしの`chat`はrepository rootから実行した場合、実利用推奨の`config/live.yaml`を選択します。profileを固定したい場合は次を使います。

```bash
uv run local-live --config config/live.yaml chat
```

起動時に次の情報を表示します。

```text
Local Live JA ASR: ...
LLM: ...
TTS: ...
AEC: ...
Microphone: ...
Speaker: ...
Listening...
```

通常実行中のdebug logは表示せず、turn/state/latency/errorの詳細は`results/chat_latest.json`と`results/artifacts/`へ保存します。

### chat options

```text
--provider local|openrouter
--tts-backend vllm_omni|python
--no-aec
--start-vllm
--max-turns N
```

- `--provider local`: 既定。Ollama `qwen3.5:9b-q4_K_M`。
- `--provider openrouter`: 交換可能なremote provider。free routerの出力・model・latencyは非決定的です。key本体をCLI引数に渡しません。
- `--tts-backend vllm_omni`: Live推奨。raw PCM streamingとpersistent `pw-cat`を使います。
- `--tts-backend python`: official Python Qwen3-TTS fallback。
- `--no-aec`: AECを無効化する診断用。通常利用では有効にします。
- `--start-vllm`: vLLM-Omniをこのchat processのownerとして起動し、終了時に停止します。既存portがあればcollisionで中断します。
- `--max-turns N`: 自動smoke test用のbounded終了。日常利用では省略します。

`Ctrl-C`、SIGTERM、内部fatal error、device/service failureでは、capture/playback/HTTP/AEC/owned processのcleanupを行って終了します。`chat`が自身でownerになっていない外部vLLMやOllamaを勝手に停止しません。

## one-shot診断: `local-live run`

`run`は既存の固定時間・one-shot診断です。連続会話には使いません。

```bash
uv run local-live run
uv run local-live run --duration 8
uv run local-live run --input-wav path/to/input.wav
uv run local-live run --provider local --no-aec
```

## ベンチマーク

```bash
uv run local-live bench app
uv run local-live bench physical-onset
uv run local-live bench phase6
uv run local-live bench unattended
uv run local-live bench interruption
uv run local-live bench stability
uv run local-live bench echo-rejection
uv run local-live bench mic-readiness
uv run local-live bench aec
uv run local-live bench asr
uv run local-live bench tts
uv run local-live bench tts-serving
uv run local-live bench llm
uv run local-live bench e2e
```

Phase 7の`bench app`は60 turnのapplication acceptanceです。synthetic inputを使い、human speechやphysical double-talkの成功を意味しません。hardware依存のcommandは利用可能なdeviceがない場合に`blocked`または`unavailable`として保存します。

## result確認

```bash
uv run python bench/summarize_results.py
python3 -m json.tool results/bench_app.json >/dev/null
```

大容量WAV、PCM、model weight、cache、credentialはGitへ追加しません。
