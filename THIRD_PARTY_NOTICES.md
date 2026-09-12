# Third-party notices

このrepositoryの自作コードはApache License 2.0です。ただし、以下の外部software、model、service、音声処理moduleはそれぞれのlicense・利用規約・model cardに従います。model weightはrepositoryへ含めていません。再配布や商用利用の前に、リンク先の最新条件を確認してください。

| 区分 | upstream | 確認先 |
| --- | --- | --- |
| ASR runtime | faster-whisper | https://github.com/SYSTRAN/faster-whisper |
| ASR model | Whisper large-v3-turbo | https://huggingface.co/Systran/faster-whisper-large-v3-turbo |
| LLM runtime | Ollama | https://github.com/ollama/ollama |
| LLM model | Qwen3.5 | https://huggingface.co/Qwen |
| TTS model/runtime | Qwen3-TTS | https://github.com/QwenLM/Qwen3-TTS |
| TTS serving | vLLM | https://github.com/vllm-project/vllm |
| TTS serving extension | vLLM-Omni | https://github.com/vllm-project/vllm-omni |
| audio graph | PipeWire | https://pipewire.org/ |
| echo cancellation | WebRTC Audio Processing | https://webrtc.googlesource.com/src/+/main/modules/audio_processing/ |
| optional remote LLM | OpenRouter | https://openrouter.ai/terms |

external componentのversion固定値と実測構成は[`docs/official-sources.md`](docs/official-sources.md)および[`docs/vllm-omni.md`](docs/vllm-omni.md)にまとめています。各upstreamのlicense本文をこのrepositoryへ再掲しているわけではありません。
