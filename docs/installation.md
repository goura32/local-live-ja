# インストール

## 対応範囲

本PoCの実測環境はLinux x86_64、Python 3.12、NVIDIA GPU、PipeWire、Ollama、USB microphone/speakerです。これは測定した環境であり、他のdistribution、GPU、audio driver、PipeWire graphでの動作を保証するものではありません。

必須条件は次のとおりです。

- Linux workstation
- Python 3.12以上
- `uv`
- NVIDIA GPUと互換CUDA runtime（推奨ASR/TTS経路）
- PipeWireと`pw-cat` / `pw-record`、音声入力・出力デバイス
- Ollamaと`qwen3.5:9b-q4_K_M`
- Hugging Faceへアクセスできる環境（ASR/TTS model取得時）
- vLLM-Omni `0.28.0`とvLLM `0.28.0`を入れた専用venv

CPU ASRとofficial Python Qwen3-TTSはfallbackとして残りますが、今回のLive推奨構成と同じ遅延・VRAM特性にはなりません。

## 依存コマンドの確認

```bash
python3 --version
uv --version
nvidia-smi
pipewire --version
pw-cat --version
ollama --version
```

`pw-cat`の表示形式はdistributionにより異なります。`doctor`は利用可能な範囲をJSONへ記録します。

## repositoryのインストール

```bash
git clone https://github.com/goura32/local-live-ja.git
cd local-live-ja
uv sync --extra dev --extra voice
```

`--extra voice`は`faster-whisper`、`qwen-tts`、`torch`、`sounddevice`、`webrtcvad`等を入れます。GPU向けwheelの選択は環境のCUDA/PyTorch互換性に依存します。依存解決に失敗した場合は、まずGPU vendorのPyTorch wheel条件を確認してください。

## model取得

ASR/TTSは初回実行時にmodel cacheへ取得される場合があります。cacheやweightはrepositoryへcommitしません。必要なら事前に各ライブラリの公式download手順で取得してください。

- ASR: `faster-whisper` の `large-v3-turbo`
- TTS: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- speaker: `Ono_Anna`
- language: `Japanese`

## Ollama

Ollamaを起動し、固定modelを準備します。

```bash
ollama serve
ollama pull qwen3.5:9b-q4_K_M
curl -fsS http://127.0.0.1:11434/api/tags
```

Ollamaは外部serviceとして扱います。`local-live`はcredentialをconfigへ埋め込みません。

## vLLM-Omni専用venv

vLLM-Omniはrepositoryの通常`.venv`へ混ぜず、ユーザーhome配下などportableな場所へ専用venvを作ります。実測で固定した組み合わせは`vllm-omni==0.28.0` + `vllm==0.28.0`です。install source・CUDA wheel・FlashInferの条件はvLLM-Omni公式recipeに従ってください。

概念的な手順は次のとおりです。

```bash
python3 -m venv ~/.venvs/local-live-vllm-omni-0.28.0
~/.venvs/local-live-vllm-omni-0.28.0/bin/python -m pip install -U pip
~/.venvs/local-live-vllm-omni-0.28.0/bin/python -m pip install "vllm==0.28.0" "vllm-omni==0.28.0"
```

PyTorch/CUDAの組み合わせは環境ごとに確認してください。`config/live.yaml`の`tts.vllm_python`は`auto`です。探索を固定したい場合は、ユーザー固有pathをconfigへ書かず、次のようにprocess環境だけへ渡します。

```bash
export LOCAL_LIVE_VLLM_PYTHON="$HOME/.venvs/local-live-vllm-omni-0.28.0/bin/python"
```

server recipe、request schema、host compatibility workaroundは [`docs/vllm-omni.md`](vllm-omni.md) にあります。

## vLLM server

Live TTSはlocalhostのvLLM-Omni serverを使います。既存serverを使う場合は公式recipe相当で次を起動します。

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
  --omni --host 127.0.0.1 --port 8091 --trust-remote-code
```

実際のentrypointは専用venvのinstall形態に合わせてください。`local-live chat --start-vllm`を使う場合は、アプリがownerとして起動し、終了時にそのprocessだけを停止します。外部serverを誤って停止しないため、port collisionはエラーにします。

## audio device

```bash
uv run local-live doctor
wpctl status
pactl list short sinks
pactl list short sources
```

`chat`はstable PipeWire/Pulse node nameを使い、USB microphoneとspeakerを解決します。手動gain tuningは検証範囲外です。

## 起動

```bash
uv run local-live doctor
uv run local-live chat
```

実利用profileを明示する場合:

```bash
uv run local-live --config config/live.yaml chat
```

通常のLive推奨値は、ASR `large-v3-turbo` GPU `int8_float16`、Ollama `qwen3.5:9b-q4_K_M`、Qwen3-TTS CustomVoice `Ono_Anna` / Japanese、vLLM-Omni HTTP raw PCM streaming、AEC enabledです。
