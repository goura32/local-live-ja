# local-live-ja 実測結果

この文書はCLIが生成したJSONを根拠に更新した。`null`、`blocked`、`unavailable`、`unsupported_or_failed`は0点ではなく、測定不能または未対応を表す。

## 判定

PoCのソフトウェア縦切りは成立した。合成ユーザー音声を入力し、VAD、Whisper Turbo、local OllamaまたはOpenRouter、文単位TTS、PipeWire playbackまで自動実行できた。ASR GPU/CPU比較、TTS GPU/CPU参考測定、LLM streaming/tool/cancel、4構成E2Eを実行済みである。

AECの実音響性能比較だけは成立していない。検出したUSB Audio mic/speakerでPipeWire WebRTC AEC moduleのload/unloadと仮想sink/source生成は確認できたが、AEC OFF/ONのcapture RMSがともに0.0だったため、相関、残留echo減衰、ASR自己再認識量は未測定として`blocked`にした。無音を性能値として扱っていない。

測定時刻は主に2026-09-11 18:00--18:10 UTC（2026-09-12 JST）。各値の完全なイベント時刻と環境snapshotは`results/*.json`、集約値は`results/summary.json`にある。

## 構成

`Microphone/PipeWire -> CPU energy VAD -> 発話終了後のutterance-final ASR -> LLM event stream -> sentence chunker -> Qwen3-TTS -> PipeWire playback`。

- VADはCPUのenergy方式。continuous incremental ASRは実装していない。
- ASRは`faster-whisper`の`large-v3-turbo`、`language=ja`。発話区間確定後に一括transcribeする。
- LLMは`LLMProvider`相当の共通イベント（text delta、tool call、tool result、completion、cancel、error）を使用し、`OllamaLLM`と`OpenRouterLLM`を差し替え可能にした。
- Ollamaは`/api/tags`で観測した`qwen3.5:9b-q4_K_M`を使用し、context targetは8192。モデル名を推測して固定していない。
- OpenRouterはrequested modelを`openrouter/free`に固定し、各completionのAPI返却`actual_model`を保存する。free routerの配下モデル比較はしていない。
- TTSは`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`、speaker=`Ono_Anna`、language=`Japanese`。公式Python APIで文チャンクごとに生成し、真のonline streamingは仮定していない。chunk上限48文字、timeout 0.8秒。
- AECは独自実装ではなく、`pactl load-module module-echo-cancel`経由でPipeWireの`libpipewire-module-echo-cancel.so`と`aec_method=webrtc`を使用する。これはこの環境で実際にnodeを生成できたPulse互換loaderである。
- `Cancellation`によりLLM generation、pending TTS、playbackを同じ外部制御で停止できる。double-talk/barge-in判定は対象外。

## 実環境・version

`doctor`の最終結果はstatus=`warn`（failureなし）だった。warnはGitHub CLI認証のみで、ローカル音声PoCの依存ではない。

- host: `ws1`
- OS/kernel: Linux x86_64、kernel `7.0.0-31-generic`、glibc 2.39
- CPU: Intel Core i7-13700、24 logical CPUs
- GPU: NVIDIA GeForce RTX 5070 Ti、driver 595.84、16,303 MiB、compute capability 12.0
- Python: 3.12.3
- PyTorch: 2.14.0、torch CUDA runtime 13.0。`nvcc`は未検出。
- faster-whisper 1.2.1、CTranslate2 4.8.2、transformers 4.57.3
- qwen-tts 0.1.1、numpy 2.5.3、soundfile 0.14.0、sounddevice 0.5.6、webrtcvad-wheels 2.0.14
- CUDA 12 compatibility wheels: nvidia-cublas-cu12 12.9.2.10、nvidia-cudnn-cu12 9.26.0.51、nvidia-cuda-nvrtc-cu12 12.9.86
- PipeWire: 1.0.5
- Ollama server API: 0.33.3（CLI binaryはPATH上には無かったが、既存server/APIは利用可能）
- Git 2.43.0、GitHub CLI 2.45.0
- OpenRouter credential: file存在、permission OK、API認証HTTP 200。値は表示・保存していない。
- PipeWire候補: USB Audio source node 71 / sink node 69、GoStream source 55 / sink 54等を検出。最終自動選択は明示名USB Audio。
- `flash-attn`は未導入、SoXは未検出。Qwen3-TTSはmanual PyTorch経路でベンチ完走した。

## ASR: synthetic ASR regression

これはQwen3-TTSで作った28.8秒の同一WAVをWhisperへ戻す回帰試験であり、実マイク音声の認識精度やMOSではない。referenceには通常会話、日本語数字、英数字、GPU/CUDA/Docker/Ollama/Python、日付・時刻、GPU/USB microphone語を含めた。CERはUnicode正規化後に計算した。

| mode | device / compute | elapsed (s) | RTF | CER | GPU peak (MiB) | ASR増分peak (MiB) |
|---|---|---:|---:|---:|---:|---:|
| GPU float16 | cuda / float16 | 1.9478 | 0.06763 | 0.21875 | 8975 | 2320 |
| GPU int8_float16 | cuda / int8_float16 | 1.9386 | 0.06731 | 0.21875 | 7983 | 1098 |
| CPU int8 | cpu / int8 | 5.4752 | 0.19011 | 0.21875 | 6885* | 0 |

3モードとも同一WAV・同一referenceでCERは0.21875だった。GPU `int8_float16`はこの試験で認識結果同等、RTFがわずかに短く、ASR増分VRAMはfloat16の約半分だったため、未指定時のGPU既定に採用した。`*` CPU行のnvidia-smi peakは別プロセスの既存GPU使用量を含むため、比較には増分0 MiBを用いる。

## TTS

公式APIはonline streaming generationを提供しないため、first-audio-equivalentは1文の生成要求からWAVが得られるまでの値である。`preload_seconds`は別計測で、音声品質は主観評価していない。

| run | device | audio (s) | first-audio-equivalent (s) | warm (s) | RTF | GPU peak (MiB) | 増分peak (MiB) | CPU load |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPU | cuda:0 | 7.76 | 5.5221 | 5.5067 | 0.7103 | 9163 | 50 | 4.57% |
| CPU参考 | cpu | 7.84 | 20.1509 | 20.1351 | 2.5688 | 7275* | 0 | 66.05% |

GPU preloadは8.2527秒、CPU preloadは4.0898秒だった。GPU RTF<1、CPU RTF>1だったので、E2E既定はGPU TTSとする。生成WAVはGit管理外である。

## LLM

通常streamではlocalが日本語短文と`stop`を返し、TTFT 0.1473秒、elapsed 0.2278秒だった。local requested/actualはともに`qwen3.5:9b-q4_K_M`。OpenRouter通常requestはrequested=`openrouter/free`、actual=`nvidia/nemotron-3.5-lightning:free`、TTFT 4.5726秒、elapsed 4.5974秒だったが、256 completion tokensをreasoningに使い切り、英語のthinking processを返して`length`終了した。日本語出力保証はできない。

mock tool chain（calculator `17*23`、fixed_test_data `status`）は、localが2 calls/2 roundsまで実行した後、Ollama HTTP 400で`failed`。OpenRouterは2 calls/2 roundsを完了し、2 round目に日本語のtool result要約を返して`success`だった。free routerではrequestごとにactual modelが変わることも確認した（tool chainでは`poolside/laguna-s-2.1:free`、`liquid/lfm-2.5-2.6b:free`）。未対応/失敗をprompt hackで補っていない。

両providerともベンチのmidstream cancelは`cancelled` eventを観測した。toolはcalculatorと固定データだけで、実サービスの破壊的操作はしていない。

## 4構成E2E

各構成は「同じ日本語synthetic user WAV -> ASR -> provider -> GPU TTS -> assistant output ASR」。GPU ASRの構成は比較を揃えるためfloat16、CPU ASRはint8。E2E latencyは最初のVAD eventから最後のplayback endまでである。

| case | ASR / LLM / TTS | ASR RTF | LLM TTFT (s) | E2E latency (s) | assistant CER |
|---|---|---:|---:|---:|---|
| A | GPU / local / GPU | 0.07144 | 0.1768 | 245.1195 | [0.0, 0.0, 0.0] |
| B | CPU / local / GPU | 0.20382 | 0.1372 | 17.0981 | [0.0, 0.0] |
| C | GPU / OpenRouter/free / GPU | 0.01454 | 1.5763 | 8.5883 | [0.0, 0.0] |
| D | CPU / OpenRouter/free / GPU | 0.14947 | 3.1065 | 20.6415 | [4.8333] |

- A local actual=`qwen3.5:9b-q4_K_M`、assistant outputは日本語で3 TTS chunks。
- B local actual=`qwen3.5:9b-q4_K_M`、2 chunks。
- C actual=`google/gemma-4-31b-it:free`、日本語短文で2 chunks、出力CER 0。
- D actual=`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`、短文1 chunkだがsynthetic referenceとのCERは4.8333。
- Aはevent上`vad_start -> vad_end`が約227.4秒となる一過性の大きな待ち時間を含んだ。直後の同じWAVの単独read+VADは3回とも0.003--0.005秒で、この遅延をVADアルゴリズム性能とは解釈していない。原因は未特定なので、Aのlatencyはそのまま実測値として記録しoutlier扱いにした。
- C/Dのactual modelはfree routerがrequestごとに選んだ値であり、品質比較やランキングではない。

別途`local-live run --input-wav ... --provider local --no-aec`もcompletedした。最終runは未指定GPU ASR=`cuda/int8_float16`、RTF=0.08752、LLMから5 TTS chunks、PipeWire `pw-play`まで通った。

## AEC OFF/ON

PipeWire `libpipewire-module-echo-cancel.so`はdoctorで存在を確認し、benchでは`pactl` wrapperで`aec_method=webrtc`、master sink=69、master source=71、virtual Echo-Cancel Sink=89 / Source=80を生成・解放できた。これはAEC moduleの構築・接続確認であり、AEC性能の成功を意味しない。

同一`aec_reference.wav`をUSB Audio sinkへ再生し、OFFはsource 71、ONはEcho-Cancel Source 80を`pw-record`した。OFF/ON WAVは作成されたが、両方のrecording RMS=0.0だった。比較結果は以下のとおり。

- reference/recording correlation: OFF=null、ON=null（無音のため未測定）
- cross-correlation lag: OFF/ONともnull
- residual echo attenuation: null
- ASR self-rerecognition: unavailable（signal floor以下）
- result status: `blocked`

`pw-record`は停止時return code 1だったがstderrは空で、問題はreturn codeではなく0.0 RMSである。USB Audio/GoStream等のcandidateは検出できているため、次回はcapture endpointに実サンプルが入る状態を確認して同じCLIを再実行する。double-talkは試験していない。

## 失敗・制約

- GitHub CLI: `gh auth status`のプロセス終了codeは0だが、認証marker検証はfalseだった。したがってこのworkspaceからGitHub private repository作成/pushは未完了である。ローカルcommit後にGitHub authを再設定して作成・pushする必要がある。
- OpenRouterは認証HTTP 200でもfree router配下modelがrequestごとに変わり、normal streamが英語reasoning/`length`終了になる場合がある。これはprovider構成の制約として記録し、local PoCの失敗とは扱っていない。
- local Qwen3.5 tool chainは2 round目のOllama HTTP 400で失敗した。provider/modelがtool result messageを受け付けない経路として記録した。
- AECは物理capture無音で性能未測定。無音から減衰量を推測していない。
- E2E Aには未特定の一過性待ち時間outlierがある。
- synthetic ASR regressionはTTS生成音声の自己回帰であり、実マイクの騒音・話者差・部屋音響の評価ではない。
- 真のonline Qwen3-TTS streaming、double-talk、有人barge-in、主観MOS、外部有料LLM選定、production UI/agent frameworkは対象外。
- flash-attn/SoX不足の警告は残るが、今回のQwen3-TTS manual PyTorchベンチとE2Eを阻害しなかった。

## 現時点の推奨既定

1. CPU VAD + 発話終了後ASR。
2. GPU ASR `large-v3-turbo`, `device=cuda`, `compute_type=int8_float16`, `language=ja`。CPU fallbackは`device=cpu`, `compute_type=int8`。
3. local Ollama `qwen3.5:9b-q4_K_M`, context 8192、短い応答。
4. Qwen3-TTS 0.6B CustomVoice `Ono_Anna`, Japanese, GPU。LLM streamを48文字/0.8秒で文chunkして逐次TTSする。
5. AECはmodule load後にcapture RMSを確認できた場合だけ有効な比較値として採用する。現machineではAEC性能値を採用しない。
6. tool callingが必要な場合はmock tool経路を使い、実運用ではproviderごとの対応状態とactual modelをログへ記録する。

## 次段階

- USB microphoneのcapture RMSとUSB speakerから室内への実音響loopを確認し、AEC OFF/ONを再測定する。追加スピーカー、double-talk試験は要求しない。
- GitHub CLIを再認証してprivate repositoryを作成し、このrepoをpushする。
- Aの待ち時間outlierをevent/IO単位で再現調査し、warm/cold、disk wait、GPU contentionを分離する。
- 実マイク日本語データで別ベンチを追加する（synthetic ASR regressionのCERと混同しない）。
- 低遅延をさらに必要とする場合だけ、sentence chunkの実測を基にTTS起動並列化またはincremental ASRを検討する。Qwen3-TTS独自streaming engineは作らない。

## 再現コマンド

```text
uv sync --extra dev --extra voice
uv run local-live doctor
uv run local-live bench asr
uv run local-live bench tts
uv run local-live bench llm
uv run local-live bench e2e
uv run local-live bench aec
uv run python bench/summarize_results.py
```
