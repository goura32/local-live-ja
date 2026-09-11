# local-live-ja 実測結果

この文書は、ローカルfilesystemの`/home/ws1/projects/local-live-ja`で実行した最新JSONを根拠にする。`null`、`blocked`、`unavailable`、`unsupported_or_failed`は0点ではなく、測定不能・未実行・未対応を表す。生成WAV、model cache、raw log、credentialはGit管理しない。

## 判定

| component | state | 根拠 |
|---|---|---|
| ASR | measured | `large-v3-turbo`、GPU 2 mode + CPU、同一synthetic WAVで完走 |
| TTS | measured | GPU cold 1回、短/中/長のwarm各3回、CPU cold reference |
| local LLM | measured | Ollama `qwen3.5:9b-q4_K_M` normal stream完了 |
| OpenRouter LLM | measured with variability | requested `openrouter/free`、actual modelはrequestごとに変動 |
| tool calling | measured | local/OpenRouterとも2 calls/2 rounds成功 |
| E2E | measured | A/B/C/Dは各3本の成功runを確保。C/Dのfailed attemptも保持 |
| AEC | blocked | raw USB captureがRMS=0、peak=0のためAECを実行していない |
| test reproducibility | pass | 単一process・直列pytest、exit code 0、32 tests pass |

総合判定は`measured_with_blockers`。ソフトウェア縦切りとlocal Live候補は成立したが、実音響AECと物理playback roundtripは未測定である。Live用途の根拠は、GPU TTS warmの短文median 1.9032秒と、local GPU E2E Aのmedian 6.2533秒である。真のonline Qwen3-TTS streamingは前提にしていない。

## 実行環境

- 測定working tree: `/home/ws1/projects/local-live-ja`
- `doctor`: status=`pass`
- host: `ws1`
- OS/kernel: Linux x86_64、kernel `7.0.0-31-generic`、glibc 2.39
- CPU: Intel Core i7-13700、24 logical CPUs
- GPU: NVIDIA GeForce RTX 5070 Ti、driver 595.84、16,303 MiB、compute capability 12.0
- Python 3.12.3、PyTorch 2.14.0、torch CUDA runtime 13.0
- faster-whisper 1.2.1、CTranslate2 4.8.2、transformers 4.57.3
- qwen-tts 0.1.1、numpy 2.5.3、soundfile 0.14.0、sounddevice 0.5.6、webrtcvad-wheels 2.0.14
- CUDA compatibility wheels: cublas 12.9.2.10、cudnn 9.26.0.51、cuda-nvrtc 12.9.86
- PipeWire 1.0.5、Ollama API 0.33.3
- `flash-attn`未導入、SoX executable未検出。ただしmanual PyTorch TTSとE2Eは完走した。

## ASR: synthetic regression

Qwen3-TTSで生成した同一28.0秒WAVを、`language=ja`の`large-v3-turbo`へ戻す回帰試験である。実マイク音声の精度、話者差、部屋音響、MOSではない。CERはUnicode正規化後に算出した。

| mode | device / compute | elapsed (s) | RTF | CER | GPU peak (MiB) | ASR増分peak (MiB) |
|---|---|---:|---:|---:|---:|---:|
| GPU float16 | cuda / float16 | 1.6009 | 0.05750 | 0.203125 | 9365 | 2090 |
| GPU int8_float16 | cuda / int8_float16 | 1.9962 | 0.07170 | 0.203125 | 8373 | 1096 |
| CPU int8 | cpu / int8 | 12.8152 | 0.46032 | 0.203125 | 7277* | 0 |

3 modeのCERは同一だった。GPU既定は、認識結果同等で増分VRAMが少ない`int8_float16`とする。CPU行のnvidia-smi peakは別プロセスの既存GPU使用量を含むため、CPU比較には増分0 MiBを使う。

## TTS cold/warm latency

Modelは`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`、speaker=`Ono_Anna`、language=`Japanese`。GPU測定は新しいengineを未loadで開始し、short cold 1回の後、同じmodel instanceをresidentにしたまま各chunkを3回生成した。warm rowsの`model_load_seconds`は全て0.0である。

`first_audio_equivalent`は公式Python APIが返したwaveformが得られる時刻であり、streamingの最初のsampleではない。coldの値はrequest開始からなのでmodel load込み、`warm_first_audio_equivalent`はinference開始からでmodel loadを含まない。`audio_complete`はWAV書込完了、`playback_possible`は呼び出し元がそのWAVを再生可能になった時刻である。これは物理speakerから音が出た時刻ではない。

### GPU cold

| chunk | chars | model load (s) | first-audio-equivalent (s) | warm inference-to-audio (s) | audio complete (s) | playback possible (s) | total (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 8 | 9.5825 | 12.5786 | 2.9961 | 12.5877 | 12.5877 | 12.5879 |

### GPU warm

| chunk | chars | 3 runs: total elapsed (s) | median total (s) | median inference (s) | median audio complete (s) | median playback possible (s) | median RTF |
|---|---:|---|---:|---:|---:|---:|---:|
| short | 8 | 3.0681, 1.9032, 1.6611 | 1.9032 | 1.8782 | 1.9021 | 1.9021 | 0.6521 |
| medium | 20 | 2.7902, 2.8429, 2.7869 | 2.7902 | 2.7614 | 2.7900 | 2.7900 | 0.6504 |
| long | 57 | 6.9308, 6.5041, 6.8689 | 6.8689 | 6.8408 | 6.8689 | 6.8689 | 0.6434 |

CPU referenceはcold 1回で、model load 4.7537秒、inference開始からaudio相当まで29.1917秒、audio complete 33.9474秒、total 33.9475秒、RTF 3.6858だった。したがって実用的な初動時間はGPU warm shortの約1.9秒を基準にし、長いLLM出力はsentence chunkingで分割する。Qwen3-TTS独自online engineは作っていない。

## LLM

| provider | requested model | normal TTFT (s) | normal total (s) | actual model | tool chain |
|---|---|---:|---:|---|---|
| local Ollama | `qwen3.5:9b-q4_K_M` | 0.1646 | 0.2453 | `qwen3.5:9b-q4_K_M` | success, 2 calls / 2 rounds |
| OpenRouter | `openrouter/free` | 0.9962 | 1.0680 | `inclusionai/ling-3.0-flash-vl:free` in this request | success, 2 calls / 2 rounds |

OpenRouterの`actual_model`はfree routerがrequestごとに選ぶため、上表の値を固定modelの品質比較には使わない。E2Eではactual modelをrunごとに保存している。

### Ollama tool-calling修正

修正前の2 round目はHTTP 400で、response bodyの安全な要約は次のとおりだった。

- status: `400`
- error: `Value looks like object, but can't find closing '}' symbol`
- 原因: Ollama native chat形式が`function.arguments`のobjectを要求するのに、実装がJSON文字列を送っていた。
- 修正: Ollama assistant tool callは`function: {name, arguments: {...}}`、tool resultは`{role: "tool", content: "..."}`とし、OpenAI互換形式を使うOpenRouterとは分離した。
- 修正後: local Ollamaは`計算結果は391...`まで2 rounds完了。HTTP 400は発生していない。

修正前の完全なベンチJSONは`results/history/bench_llm_baseline_869809d.json`、本文要約と公式形式との差分は`results/history/ollama_tool_format_diagnostic.json`に残した。credentialやraw response body全体は保存していない。

## E2E A/B/C/D

同一synthetic user WAVに対して、各構成はfull warm-up後に成功run 3本を目標にした。C/Dのvisible textなしrunは失敗として保持し、medianからは成功runだけを使った。outlierは削除していない。GPU ASRはPoC既定の`int8_float16`、CPU ASRは`int8`、TTSはGPUである。

| case | ASR / LLM / TTS | attempts | successful | 個別total E2E (s) | median total E2E (s) |
|---|---|---:|---:|---|---:|
| A | GPU / local / GPU | 3 | 3 | 5.2876, 6.2533, 6.4607 | 6.2533 |
| B | CPU / local / GPU | 3 | 3 | 34.6375, 36.3003, 33.6340 | 34.6375 |
| C | GPU / OpenRouter/free / GPU | 7 | 3 | 5.5378 failed, 13.5509, 6.5473 failed, 2.5966 failed, 10.2914, 2.4427 failed, 8.6235 | 10.2914 |
| D | CPU / OpenRouter/free / GPU | 5 | 3 | 27.2507, 15.3549 failed, 402.4608, 17.4276 failed, 36.9513 | 36.9513 |

各構成の成功run median stageは次のとおり。`playback_roundtrip_duration_s=null`は、E2Eベンチが生成WAVとassistant ASRの論理縦切りを測り、物理speaker/microphone roundtripはAECベンチへ分離しているためである。nullを0秒としてtotalに足していない。

| case | ASR duration (s) | LLM TTFT (s) | LLM total (s) | TTS duration (s) | playback/roundtrip (s) | total E2E (s) | peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.4146 | 0.0826 | 0.1837 | 5.1900 | null | 6.2533 | 10353 |
| B | 11.1497 | 0.0955 | 0.1595 | 4.8087 | null | 34.6375 | 9369 |
| C | 0.4103 | 1.1354 | 1.9694 | 5.9084 | null | 10.2914 | 10373 |
| D | 11.1897 | 3.5243 | 3.6341 | 3.4492 | null | 36.9513 | 10249 |

各runには上表の全項目に加え、input read、VAD、assistant-ASR、event timing、actual model、failed reasonを保存した。current JSONのoutlier annotationは次のとおり。

- C failed attemptsはOpenRouterがvisible textを返さず、`LLM returned no visible text`となった。Cは7 attempts中3成功で、failed 4本を削除していない。
- D run 3は402.4608秒。OpenRouterが英語のthinking processをvisible outputとして返し、28 TTS chunksを生成した。stageはTTS 114.7151秒、assistant-ASR 268.3298秒で、dominant stageはassistant-ASR。`tts_duration_s`と`assistant_asr_duration_s`をoutlier flagにした。Dのmedianはこのrunを削除せず、3成功runのmedianとして36.9513秒を保存した。
- 基準commit 869時点のA 245.1195秒outlierも削除していない。旧JSONは`results/history/bench_e2e_baseline_869809d.json`にあり、旧eventでは約227.4秒がVAD区間に現れていた。同一WAVの単独read+VADが0.003--0.005秒だったため、VAD性能値と解釈していない。

### assistant CERの比較対象

assistant CERは全runで、実際のTTS rowの`text`を`spoken_text`として、その同じTTS WAVをASRしたtextを`asr_text`として、対応indexごとに比較している。LLMの期待回答、reasoning、tool-call JSON、発話していないtextは比較対象にしていない。最新JSONの`assistant_roundtrip`に両方の文字列とCERを保存した。

- A成功runのCER arrays: `[0.0, 0.0]`, `[0.0, 0.2353]`, `[0.0, 1.6471]`
- B成功runのCER arrays: `[0.0, 0.0]`, `[0.0, 0.6471]`, `[0.0, 0.1176]`
- C成功runのCER arrays: `[0.1111]`, `[0.0, 0.0, 0.0]`, `[0.0, 0.0]`
- D成功runのCER arrays: `[0.1538]`, 28-chunk long-output array, `[0.0, 0.0]`

基準JSONのD=`4.8333`は、old runで実際にspokenだった`了解です、テスト開始します。`と、その生成WAVをASRした長い誤認識列を比較した値だった。比較対象の取り違えではないが、old schemaではTTS rowとassistant-ASR rowの対応path/timingが弱く、今回index対応の`assistant_roundtrip`へ修正した。最新Dの高CER/402秒runは、actual free modelがthinking processを発話textとして返したことと、長文TTS/ASRの品質・時間変動が原因であり、CERロジックの期待回答混入ではない。

## AEC

最新run IDは`aec_20260911T200521Z_222082807004148`。raw captureを最初に1回だけ実行し、物理信号を確認できなかったためAEC moduleをloadしていない。

- playback target: runtime解決したnode.name=`USB Audio アナログステレオ`、node_id=69（sink）
- capture target: runtime解決したnode.name=`USB Audio アナログステレオ`、node_id=71（source）
- raw recording duration: 7.016875秒
- raw RMS: 0.0
- raw peak: 0.0
- signal floor RMS: 0.000001
- `aec_attempted`: false
- AEC OFF/ON: null / null
- correlation、cross-correlation lag、residual echo attenuation、Whisper self-rerecognition: null/blocked
- result status: `blocked`

node IDは設定ファイルへ永続化していない。各runでPipeWire inventoryを再取得し、node.name/propertyから現在のnodeを解決して、録音時だけruntime IDを渡す。旧AEC試行の不確定値と固定filename時代の結果は`results/history/bench_aec_baseline_869809d.json`とGit履歴に残した。raw captureが閾値を超えるまではOFF/ON比較を実行しない。

## テストと成果物

- pytest command: `.venv/bin/python -m pytest -q`（環境変数でbytecode/BLAS threadを抑制）
- working tree: `/home/ws1/projects/local-live-ja`
- execution: single process, serial
- exit code: 0
- passed: 32（基準の28 tests + 測定回帰4 tests）
- 過去のSIGTERM/SIGKILL実行はpassに算入していない。
- `uv build`: baseline commitで成功済み。今回のsource変更後も下記最終検証で再実行する。
- JSON: `results/doctor.json`, `bench_asr.json`, `bench_tts.json`, `bench_llm.json`, `bench_e2e.json`, `bench_aec.json`, `run_latest.json`, `summary.json`
- 履歴: `results/history/`。current E2E/AEC/LLM baselineとfree-router retry attemptを保存している。
- schema: current benchmark JSONはv2、summaryはv2。

## 残ったblocker

1. USB microphone raw captureが無音。speaker再生中でもRMS/peak=0のため、AEC性能値を出せない。実入力経路を直すまでAECはblocked。
2. E2Eの物理playback/roundtripは未測定。論理WAV-readyとassistant-ASRは測ったが、物理音響性能はAECのraw-first gateにより分離している。
3. OpenRouter free routerはactual model、visible text、reasoning、出力長、TTFTがrequestごとに変動する。local PoCの失敗とは混同しない。C/D failed attemptsとD長文outlierは保存済み。
4. GPU warm shortの初動は約1.9秒であり、sub-second live responseではない。sentence chunkingで初動を確保するが、Qwen3-TTS公式Python API自体のtrue online streamingは未提供。
5. flash-attnとSoX executableは未導入。manual TTS経路は完走しているため、現測定のblocking issueではない。

## 推奨既定

1. CPU energy VAD + 発話終了後のutterance-final ASR。
2. `large-v3-turbo`、`language=ja`、GPU `int8_float16`、CPU fallback `int8`。
3. local Ollama `qwen3.5:9b-q4_K_M`を通常経路。OpenRouterは交換可能な比較経路としてactual modelを記録する。
4. Qwen3-TTS CustomVoice `Ono_Anna`、Japanese、GPU。LLM streamを最大48文字/0.8秒でsentence chunkする。
5. AECはraw captureが成立したrunだけOFF/ONを実行し、無音から数値を推測しない。
6. tool callingはcalculator/fixed dataのdeterministic mockだけを使う。providerが失敗した場合はunsupported/failedとして保存する。

## 再現コマンド

```text
cd ~/projects/local-live-ja
UV_LINK_MODE=copy uv sync --extra dev --extra voice
.venv/bin/local-live doctor
.venv/bin/local-live bench asr --force-audio
.venv/bin/local-live bench tts
.venv/bin/local-live bench llm
.venv/bin/local-live bench e2e
.venv/bin/local-live bench aec
.venv/bin/python -m pytest -q
.venv/bin/python bench/summarize_results.py
```
