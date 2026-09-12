# local-live-ja 実測結果

この文書は、ローカルfilesystemの`/home/ws1/projects/local-live-ja`で実行した最新JSONを根拠にする。`null`、`blocked`、`unavailable`、`unsupported_or_failed`は0点ではなく、測定不能・未実行・未対応を表す。生成WAV、model cache、raw log、credentialはGit管理しない。

## 判定

| component | state | 根拠 |
|---|---|---|
| ASR | measured | `large-v3-turbo`、GPU 2 mode + CPU、同一synthetic WAVで完走 |
| TTS | measured | GPU cold 1回、6長さ×warm各5回、CPU cold reference |
| local LLM | measured | Ollama `qwen3.5:9b-q4_K_M` normal stream完了 |
| OpenRouter LLM | measured with variability | requested `openrouter/free`、actual modelはrequestごとに変動 |
| tool calling | measured | local/OpenRouterとも2 calls/2 rounds成功 |
| E2E | measured | A/B/C/Dは各3本の成功runを確保。C/Dのfailed attemptも保持 |
| AEC | measured with limitations | stable targetで16条件matrixと3候補×3 repeatを測定。VAD false triggerは残る |
| test reproducibility | pass | 単一process・直列pytest、exit code 0、41 tests pass |

総合判定は`measured_with_limitations`。phase-2でsynthetic user endからraw USB microphone acoustic onsetまでの物理latency、TTS length matrix、AEC volume/gain matrix、CPU ASR resident profileを追加した。live latencyは3.4000秒で2秒目標未達、AECは減衰改善を確認したがassistant-only VAD false triggerが残る。真のonline Qwen3-TTS streamingは前提にしていない。

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

## Phase-1 baseline: ASR synthetic regression

Qwen3-TTSで生成した同一28.0秒WAVを、`language=ja`の`large-v3-turbo`へ戻す回帰試験である。実マイク音声の精度、話者差、部屋音響、MOSではない。CERはUnicode正規化後に算出した。

| mode | device / compute | elapsed (s) | RTF | CER | GPU peak (MiB) | ASR増分peak (MiB) |
|---|---|---:|---:|---:|---:|---:|
| GPU float16 | cuda / float16 | 1.6009 | 0.05750 | 0.203125 | 9365 | 2090 |
| GPU int8_float16 | cuda / int8_float16 | 1.9962 | 0.07170 | 0.203125 | 8373 | 1096 |
| CPU int8 | cpu / int8 | 12.8152 | 0.46032 | 0.203125 | 7277* | 0 |

3 modeのCERは同一だった。GPU既定は、認識結果同等で増分VRAMが少ない`int8_float16`とする。CPU行のnvidia-smi peakは別プロセスの既存GPU使用量を含むため、CPU比較には増分0 MiBを使う。

## Phase-1 baseline: TTS cold/warm latency

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

## Phase-1 baseline: LLM

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

## Phase-1 baseline: E2E A/B/C/D

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

## Phase-1 baseline: AEC single-run

最新run IDは`aec_20260911T232048Z_233809883876679`。従来の`wpctl` runtime node ID指定ではraw captureが無音になるrunがあったため、`pactl`から得た安定した`node.name`をtargetとして保持するよう修正した。

- playback target: `alsa_output.usb-Generic_USB_Audio_201405280001-00.analog-stereo`
- capture target: `alsa_input.usb-Generic_USB_Audio_201405280001-00.analog-stereo`
- raw capture: duration 7.3582秒、RMS 0.02321、peak 0.7110 → `pass`
- AEC OFF: RMS 0.03353、peak 0.28247、reference correlation 0.5935
- AEC ON: RMS 0.02770、peak 0.23416、reference correlation 0.3122
- residual echo attenuation: 1.659 dB
- Whisper自己再認識 CER: OFF 0.2353、ON 0.2941
- result status: `measured`

AEC経路とWebRTC echo-cancel moduleの動作は確認できた。一方、今回の単一物理runではRMS減衰は約1.66 dBに留まり、自己音声のASR認識もOFF/ONとも残ったため、「十分なecho suppression」とは判定しない。机上配置、speaker音量、microphone方向、AEC sample-rate/latency等の調整は次段階の課題とする。

今回の診断ではUSB microphone自体は有効信号を取得でき、無音の主因は一時的な数値node IDを`pw-record`/`pw-play` targetとして使用していた点だった。物理nodeは`pactl`のstable nameで指定し、数値IDは診断情報としてのみ記録する。旧blocked結果はGit履歴と`results/history/bench_aec_baseline_869809d.json`に残している。

## Phase-1 baseline: テストと成果物

- pytest command: `.venv/bin/python -m pytest -q`（環境変数でbytecode/BLAS threadを抑制）
- working tree: `/home/ws1/projects/local-live-ja`
- execution: single process, serial
- exit code: 0
- passed: 33（phase-1 baseline。current phase-2 finalは41 tests）
- 過去のSIGTERM/SIGKILL実行はpassに算入していない。
- `uv build`: baseline commitで成功済み。今回のsource変更後も下記最終検証で再実行する。
- JSON: `results/doctor.json`, `bench_asr.json`, `bench_tts.json`, `bench_llm.json`, `bench_e2e.json`, `bench_aec.json`, `run_latest.json`, `summary.json`
- 履歴: `results/history/`。current E2E/AEC/LLM baselineとfree-router retry attemptを保存している。
- schema: current benchmark JSONはv2、summaryはv2。

## 残った制約

1. AECは物理経路まで測定できたが、今回の単一runの残留echo減衰は約1.66 dBで、Whisper自己再認識も残った。AEC品質改善は次段階。
2. E2E A/B/C/Dは論理WAV-readyまでの比較であり、全会話ターンをspeaker→room→microphoneへ戻す物理roundtrip latencyは別測定として残る。
3. OpenRouter free routerはactual model、visible text、reasoning、出力長、TTFTがrequestごとに変動する。local PoCの失敗とは混同しない。C/D failed attemptsとD長文outlierは保存済み。
4. GPU warm shortの初動は約1.9秒であり、sub-second live responseではない。sentence chunkingで初動を確保するが、Qwen3-TTS公式Python API自体のtrue online streamingは未提供。
5. flash-attnとSoX executableは未導入。manual TTS経路は完走しているため、現測定のblocking issueではない。

## 推奨既定

1. CPU energy VAD + 発話終了後のutterance-final ASR。
2. `large-v3-turbo`、`language=ja`、GPU `int8_float16`、CPU fallback `int8`。
3. local Ollama `qwen3.5:9b-q4_K_M`を通常経路。OpenRouterは交換可能な比較経路としてactual modelを記録する。
4. Qwen3-TTS CustomVoice `Ono_Anna`、Japanese、GPU。LLM streamを最大48文字/0.8秒でsentence chunkする。
5. AECはstable `node.name`で物理USB source/sinkを指定し、raw capture成立後だけOFF/ONを実行する。現在の測定では動作するが抑圧効果は弱い。
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
.venv/bin/local-live bench live-latency
.venv/bin/local-live bench aec-matrix
.venv/bin/python -m pytest -q
.venv/bin/python bench/summarize_results.py
```

## 次フェーズ検証（基準commit `39610a58e1cf16dc49ffa08b3f22f3cb34cb6c0e`）

今回の追加結果は `results/bench_live_latency.json`、`results/bench_aec_matrix.json`、
`results/bench_tts.json`、`results/bench_asr.json` に保存した。生成WAVは従来どおりGit管理外である。

### P0: speech-end → first physical audio

指標名は `synthetic_user_end_to_first_physical_assistant_audio`。人間の発話終了ではなく、
synthetic user WAVをASRへ直接handoffした時刻から、USB speakerの音をraw USB microphoneで検出した
acoustic onsetまでを測った値である。raw recorderはsynthetic_user_end直前に開始し、WAV ready後の
`pw-play`はmeasurement leadを挿入せず起動した。AEC sourceは使用していない。

5回すべてphysical onsetを検出した。

- 個別値: 3.0500 / 3.7500 / 3.5700 / 3.4000 / 2.3500 秒
- median: 3.4000 秒
- p95相当（inclusive、5点）: 3.7140 秒
- min/max: 2.3500 / 3.7500 秒
- ASR median: 0.3750 秒
- local Ollama TTFT median: 0.0755 秒
- sentence buffering median: 0.0091 秒
- TTS median: 1.6754 秒
- WAV ready → pw-play median: 0.0008 秒
- pw-play → acoustic onset median: 1.2598 秒
- 支配stage: TTS（次点は物理再生経路）

第一目標の2.0秒、良好目標の1.5秒はいずれも未達である。outlierは削除していない。
以前のcapture開始後に0.4秒leadを挿入した測定は、比較用に
`results/history/bench_live_latency_lead_included_pre_active_capture.json` として残した。

chunking候補のphysical比較も同じ代表LLM出力で実施した。`48/0.8`、`32/0.5`、`24/0.5`、
`16/0.3`を総当たりせず比較し、各候補のchunk数、TTS ready、physical onset、細切れproxyを
JSONへ保存した。これは1回ずつの候補比較であり、候補の絶対最適性やMOSを意味しない。

### P0: Qwen3-TTS warm latency

GPU coldはchars_8で、model loadとwarm inferenceを混在させていない。

- cold chars_8 total: 9.9575秒
- cold model load: 6.1666秒
- cold inference start → first-audio-equivalent: 3.7818秒
- cold WAV/playback-ready: 9.9574秒

resident modelで各5回測ったwarm medianは以下のとおりである。全warm runの
`model_load_seconds`は0.0秒である。

| chunk | 実文字数 | total | inference start → first audio-equivalent | RTF |
|---|---:|---:|---:|---:|
| chars_5 | 5 | 1.4283 s | 1.3998 s | 0.6482 |
| chars_8 | 8 | 2.3944 s | 2.3673 s | 0.6419 |
| chars_12 | 12 | 1.8237 s | 1.7989 s | 0.6425 |
| chars_20 | 20 | 2.8286 s | 2.8012 s | 0.6380 |
| chars_30 | 29 | 3.7079 s | 3.6827 s | 0.6394 |
| chars_50 | 61 | 6.8982 s | 6.8687 s | 0.6381 |

今回の最短条件はchars_5の1.4283秒。Qwen3-TTS公式Python APIのtrue online streamingは
仮定せず、既存のsentence chunking方式を維持する。ただし実LLMの最初のchunk（今回6文字）では
TTS medianが1.6754秒であり、physical latencyも3.4000秒だった。2秒未満を狙う次フェーズでは、
Qwen3-TTSのserving方式または別TTSの比較が必要である。

### P1: microphone volume × speaker volume echo-only matrix

同一 `results/artifacts/aec_reference.wav` を再利用し、各条件を
`raw capture → AEC OFF → AEC ON` の順に直列測定した。Pulse/PipeWireのstable nameを使い、
speaker/source volumeは25/50/75/100%の範囲だけを設定した（boostなし）。16/16条件がmeasuredで、
75% gateによるsafety skipはなかった。

- residual echo attenuation: median 5.9927 dB、range 2.4633–8.1968 dB
- reference correlation: OFF median 0.6002、ON median 0.2500
- assistant-only VAD false-trigger duration ratio: OFF median 0.6640、ON median 0.4123
- assistant-only VAD false-trigger run rate: OFF 15/16 (0.9375)、ON 13/16 (0.8125)
- peak/clipping: 全条件のclipping ratioは0.0。最大peakもraw 0.5896、OFF 0.4438、ON 0.3272で、100%条件をskipしなかった

これはRMSだけで選んだ値ではない。VAD false trigger低減、Whisper自己再認識、attenuation、
correlation、clipの順に候補を選び、単発採用を避けるため3候補を各3回再測定した。

repeat測定で今回のecho-only operating envelopeとして観測された範囲は、speaker 75–100%、
microphone 25–50%である。これは人間音声の最適gainでも絶対最適条件でもない。

| 条件 | attenuation median (min–max) | AEC ON VAD ratio median | ON false-trigger rate | ON Whisper self-rerecognition rate | ON CER median | clipping |
|---|---:|---:|---:|---:|---:|---|
| speaker 75 / mic 25 | 7.1227 (6.7235–9.0895) dB | 0.2630 | 3/3 | 3/3 | 0.1765 | none |
| speaker 100 / mic 25 | 6.8942 (5.9480–8.4676) dB | 0.4058 | 3/3 | 3/3 | 0.1765 | none |
| speaker 75 / mic 50 | 6.9825 (6.3684–7.6316) dB | 0.3442 | 3/3 | 3/3 | 0.3235 | none |

現行baseline約1.6591 dBに対して、repeat-tested候補のattenuation medianは明確に上回った。
一方、VAD false triggerは理想のゼロではなく、Whisper ON transcriptも毎回空になるわけではない。
従ってAECは改善を確認したが、assistant自己音声による反応を解消したとは判定しない。

### P2: CPU ASR 11秒問題

同一 `synthetic_asr_regression.wav` をresidentの同一CPU `large-v3-turbo` modelで処理し、
cold 1回とwarm 5回を分離した。

- model load: 1.9547秒
- 音声全体: 27.84秒、VAD後: 25.94秒
- warm inference median: 10.8950秒
- warm total median: 10.9033秒
- warm file decode/resample median: 0.0052秒
- warm VAD median: 0.0003秒
- warm RTF（VAD後音声基準）: 0.4200
- E2E Bの既存CPU ASR stage median: 11.1497秒

従って約11秒の原因はfile decode/resampleやVADではなく、25.94秒のVAD後utteranceに対する
faster-whisper/CTranslate2 inferenceである。E2Eとの差は約0.25秒で、同一modelのwarm profileと
整合する。thread tuning sweepは行っていない。CPU ASRは品質確認用fallbackとしては採用可能だが、
現在のLive latencyの主経路には遅すぎる。

## Phase 3A: Qwen3-TTS low-latency optimization

Phase 3Aでは、新しいTTS engine、serving方式、独自streaming engineを導入せず、公式Qwen3-TTS Python APIの同一resident modelだけを比較した。GPU ASRは`large-v3-turbo / int8_float16`、LLMはlocal Ollama、AEC algorithmとphase-2のvolume envelopeは変更していない。

### 生成WAV onsetとsafe trim

生成WAVは10 ms frame RMSで測定した。thresholdは`max(absolute floor 0.004, peakの2%, noise floorの4倍, noise floor + 6 MAD)`、stable onsetは3連続active frameとした。`first_nonzero_sample`だけでtrimせず、stable onsetからpre-rollを残した。

代表5文（肯定、説明、数字、技術用語、2文）のtrim前leading silenceはmedian 0.5100秒、範囲0.2400–1.2600秒だった。50/100/150/200/250 ms pre-rollを同一生成WAVから作り、元WAVとtrim WAVを同じGPU Whisperへ戻した。

| pre-roll | 全5文 quality OK | trim-induced CER悪化 | trim-induced語頭回帰 | removed duration median |
|---:|---:|---:|---:|---:|
| 0 ms | 5/5 | 0 | 0 | 0.0000 s |
| 50 ms | 5/5 | 0 | 0 | 0.4600 s |
| 100 ms | 5/5 | 0 | 0 | 0.4100 s |
| 150 ms | 5/5 | 0 | 0 | 0.3600 s |
| 200 ms | 5/5 | 0 | 0 | 0.3100 s |
| 250 ms | 5/5 | 0 | 0 | 0.2600 s |

latency削減を優先しつつ品質劣化のない最小値として、`trim + 50 ms pre-roll`を採用した。technical terms文の絶対CERはbaselineから高かったが、trim前後の差分は0であり、trimによる悪化ではない。

### Qwen3-TTS warm matrix

GPU cold chars_8はmodel load `6.2181 s`、total `9.0130 s`、inference start → first-audio-equivalent `2.7848 s`。warmは同一resident model、各5回、公式sampling policyで測定した。

| length | chars | total median | inference median | RTF median |
|---|---:|---:|---:|---:|
| chars_5 | 5 | 1.4941 s | 1.4688 s | 0.6129 |
| chars_8 | 8 | 1.5819 s | 1.5567 s | 0.6081 |
| chars_12 | 12 | 1.9154 s | 1.8889 s | 0.6077 |
| chars_20 | 20 | 3.0012 s | 2.9833 s | 0.6055 |
| chars_30 | 29 | 3.4135 s | 3.3844 s | 0.6044 |
| chars_50 | 61 | 6.6719 s | 6.6431 s | 0.6015 |

### official generation policy comparison

同じresident official modelで`こんにちは。`を各3回測定した。sampled baselineを変更せず、短いtoken capとdeterministicを採否比較した。

| policy | inference median | total median | audio duration median | onset |
|---|---:|---:|---:|---:|
| sampled_2048（採用） | 1.4555 s | 1.4802 s | 2.32 s | 3/3 |
| sampled_1024 | 1.5645 s | 1.5928 s | 2.56 s | 3/3 |
| deterministic_1024（不採用） | 47.1753 s | 47.2204 s | 81.84 s | 0/3 |

`do_sample=False`は品質以前に極端な長時間/長音声となった。`sampled_1024`もbaselineより速くならなかったため、generation policyは公式sampling defaults相当の`sampled_2048`を維持する。

### P0: 分解後のphysical latency

指標は`synthetic_user_end → first physical assistant audio`であり、人間の実発話latencyではない。各runで、TTS inference start、waveform ready、WAV ready、generated WAV speech onset、trim ready、`pw-play` start、raw USB microphone onsetを保存した。raw side-channelはAEC sourceではなく、stable Pulse/PipeWire `node.name` targetを使用した。

最新10回は全てmeasuredで、trim/pre-rollは50 ms、first chunk policyは句読点/自然なphrase境界優先（語中強制splitなし、48 chars/0.8 s）だった。

- individual: 2.5800 / 2.6300 / 3.3100 / 3.3200 / 3.2700 / 2.8800 / 1.7200 / 3.3700 / 2.2500 / 2.2400 s
- median: 2.7550 s
- p95相当: 3.3475 s
- mean / standard deviation: 2.7570 / 0.5700 s
- min / max: 1.7200 / 3.3700 s
- trim前generated WAV leading silence（この10回）: median 0.6000 s
- trim後speaker speech pre-roll: median 0.0500 s
- WAV ready → pw-play: median 0.0027 s
- expected speaker onset → measured raw microphone: median 0.2397 s
- volume/mute/default sink/source restore: errorなし

### pure playback path

TTS生成と分離するため、先頭の期待信号位置が既知の低振幅probe WAVを同一条件で5回再生した。`expected signal → raw microphone onset`は`0.8382 / 0.2186 / 0.2185 / 0.2384 / 0.2184` s、median `0.2186 s`、p95相当 `0.7183 s`、mean `0.3464 s`、std `0.2751 s`だった。1回のpeak=1.0/clipping ratio約3e-5は削除せずoutlierとして保持した。

### latency budget

| stage | median | share of total |
|---|---:|---:|
| ASR | 0.3771 s | 13.69% |
| LLM TTFT | 0.0754 s | 2.74% |
| first chunk buffering | 0.0091 s | 0.33% |
| TTS inference | 1.9803 s | 71.88% |
| TTS file postprocess | 0.0287 s | 1.04% |
| TTS leading audio after trim | 0.0500 s | 1.81% |
| WAV ready → pw-play | 0.0027 s | 0.10% |
| expected speaker onset → measured microphone | 0.2397 s | 8.70% |

最大支配stageはTTS inferenceである。trimで元leading silenceの中央値約0.60秒を50 ms pre-rollまで削減できたが、公式APIの短文inference中央値が約1.98秒残るため、physical medianは2秒を安定して切らなかった。stage shareの合計とtotalの差は、イベント境界・録音検出・LLM/TTS間の未分類overheadを含む。

### Phase 3A判定

median `<2.0 s` は未達、stretch goal `<1.5 s` も未達である。trimは低コストで約0.5秒規模の改善余地を回収したが、現行official Qwen3-TTS生成方式自体が最大bottleneckとして残る。従って次フェーズでは、P0としてQwen3-TTS serving方式または別TTS engineの比較へ進む。Phase 3Aの生成方式は公式Python pathのまま継続し、新engine移行はこのphaseでは行っていない。

### 状態と履歴

今回の状態は `results/summary.json` の `component_judgement` に記録した。
live latencyはphysical測定として成立したが2秒目標未達、AEC matrixは測定成立したがfalse trigger
残存のため、総合判定は `measured_with_limitations` とする。基準時点のAEC/E2E/TTS/ASR結果と
過去outlierは `results/history/` から削除していない。
