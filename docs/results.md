# local-live-ja 実測結果

## 最新総合結果（Phase 7）

- release metadata: `v0.1.1`（測定JSONの`local-live-ja: 0.1.0`は、再測定せず保持している実測時点のprovenance）
- overall: `unattended_poc_complete`
- application acceptance: 60 scripted turns、application failure `0`
- same application path: `SessionController` + `LivePipeline` + incremental `SentenceChunker`
- continuous capture: `RealMicrophoneSource` callback interface、`FixtureAudioSource` application harness
- multi-turn history: system/user/assistant order、spoken-only assistant commit、bounded trim、next-request forwarding
- incremental LLM→TTS: `TextDelta`受信中にnatural sentenceをTTSへ渡すことを自動確認
- Live default candidate: vLLM-Omni `0.28.0` HTTP raw PCM streaming + persistent `pw-cat`
- synthetic application-level barge-in: first/middle/lateの3条件を各3回（計9 run）、playback/TTS/LLM cancellationと次turn recoveryを確認
- bounded recovery: temporary Ollama/TTS/playback failure、empty ASR、too-short utteranceを確認
- resource/application harness: FD・child/playback process・active HTTPのmonotonic growthなし、stale PCMなし
- final gates: pytest、compileall、`uv build`、`git diff --check`、CI workflow、reachable-history secret auditを確認

この判定は`production_ready`を意味しない。人間の実発話品質、physical double-talk、physical barge-in体験、MOS、subjective評価、manual gain tuning、production approvalは`deferred_manual`であり、無人PoC受入範囲外である。Phase 6のphysical測定結果とPhase 1〜5の履歴は以下に保持する。

この文書は、ローカルfilesystemの`repository root`で実行した最新JSONを根拠にする。`null`、`blocked`、`unavailable`、`unsupported_or_failed`は0点ではなく、測定不能・未実行・未対応を表す。生成WAV、model cache、raw log、credentialはGit管理しない。

## 過去の総合判定（Phase 1〜2時点）

| component | state | 根拠 |
|---|---|---|
| ASR | measured | `large-v3-turbo`、GPU 2 mode + CPU、同一synthetic WAVで完走 |
| TTS | measured | GPU cold 1回、6長さ×warm各5回、CPU cold reference |
| local LLM | measured | Ollama `qwen3.5:9b-q4_K_M` normal stream完了 |
| OpenRouter LLM | measured with variability | requested `openrouter/free`、actual modelはrequestごとに変動 |
| tool calling | measured | local/OpenRouterとも2 calls/2 rounds成功 |
| E2E | measured | A/B/C/Dは各3本の成功runを確保。C/Dのfailed attemptも保持 |
| AEC | measured with limitations | stable targetで16条件matrixと3候補×3 repeatを測定。VAD false triggerは残る |
| test reproducibility | pass | 単一process・直列pytest、exit code 0、118 tests pass（Phase 7最終） |

Phase 1〜2時点の総合判定は`measured_with_limitations`だった。これは過去snapshotであり、現在の総合判定ではない。phase-2でsynthetic user endからraw USB microphone acoustic onsetまでの物理latency、TTS length matrix、AEC volume/gain matrix、CPU ASR resident profileを追加した。live latencyは3.4000秒で2秒目標未達、AECは減衰改善を確認したがassistant-only VAD false triggerが残る。真のonline Qwen3-TTS streamingは前提にしていない。

## 実行環境

- 測定working tree: `repository root`
- `doctor`: status=`pass`
- host: local workstation（固有hostnameは非公開）
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
- working tree: `repository root`
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

## Phase 3B: Qwen3-TTS serving comparison

Phase 3Bは同じ`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`、speaker=`Ono_Anna`、language=`Japanese`、24 kHzを、official Python API、vLLM-Omni non-streaming HTTP、vLLM-Omni HTTP raw-PCM streamingで比較した。Phase 3Bの実装開始基準は指定された`8a4981801f44643d005f8c31549858f44bc60c0c`で、作業treeにはその後のPhase 3A metadata correction `8fd208c1c50f45122b73bffc3c231453d1d78e5f`が含まれる。

### 公式仕様・version

確認した公式sourceは`docs/vllm-omni.md`にも固定記録している。

- repository main HEAD at inspection: `bc0c9f4b45c45c59aa2f92471842e8c18ae403ca`
- benchmark package: `vllm-omni==0.28.0`、official tag commit `eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c`
- companion package: `vllm==0.28.0`
- official source: `docs/serving/speech_api.md`、`vllm_omni/deploy/qwen3_tts.yaml`、`recipes/Qwen/Qwen3-TTS.md`
- official speech API: `POST /v1/audio/speech`; non-stream responseはWAV、streamは`stream=true`、`stream_format=audio`、`response_format=pcm`、`speed=1.0`、`sample_rate=24000`でraw PCM chunkを返す
- raw PCM benchmark format: signed 16-bit、mono、24 kHz。HTTP chunk境界はsample boundaryを仮定せず、奇数byteを次chunkへcarryした
- official Qwen3-TTS recipeは0.6B CustomVoiceを含む。voices endpointは`ono_anna`をadvertiseするが、request payloadは要求どおり`Ono_Anna`を保持した
- deploy default: `async_chunk=true`、`initial_codec_chunk_frames=1`。API field omitted（null相当）、explicit `1/2/4`を比較した。async OFFはserver再起動を伴うため今回未実施
- official WebSocket routeは`/v1/audio/speech/stream`で、`session.config`、`input.text`、`input.done`、`session.close`を受けてsentence-scoped audioをstreamする。server→clientの`text.delta`は無い。HTTP full-text PCMで先に効果を確認する方針のため未実装・未測定

導入は既存`.venv`と分離した`~/.venvs/local-live-vllm-omni-0.28.0`へ行った。serverは`127.0.0.1:8091`、single GPU、official deploy YAMLで起動し、readinessは`GET /v1/audio/voices`で確認した。最初のofficial FlashInfer sampler defaultはhostに`nvcc`が無いためJIT compile前に失敗した。modelやengineを変えず、server processだけ`VLLM_USE_FLASHINFER_SAMPLER=0`のPyTorch sampler fallbackで再起動し、readiness・request・停止が成功した。このoverrideは性能最適化ではなくhost compatibility workaroundである。standalone 25×3の本測定はserverのspeed/sample-rate default（1.0/24 kHz）を使用し、その後のcontract smokeでclientが明示する`speed=1.0`・`sample_rate=24000` payloadの受理を確認した。

server startupは`56.37 s`（process start → voices readiness）、logの2 stage model-load合計は`4.49 s`（`2.61 s + 1.88 s`）。standalone server readiness時VRAMは`8,614 MiB`、server停止returncodeは0だった。

### standalone serving benchmark

5種類の同一日本語response（短い肯定、短文、数字、技術用語、2文）を各方式で5回測った。Python modeは最初のmodel load rowを保持し、warm-only集計では`model_load_seconds=0`の24 rowsを使った。vLLM modeはserver resident状態の25 rowsである。physical rowsはすべてstable USB speaker/microphone target、raw USB capture、outlier保持である。

| mode | measured rows | request → first PCM/full audio median | request → first actual speech PCM median | physical first audio median |
|---|---:|---:|---:|---:|
| official Python API (warm-only) | 24 | 2.3128 s | 2.8472 s | 2.6950 s |
| vLLM-Omni non-streaming | 25 | 0.4451 s | 1.0451 s | 1.2700 s |
| vLLM-Omni HTTP streaming | 25 | 0.0376 s | 0.5479 s | 0.9600 s |

vLLM HTTP streamingは全25 rowsで`first_audio_chunk_received`、`first_audio_chunk_queued`、`playback_stream_started`、`last_audio_chunk_received`、`playback_completed`を保存した。chunkごとの`pw-play`起動は行わず、各runの1本のpersistent `pw-cat` stdinへPCMを継続供給した。stream generated audioのstable onsetはmedian`0.5479 s`相当であり、network first chunk `0.0376 s`と区別した。

### initial_codec_chunk_frames

同じ短文を各3回、stream playback sinkへ流して比較した。全候補12/12 rowsがmeasured、clipping ratioは0、PCM continuity errorは無かった。

| request value | first PCM median | final duration median | CER代表row |
|---:|---:|---:|---:|
| omitted / null | 0.0386 s | 3.12 s | 0.0000 |
| 1 | 0.0429 s | 3.20 s | 0.0000 |
| 2 | 0.0635 s | 2.96 s | 0.0000 |
| 4 | 0.0730 s | 2.00 s | 0.0000 |

最短はrequest field omitted（server deploy defaultを使う）だった。explicit `4`は速さだけでなくdurationも短くなるため、今回の採用値にはしない。main streaming benchmarkはfield omittedで測定した。

### final live E2E

経路はsynthetic user WAV → GPU `large-v3-turbo/int8_float16` → local Ollama `qwen3.5:9b-q4_K_M` → natural sentence boundary first chunk → vLLM-Omni HTTP streaming → first PCM → persistent playback → raw USB microphoneである。10 measuredを得るまで最大14 attemptsを許容し、physical onset未検出のattemptも削除しなかった。

| attempt | result |
|---:|---:|
| 1 | 6.9400 s |
| 2 | 1.6200 s |
| 3 | blocked: raw microphone acoustic onset not detected |
| 4 | 1.8200 s |
| 5 | 1.4100 s |
| 6 | 1.2500 s |
| 7 | 1.8700 s |
| 8 | 1.7300 s |
| 9 | 1.5800 s |
| 10 | 1.8400 s |
| 11 | 1.8900 s |

10 measured rowsのindividual valuesは`6.9400 / 1.6200 / 1.8200 / 1.4100 / 1.2500 / 1.8700 / 1.7300 / 1.5800 / 1.8400 / 1.8900 s`。summaryはmedian`1.7750 s`、p95相当`4.6675 s`、mean`2.1950 s`、standard deviation`1.6804 s`、min/max`1.2500 / 6.9400 s`である。6.94秒は削除していないoutlierで、3回目のblocked rowもJSONに残した。

vLLM streaming final E2Eのrequest→first PCM medianは`0.0588 s`、request→first actual speech PCM medianは`1.0039 s`。standalone servingのstream request→first actual PCMは`0.5479 s`である。同じ機器で測ったpure physical playback path medianは`0.2186 s`。Phase 3A Python baseline median`2.7550 s`に対する改善率は`35.57%`である。

### GPU memory / quality / cancellation

| observation | VRAM |
|---|---:|
| vLLM server standalone ready | 8,614 MiB |
| vLLM server + Ollama resident + Whisper loaded | 13,790 MiB |
| live pipeline during E2E | 15,085 MiB |
| after server stop (Ollama remains) | 6,043 MiB |

16,303 MiB GPUでOOMは発生しなかったが、live pipelineのfree memoryは約1.2 GiBまで減った。既存Ollama processは停止・再設定していない。server停止後にvLLM process/resource trackerは消え、audio settings（speaker/microphone volume、mute、default sink/source）はread-backで復元確認した。

同じmodel/speaker/languageを使った3方式のWhisper roundtripは15 representative rowsで測定し、各rowにtranscript、CER、duration、RMS、peak、clipping ratioを保存した。clipping ratioは全方式0。絶対CERはresponse/textや生成samplingの差で変動したが、vLLM serving方式だけに一貫した明らかな悪化は観測されなかった。initial frame probe代表CERも全候補0である。MOSは実施していない。

stream parser、PCM format、chunk ordering、empty chunk、connection interruption、cancellation、persistent playback cleanup、backend switch、server unavailable、timing aggregationは`tests/test_phase3b_contract.py`を含むserial pytestで自動検証した。midstream cancellationではfake persistent processがterminateされ、古いPCMをqueueし続けないことを確認した。

### Phase 3B判定

vLLM-Omni HTTP streamingのfinal physical medianは`1.7750 s`で、第一目標`<2.0 s`を達成した。stretch goal`<1.5 s`は未達である。判定帯は1.5–2.0秒なので、固定モデルのままQwen3-TTS + vLLM-Omni streamingをLive既定候補として継続し、official Python backendをfallbackとして残す。別TTS model/engine比較へ直ちに進む条件（serving変更後も`>=2.5 s`）には該当しない。

incremental text WebSocketはHTTP streamingが既に2秒未満のため、次の必須作業にはしない。6.94秒outlierやLLM first-chunk変動の安定化が必要になった場合に、公式WebSocket `input.append`/`input.done`を次の比較候補とする。async OFF比較も同じ理由でP0ではなく、現行結果はofficial async default ONのみである。

Phase 3Bの機械可読結果は`results/bench_tts_serving.json`、final live E2Eは`results/bench_live_latency.json`、集約は`results/summary.json`である。生成WAV/PCM、raw capture、server log、model weight、cache、credentialはGit管理しない。

## Phase 4: Continuous Live Stability + Self-Echo Rejection

Phase 4はPhase 3Bの固定経路を継続した。ASRは`faster-whisper large-v3-turbo` GPU `int8_float16`、LLMはlocal Ollama `qwen3.5:9b-q4_K_M`、TTSは`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` / `Ono_Anna` / `Japanese` / 24 kHz、servingはmatching `vllm-omni==0.28.0` + `vllm==0.28.0` HTTP raw-PCM streaming、persistent PCM playbackである。Python Qwen3-TTS backendはfallbackとして維持し、vLLM 0.29/main、別model、FlashInfer/nvcc対応は行っていない。

### Continuous stability

同一質問の反復を避け、8種類の短い日本語fixture WAVを循環し、resident serverで50 attemptsを実行した。physical onset未検出turnは削除せずblockedとして保存した。

| metric | result | target / interpretation |
|---|---:|---|
| attempts | 50 | target 50 attempts |
| measured turns | 48 | 2 turns blocked by physical onset detector |
| failed turns | 0 | no pipeline error or hang |
| blocked turns | 2 | turns 1–2; retained in JSON |
| physical first audio median | 1.7850 s | target <2.0 s: met |
| p90 | 2.2860 s | recorded |
| p95 | 2.3990 s | target <2.5 s: met |
| p99-equivalent | 2.5937 s | recorded |
| max | 2.7300 s | recorded |
| failure rate | 4.0% | blocked onset attempts included; failed=0 |
| physical onset success | 96.0% | target >=95%: met |

Turn-level JSON includes ASR duration, LLM TTFT/total, first text chunk, TTS request, first PCM, first actual speech PCM, playback start/complete, physical onset, total turn latency, VRAM/RAM/CPU, vLLM/Ollama health, cancellation state, retry state, and error fields. Stage medians were `ASR 0.2577 s` (ASR-only duration `0.2191 s`), LLM TTFT `0.1937 s`, LLM total `0.6069 s`, first-sentence buffering `0.4149 s`, TTS request→first PCM `0.0647 s`, TTS request→first actual speech PCM `0.5540 s`, and total turn `1.7850 s`.

Outlier classification uses component median ratios and a minimum delta, not a fixed 3-second rule. Among 15 flagged rows, the dominant causes were `first_pcm_to_actual` 7, `first_sentence_buffering` 6, and acoustic onset detection 2. The first turn also exposed cold Ollama/LLM behavior (`LLM TTFT 23.7647 s`) and was blocked by the physical detector; this is recorded separately from the Phase 3B 6.94-second event.

The Phase 3B 6.9400-second row was reclassified from its saved stage values: `tts_request_to_first_pcm=2.2791 s`, 38.8x its component median, was the dominant relative cause; ASR was also elevated at 2.7188 s (9.4x), while LLM TTFT and downstream stream-to-physical were not dominant. This identifies the primary hypothesis as a TTS first-PCM/server contention or cold-path event, not an unclassified physical-only event; the measurement does not prove a single internal vLLM kernel cause.

Warm-state medians by 10-turn window were:

| turns | TTFA (TTS request→first actual PCM) | physical first audio | VRAM peak | process RAM peak |
|---|---:|---:|---:|---:|
| 1–10 | 0.9206 s | 2.1500 s | 15,101 MiB | 4,159.24 MiB |
| 11–20 | 0.5018 s | 1.7400 s | 15,101 MiB | 4,159.25 MiB |
| 21–30 | 0.5176 s | 1.7300 s | 15,101 MiB | 4,159.26 MiB |
| 31–40 | 0.5167 s | 1.7700 s | 15,101 MiB | 4,159.27 MiB |
| 41–50 | 0.5681 s | 1.8200 s | 15,101 MiB | 4,159.30 MiB |

The first-window TTFA includes the cold path. First-to-last 10-turn drift was 0 MiB VRAM and +0.06 MiB process RAM; no clear memory leak was detected. No GC or cache clear was forced per turn. The component remains `measured_with_limitations`, because 48/50 attempts produced measurable physical onset rather than 50/50 measured onsets, even though the attempt count and three numeric latency targets passed.

### Server lifecycle and interruption

The benchmark performed server start/readiness, 50-turn use, clean stop, restart, and three post-restart fixture turns. Restart readiness was HTTP 200 and all 3 restart turns were measured. The vLLM server used the isolated Python environment and FlashInfer sampler fallback `VLLM_USE_FLASHINFER_SAMPLER=0`; this was unchanged from Phase 3B.

A real vLLM streaming turn was programmatically interrupted after the first PCM queue. `queued_pcm_discarded=true` and the standalone blocking-LLM cancellation probe observed `llm_stream_cancelled=true` with a 5.3 ms cancel-to-return. `pipeline.cancel()` stopped the persistent PCM process, discarded stale queued audio, propagated cancel to the active vLLM HTTP client, and returned the pipeline to cancelled state with empty `spoken_text`. Software playback stop was `0.1986 s`, meeting the `<=0.200 s` target; `vllm_http_stream_cancelled=true` was observed. No separable post-interrupt physical microphone interval was detected, so physical stop latency is `null`, not fabricated. This is a physical-stop measurement limitation, not a claim of zero acoustic tail.

### Self-echo rejection

The adopted candidate is a post-VAD reference-aware layer, not a replacement AEC: 80 ms windows / 40 ms hop, bounded normalized reference correlation with lag, microphone/reference energy ratio, and residual unexplained energy after reference projection. When assistant playback is active and reference explains the AEC-output candidate, it returns `probable_self_echo` and rejects it; otherwise it returns `possible_user_speech`. Thresholds were selected by grid search over the measured fixture distributions rather than fixed before measurement.

The automated dataset used 10 assistant-only fixtures (delayed/scaled assistant reference plus noise) and 10 synthetic user-like fixtures (the same residual echo plus an independent injected signal). It is a regression fixture, not a physical double-talk success test and not a production guarantee.

| metric | result | target |
|---|---:|---:|
| assistant-only VAD-positive fixtures | 10 | recorded |
| assistant-only echo rejects | 10/10 | recorded |
| assistant-only false accept | 0/10 = 0% | <=10%: met |
| synthetic user-like accepted | 10/10 = 100% | >=90%: met |
| synthetic user-like false reject | 0% | recorded |

Measured thresholds were correlation `0.9642`, max lag `66 ms`, energy ratio `0.1417–0.4390`, and max short-window residual ratio `0.9`. The simple mute reference gives 0% assistant-only false accept by disabling VAD, but it was not adopted because it removes future barge-in candidates. Phase 4 uses the echo-aware candidate instead; AEC filter parameters were not retuned.

### Real microphone readiness

A short automated readiness recording resolved the raw USB microphone target and an AEC source, measured RMS/peak/clipping/VAD, loaded and transcribed the fixed GPU ASR, and restored the audio settings. Raw capture RMS/peak were `0.02542 / 0.11334`, clipping `0`; AEC capture RMS/peak were `0.003285 / 0.01990`, clipping `0`; AEC VAD speech ratio was `0.0` with no human speech required; ASR loaded/transcribed successfully with empty content. The raw/AEC recorder returned code 1 after controlled stop but produced valid duration-bearing WAVs; this return code is retained in the artifact. The AEC readiness master used the available GoStream sink while leaving the physical playback target unchanged. `mic-readiness` is `measured`, not a human conversation approval.

### Phase 4 files and judgement

- `results/bench_stability.json`: 50 continuous attempts, retained blocked rows, stage/outlier classification, warm windows, server restart, memory, and interruption link
- `results/bench_echo_rejection.json`: fixture rows, correlation/lag/energy distributions, calibrated thresholds, and mute reference
- `results/bench_mic_readiness.json`: raw USB/AEC/VAD/ASR readiness
- `results/bench_interruption.json`: independent resident-server interruption measurement
- `results/summary.json`: aggregate and component judgement
- `docs/echo-rejection.md`: algorithm and limitation notes

Component judgement is `continuous_stability=measured_with_limitations`, `echo_rejection=pass`, `interruption=pass`, `server_restart=pass`, and `microphone_readiness=measured`. Phase 4 overall is `measured_with_limitations`: continuous latency and onset thresholds passed, but two initial physical-onset blocks and the unresolved physical-stop tail keep the result from being a blanket production claim. Phase 6 subsequently completed the unattended physical-onset measurement hardening; human real-microphone conversation remains `deferred_manual`, not an automatically scheduled next phase.

## Phase 5: unattended robustness and synthetic double-talk

Phase 5 was run without human speech, human barge-in, subjective listening, manual gain/device changes, or production-readiness approval. The fixed configuration remained `large-v3-turbo` / GPU `int8_float16`, Ollama `qwen3.5:9b-q4_K_M`, Qwen3-TTS CustomVoice `Ono_Anna` / Japanese, vLLM-Omni/vLLM `0.28.0`, HTTP raw PCM streaming, and the existing AEC path. Generated WAVs and logs remain ignored artifacts.

### Extended echo-rejection dataset

The dataset contains 40 deterministic rows: 20 assistant-only and 20 synthetic user-like rows. It uses the existing distinct Japanese `phase4_input_technical.wav` fixture (`PipeWireとCUDAの状態を教えてください。`) rather than copying the assistant reference. The 20 conditions cover assistant-start, assistant-middle, assistant-end, and assistant-end-or-after offsets; weak-user/strong-echo, equal-level, and strong-user/weak-echo levels; noise RMS `0.0005/0.002/0.006`; and lag `20/40/60 ms`. The first evaluation applied the Phase 4 threshold unchanged. Calibration and validation were stratified 50/50 (10 assistant-only + 10 user-like rows per split); the calibrated candidate was reported separately and was not used to hide fixed-threshold results.

| validation metric | fixed Phase 4 threshold | calibration-only threshold applied to validation | target |
|---|---:|---:|---:|
| assistant-only false accept | 0/10 = 0% | 0/10 = 0% | <=5% |
| synthetic user-like acceptance | 10/10 = 100% | 10/10 = 100% | >=95% |
| synthetic user-like false reject | 0% | 0% | recorded |

The fixed threshold remained correlation `0.9642105263`, lag `<=66 ms`, energy ratio `0.1416587676–0.438991174`, residual ratio `0.9`, 80 ms window, and 40 ms hop. Score distributions and per-row threshold margins are in `results/bench_echo_rejection.json`. The analyzer now includes active-reference window correlation and evaluates strong unmatched trailing/leading microphone energy; it does not introduce an ML classifier or replace AEC.

### Physical onset repeat and detector sensitivity

The same fixed 3.04 s assistant reference was played 30 times. All 30 playback commands succeeded, all 30 captures had valid duration and nonzero RMS/peak, PipeWire reported 6 devices (3 sinks/3 sources) on each checked repeat, and no repeat was blocked or failed. The raw recorder returned `1` after controlled SIGTERM while valid WAVs were produced; this return code is retained and classified as expected for this path. Onset detection was 30/30 (100%), with no blocked-cause classification. The repeated rows retain expected onset (`1.04 s`), detected onset, recording length, raw RMS/peak, cross-correlation, device availability, PipeWire health, and threshold sensitivity.

The bounded synthetic detector suite at 3x/4x/5x noise multipliers produced 0 false negatives, 0 false positives, and 0.0 s timing error. Physical replay showed one early false-positive candidate and 14 late-onset candidates against the generated-signal expectation; the largest timing error was `0.97 s`, while median absolute timing error was `0.25 s`. These candidates had valid capture and playback, so they are physical timing/reference-alignment limitations rather than missing-speaker or missing-microphone evidence. No production threshold relaxation was adopted.

### 100-turn stability, outliers, and resources

The complete unattended orchestrator run completed 100/100 measured turns, 0 blocked, and 0 failed. A follow-up resource-instrumented stability remeasurement completed 100 attempts, 99 measured, 1 blocked, and 0 failed; the blocked row was turn 57 with `physical_audio_not_detected`, not a resource-monitor failure. The remeasurement's physical first audio was median `1.739999006 s`, p95 `2.3839988367 s`, p99 `4.0799983973 s`, and max `5.059998266 s`. The 18 remeasurement outliers were retained; dominant classification was `first_sentence_buffering` (11), with acoustic onset (2), first-PCM-to-actual (4), and LLM TTFT (1) also recorded. The earlier 28.45 s maximum remains in the complete orchestrator artifact and was not discarded or relabeled as normal latency.

VRAM for the complete orchestrator was baseline `9,830 MiB`, peak `15,091 MiB`, free minimum `751 MiB`, and OOM count `0`. The resource-instrumented remeasurement was baseline `15,015 MiB`, peak `15,123 MiB`, free minimum `719 MiB`, and OOM count `0`; neither crossed the <500 MiB warning. First-vs-last-10-turn drift in the remeasurement was `0 MiB` VRAM and `+3.066 MiB` RAM. Across all 100 remeasurement rows, FD start/end was `43 -> 43`, per-turn delta max `0`, and clear monotonic-growth turns `0`; child-process and playback-process monotonic-growth turns were also `0`. The 1,990 process-resource samples therefore explain the earlier outer `4 -> 43` as a one-time warm-cache/runtime step, not a continuing descriptor leak. The active-HTTP counter in the new series excludes TIME_WAIT and records a stable per-turn value. No GC or cache clear was forced per turn.
Server lifecycle completed start/readiness, 100 turns, clean stop, restart, 5 post-restart turns, and clean stop. Restart readiness was HTTP 200; completed restart runs were `5/5`. The vLLM FlashInfer fallback environment remained `VLLM_USE_FLASHINFER_SAMPLER=0`.

### Fault and cancellation resilience

The bounded fault matrix measured all six probes: vLLM HTTP stream disconnect, playback process premature exit, empty/invalid PCM, injected Ollama request failure, unavailable vLLM health, and mock unavailable microphone target. Each returned without a hang; playback cleanup and error/state classification were retained, and the matrix reported next-turn recovery.

Cancellation stress ran first-PCM, playback-middle, and playback-end queue boundaries five times each (15/15). Software stop median was `0.003183324 s` (3.18 ms), pipeline IDLE rate `100%`, recovery rate `100%`, and stale PCM count `0`. `spoken_text_correct` was true for all 15 rows: canceled/incomplete PCM was not committed as spoken text, while the post-cancel recovery probe committed only its completed response. HTTP stream was active and its close cancellation was observed at all 15 interrupt points; no late PCM was successfully queued after cancellation.

### Phase 5 judgement and artifacts

Phase 5 remains `measured_with_limitations`, not a production-readiness claim. The numeric stability, echo validation, cancellation, fault, server-restart, audio-state restore, VRAM-margin, and per-turn resource candidates passed. The remaining measured limitation is physical onset timing/reference alignment, including one blocked row in the resource-instrumented 100-attempt rerun; the earlier complete orchestrator run had 100 measured rows. The per-turn series found no continuing FD, child-process, or playback-process growth. Human speech, physical double-talk/barge-in, MOS/listening, manual gain tuning, and production approval are `deferred_manual` by design and are not blockers for this unattended phase.

- `results/bench_unattended.json`: complete Phase 5 orchestration, all 100 turn rows, 30 onset rows, resources, VRAM, fault and cancellation links
- `results/bench_echo_rejection.json`: 40 fixture rows, fixed-threshold first evaluation, calibration/validation split, distributions, margins, and calibrated comparison
- `results/bench_stability.json`: 100-turn rows, retained outliers, restart lifecycle, free-VRAM and memory summaries
- `results/bench_interruption.json`: 15 cancellation rows, stale-PCM and `spoken_text` checks, recovery probe
- `results/summary.json`: compact Phase 5 aggregate
- `docs/unattended-validation.md`: unattended protocol and exclusions

## Phase 6: physical onset measurement hardening

Phase 6はturn 57の既存artifactを先に診断し、measurement evidenceをplayback、raw microphone、actual-reference alignmentへ分離する。固定構成、ASR/LLM/TTS、speaker、vLLM version、AEC、Phase 4 echo thresholdは変更しない。人間発話、human double-talk/barge-in、MOS、主観音質、manual gain/device tuning、production approvalは`deferred_manual`とする。

`results/turn57_diagnosis.json`では、旧turn 57にraw WAVとaggregate timingはあったが、実PCM write count、persistent playback exit status、short-frame energy series、actual-playback-PCM confidence/marginが`not_recorded`だったことを明示した。Phase 6ではHTTP chunk境界ではなく、persistent `pw-cat`へ実際にqueueした連結PCMをreferenceにする。

固定replayの実測は100/100 measured、confirmed 100、correlation_recovered 0、blocked 0、failed 0、unknown 0だった。追加fixtureは5種類×10回=50/50 measured、confirmed 50。no-playback negativeは30/30 measured、false positive 0/30（0%）、unknown 0/30だった。全positive 150件でdevice availabilityはtrue、PipeWire healthはready、persistent playback process exitは0だった。既知physical path分布5件（median 0.218581 s、p95 0.718257 s、max 0.838216 s）からexpected onset windowを導出し、固定値だけに依存しなかった。

最終100-turn stabilityは100 attempts、application success 100/100、application failed 0、physical measurement confirmed 100/100だった。confirmed-only latencyはmedian 0.365884 s、p95 0.690515 s、p99 0.832720 s、max 0.847195 s。confirmed+recoveredも同じ100件で同値だった。旧energy detectorだけでなくactual PCM alignmentを保存し、turn 57はapplication successかつ`confirmed`へ再分類された。energy detectorのwindow外7件はactual PCM alignmentで回復し、`energy_detector_late_reference_recovered`として記録した。Phase 6でphysical path window外へ残ったlate onsetは0件だった。

resourceはPhase 5のper-turn monitorを回帰確認として再利用した。FD/child/playback/active HTTPのper-turn monotonic growthは全て0、VRAM peak 15,089 MiB、free minimum 753 MiB、OOM 0、memory leak suspected falseだった。外側snapshotのFD `4→43`はwarm-cache/runtime初期化差分として注記し、leak判定には使っていない。server restartは5/5、owned stop returncode 0、audio snapshot restore errorはnull、8091は終了後解放された。

first_sentence_bufferingは100件観測、median 0.349629 s、max 0.775215 s、outlier dominant 10件だった。自然文boundaryは100/100、timeout fallbackの原因タグは`not_recorded`だが、run中のhang/errorはなく、既存chunkerのbounded behaviorと自然文境界variationでありbugは確認しなかった。総合判定は`unattended_validation_complete`である。

- `results/bench_physical_onset.json`: Phase 6 fixed replay、fixture replay、negative controls、三系統evidence、expected window、分類、audio state restore
- `results/bench_stability.json`: 100-turn application/measurement status分離、actual PCM alignment、confirmed-only/confirmed+recovered latency
- `results/bench_unattended.json`: Phase 6総合判定、physical onset、stability、resource regression、restart、deferred_manual
- `results/summary.json`: compact Phase 4/5/6 aggregate
- `docs/physical-onset.md`: protocol、turn 57 diagnosis、分類、PASS条件
