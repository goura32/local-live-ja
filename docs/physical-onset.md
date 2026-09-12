# Phase 6 physical onset measurement

Phase 6は、実人間発話を使わずに、physical onsetの測定器と実アプリケーションの成否を分離するための無人検証である。人間発話、human double-talk/barge-in、MOS、聞き比べ、manual gain/device tuning、production approvalは実施せず、`deferred_manual`に記録する。

## turn 57 diagnosis

既存`results/bench_stability.json`のturn 57は、旧energy detectorでは`physical_audio_not_detected`だった。保存artifactから確認できた範囲は次のとおり。

- TTS/network first PCM、first actual speech PCM、PCM aggregate、playback stream start/end、record start/end、PipeWire inventory/healthは記録済み。
- raw mic WAVは存在し、4.904875秒、RMS `0.0303691`、peak `0.457825`、clipping `0`だった。
- detector Aのnoise floor、threshold、search start、correlation scalar、lagは記録済みだが、short-frame energy seriesはなかった。
- `pw-cat`へ実際に書いたbytes、write count、persistent playback processのexit status、実PCM基準のalignment confidence/marginは記録されていなかった。
- したがって、旧artifactだけからspeaker欠落・capture欠落・detector false negative・reference mismatchを確定してはいけない。未保存項目は`results/turn57_diagnosis.json`で`not_recorded`とした。

## evidence

各positive replayは以下を別フィールドで保存する。

A. playback evidence

- actual playback PCM bytesをHTTP chunk境界から独立して連結
- first PCM received / first actual speech PCM
- `pw-cat`へqueueしたbytes、write count、先頭write timestamps
- persistent playback process alive-at-start、exit status、completed

B. raw microphone evidence

- recording duration、RMS、peak、clipping
- 10 ms short-frame RMS series（partial final frameはpaddingして保存）
- adaptive noise floor、threshold、detected onset、search start
- device availability、PipeWire health

C. actual-reference alignment evidence

- authoritative PCM（persistent playbackへ実際にqueueしたPCM）を16-bit PCMとして参照
- recordingとのnormalized correlation
- reference active windowの探索位置、best lag、aligned onset
- signed correlation、absolute confidence、confidence margin

referenceはTTS request text、予定生成音声、HTTP chunk境界ではない。`pw-cat`へ渡した連結PCM byte streamをauthoritativeとする。

## dual detector and classification

Detector Aは既存adaptive short-frame RMS detectorを使用し、thresholdをPhase 4/5の目的で緩めない。Detector Bはactual playback PCMの複数短窓normalized correlation/alignmentであり、音響経路の位相反転はsigned値とabsolute値を併記する。

最終分類は単純なORではない。

- `confirmed`: energy onsetあり、reference alignmentあり、expected window内
- `correlation_recovered`: energy onsetはmissだがreference alignmentが十分でexpected window内
- `energy_only`: energy onsetあり、reference alignmentが弱い
- `microphone_capture_failure`: playbackは正常だがrecording duration/levelが異常
- `playback_failure`: PCM write/process/completionが異常
- `late_outside_window`: reference alignmentが既知のpath window外。energy onsetだけが遅く、reference alignmentがwindow内なら`confirmed`のまま`energy_detector_late_reference_recovered`を別タグにする
- `no_physical_match`: recordingは存在するがreference matchなし
- `no_playback_negative`: playbackなしでenergy onsetなし
- `false_positive`: playbackなしでenergy onsetあり
- `unknown`: 証拠不足または解析error

`application_status`はplayback/recording処理の成功を表し、`physical_measurement_status`は上記分類を表す。detector missだけでapplicationをfailedにしない。

## expected onset window

`first_pcm_written + generated reference onset + prior physical replay path distribution`からwindowを生成する。固定秒数を唯一の根拠にしない。Phase 3Aの`bench_playback_path`に保存された分布をsourceとして、低値・高値・中央値・p95を保存し、marginを明示する。record tailはplayback durationと分布上限に基づき、毎回無制限に延長しない。

## run matrix

- fixed reference: 100 replay以上
- no-playback negative capture: 30回
- independent Japanese fixtures: 5種類 × 10回
- final stability: 100 attempts

raw WAV/PCMとshort-frame seriesは`results/artifacts/`へ置き、Git管理対象外とする。失敗artifactは削除しない。

## PASS判定

- fixed replay classifiable >= 99%
- fixed replay `confirmed + correlation_recovered` >= 99%
- negative false positive = 0
- unknown <= 1%
- final stability application failed = 0、physical confirmed/recovered >= 99%
- confirmed-onlyとconfirmed+recoveredのlatencyを別集計
- Phase 5で確立したFD/child/playback/live-HTTP/VRAM/RAM監視にregressionなし

全条件を満たした場合のみ`unattended_validation_complete`とする。未達時は具体的なclassificationとartifactを残し、推測でPASSにしない。

## measured result

- fixed reference: 100/100 measured、100 confirmed、0 blocked/failed/unknown
- independent fixtures: 50/50 measured、50 confirmed
- no-playback negative: 30/30 measured、false positive 0、unknown 0
- final stability: 100/100 application success、100/100 physical confirmation
- confirmed-only latency: median `0.365884 s`、p95 `0.690515 s`、p99 `0.832720 s`、max `0.847195 s`
- legacy energy onsetのwindow外候補7件はactual PCM alignmentでwindow内へ回復し、`energy_detector_late_reference_recovered`として記録。Phase 6のphysical path window外は0件。
- per-turn FD/child/playback/live-HTTP monotonic growthは全て0、VRAM free minimum `753 MiB`、OOM 0、audio restore errorなし。
- 判定: `unattended_validation_complete`
