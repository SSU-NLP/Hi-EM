# 벤치마크 정리

## 평가 축 구분 (중요)

| 목적 | 벤치마크 | 메트릭 |
|---|---|---|
| **Topic 경계 감지** (Phase 1) | **TopiOCQA**, **TIAGE** | turn-transition binary F1 |
| **Downstream QA** (Phase 4) | LoCoMo, LongMemEval | QA accuracy (GPT-4o judge) |

**LongMemEval / LoCoMo를 topic 경계 감지에 쓰면 안 된다** — turn-level topic label이 없고 session 경계는 weak proxy. 내부적으로 한 세션에 여러 subtopic이 공존할 수 있으므로 session-as-topic 가정은 over-segmentation을 FP로 잘못 처벌한다.

## Tier 1 (Downstream QA — Phase 4): 장기 대화 메모리

### LoCoMo
- 경로: `benchmarks/locomo/`
- 데이터: `benchmarks/locomo/data/locomo10.json` (레포 내 포함)
- 구조: 10개 대화, multi-day 세션 구성
- 평가: QA accuracy, event summarization
- 논문: ACL 2024
- 레포: https://github.com/snap-research/locomo

### LongMemEval
- 경로: `benchmarks/LongMemEval/`
- 데이터: HuggingFace (레포 README 참조)
- 구조: 500 질문, 5개 메모리 능력
  1. Information extraction
  2. Multi-session reasoning
  3. Temporal reasoning
  4. Knowledge updates
  5. Abstention
- 평가: QA accuracy (GPT-4o judge). **Topic 경계 감지 용도 아님.**
- 논문: ICLR 2025
- 레포: https://github.com/xiaowu0162/LongMemEval
- **(2026-05-16) evidence/retrieval 지표 미산출 한계**: `scripts/analyze_evidence_topics.py`
  의 evidence→topic concentration 및 H@k/R@k/P@k 는 LoCoMo `dia_id`
  evidence 구조 전제. LongMemEval 은 turn-level `has_answer` 구조라
  `evidence_topic_summary.json` 이 전부 null/0 으로 나온다 — LongMemEval
  run 에서는 "왜 졌나"를 retrieval 수준으로 정량 추적 불가. 단일 idx
  haystack 진단은 `scripts/inspect_longmemeval_segmentation.py` (turn /
  question / `--user-only` / `--version v2|v3.3.4|v3.3.5|v3.3.6|v3.3.7|v3.3.8`)
  로 별도 수행. cross-idx user-turn concat 은 idx seam artifact 때문에 금지
  (decision-log 2026-05-16 segmentation 평가 프로토콜).
- **(2026-05-17) 변형(oracle vs s)**: 로컬엔 `longmemeval_oracle.json`
  (증거 세션만, 질문당 ~6세션/36턴)만 있었음. retrieval 난이도 제거판이라
  segmentation/추론만 측정 가능. `longmemeval_s_cleaned.json`(277MB, HF
  `xiaowu0162/longmemeval-cleaned`) 추가 — 질문당 ~48세션(증거+distractor)
  /~245 user턴, distractor 는 ShareGPT/UltraChat 재활용(세션당 user턴
  0~66 불규칙). `has_answer`/`answer_session_ids` 동일. **s 의 idx 정렬이
  로컬 oracle 과 다름 → qid 매칭 필수** (로컬 oracle 은 cleaned 이전
  구버전 가능). 증거/distractor 분리 진단 = `scripts/inspect_longmemeval_s.py`
  (`--qid`/`--idx`/`--with-distractors`/`--out`). 평가 metric =
  `evidence_recall@K`(topic-level primary, session-level 보조);
  `evidence_cohesion` 폐기 (decision-log 2026-05-17).

## Tier 1 (Topic 경계 감지 — Phase 1): Hi-EM segmentation 검증

### TopiOCQA
- 경로: `benchmarks/topiocqa/`
- 데이터: `python download_data.py --resource data.topiocqa_dataset.dev`
- 구조: dev 2514 turns, 205 conv, 평균 12턴, 3.3 shift/conv
- Topic 정의: Wikipedia document 경계 (명시 annotation)
- 평가: topic shift detection F1
- 레포: https://github.com/McGill-NLP/topiocqa
- **특성**: factoid QA, shift rate 28%/transition — frequent-shift regime

### TIAGE
- 경로: `benchmarks/tiage/`
- 데이터: `benchmarks/tiage/data/personachat/anno/{train,dev,test}/anno_*.json` (레포 내 포함)
- 구조: train 300 / dev 100 / test 100 conv, 평균 15.6턴, 3.15 shift/conv (test 기준)
- Turn label: `-1`(첫 턴) / `0`(continue) / `1`(shift). 인간 주석, Cohen's Kappa 0.48.
- 평가: topic shift detection F1 (turn-transition binary)
- 레포: https://github.com/HuiyuanXie/tiage
- **특성**: PersonaChat 기반 chit-chat, shift rate 20%/transition, 짧은 utterance (~50자)

---

## 데이터 준비 체크리스트
- [x] LoCoMo clone (`benchmarks/locomo/`)
- [x] TopiOCQA clone (`benchmarks/topiocqa/`)
- [x] LongMemEval clone + HF 데이터 다운 (`benchmarks/LongMemEval/data/`)
- [x] TIAGE clone (`benchmarks/tiage/`)
- [x] LoCoMo 데이터 확인 (레포 내 포함)
- [x] TopiOCQA train/dev download (via `download_data.py`)

---

## 데이터 분석 체크리스트 (Phase 0 완료 기준)

각 벤치마크별 필수 분석:
- [ ] 샘플 1~2개 JSON 구조 직접 확인
- [ ] 평균/중간값/최대값: 세션 수, 턴 수, 쿼리 토큰 길이
- [ ] Topic 전환 패턴 (명시 annotation 여부, 전환 빈도, 유형)
- [ ] Claude-유사 대화(코딩/글쓰기/브레인스토밍)와의 유사성 평가
- [ ] 해당 벤치마크에서 **어떤 사건 모델이 유리할지** 판단

분석 결과 → `outputs/benchmark-analysis.md`

---

## 주의: 벤치마크별 bias

각 벤치마크는 서로 다른 대화 성격을 가진다:
- **TopiOCQA**: Wiki 기반 정보 검색 QA. Named entity 풍부.
- **TIAGE**: PersonaChat 기반 chit-chat. Cue phrase 많음.
- **LoCoMo**: Multi-day 긴 대화. 세션 경계 명확.
- **LongMemEval**: Chat assistant 시뮬레이션. Needle-in-haystack 구조.

**한 벤치마크의 특성만 보고 설계를 고정하지 마라.** 여러 벤치마크에서 공통으로 유효한 신호를 찾거나, 벤치마크별 적응형 설계를 고려할 것.