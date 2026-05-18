# REPORT — LLM pseudo-label segmentation sweep (LongMemEval idx 374)

## 1. 실험 setup

- **목적**: α/λ 가 segmentation 의 evidence-topic cohesion 을 어디서 깨는지,
  Qwen pseudo-label 대비 boundary/cluster 정합과 함께 직접 확인.
  (codex 2026-05-16 수정 프로토콜 — LLM 라벨은 gold 아님, pseudo-label.)
- **데이터**: `benchmarks/LongMemEval/data/longmemeval_oracle.json` idx=374
  단일 haystack (cross-idx concat 없음 — seam 회피).
  qid=gpt4_a1b77f9c · qtype=temporal-reasoning ·
  user턴 36개 · 6 sessions.
- **정답 evidence**: LongMemEval `has_answer=true` user턴 6개 — [0]sess1, [6]sess2, [15]sess3, [18]sess4, [28]sess5, [30]sess6.
  질문 1개당 evidence 가 같은 topic 에 모이면 retrieval atomicity 보존.
- **Qwen pseudo-label**: **생략 (--no-llm)** — Crts qwen3.5-9b 가 이 segmentation 프롬프트에서 reasoning 으로 max_tokens 소진 → content 빈응답, enable_thinking/`/no_think` 어떤 것으로도 thinking 차단 불가 (probe 로 확정: finish_reason=length, reasoning_tokens 수천). 따라서 bF1/ARI/NMI 미산출.
- **embedding**: `make_encoder()` env backend (실험과 동일).
- **segmenter**: v2 (`HiEMSegmenter`, σ₀²=0.01) · v3.3.4 (`HiEMSegmenterV334`).
- **HP grid**: α∈[1.0, 3.0, 10.0, 30.0, 100.0] × λ∈[0.0, 1.0, 3.0, 10.0] (각 segmenter 20 run, 총 40).
- **metric**:
  - 1차 `evidence_cohesion` = 1[모든 evidence 턴이 동일 topic]; `ev_topics`=evidence 가 흩어진 topic 수.
  - 보조 `bF1`/`ARI`/`NMI` = Qwen pseudo-label 대비.
  - 진단 `raw_topics`/`new_rate`/`max_share`; v3.3.4 `gru_used`(rnn_min_history=2 충족 턴 수)/`cosGRU`/`cosCen`.

## 2. 결과 — v2 (SEM core)

| α | λ | ev_cohesion | ev_topics | raw_topics | new_rate | max_share | bF1 | ARI | NMI | gru_used | cosGRU | cosCen |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0 | 0 | 4 | 12 | 0.333 | 0.111 | - | - | - | 0 | - | - | ⭐
| 1 | 1 | 0 | 4 | 12 | 0.333 | 0.111 | - | - | - | 0 | - | - |
| 1 | 3 | 0 | 4 | 12 | 0.333 | 0.111 | - | - | - | 0 | - | - |
| 1 | 10 | 0 | 4 | 12 | 0.333 | 0.111 | - | - | - | 0 | - | - |
| 3 | 0 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |
| 3 | 1 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |
| 3 | 3 | 0 | 4 | 12 | 0.333 | 0.111 | - | - | - | 0 | - | - |
| 3 | 10 | 0 | 4 | 12 | 0.333 | 0.111 | - | - | - | 0 | - | - |
| 10 | 0 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |
| 10 | 1 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |
| 10 | 3 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |
| 10 | 10 | 0 | 4 | 12 | 0.333 | 0.111 | - | - | - | 0 | - | - |
| 30 | 0 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |
| 30 | 1 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |
| 30 | 3 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |
| 30 | 10 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |
| 100 | 0 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |
| 100 | 1 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |
| 100 | 3 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |
| 100 | 10 | 0 | 4 | 13 | 0.361 | 0.111 | - | - | - | 0 | - | - |


## 3. 결과 — v3.3.4

| α | λ | ev_cohesion | ev_topics | raw_topics | new_rate | max_share | bF1 | ARI | NMI | gru_used | cosGRU | cosCen |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0 | 0 | 5 | 28 | 0.778 | 0.056 | - | - | - | 0 | - | - |
| 1 | 1 | 0 | 5 | 19 | 0.528 | 0.111 | - | - | - | 2 | 0.751 | 0.822 |
| 1 | 3 | 0 | 5 | 12 | 0.333 | 0.389 | - | - | - | 12 | 0.72 | 0.767 |
| 1 | 10 | 0 | 5 | 12 | 0.333 | 0.389 | - | - | - | 12 | 0.72 | 0.767 |
| 3 | 0 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |
| 3 | 1 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |
| 3 | 3 | 0 | 6 | 26 | 0.722 | 0.056 | - | - | - | 0 | - | - |
| 3 | 10 | 0 | 5 | 12 | 0.333 | 0.389 | - | - | - | 12 | 0.726 | 0.767 |
| 10 | 0 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |
| 10 | 1 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |
| 10 | 3 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |
| 10 | 10 | 0 | 6 | 32 | 0.889 | 0.056 | - | - | - | 0 | - | - |
| 30 | 0 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |
| 30 | 1 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |
| 30 | 3 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |
| 30 | 10 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |
| 100 | 0 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |
| 100 | 1 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |
| 100 | 3 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |
| 100 | 10 | 0 | 6 | 36 | 1.0 | 0.028 | - | - | - | 0 | - | - |


⭐ = 선택 기준(evidence_cohesion↑ → ev_topics↓ → raw_topics↓) 1위:
**v2 α=1 λ=0** —
evidence_cohesion=0, ev_topics=4,
raw_topics=12, bF1=-, ARI=-.

## 4. 해석

- evidence_cohesion=1 인 HP 가 있나? 없으면 어떤 α/λ 에서도 단일 질문의
  정답 근거가 한 topic 에 안 모인다는 뜻 → 과분절이 HP tuning 으로 안 풀림
  (codex 진단: 진짜 병목은 segmentation atomicity).
- v3.3.4 `gru_used`: 0 이면 그 HP 에서 GRU dynamics 死문 (α 큰 영역 예상).
  cosGRU < cosCen 이면 GRU 가 centroid 보다 예측 열세.
- bF1/ARI 최적점과 evidence_cohesion 최적점이 어긋나면 "Qwen 에 맞추기"가
  downstream 병목과 무관함을 보임.

## 5. 한계 / 검증 미해결

- Qwen pseudo-label 은 human gold 가 아니다. 1회 호출, temp=0 이라 재현되나
  Qwen 의 주관적 topic ontology 에 의존. boundary 는 cluster id 변화에서 파생.
- 단일 idx·단일 질문(evidence_cohesion 이 0/1 binary) → 표본 1. 경향 참고용.
- assistant 턴 제외(orchestrator 와 동일), STM/RoundProcessor 미적용 — 순수
  segmenter-level 진단. STM importance eviction 효과는 별도.
- 데이터 늘려 (다른 idx, LoCoMo conv0 단일대화) 일반화 필요.

## 6. 정정 / Addendum (2026-05-17, codex 위임 — 원본 §1~5 보존)

후속 idx374 심층 진단에서 본 REPORT 의 핵심 전제 2가지가 정정됨:

1. **`evidence_cohesion` metric 오류 → 폐기**. evidence 가 6 이질 세션에 1턴씩
   본질 분산되므로, 전 evidence 단일 topic 을 *설계 목표* 로 삼는 것은 SEM
   scene atomicity 위배(mega-merge 강요)임이 확정. §2~4 의 "evidence_cohesion=0
   = 실패" 해석은 무효 — ev_topics 4~6 은 정상. 대체 metric =
   `evidence_recall@K`(topic-level primary, session-level 보조).
   decision-log 2026-05-17 evidence_cohesion 폐기 entry.

2. **표의 v3.3.4 수치는 unseeded(비결정적)**. EventRNN random init 이
   v3.3.4 시점 seed 미고정 → 본 REPORT 의 raw_topics/ev_topics 등은
   실행마다 변동. 정확 수치로 재현 불가(경향만 유효). v3.3.6+ 에서 seed
   도입으로 해소. decision-log 2026-05-17 v3.3.6 entry / methodology
   infrastructure §10.

3. 후속 버전(v3.3.5 f_is_trained / v3.3.6 persistence+replay+seed / v3.3.7
   map_variance[#14|15 반증] / v3.3.8 pe_prior)이 본 REPORT 의 과분절
   진단을 순차 처치. idx374 topic 수 23(v3.3.4)→11→9→9→7(v3.3.8 pp0.4).
   상세 = methodology v3.3.5~8.md. pe_prior 확정은 longmemeval_s 평가 후속.
