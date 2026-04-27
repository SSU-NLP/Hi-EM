# Phase 4 Sanity Report — LongMemEval Oracle (30 questions, stratified)

**날짜**: 2026-04-26
**Scale**: subset 30 questions = 5/qtype × 6 question_type
**Data**: `benchmarks/LongMemEval/data/longmemeval_oracle.json`
**Model**: `Qwen/Qwen3-8B` (vLLM 로컬, 응답·judge 동일 endpoint, 비용 0)
**Hi-EM HP**: persistence (α=1, λ=10, σ²=0.01) — 옵션 5 결정 그대로
**Hi-EM Memory window**: k_topics=3, k_turns_per_topic=5 (= max 15 turns)
**RAG K**: 10
**Sliding K**: 20

---

## 1. 결과 — 4-way Accuracy 비교

| Method | Overall | knowledge | **multi** | ssa | ssp | ssu | temporal |
|---|---|---|---|---|---|---|---|
| sliding K=20 | 0.700 (21/30) | 0.80 (4/5) | 0.20 (1/5) | 1.00 (5/5) | 0.80 (4/5) | 0.80 (4/5) | 0.60 (3/5) |
| **full** | **0.833 (25/30)** | 0.80 | 0.60 | 1.00 | 1.00 | 0.80 | 0.80 |
| rag K=10 | 0.733 (22/30) | 0.60 | 0.20 | 1.00 | 0.80 | 0.80 | **1.00** |
| **hi-em** | **0.633 (19/30)** ⚠️ | **0.20** ⚠️ | **0.00** ⚠️ | 1.00 | 0.80 | **1.00** | 0.80 |

> **Hi-EM이 4 method 중 꼴찌 overall.** multi-session 5/5 fail, knowledge-update 4/5 fail.

## 2. 진단 — 파이프라인 vs 알고리즘

### 2.1 파이프라인 정상 (검증됨)

- 단위 테스트 58/58 PASS (preload_history, return_debug, A→B→A 토픽 복귀 포함)
- ssu 5/5, ssa 5/5, temporal 4/5 — **단일 세션 + 연속 정보에선 baseline 수준 동작**
- LTM state.json 직접 검사: history 24~46 turns가 모두 정확히 저장, segmenter가 topic 부여

→ **코드 결함 아님**.

### 2.2 알고리즘/HP 한계 (multi-session LTM state 분석)

multi-session 5/5 fail의 LTM state:

| QID | n_topics | max topic count | history turns | 패턴 |
|---|---|---|---|---|
| gpt4_59c863d7 | **1** | [24] | 46 | 모든 user turn이 한 cluster — persistence HP 너무 sticky |
| 6d550036 | 9 | [3,3,3,3,3] | 44 | 적당히 분리, but k_topics=3 → 6 topic 누락 |
| b5ef892d | 7 | [3,3,3,3,3] | 36 | 동일 |
| e831120c | 5 | [3,3,3,3,1] | 24 | 동일 |
| 0a995998 | 6 | [3,3,3,3,3] | 34 | 동일 |

두 패턴:
- **(case 1)** persistence HP λ=10이 너무 강해 모든 turn이 한 토픽 → topic ID 부여 의미 무효 → 5 turns만 보고 답
- **(case 2~5)** 분리는 됐으나 `k_topics=3 × k_turns_per_topic=5 = 15 turns` budget이 multi-session 정답 분포(history 곳곳)에 부족

knowledge-update 4/5 fail도 비슷한 원인 추정 — 같은 주제 update가 한 topic 안에 stale + new 둘 다, recency tail 5 turn 안에 new 안 들어가면 fail.

### 2.3 강점

- ssu 1.00 (full=0.80, rag=0.80) — **단일 user 세션에선 Hi-EM이 다른 baseline보다 약간 우위**
- temporal 0.80 — full과 동등 (RAG 1.00은 cosine top-K가 시간 정보 잘 잡는 듯)
- ssa 1.00, ssp 0.80 — baseline과 동등

→ Hi-EM의 토픽 ID·centroid 메커니즘은 **단일 세션 + 분명한 주제 영역**에서 정상 동작.

## 3. 한계

1. **표본 5/qtype은 noise 큼** (±20% 변동). 전체 oracle 500으로 검증 필요.
2. **prefill_tokens 미기록**: 현재 jsonl은 hypothesis만. token 효율 axis (Hi-EM이 적은 token으로 비슷한 acc?) 비교 불가. wandb엔 있으나 사용자 환경에서 wandb 연결 이슈로 chart 미확인.
3. **HP 단일 측정**: persistence HP만 시도. freq-shift HP (α=10, λ=1, σ²=0.1)는 보류 — 옵션 5 ARI 기준으론 약했지만 multi-session에선 다를 수도.
4. **k_topics=3, k_turns_per_topic=5 단일 측정**: token budget이 RAG K=10보다 1.5x 크지만 multi-session엔 부족할 수 있음.

## 4. Phase 1-6 reframing 정합성 재검토

옵션 3 reframing의 핵심 가정:
> "boundary F1·ARI는 Hi-EM 가치 지표 아님. **downstream QA에서 진짜 측정**."

본 sanity 결과:
- **현재 HP·k 조합으로는 Hi-EM이 baseline에 명확히 패배**
- 단 강점 영역(ssu, temporal) + 약점 영역(multi, knowledge-update) 분명
- "Hi-EM은 단일 세션 + 한 주제 깊이 추적엔 강하지만, 분산 정보 + 업데이트엔 RAG/full보다 약하다"는 잠정 결론

Phase 5 (논문) 진입 자격은 **HP sweep 후 best Hi-EM이 한 능력 이상에서 baseline 우위 또는 동등 + token 효율 우위** 충족 시.

## 5. 다음 단계 후보

| 옵션 | 비용 | 기대 |
|---|---|---|
| **A. HP/k sweep** | 1~2시간 | k_topics=5, k_turns_per_topic=10 + freq-shift HP. multi-session winning region 탐색 |
| **B. jsonl에 prefill_tokens 추가 + 재실행** | 5분 fix + 재실행 | token 효율 axis 확보 — Hi-EM이 적은 token으로 비슷한 acc면 의미 있음 |
| **C. oracle 전체 500으로 검증** | ~수 시간 | 5/qtype noise vs 진짜 차이 분리 |
| **D. 결과 인정 + Phase 4 reframing 재서술** | 즉시 | "Hi-EM은 ssu/temporal 강점, multi/knowledge 약점 명확 — 정직 기록 후 Phase 5 plan 재고" |

**권고 순서**: B → A → C → D
1. B: token 효율 axis 없으면 Hi-EM 가치 평가 불가능한 채로 결정 못 내림
2. A: HP sweep으로 winning region 있나 확인 (없으면 D)
3. C: best HP로 큰 sample 검증
4. D: 결과 위에 정직한 결론

## 6. 검증 미해결

- **prefill_tokens 정량**: 즉시 fix 가능 (B)
- **HP regime 다양성**: persistence vs freq-shift sweep 필요 (A)
- **Sample size**: 5/qtype → 100/qtype 필요 (C)
- **knowledge-update 0.20의 직접 원인**: stale-vs-new 가설 미검증 — case별 LTM 검사 필요

## 7. 참고 산출물

- 4 hypothesis jsonl: `outputs/phase-4-sanity-{sliding,full,rag,hi-em}.jsonl`
- 4 judge result: `outputs/phase-4-sanity-{sliding,full,rag,hi-em}.jsonl.judged.jsonl`
- LTM state per question: `data/ltm/longmemeval/<conv_id>/{*.jsonl, *.state.json}` (gitignored)
- Codex review 메트릭 결정: `context/06-decision-log.md` 2026-04-26 entry
