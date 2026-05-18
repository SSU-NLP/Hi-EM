# REPORT — LongMemEval: hi-em-full-v3.3.4 vs rag (500Q)

experiment: `2026-05-16_v334_rag_longmemeval`
generated: 2026-05-16 (정식판; 자동 생성 단축표를 본 문서로 대체)
path: `outputs/experiments/2026-05-16_v334_rag_longmemeval/`

---

## 1. 실험 setup

- **목적**: LongMemEval 전체에서 Hi-EM 최신 계열(v3.3.4)이 단순 RAG 대비
  실제 QA 정확도 우위가 있는지, qtype 별로 어디서 이기고 어디서 지는지 확인.
- **데이터**: `benchmarks/LongMemEval/data/longmemeval_oracle.json` —
  500 question, 각 question = 하나의 multi-session haystack (oracle: 정답
  관련 3~6 session). qtype 분포 (n/500):

  | qtype | n |
  |---|--:|
  | temporal-reasoning | 133 |
  | multi-session | 133 |
  | knowledge-update | 78 |
  | single-session-user | 70 |
  | single-session-assistant | 56 |
  | single-session-preference | 30 |

- **방법**:
  - `rag` — per-haystack chunk + cosine top-k (rag_k=10) prefill.
  - `hi-em-full-v3.3.4` — `HiEMSegmenterV334` segmentation + STM(importance)
    retrieval. seg HP: **α=1, λ=10, cos=0.7, β=0.5, rnn_train_steps=1**
    (= `experiment.py`/`run_experiment.py` config default; ⚠️ v3.3.4
    segmenter 클래스 default 인 α=100 이 아님 — 본 run 은 GRU 가 발동
    가능한 α=1 regime). mem: k_top=3, k_turn=5, round_size/STM 기본값.
- **LLM / judge**: 생성·judge 모두 `qwen/qwen3.5-9b` (Crts 프록시,
  `--no-thinking`). embedding = env backend (api, bge-base-en-v1.5).
- **seed / 반복**: 단일 run (seed 명시 안 함, 3-run 아님 → std 없음).
- **metric**: LLM-judge 기반 question accuracy (overall + qtype별).
  보조: prefill 메시지 수, 생성 latency p50, wall.
- **baseline**: `rag` (동일 데이터·LLM·judge, 같은 experiment 안에서 실행).

## 2. 결과 (500Q, 단일 run)

| qtype (n) | rag | v3.3.4 | Δ(v334−rag) |
|---|--:|--:|--:|
| **overall (500)** | **0.708** | **0.722** | **+0.014** |
| temporal-reasoning (133) | 0.481 | 0.466 | −0.015 |
| multi-session (133) | 0.662 | 0.729 | **+0.068** |
| knowledge-update (78) | 0.795 | 0.821 | +0.026 |
| single-session-user (70) | 0.986 | 0.943 | −0.043 |
| single-session-assistant (56) | 0.893 | 0.946 | +0.054 |
| single-session-preference (30) | 0.700 | 0.633 | −0.067 |

| 운영 지표 | rag | v3.3.4 |
|---|--:|--:|
| error_rate | 0.00 | 0.00 |
| prefill msgs (avg) | 10.7 | 22.2 |
| gen latency p50 (s) | 6.10 | 6.61 |
| STM_n_topics (mean/round) | – | 6.14 |
| wall | 29m 50s | 26m 17s |

## 3. 해석

- **overall 은 v3.3.4 가 +1.4pt (0.722 vs 0.708)** 로 근소 우위. 단 균일
  우위가 아니라 qtype 별로 명확히 갈림.
- **v3.3.4 가 크게 이기는 곳**: multi-session (+6.8pt), single-session-
  assistant (+5.4), knowledge-update (+2.6). 여러 session 에 정보가
  흩어진 질문에서 STM importance 가 관련 topic 들을 모아 prefill 하는
  것이 cosine top-k 보다 유효 — segmentation+STM 의 설계 의도가 작동하는
  영역.
- **v3.3.4 가 지는 곳**: single-session-preference (−6.7), single-session-
  user (−4.3), temporal-reasoning (−1.5). 단일 session·단일 turn 에 답이
  있는 질문에서는 RAG 의 query-cosine top-k 가 그 한 turn 을 더 정확히
  집고, Hi-EM 은 importance 기반이라 정작 관련 단일 turn 을 누락/희석.
- **temporal-reasoning 열세**는 앞선 segmentation inspection 과 일치:
  "여러 시점 정보를 묶어야" 하는 질문에서 evidence 가 서로 다른 topic 으로
  분절돼 atomic prefill 이 깨짐 (LongMemEval idx0 사례에서 직접 확인).
- **비용 대비**: v3.3.4 는 prefill 메시지를 RAG 의 ~2배(22.2 vs 10.7)
  쓰고도 +1.4pt. context 효율은 RAG 가 더 높음. wall 은 v3.3.4 가 더
  짧은데, 이는 prefill 구조 차이 + RAG latency p95 가 큰 탓.

## 4. 판정

- **iteration 내 분류**: v3.3.4 = overall **소폭 향상**(+0.014, 단일 run
  이라 noise 가능성 배제 불가) / multi-session·assistant·knowledge-update
  **향상** / temporal·single-session-preference·user **회귀**.
- **다음 iteration 결정**: **보류 (재검증 필요)**. 이유:
  1. 단일 run·seed 미고정 → ±noise 범위 미상. overall +1.4pt 가 유의한지
     판단하려면 ≥3-run std 필요.
  2. temporal 회귀가 segmentation 과분절 진단(2026-05-16 decision-log
     라인)과 방향 일치 → HP/atomicity 손보기 전엔 v3.3.4 채택 근거 약함.
  3. 단일-session qtype 회귀는 "importance retrieval 이 단일 정답 turn 을
     RAG 만큼 못 집는다"는 별도 약점 — importance policy 점검 대상.

## 5. 한계 / 검증 미해결

- **단일 run, seed 미고정, std 없음.** overall Δ=+0.014 가 noise 내인지
  미확정. CLAUDE.md "3-run 이상" 미충족 — 정식 채택 판정 불가.
- **evidence co-location metric 부재**: `evidence_topic_summary.json` 이
  전부 null/0 (analyzer 가 LoCoMo `dia_id` 전제 — LongMemEval 은 turn
  `has_answer` 구조라 미산출). retrieval H@k/R@k/P@k 도 N/A. 따라서 "왜
  졌나"를 retrieval 수준에서 본 run 으로는 정량 추적 불가.
- **HP regime 주의**: 본 run 은 α=1 (config default). 앞서 진단한
  "α=100 GRU dead-code" 와 다른 영역 — 본 결과로 GRU 유효성 결론 금지.
  단, α=1 LongMemEval inspection 에서도 18 user턴→12 topic 과분절 관찰됨.
- **judge=생성 동일 모델(qwen3.5-9b)**: self-judge bias 가능. baseline
  과 동일 조건이라 상대 비교는 유효하나 절대값 해석 주의.
- **oracle 변형**: haystack 이 정답 관련 session 만(3~6) → distractor 적음.
  full LongMemEval_s/_m 대비 retrieval 난이도 낮음. 일반화엔 large variant
  필요.
- **full-context 천장 부재 (cross-ref)**: 본 run 은 full 미포함. 단,
  과거 full-context baseline 을 oracle **500Q** 에 돌린 결과가
  `archive/2026-04-28/20260428_freq_shift_full_*` 에 존재 (Qwen3-8B,
  `--no-thinking`, seg α=10/λ=1): **full 0.734 / rag 0.728 /
  hi-em-full-v1 0.726 / sliding 0.636 / hi-em 0.562**. 즉 oracle 에서
  천장(full)이 RAG 대비 **+0.6pt** 뿐 — distractor 부재로 RAG 도 증거
  대부분 회수, "관련 세션 통째 prefill" 헤드룸 자체가 작다. full prefill
  p50 5850 vs rag 2175 tok (~2.7×). 본 500Q(qwen3.5-9b) 와 LLM 이 달라
  직접 비교 불가하나 패턴 일관(full≈rag). 분절 전략 가치는 distractor 多
  인 `longmemeval_s` 에서 full 천장 재측정해야 검증 가능.
  (decision-log 2026-05-16 entry 의 동일 cross-ref 참조.)
