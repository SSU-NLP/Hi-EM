# Phase 4 Sanity — Qwen3 Thinking Off (no-think) 비교 리포트

**날짜**: 2026-04-26
**Scale**: 30 questions × 4 method, stratified (5/qtype × 6 type)
**Data**: `benchmarks/LongMemEval/data/longmemeval_oracle.json`
**Model**: `Qwen/Qwen3-8B` (vLLM, 응답·judge 동일 endpoint)
**변경**: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`로 `<think>` 블록 비활성화
**HP/HP 동일**: persistence (α=1, λ=10, σ²=0.01), k_topics=3, k_turns_per_topic=5, sliding K=20, RAG K=10
**비교 대상**: 어제 sanity (think on, `outputs/phase-4-sanity-*.judged.jsonl`)

---

## 1. Overall 결과

| Set | sliding | full | rag | hi-em |
|---|---|---|---|---|
| think (어제) | 0.700 (21/30) | **0.833** (25/30) | 0.733 (22/30) | **0.633** (19/30) |
| **nothink (오늘)** | 0.867 (26/30) | **0.900** (27/30) | 0.867 (26/30) | **0.767** (23/30) |
| **Δ** | **+0.167** | +0.067 | +0.133 | **+0.133** |

→ **모든 method 정확도 상승**. nothink가 LongMemEval에서 더 적합. Hi-EM은 +0.133이지만 **여전히 4 method 중 꼴찌** (0.767 < 다른 셋 0.867~0.900).

## 2. Per-qtype (NoThink)

| Method | knowledge | multi | ssa | ssp | ssu | temporal |
|---|---|---|---|---|---|---|
| sliding K=20 | 0.80 | 0.60 | 1.00 | 1.00 | 1.00 | 0.80 |
| **full** | 0.80 | **0.60** | 1.00 | 1.00 | 1.00 | **1.00** |
| rag K=10 | 0.80 | 0.60 | 1.00 | 1.00 | 1.00 | 0.80 |
| **hi-em** | 0.80 ⬆⬆ | **0.20** ⚠️ | 1.00 | 0.80 | 0.80 | 1.00 |

### Per-qtype Δ (nothink − think)

| Method | knowledge | multi | ssa | ssp | ssu | temporal |
|---|---|---|---|---|---|---|
| sliding | +0.00 | **+0.40** | +0.00 | +0.20 | +0.20 | +0.20 |
| full | +0.00 | +0.00 | +0.00 | +0.00 | +0.20 | +0.20 |
| rag | +0.20 | **+0.40** | +0.00 | +0.20 | +0.20 | -0.20 |
| **hi-em** | **+0.60** | +0.20 | +0.00 | +0.00 | -0.20 | +0.20 |

## 3. 핵심 발견

### 3.1 모든 method ↑ — LongMemEval은 retrieval-heavy
- 6 question_type 중 5개 (ssu/ssa/ssp/multi/knowledge-update)는 **fact recall**
- 1개 (temporal-reasoning)만 reasoning
- Sprague et al. 2024 ("To CoT or not to CoT?", arXiv:2409.12183): CoT는 **math/symbolic에서만 의미**, recall은 손해
- → think off가 본 벤치마크 특성에 정합

### 3.2 Hi-EM `knowledge-update +0.60` 큰 개선
- think on: 0.20 (4 method 중 최악) → nothink: 0.80 (baseline 수준)
- Chen et al. 2024 ("Do NOT Think That Much for 2+3=?", arXiv:2412.21187): reasoning model의 **answer-switching** — 정답 발견 후 second-guess
- 추정: think on에선 prefill의 정답을 보고도 "wait, but..."로 다른 답 도출. nothink는 직접 응답.

### 3.3 Hi-EM `multi-session 0.20` 여전히 약점
- think on 0.00 → nothink 0.20: 1개 회복
- 다른 method (sliding/full/rag) 모두 0.60 → **0.40 격차 그대로**
- think 무관한 **알고리즘 한계**

### 3.4 `temporal-reasoning`: 예외 (rag −0.20)
- rag는 cosine top-K로 시간 정보 잘 찾고 think이 도움 됐던 듯 (어제 1.00 → 오늘 0.80)
- 다른 method (full +0.20, sliding +0.20, hi-em +0.20)는 nothink가 도움
- 표본 5개라 noise 가능성 큼

## 4. Hi-EM 진단 (multi-session 약점은 알고리즘 본질)

### 4.1 LTM state 분석 (어제 그대로 재확인)
| QID 패턴 | n_topics | top topic count | history turns |
|---|---|---|---|
| (1 케이스) | **1** | [24] | 46 |
| (4 케이스) | 5~9 | [3,3,3,3,3] | 24~44 |

### 4.2 root cause
- **n_topics=1**: persistence HP λ=10 너무 강 → 모든 turn 한 cluster → 5 turn만 보고 답
- **n_topics 5~9 + k_topics=3**: 6 topic 누락
- 두 패턴 모두 **k_topics × k_turns_per_topic = 15 budget이 multi-session 정답 분포보다 작음**

### 4.3 RAG와의 비교 (token 효율 axis)
| Method | budget (turns) | multi-session acc |
|---|---|---|
| sliding | 20 | 0.60 |
| **rag** | **10** | **0.60** ← 더 적은 budget으로 60% |
| **hi-em** | **15** | **0.20** ← 더 큰 budget인데 20% |

→ **Hi-EM은 RAG에 두 axis 모두 패배** (accuracy + token 효율). 의도한 winning thesis (적은 token으로 baseline 동등)가 sanity에선 미충족.

## 5. Phase 1-6 reframing 정합성

옵션 3 reframing 가설:
> "boundary F1·ARI는 Hi-EM 가치 지표 아님. 진짜 가치는 downstream QA."

본 nothink sanity 결과:
- **knowledge-update +0.60, multi-session +0.20**: 정보가 prefill에 있으면 hi-em도 답할 수 있음 → **prefill 자체는 정확** (think 한계 제거 후)
- **multi-session 0.20**: prefill에 정답 정보 안 들어가는 케이스 — algorithm 한계
- **token 효율 axis 정량 측정 미실시** (jsonl엔 token 정보 없음, wandb만)

## 6. 한계

1. **표본 5/qtype** — ±20% noise 가능. multi-session 0.20 vs 0.60이 진짜 차이인지 sample 효과인지 미확정.
2. **HP 단일 측정** — persistence (α=1, λ=10) 외 freq-shift / k 조합 미시도.
3. **prefill_tokens 정량 부재** — token 효율 axis는 wandb에만, jsonl 분석 불가.
4. **Step 2-4 보류 상태** (importance / merge / adaptive K) — multi-session 약점과 직접 연결되는 기능 미구현.
5. **데이터 size 제약** — oracle (작은 history). `s_cleaned` (115k tokens) / `m_cleaned` (500 sessions)에선 양상 다를 수 있음.

## 7. 다음 단계 후보

| 옵션 | 비용 | 기대 |
|---|---|---|
| **A. oracle 전체 500 nothink** | ~2시간 | 5/qtype noise 제거, 진짜 차이 확정 |
| **B. HP/k sweep** (persistence/freq-shift × k_topics ∈ {3,5,8} × k_turns_per_topic ∈ {5,10,15}) | 4~8시간 | Hi-EM winning region 있나 탐색 |
| **C. Step 2-4 구현** (topic merge + adaptive K_window) | 1~2일 (코드 + 테스트) | n_topics=1 케이스 분할, history 크기 적응 |
| **D. jsonl에 prefill_tokens 추가** | 5분 fix | token 효율 axis 정량 가능 |
| **E. 정직 reframing** | 즉시 | "현재 algorithm은 RAG에 두 axis 모두 패배. 토픽 ID 부여는 ssu/ssa에서만 의미" |
| **F. `s_cleaned` 추가 다운로드 + 평가** | ~6시간 | Hi-EM 진짜 시나리오(긴 history) |

### 권고 우선순위

1. **D + A 묶음** (즉시): D는 5분, A는 noise 제거 — 본 평가로 하나 명확히
2. A 결과로 분기:
   - Hi-EM이 multi-session에서 0.20 ± 0.05면 **알고리즘 한계 확정** → C 또는 E
   - 0.40+로 회복하면 **표본 noise** → B (HP sweep)으로 winning region 탐색

3. **F (s_cleaned)**: oracle은 evidence sessions만이라 짧음. Hi-EM의 차별점이 가장 빛날 곳은 **긴 history + 토픽 복귀**가 잦은 m_cleaned. 본 평가의 일부에 포함 권고.

## 8. 참고 산출물

- 8 hypothesis jsonl: `outputs/phase-4-sanity{,-nothink}-{sliding,full,rag,hi-em}.jsonl`
- 8 judge result: `... .judged.jsonl`
- LTM state per question (Hi-EM only): `data/ltm/longmemeval/<conv_id>/{*.jsonl, *.state.json}` (gitignored)
- 어제 think sanity 분석: `outputs/phase-4-sanity.md`
- W&B run group: `longmemeval_oracle-<UTC ts>` (사용자 환경에서 부분 sync)

## 9. 인용

- Sprague et al. 2024. "To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning." [arXiv:2409.12183](https://arxiv.org/abs/2409.12183)
- Chen et al. 2024. "Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs." [arXiv:2412.21187](https://arxiv.org/abs/2412.21187)
- Liu et al. 2024. "Lost in the Middle: How Language Models Use Long Contexts." [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
- Wu et al. 2024. "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory." [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)
