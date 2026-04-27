# Phase 1-3 — TopiOCQA dev segmentation 결과

실행 파라미터: α=10.0, λ=1.0, σ₀²=0.1, device=auto, limit_convs=None

## 데이터
- conversations: 205
- turns: 2514
- transitions (turn 쌍): 2309
- ground-truth shifts (`Topic` 필드 변화): 672
- shift rate per transition: 0.291

## Topic shift F1 (turn-transition 단위 binary)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| (a) all-boundary | 0.291 | 1.000 | 0.451 |
| (b) cosine-threshold (θ=0.7) | 0.307 | 0.970 | 0.467 |
| (c) Hi-EM (sCRP + option A) | 0.318 | 0.906 | 0.471 |

- cosine-threshold sweep 후보: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

## Latency (Hi-EM overhead)
- embedding (bge-base-en-v1.5): 4.22s total, 1.68 ms/turn
- HiEMSegmenter.assign(): 0.085s total, 0.034 ms/turn
- Hi-EM 총 overhead ≈ 1.71 ms/turn

## 해석
- **Ground truth**: `Topic` 필드 (Wikipedia doc) 변화만. `Topic_section` 변화는 noise → Hi-EM이 해당 경계에서 분할 시 False Positive.
- **한계**: TopiOCQA 평균 12턴 → Hi-EM variance($\sigma^2_k$)가 $n_e \geq 3$ 이후에 학습되므로 본 Step은 **centroid 부분만 실측** 검증. variance 효과는 Phase 4 LongMemEval QA에서 간접 측정.
- **cosine-threshold가 dev에서 sweep됨** — Hi-EM은 그 best θ를 이긴 것.

## Gate 판정 (plan.md Step 1-4)
- Hi-EM F1 > cosine baseline F1: **True** (0.471 vs 0.467)
- Hi-EM F1 > 0.4: **True** (0.471)
- 턴당 overhead ≤ 200ms (LLM 1000ms의 20% 기준): **True** (1.71 ms)

**Gate 결과: PASS**

→ **Phase 2 진입 가능**
