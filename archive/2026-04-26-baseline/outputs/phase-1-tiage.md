# Phase 1-3 (augment) — TIAGE topic-shift detection

실행: `python scripts/run_tiage_segmentation.py --split test` (device=cpu)

## 데이터 — TIAGE test
- dialogs: 100
- total turns: 1564
- topic-shift labels ('1'): 315 (shift rate 0.215 / transition)

## 지표 — topic-shift F1 (turn-transition binary)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| (a) all-boundary | 0.215 | 1.000 | 0.354 |
| (b) cosine-threshold (θ=0.525) | 0.332 | 0.575 | 0.421 |
| (c) Hi-EM persistence (α=10.0, λ=1.0, σ₀²=0.1) | 0.239 | 0.895 | 0.377 |
| (c') Hi-EM freq-shift (α=10, λ=1, σ₀²=0.1) | 0.239 | 0.895 | 0.377 |

## Latency
- embed: 3.6s / 2.30 ms/turn
- Hi-EM assign: 63.7 ms total / 0.041 ms/turn
- 총 overhead: 2.34 ms/turn

## Gate 판정 (plan.md Phase 1-4 criteria, TIAGE 적용)

- Hi-EM F1 > cosine baseline F1: **False** (0.377 vs 0.421)
- Hi-EM F1 > 0.4: **False** (0.377)
- 턴당 overhead ≤ 200ms (LLM 1000ms의 20%): **True** (2.34 ms)

**Gate**: FAIL
