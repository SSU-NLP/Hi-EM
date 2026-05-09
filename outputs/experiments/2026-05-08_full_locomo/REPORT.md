# Sweep: 2026-05-08_full_locomo

Generated: 2026-05-09T18:43:26

Path: `outputs/experiments/2026-05-08_full_locomo/`

n_questions: **1986** (uniform across runs)

Columns: T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock (h/m/s); notes — HPs the method actually consumes (incl. overrides).

| method | notes | acc | mh | sh | tr | adv | od | T1μ | T2μ | T3μ | T1max | T2max | T1var | n_topics | gen_p50(s) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full (full) | (full context) | 0.248 | 0.219 | 0.140 | 0.067 | 0.646 | 0.034 | - | - | - | - | - | - | 0.00 | 18.05 | 22m 27s |
| rag (rag) | rag_k=10 | 0.286 | 0.137 | 0.120 | 0.063 | 0.904 | 0.057 | - | - | - | - | - | - | 0.00 | 1.99 | 49m 20s |
| rag-observation (rag_observation) | rag_k=10 | 0.294 | 0.159 | 0.094 | 0.080 | 0.962 | 0.053 | - | - | - | - | - | - | 0.00 | 1.94 | 47m 0s |
| rag-summary (rag_summary) | rag_k=10 | 0.274 | 0.177 | 0.057 | 0.055 | 0.953 | 0.037 | - | - | - | - | - | - | 0.00 | 3.59 | 48m 57s |
| sliding (sliding) | sliding_k=20 | 0.252 | 0.065 | 0.030 | 0.020 | 1.000 | 0.051 | - | - | - | - | - | - | 0.00 | 2.37 | 2m 11s |
| hi-em-full-v1 (v1_best) | seg: α=1, λ=10, cos=0.7, β=0.5 · mem: k_top=3, k_turn=5 | 0.122 | 0.089 | 0.061 | 0.036 | 0.341 | 0.021 | 419.8 | 0.0 | 0.0 | 527 | 0 | 4978.4 | 1.00 | 3.49 | 57m 8s |
| hi-em-full-v3.1.1 (v3p1p1_best) | seg: α=10, λ=10, cos=0.3, β=0.5 · mem: k_top=3, k_turn=5 | 0.243 | 0.210 | 0.137 | 0.073 | 0.628 | 0.045 | 609.1 | 0.0 | 0.0 | 689 | 0 | 10043.4 | 1.00 | 2.75 | 50m 40s |
| hi-em-full-v3.3.1 (v3p3p1_best) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=1 · mem: k_top=3, k_turn=5 | 0.263 | 0.133 | 0.073 | 0.042 | 0.908 | 0.049 | 30.6 | 21.5 | 19.7 | 62 | 31 | 207.6 | 9.77 | 2.79 | 1h 3m 14s |
| hi-em-full-v3.3.2 (v3p3p2_best) | seg: α=100, λ=10, cos=0.9, β=0.25, rnn_train_steps=3 · mem: k_top=3, k_turn=5 | 0.263 | 0.134 | 0.081 | 0.041 | 0.895 | 0.039 | 28.0 | 21.5 | 18.9 | 38 | 29 | 39.5 | 9.89 | 2.63 | 58m 35s |

**best (acc)**: `rag_observation` — 0.294
