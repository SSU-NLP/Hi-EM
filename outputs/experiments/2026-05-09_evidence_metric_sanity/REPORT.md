# Sweep: 2026-05-09_evidence_metric_sanity

Generated: 2026-05-09T16:59:34

Path: `outputs/experiments/2026-05-09_evidence_metric_sanity/`

Columns: ev_R/ev_P — evidence recall/precision over LLM-context turns (adversarial cat-5 excluded); ev_R_strict_mh — multi-hop all-evidence-included rate; T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock.

| method | overrides | n | acc | mh | sh | tr | adv | od | ev_R | ev_P | ev_R_strict_mh | T1μ | T2μ | T3μ | T1max | T2max | T1var | n_topics | gen_p50(s) | wall(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full (full) | — | 49 | 0.276 | 0.188 | 0.099 | 0.050 | 1.000 | 0.017 | 0.991 | 0.003 | 1.000 | - | - | - | - | - | - | 0.00 | 14.87 | 45.5 |
| rag (rag) | — | 49 | 0.244 | 0.093 | 0.065 | 0.027 | 1.000 | 0.013 | 0.463 | 0.074 | 0.200 | - | - | - | - | - | - | 0.00 | 1.67 | 310.2 |
| sliding (sliding) | — | 49 | 0.220 | 0.052 | 0.007 | 0.007 | 1.000 | 0.016 | 0.000 | 0.000 | 0.000 | - | - | - | - | - | - | 0.00 | 2.74 | 16.6 |

**best (acc)**: `full` — 0.276
