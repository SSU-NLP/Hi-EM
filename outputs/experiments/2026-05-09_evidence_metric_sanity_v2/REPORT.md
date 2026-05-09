# Sweep: 2026-05-09_evidence_metric_sanity_v2

Generated: 2026-05-09T18:22:00

Path: `outputs/experiments/2026-05-09_evidence_metric_sanity_v2/`

Columns: T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; STM_n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock (h/m/s).

| method | accuracy_overall | multi-hop | single-hop | temporal-reasoning | adversarial | open-domain | T1μ | T2μ | T3μ | T1max | T2max | T1var | STM_n_topics | gen_p50(s) | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full (full) | 0.204 | 0.217 | 0.096 | 0.066 | 0.600 | 0.021 | - | - | - | - | - | - | 0.00 | 13.61 | 52s |
| rag (rag) | 0.225 | 0.097 | 0.055 | 0.036 | 0.900 | 0.015 | - | - | - | - | - | - | 0.00 | 1.61 | 5m 23s |
| sliding (sliding) | 0.220 | 0.041 | 0.008 | 0.009 | 1.000 | 0.021 | - | - | - | - | - | - | 0.00 | 2.70 | 20s |

**best (acc)**: `rag` — 0.225
