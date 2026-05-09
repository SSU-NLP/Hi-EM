# Sweep: 2026-05-09_hiem_evidence_metric_sanity

Generated: 2026-05-09T17:40:05

Path: `outputs/experiments/2026-05-09_hiem_evidence_metric_sanity/`

Columns: H@k/R@k/P@k — evidence hit/recall/precision over LLM context (k = method-specific context size: RAG=rag_k, sliding=sliding_k, Hi-EM=STM, full=full history; adversarial cat-5 excluded); R-multi-hop@k — recall on multi-hop only; T1μ/T2μ/T3μ — mean STM top-1/2/3 topic turn-counts; T1var — variance of top-1 across rounds; n_topics — mean STM topic count per round; gen_p50 — per-question generation latency p50; wall — total run wall-clock.

| method | overrides | n | acc | mh | sh | tr | adv | od | H@k | R@k | R-multi-hop@k | P@k | T1μ | T2μ | T3μ | T1max | T2max | T1var | n_topics | gen_p50(s) | wall(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi-em-full-v1 (hi_em_full_v1) | — | 49 | 0.241 | 0.121 | 0.025 | 0.022 | 1.000 | 0.016 | 0.256 | 0.187 | 0.430 | 0.001 | 503.7 | 0.0 | 0.0 | 654 | 0 | 11697.8 | 1.00 | 4.11 | 410.9 |

**best (acc)**: `hi_em_full_v1` — 0.241
