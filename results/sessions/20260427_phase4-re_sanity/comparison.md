# Session 20260427_phase4-re_sanity

- data: `benchmarks/LongMemEval/data/longmemeval_oracle.json`
- no_thinking: True
- questions_per_round: 10
- limit: 30
- methods: sliding, full, rag, hi-em

## Comparison

| Method | Overall | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sliding | 0.867 | 0.80 | 0.40 | 1.00 | 1.00 | 1.00 | 1.00 |
| full | 0.833 | 0.80 | 0.60 | 1.00 | 1.00 | 0.80 | 0.80 |
| rag | 0.767 | 0.40 | 0.40 | 1.00 | 1.00 | 1.00 | 0.80 |
| hi-em | 0.700 | 0.60 | 0.00 | 1.00 | 1.00 | 0.60 | 1.00 |

## Per-experiment artifacts
- `sliding`: `results/experiments/20260427_phase4-re_sanity_sliding/`
- `full`: `results/experiments/20260427_phase4-re_sanity_full/`
- `rag`: `results/experiments/20260427_phase4-re_sanity_rag/`
- `hi-em`: `results/experiments/20260427_phase4-re_sanity_hi-em/`