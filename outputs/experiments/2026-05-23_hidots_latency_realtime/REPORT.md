# Hi-DoTS per-turn latency (realtime, no encoder cache)

## 1. 측정 정의

- per-turn = **encode(단일 발화, 캐시 없음) + HiDoTS.assign**.
- 매 turn perf_counter 로 encode 와 assign 시간을 따로 측정 (ms).
- 첫 발화 (turn 0) 는 baseline 들과 동일하게 표본 제외.
- encoder warmup 10 forwards (cold-start, model load 비용 분리).
- encoder = `sentence-transformers/multi-qa-mpnet-base-dot-v1` (CPU).
- HP: m=2, ρ=0.7, a=0.5. seed=0.
- δ\* 무관 — encode+assign 시간은 boundary 결정 여부와 독립이므로
  Ours 의 모든 percentile/oracle 행에 동일 latency 적용.

## 2. 결과 (벤치별)

| 벤치 | n turn (timed) | Pre. encode (ms) mean / p50 / p90 | Seg. assign (ms) mean / p50 / p90 | Total (ms) mean / p50 / p90 |
|---|---:|---:|---:|---:|
| tiage | 468 | 1128.26 / 881.05 / 2304.18 | 0.4197 / 0.2507 / 0.3927 | 1128.68 / 881.34 / 2304.42 |
| dialseg711 | 498 | 849.00 / 737.91 / 1541.04 | 0.3374 / 0.2328 / 0.3806 | 849.34 / 738.34 / 1541.28 |
| superseg | 468 | 932.30 / 665.23 / 1921.17 | 0.3530 / 0.2364 / 0.3466 | 932.65 / 665.44 / 1921.44 |

## 3. cross-benchmark 평균 (Ours 행에 단일 latency 보고용)

- **Pre. (encode)**: mean = 967.33 ms · p50 747.43 · p90 1920.15 (n=1434)
- **Seg. (assign)**: mean = 0.3694 ms · p50 0.2416 · p90 0.3705
- **Total (Pre.+Seg.)**: mean = 967.69 ms · p50 747.65 · p90 1920.39
- cold-start (model load): 157.49 s — per-turn 과 분리.

## 4. 표 셀 매핑 (dts_result.md 의 Online Ours 4 행)

- **Pre.** 셀 = 967 ms (cross-bench mean, encoder single-utt forward).
- **Seg.** 셀 = 0.369 ms (cross-bench mean, HiDoTS.assign).
- p70/p75/p80/oracle 4 행 모두 동일 (δ\* 무관).

## 5. 한계 / 검증 미해결

- 표본 = 벤치당 500 turn budget subsample (seed=0). content-independent 한 latency 특성상 전수 측정 대비 편차 작을 것이나, 완전 일치 보장은 분포 가정에 의존.
- CPU only (machine: 본 측정 머신). GPU 환경에서는 encode 비용 한 자릿수 ms 수준으로 떨어질 가능성 — 그 경우 별도 측정 필요.
- 인코더 lazy init / jit / GC 영향 — warmup 10 fwd 로 1차 제거, 잔여 노이즈는 p50 사용시 영향 작음.
- baseline (GreedySeg 13 ms 등) 와 동일 single-utt forward 정의로 비교 가능. metric 가족 차이는 별도 — 본 REPORT 는 latency 만.
