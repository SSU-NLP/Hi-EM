#!/usr/bin/env bash
# locomo sanity (--limit 50 --stratify) × 3 methods × 2 sigma configs.
# (alpha, lambda, sigma) = (1000, 0, 0.001) / (1000, 0, 0.01)
# methods: hi-em-full-v1, hi-em-full-v3.1.1, hi-em-full-v3.2.1
#
# Per-config STM top-K stats files (so v3.1/v3.2 가 RoundProcessor 의 record_round
# 를 env-var 만으로 켜고, file-existence gate 가 run 간 충돌 안 나도록).
set -u
cd "$(git rev-parse --show-toplevel)"

OUT_DIR=outputs/sweep_2026-05-04_locomo_sanity50
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

methods=("hi-em-full-v1" "hi-em-full-v3.1.1" "hi-em-full-v3.2.1")
sigmas=(0.001 0.01)

echo "=== START $(date -Is) ===" | tee -a "$LOG"

for sigma in "${sigmas[@]}"; do
  for method in "${methods[@]}"; do
    method_tag="${method//./_}"
    EXP_ID="20260504_locomo_sanity50_a1000_l0_s${sigma}_${method_tag}"
    TOPK="$OUT_DIR/stm_topk_${method_tag}_s${sigma}.json"
    echo "" | tee -a "$LOG"
    echo "=== Config: method=${method}  alpha=1000 lmda=0 sigma=${sigma}  topk=${TOPK}  $(date -Is) ===" | tee -a "$LOG"
    HIEM_STM_TOPK_STATS_PATH="$TOPK" \
    uv run python scripts/run_experiment.py \
      --method "$method" \
      --benchmark locomo \
      --data benchmarks/locomo/data/locomo10.json \
      --limit 50 --stratify \
      --alpha 1000 --lmda 0 --sigma0-sq "$sigma" \
      --exp-id "$EXP_ID" \
      --questions-per-round 50 \
      2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    echo "[exit] ${EXP_ID} rc=${rc}" | tee -a "$LOG"
  done
done

echo "=== END $(date -Is) ===" | tee -a "$LOG"
