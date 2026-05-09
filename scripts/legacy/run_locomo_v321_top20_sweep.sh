#!/usr/bin/env bash
# v3.2.1 (Bounded Cosine MAP + sub-linear sCRP) sweep over top-20 configs
# from v1 ∪ v3.1.1 (LoCoMo α×λ×cos main sweep).
#
# 20 configs × 3 β values = 60 runs.
# β grid: 0.25 (강), 0.5 (중간/sqrt), 1.0 (control = v3.1.1 회귀 검증)
#
# Prereq: scripts/aggregate_locomo_alphalambda_results.py 가 먼저 돌아서
# summary_table.csv 가 생성된 상태여야 함. 그 위에 pick_top20_for_v321.py
# 가 top20_for_v321.txt 를 만든 뒤 이 스크립트가 그 list 를 읽음.
set -u
cd "$(git rev-parse --show-toplevel)"

OUT_DIR=outputs/sweeps/2026-05-05_locomo_alpha_lambda_cos
LOG="$OUT_DIR/run_v321.log"
TOP_FILE="$OUT_DIR/top20_for_v321.txt"
BETAS=(0.25 0.5 1.0)

if [ ! -f "$TOP_FILE" ]; then
  echo "ERROR: $TOP_FILE not found. Run scripts/pick_top20_for_v321.py first." >&2
  exit 1
fi

echo "=== v3.2.1 top-20 × 3β START $(date -Is) ===" | tee -a "$LOG"

while IFS= read -r line; do
  # Skip comments / blank
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  [[ -z "${line// }" ]] && continue
  # Strip trailing comment
  cfg="${line%%#*}"
  read -r alpha lmda cos <<< "$cfg"
  [ -z "${cos:-}" ] && continue

  for beta in "${BETAS[@]}"; do
    EXP_ID="20260505_locomo_aL_top20_a${alpha}_l${lmda}_c${cos}_b${beta}_hi-em-full-v3_2_1"
    TOPK="$OUT_DIR/stm_topk_v3_2_1_a${alpha}_l${lmda}_c${cos}_b${beta}.json"
    echo "" | tee -a "$LOG"
    echo "=== Config: method=hi-em-full-v3.2.1  alpha=${alpha} lmda=${lmda} cos=${cos} beta=${beta}  topk=${TOPK}  $(date -Is) ===" | tee -a "$LOG"
    HIEM_STM_TOPK_STATS_PATH="$TOPK" \
    uv run python scripts/run_experiment.py \
      --method hi-em-full-v3.2.1 \
      --benchmark locomo \
      --data benchmarks/locomo/data/locomo10.json \
      --limit 50 --stratify \
      --alpha "$alpha" --lmda "$lmda" --sigma0-sq 0.01 \
      --cos-threshold "$cos" \
      --beta "$beta" \
      --exp-id "$EXP_ID" \
      --questions-per-round 50 \
      2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    echo "[exit] ${EXP_ID} rc=${rc}" | tee -a "$LOG"
  done
done < "$TOP_FILE"

echo "=== v3.2.1 END $(date -Is) ===" | tee -a "$LOG"
