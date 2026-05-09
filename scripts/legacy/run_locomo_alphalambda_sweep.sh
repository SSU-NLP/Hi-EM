#!/usr/bin/env bash
# locomo sanity α × λ [× cos_threshold] sweep.
#  • hi-em-full-v1          σ²₀=0.01 (fixed), α ∈ {1,10,100,1000}, λ ∈ {0,1,10,100}      → 16 runs
#  • hi-em-full-v3.1.1    σ²₀ dead, α × λ × cos_threshold ∈ {0.3,0.5,0.7,0.9}            → 64 runs
# Total 80 runs × ~5min = ~6-7h.
#
# Per-config STM top-K stats files. flush_aggregate is gated on hi-em-full-v1
# in run_experiment.py, so v3.1 runs only emit the .rounds.jsonl (per-round)
# file; aggregate is built post-hoc from that.
set -u
cd "$(git rev-parse --show-toplevel)"

OUT_DIR=outputs/sweeps/2026-05-05_locomo_alpha_lambda_cos
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

ALPHAS=(1 10 100 1000)
LAMBDAS=(0 1 10 100)
COS_THRS=(0.3 0.5 0.7 0.9)

echo "=== START $(date -Is) ===" | tee -a "$LOG"

# --- Block 1: hi-em-full-v1 (Gaussian, σ²₀=0.01 fixed) ---
for alpha in "${ALPHAS[@]}"; do
  for lmda in "${LAMBDAS[@]}"; do
    EXP_ID="20260505_locomo_aL_a${alpha}_l${lmda}_hi-em-full-v1"
    TOPK="$OUT_DIR/stm_topk_hi-em-full-v1_a${alpha}_l${lmda}.json"
    echo "" | tee -a "$LOG"
    echo "=== Config: method=hi-em-full-v1  alpha=${alpha} lmda=${lmda} sigma=0.01  topk=${TOPK}  $(date -Is) ===" | tee -a "$LOG"
    HIEM_STM_TOPK_STATS_PATH="$TOPK" \
    uv run python scripts/run_experiment.py \
      --method hi-em-full-v1 \
      --benchmark locomo \
      --data benchmarks/locomo/data/locomo10.json \
      --limit 50 --stratify \
      --alpha "$alpha" --lmda "$lmda" --sigma0-sq 0.01 \
      --exp-id "$EXP_ID" \
      --questions-per-round 50 \
      2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    echo "[exit] ${EXP_ID} rc=${rc}" | tee -a "$LOG"
  done
done

# --- Block 2: hi-em-full-v3.1.1 (Bounded Cosine MAP, σ²₀ dead) ---
for alpha in "${ALPHAS[@]}"; do
  for lmda in "${LAMBDAS[@]}"; do
    for cos in "${COS_THRS[@]}"; do
      method_tag="hi-em-full-v3_1_1"
      EXP_ID="20260505_locomo_aL_a${alpha}_l${lmda}_c${cos}_${method_tag}"
      TOPK="$OUT_DIR/stm_topk_${method_tag}_a${alpha}_l${lmda}_c${cos}.json"
      echo "" | tee -a "$LOG"
      echo "=== Config: method=hi-em-full-v3.1.1  alpha=${alpha} lmda=${lmda} cos=${cos}  topk=${TOPK}  $(date -Is) ===" | tee -a "$LOG"
      HIEM_STM_TOPK_STATS_PATH="$TOPK" \
      uv run python scripts/run_experiment.py \
        --method hi-em-full-v3.1.1 \
        --benchmark locomo \
        --data benchmarks/locomo/data/locomo10.json \
        --limit 50 --stratify \
        --alpha "$alpha" --lmda "$lmda" --sigma0-sq 0.01 \
        --cos-threshold "$cos" \
        --exp-id "$EXP_ID" \
        --questions-per-round 50 \
        2>&1 | tee -a "$LOG"
      rc=${PIPESTATUS[0]}
      echo "[exit] ${EXP_ID} rc=${rc}" | tee -a "$LOG"
    done
  done
done

echo "=== END $(date -Is) ===" | tee -a "$LOG"
