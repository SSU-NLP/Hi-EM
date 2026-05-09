#!/usr/bin/env bash
# locomo sanity 50 — 3 RAG variants (rag / rag-summary / rag-observation).
# 같은 sweep 디렉토리에 결과 떨어뜨려서 α×λ×cos 결과랑 같이 표 정리.
set -u
cd "$(git rev-parse --show-toplevel)"

OUT_DIR=outputs/sweep_2026-05-05_locomo_alpha_lambda_cos
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run_rag.log"

methods=("rag" "rag-summary" "rag-observation")

echo "=== RAG START $(date -Is) ===" | tee -a "$LOG"

for method in "${methods[@]}"; do
  EXP_ID="20260505_locomo_aL_${method//-/_}"
  echo "" | tee -a "$LOG"
  echo "=== Config: method=${method}  $(date -Is) ===" | tee -a "$LOG"
  uv run python scripts/run_experiment.py \
    --method "$method" \
    --benchmark locomo \
    --data benchmarks/locomo/data/locomo10.json \
    --limit 50 --stratify \
    --exp-id "$EXP_ID" \
    --questions-per-round 50 \
    2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "[exit] ${EXP_ID} rc=${rc}" | tee -a "$LOG"
done

echo "=== RAG END $(date -Is) ===" | tee -a "$LOG"
