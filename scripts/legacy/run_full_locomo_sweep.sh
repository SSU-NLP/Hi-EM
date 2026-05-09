#!/usr/bin/env bash
# Full LoCoMo (10 conv / 1986 questions, no --limit, no --stratify) sweep
# of best-known Hi-EM configs vs all RAG/sliding baselines.
#
# Best configs (from prior sweeps):
#   - v1     : α=1,   λ=10, σ²₀=0.01            (decision-log 2026-04-25)
#   - v3.1.1 : α=10,  λ=10, cos=0.3             (sweep_2026-05-05, acc=0.292 / sanity50)
#   - v3.3.1 : α=100, λ=10, cos=0.9, β=0.25     (sweep_2026-05-08 HP)
#   - v3.3.2 : v3.3.1 + pe=0.5, train_steps=3   (sweep_2026-05-08 PE)
#
# v3.2.1 intentionally skipped per user request 2026-05-08.
# Captures STM top-K (T1/T2/T3) for hi-em-full-* methods.
set -u
cd "$(git rev-parse --show-toplevel)"

OUT_DIR="outputs/sweeps/2026-05-08_full_locomo"
DATA="benchmarks/locomo/data/locomo10.json"
mkdir -p "$OUT_DIR"

run_one() {
  local method="$1"
  local label="$2"
  shift 2
  local extra=("$@")
  local run_dir="$OUT_DIR/$label"
  local exp_id="20260508_full_locomo_$label"
  local log="$run_dir/run.log"
  local topk="$run_dir/stm_topk.json"

  # Resume guard: if this method already finished cleanly, skip.
  if [ -f "$run_dir/exit_code.txt" ] && [ "$(cat "$run_dir/exit_code.txt")" = "0" ]; then
    echo "[skip] $label — exit_code.txt=0 (already done)"
    return 0
  fi

  mkdir -p "$run_dir"
  rm -f "$topk" "${topk%.json}.rounds.jsonl" 2>/dev/null || true
  {
    echo "=== START $(date -Is) ==="
    echo "method=${method}"
    echo "label=${label}"
    echo "extra=${extra[*]}"
    echo "exp_id=${exp_id}"
    echo
  } > "$log"

  WANDB_MODE=disabled \
  UV_CACHE_DIR=/tmp/uv-cache \
  HIEM_STM_TOPK_STATS_PATH="$topk" \
  uv run python scripts/run_experiment.py \
    --method "$method" \
    --benchmark locomo \
    --data "$DATA" \
    --questions-per-round 200 \
    --exp-id "$exp_id" \
    --results-root "$run_dir/results" \
    --no-token-count --no-thinking --workers 100 \
    "${extra[@]}" \
    >> "$log" 2>&1
  local rc=$?
  echo "$rc" > "$run_dir/exit_code.txt"
  echo "[exit] ${exp_id} rc=${rc}" >> "$log"
  echo "=== END $(date -Is) ===" >> "$log"
}

# Hi-EM lineage best configs.
run_one hi-em-full-v1     v1_best     --alpha 1   --lmda 10 --sigma0-sq 0.01
run_one hi-em-full-v3.1.1 v3p1p1_best --alpha 10  --lmda 10 --cos-threshold 0.3 --tau 50
run_one hi-em-full-v3.3.1 v3p3p1_best --alpha 100 --lmda 10 --cos-threshold 0.9 --beta 0.25
run_one hi-em-full-v3.3.2 v3p3p2_best --alpha 100 --lmda 10 --cos-threshold 0.9 --beta 0.25 \
                                       --pe-threshold 0.5 --rnn-train-steps 3

# Baselines (no segmentation HPs).
run_one rag             rag
run_one rag-summary     rag_summary
run_one rag-observation rag_observation
run_one sliding         sliding
run_one full            full

# Aggregate.
uv run python scripts/aggregate_full_locomo.py
