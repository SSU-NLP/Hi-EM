#!/usr/bin/env bash
# Resume the α×λ×cos main sweep (idempotent — completed exp-ids skip)
# 그리고 main 끝나면 RAG 3 variants 자동 trigger.
set -u
cd "$(git rev-parse --show-toplevel)"

echo "=== RESUME main α×λ×cos sweep $(date -Is) ==="
bash scripts/run_locomo_alphalambda_sweep.sh

echo ""
echo "=== Main sweep done. Starting RAG 3 variants $(date -Is) ==="
bash scripts/run_locomo_sanity50_rag.sh

echo ""
echo "=== Aggregating final summary table $(date -Is) ==="
python3 scripts/aggregate_locomo_alphalambda_results.py

echo ""
echo "=== ALL DONE $(date -Is) ==="
