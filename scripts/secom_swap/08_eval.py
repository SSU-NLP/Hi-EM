"""Evaluate predictions against ground-truth answers.

Metrics:
- QA F1 (token overlap, normalized) — from SeCom's metrics.py
- Best Subspan EM — from SeCom's metrics.py
- BERTScore F1 (roberta-large, ~1.4 GB auto-download, CPU OK)
- ROUGE-L f1

Writes a summary JSON per method to ``--save_path``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks/SeCom/experiment"))

from metrics import evaluate_match  # noqa: E402  qa_f1 + subspan_em


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_path", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--skip_bertscore", action="store_true")
    ap.add_argument("--skip_rouge", action="store_true")
    args = ap.parse_args()

    data = []
    with open(args.load_path) as f:
        for line in f:
            data.append(json.loads(line))

    preds = []
    gts = []
    for sample in data:
        for p, g in zip(sample["predictions"], sample["answers"]):
            preds.append(p)
            gts.append(g)
    print(f"n_qa_pairs: {len(preds)}", flush=True)

    out = {
        "n_conv": len(data),
        "n_qa": len(preds),
    }

    m = evaluate_match(preds, gts, truncate_pred=False)
    out["qa_f1_score"] = m["qa_f1_score"]
    out["best_subspan_em"] = m["best_subspan_em"]
    print(f"qa_f1={m['qa_f1_score']:.2f}  subspan_em={m['best_subspan_em']:.2f}", flush=True)

    if not args.skip_rouge:
        from rouge import Rouge
        rouge = Rouge()
        scores = []
        for p, g in zip(preds, gts):
            try:
                s = rouge.get_scores([p or " "], [g or " "], avg=True)
                scores.append(s["rouge-l"]["f"])
            except Exception:
                scores.append(0.0)
        out["rouge_l_f1"] = sum(scores) / len(scores) * 100
        print(f"rouge_l={out['rouge_l_f1']:.2f}", flush=True)

    if not args.skip_bertscore:
        try:
            from bert_score import score as bert_score
            P, R, F = bert_score(preds, gts, lang="en", verbose=False, batch_size=16)
            out["bertscore_f1"] = float(F.mean()) * 100
            out["bertscore_p"] = float(P.mean()) * 100
            out["bertscore_r"] = float(R.mean()) * 100
            print(f"bertscore_f1={out['bertscore_f1']:.2f}", flush=True)
        except Exception as e:
            print(f"bertscore failed: {e}", flush=True)
            out["bertscore_f1"] = None

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_path).write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {args.save_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
