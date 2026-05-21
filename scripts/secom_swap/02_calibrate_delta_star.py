"""Quick δ* calibration for v4.1.3 on Long-MT-Bench+ with mpnet encoder.

v4.1.x's default ``delta_star=0.5557`` was tuned on TIAGE *train* with
Hi-EM's bge encoder. The mpnet encoder used by SeCom has a different
embedding geometry (multi-qa-mpnet-base-dot-v1 is dot-product tuned), so
the cosine-distance distribution between consecutive turns is different
→ ``delta_star`` must be re-estimated.

Strategy
--------
Long-MT-Bench+ has no ground-truth topic boundaries, so we cannot do
F1-supervised calibration. Instead we use the same *unsupervised* heuristic
SeCom's prompt implicitly uses ("split each session into ~3-turn chunks
when topic shifts"):

  1. Encode all exchanges (L2-normalized mpnet).
  2. Compute prev-cosine distance δ_prev = 1 - cos(s_{t-1}, s_t) per turn.
  3. Report the distribution + a few candidate δ* values:
       - p70  (boundary ~30% of turns — too dense)
       - p80  (~20% boundaries — comparable to SeCom's typical 3-5 segs/session)
       - p85  (~15%)
       - p90  (~10%)

We then pick the δ* whose post-v4.1.3 segment count matches SeCom's
baseline most closely (run after segment.py baseline is done). For the
initial smoke pass we start with p80.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_mtbp(path: Path) -> list[dict]:
    data = []
    with path.open() as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=str(REPO_ROOT / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"),
    )
    ap.add_argument(
        "--encoder",
        default="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    )
    ap.add_argument("--n_conv", type=int, default=11)
    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "outputs/experiments/2026-05-21_v413_secom_swap/delta_star_calibration.json"
        ),
    )
    args = ap.parse_args()

    data = load_mtbp(Path(args.data))[: args.n_conv]
    print(f"calibrating on {len(data)} conversations", flush=True)

    enc = SentenceTransformer(args.encoder)
    print(f"encoder loaded: {args.encoder}", flush=True)

    # Flatten all sentences once + remember session boundaries (much faster
    # than per-session encode calls: 1 batched forward vs N small forwards).
    flat_sents: list[str] = []
    session_spans: list[tuple[int, int]] = []  # (start_idx, length) in flat_sents
    for conv in data:
        for sess in conv["sessions"]:
            if len(sess) < 2:
                continue
            session_spans.append((len(flat_sents), len(sess)))
            flat_sents.extend(sess)
    print(f"total sentences to encode: {len(flat_sents)} "
          f"across {len(session_spans)} sessions", flush=True)

    vecs = enc.encode(
        flat_sents,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float64)
    print(f"encoded shape={vecs.shape}", flush=True)

    all_deltas = []
    per_session_n_turns = []
    for start, n in session_spans:
        sess_vecs = vecs[start : start + n]
        cos = (sess_vecs[:-1] * sess_vecs[1:]).sum(axis=1)
        all_deltas.extend((1.0 - cos).tolist())
        per_session_n_turns.append(n)

    arr = np.array(all_deltas)
    summary = {
        "encoder": args.encoder,
        "n_conv": len(data),
        "n_sessions": len(per_session_n_turns),
        "n_delta_samples": int(arr.size),
        "delta_prev": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p70": float(np.percentile(arr, 70)),
            "p80": float(np.percentile(arr, 80)),
            "p85": float(np.percentile(arr, 85)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        },
        "session_turns": {
            "mean": float(np.mean(per_session_n_turns)),
            "min": int(np.min(per_session_n_turns)),
            "max": int(np.max(per_session_n_turns)),
        },
        "candidate_delta_star": {
            "p70": float(np.percentile(arr, 70)),
            "p80": float(np.percentile(arr, 80)),
            "p85": float(np.percentile(arr, 85)),
            "p90": float(np.percentile(arr, 90)),
        },
        "recommended_initial": float(np.percentile(arr, 80)),
        "note": (
            "p80 = boundary at ~20% of turns ≈ 2-3 segments per 13-turn session. "
            "Tune downward (more boundaries) if SeCom baseline produces more segments, "
            "upward if fewer. v4.1.3 also uses causal-window blending (a=0.5, ctx_window=3) "
            "so effective δ may differ slightly from raw δ_prev."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
