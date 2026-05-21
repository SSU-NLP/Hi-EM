"""v4.1.3 segmentation runner — mirrors SeCom's ``experiment/segment.py``.

Reads the same input JSONL (``data/mtbp/mtbp.jsonl``) and writes the same
output schema (each sample gains ``sample["segments"]: List[List[str]]``),
so SeCom's downstream pipeline (``compress.py`` → ``retrieve.py`` →
``chat.py``) consumes our segments transparently.

Differences vs SeCom's segment.py:
- No LLM call. Uses :class:`hi_em.secom_adapter.HiEMSecomSegmenter`.
- Records per-conversation latency (encode + assign).
- Records boundary-strength histogram (very_weak / weak / normal / strong).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from hi_em.secom_adapter import HiEMSecomSegmenter

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--load_path",
        default=str(REPO_ROOT / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"),
    )
    ap.add_argument(
        "--save_path",
        default=str(
            REPO_ROOT
            / "benchmarks/SeCom/experiment/result/mtbp/v413seg_mtbp.jsonl"
        ),
    )
    ap.add_argument(
        "--latency_path",
        default=str(
            REPO_ROOT
            / "outputs/experiments/2026-05-21_v413_secom_swap/latency_v413.json"
        ),
    )
    ap.add_argument(
        "--encoder",
        default="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    )
    ap.add_argument("--delta_star", type=float, required=True)
    ap.add_argument("--dim", type=int, default=768)
    args = ap.parse_args()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.latency_path).parent.mkdir(parents=True, exist_ok=True)

    data = load_jsonl(Path(args.load_path))
    print(f"n_conv: {len(data)}")

    encoder = SentenceTransformer(args.encoder)
    seg = HiEMSecomSegmenter(
        encoder=encoder,
        dim=args.dim,
        delta_star=args.delta_star,
    )

    per_conv_latency: list[dict] = []
    results: list[dict] = []

    for idx, sample in enumerate(tqdm(data, desc="segmenting")):
        segments = seg.segment(sample["sessions"])
        sample["segments"] = segments
        results.append(sample)
        lat = seg.last_latency.asdict()
        lat["conversation_id"] = sample["conversation_id"]
        lat["n_segments"] = len(segments)
        per_conv_latency.append(lat)
        print(
            f"  conv {idx} ({sample['conversation_id']}): "
            f"{lat['n_exchanges']} ex → {len(segments)} segs, "
            f"{lat['total_sec']*1000:.1f}ms total, "
            f"{lat['sec_per_exchange']*1000:.2f}ms/ex"
        )

    with open(args.save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ex_total = sum(lat["n_exchanges"] for lat in per_conv_latency)
    n_seg_total = sum(lat["n_segments"] for lat in per_conv_latency)
    total_sec = sum(lat["total_sec"] for lat in per_conv_latency)
    encode_sec = sum(lat["encode_sec"] for lat in per_conv_latency)
    segment_sec = sum(lat["segment_sec"] for lat in per_conv_latency)

    summary = {
        "encoder": args.encoder,
        "delta_star": args.delta_star,
        "n_conv": len(per_conv_latency),
        "n_exchanges": n_ex_total,
        "n_segments": n_seg_total,
        "avg_exchanges_per_segment": n_ex_total / max(1, n_seg_total),
        "boundary_strength_total": seg.boundary_strength_total,
        "total_sec": total_sec,
        "encode_sec": encode_sec,
        "segment_sec": segment_sec,
        "ms_per_exchange_total": total_sec * 1000 / max(1, n_ex_total),
        "ms_per_exchange_segment_only": segment_sec * 1000 / max(1, n_ex_total),
        "per_conv": per_conv_latency,
    }
    with open(args.latency_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nlatency summary -> {args.latency_path}")
    print(
        f"v4.1.3: {n_seg_total} segments / {n_ex_total} exchanges, "
        f"{summary['ms_per_exchange_total']:.2f} ms/exchange total, "
        f"{summary['ms_per_exchange_segment_only']:.3f} ms/exchange (assign-only)"
    )


if __name__ == "__main__":
    main()
