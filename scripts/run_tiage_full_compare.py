#!/usr/bin/env python3
"""TIAGE full compare — v1 / v3.1.1 / v3.3.1 / v3.3.2 / v3.3.3 / v3.3.4 /
v3.3.3-2 / v3.3.4-2 (8 method × N seed).

Default HP: each method's recommended config (matches LoCoMo decision-log).
Random source: ``torch.manual_seed`` + ``numpy.random.seed`` per (method, seed).
Output: REPORT.md with mean ± std F1 + n_topics + wall_per_turn.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.embedding import QueryEncoder  # noqa: E402
from hi_em.sem_core import HiEMSegmenter  # noqa: E402
from hi_em.sem_core_optimize import HiEMSegmenterV3  # noqa: E402
from hi_em.sem_core_v331_rnn import HiEMSegmenterV331  # noqa: E402
from hi_em.sem_core_v332_rnn_pe import HiEMSegmenterV332  # noqa: E402
from hi_em.sem_core_v333_rnn_f0 import HiEMSegmenterV333  # noqa: E402
from hi_em.sem_core_v334_rnn_var import HiEMSegmenterV334  # noqa: E402
from hi_em.sem_core_v333_2 import HiEMSegmenterV333_2  # noqa: E402
from hi_em.sem_core_v334_2 import HiEMSegmenterV334_2  # noqa: E402


DATA_DIR = REPO_ROOT / "benchmarks" / "tiage" / "data" / "personachat" / "anno"


def load_split(split: str):
    path = DATA_DIR / split / f"anno_{split}.json"
    raw = json.loads(path.read_text())
    return {cid: [(t[0], t[1]) for t in dialog] for cid, dialog in raw.items()}


def gt_shifts(dialog):
    labels = [t[1] for t in dialog]
    return [labels[i] == "1" for i in range(1, len(labels))]


# Method configs: (label, factory, uses_rng).
# factory(dim) → segmenter; HP = LoCoMo 2026-05-08~ recommended defaults.
def _factory_v1(dim):
    return HiEMSegmenter(dim=dim, alpha=1.0, lmda=10.0, sigma0_sq=0.01)


def _factory_v311(dim):
    return HiEMSegmenterV3(dim=dim, alpha=1.0, lmda=10.0, tau=50.0, cos_threshold=0.7)


def _factory_v331(dim):
    return HiEMSegmenterV331(
        dim=dim, alpha=100.0, lmda=10.0, tau=50.0,
        cos_threshold=0.9, beta=0.25,
        rnn_hidden_dim=32, rnn_lr=1e-3, rnn_train_steps=3,
    )


def _factory_v332(dim):
    return HiEMSegmenterV332(
        dim=dim, alpha=100.0, lmda=10.0, tau=50.0,
        cos_threshold=0.9, beta=0.25, pe_threshold=0.5,
        rnn_hidden_dim=32, rnn_lr=1e-3, rnn_train_steps=3,
    )


def _factory_v333(dim):
    return HiEMSegmenterV333(
        dim=dim, alpha=100.0, lmda=10.0, tau=50.0,
        cos_threshold=0.9, beta=0.25, pe_threshold=0.5,
        rnn_hidden_dim=32, rnn_lr=1e-3, rnn_train_steps=3,
    )


def _factory_v334(dim):
    return HiEMSegmenterV334(
        dim=dim, alpha=100.0, lmda=10.0, tau=50.0,
        cos_threshold=0.9, beta=0.25,
        rnn_hidden_dim=32, rnn_lr=1e-3, rnn_train_steps=3,
    )


def _factory_v333_2(dim):
    return HiEMSegmenterV333_2(
        dim=dim, alpha=100.0, lmda=10.0, tau=50.0,
        cos_threshold=0.9, beta=0.25, pe_threshold=0.5,
        rnn_hidden_dim=32, rnn_lr=1e-3, rnn_train_steps=3,
    )


def _factory_v334_2(dim):
    return HiEMSegmenterV334_2(
        dim=dim, alpha=100.0, lmda=10.0, tau=50.0,
        cos_threshold=0.9, beta=0.25,
        rnn_hidden_dim=32, rnn_lr=1e-3, rnn_train_steps=3,
    )


METHODS = [
    ("v1", _factory_v1, False),
    ("v3.1.1", _factory_v311, False),
    ("v3.3.1", _factory_v331, True),
    ("v3.3.2", _factory_v332, True),
    ("v3.3.3", _factory_v333, True),
    ("v3.3.4", _factory_v334, True),
    ("v3.3.3-2", _factory_v333_2, True),
    ("v3.3.4-2", _factory_v334_2, True),
]


def run_one(factory, dialogs, embeddings, *, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gt_all, pred_all = [], []
    n_topics = []
    t0 = time.perf_counter()
    n_turns = 0
    for (cid, dialog), emb in zip(dialogs.items(), embeddings):
        torch.manual_seed(seed)
        seg = factory(emb.shape[1])
        ids = [seg.assign(s)[0] for s in emb]
        gt_all.extend(gt_shifts(dialog))
        pred_all.extend(ids[i] != ids[i - 1] for i in range(1, len(ids)))
        n_topics.append(len(set(ids)))
        n_turns += len(dialog)
    f1 = f1_score(gt_all, pred_all)
    p, r, _, _ = precision_recall_fscore_support(
        gt_all, pred_all, average="binary", zero_division=0
    )
    return {
        "f1": f1,
        "precision": p,
        "recall": r,
        "n_topics_mean": float(np.mean(n_topics)),
        "wall_s": time.perf_counter() - t0,
        "n_turns": n_turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--name", type=str, default="2026-05-10_tiage_iter1",
        help="Experiment name → outputs/experiments/<name>/REPORT.md",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output path. Default = outputs/experiments/<name>/REPORT.md",
    )
    args = parser.parse_args()

    print(f"[1/3] loading TIAGE {args.split}...")
    dialogs = load_split(args.split)
    n_convs = len(dialogs)
    n_turns = sum(len(d) for d in dialogs.values())
    n_shifts = sum(sum(gt_shifts(d)) for d in dialogs.values())
    print(f"  {n_convs} conv / {n_turns} turns / {n_shifts} GT shifts")

    print("[2/3] encoding...")
    enc = QueryEncoder(device=args.device)
    embeddings = []
    for cid, dialog in dialogs.items():
        utts = [t[0] for t in dialog]
        embeddings.append(np.asarray(enc.encode(utts)))
    print(f"  encoded — dim {embeddings[0].shape[1]}")

    print("[3/3] running methods...")
    results = {}  # name → list of dicts (one per seed)
    for name, factory, uses_rng in METHODS:
        per_seed = []
        seeds = args.seeds if uses_rng else [args.seeds[0]]
        for s in seeds:
            r = run_one(factory, dialogs, embeddings, seed=s)
            per_seed.append(r)
            print(
                f"  {name:10s} seed={s} F1={r['f1']:.3f} "
                f"P={r['precision']:.3f} R={r['recall']:.3f} "
                f"n_topics={r['n_topics_mean']:.1f} wall={r['wall_s']:.1f}s"
            )
        results[name] = per_seed

    # Aggregate.
    rows = []
    for name, per_seed in results.items():
        f1s = [x["f1"] for x in per_seed]
        ps = [x["precision"] for x in per_seed]
        rs = [x["recall"] for x in per_seed]
        nts = [x["n_topics_mean"] for x in per_seed]
        wall = [x["wall_s"] for x in per_seed]
        rows.append({
            "method": name,
            "n_seeds": len(per_seed),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "p_mean": float(np.mean(ps)),
            "r_mean": float(np.mean(rs)),
            "n_topics_mean": float(np.mean(nts)),
            "wall_per_turn_ms": float(np.mean(wall)) / per_seed[0]["n_turns"] * 1000,
        })

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = REPO_ROOT / "outputs" / "experiments" / args.name / "REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# TIAGE {args.split} — full compare ({len(METHODS)} methods × {len(args.seeds)} seeds)\n",
        f"n_convs={n_convs} · n_turns={n_turns} · n_shifts={n_shifts}\n",
        "| method | n_seeds | F1 (mean ± std) | P (mean) | R (mean) | n_topics (mean) | ms/turn |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows.sort(key=lambda r: r["f1_mean"], reverse=True)
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['n_seeds']} | "
            f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f} | "
            f"{r['p_mean']:.3f} | {r['r_mean']:.3f} | "
            f"{r['n_topics_mean']:.1f} | {r['wall_per_turn_ms']:.2f} |"
        )
    lines.append("")
    best = max(rows, key=lambda r: r["f1_mean"])
    lines.append(f"**best F1**: `{best['method']}` — {best['f1_mean']:.3f} ± {best['f1_std']:.3f}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nreport → {out_path}")


if __name__ == "__main__":
    main()
