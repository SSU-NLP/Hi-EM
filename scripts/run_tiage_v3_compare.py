#!/usr/bin/env python3
"""TIAGE topic-shift detection: all hi-em variants in one comparison table.

Variants compared:
    (a) all-boundary baseline
    (b) cosine-threshold baseline (sweep, report best F1)
    (c) v2 persistence:  α=1,  λ=10, σ₀²=0.01    (current default)
    (d) v2 freq-shift:   α=10, λ=1,  σ₀²=0.1     (alternative regime)
    (e) v3.1 Bounded Cosine MAP    (τ, cos_threshold sweep)
    (f) v3.2 Cosine Prediction Error (τ, cos_threshold, ρ sweep)

Output: a single markdown table at outputs/tiage_v3_compare.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.embedding import QueryEncoder  # noqa: E402
from hi_em.sem_core import HiEMSegmenter  # noqa: E402
from hi_em.sem_core_optimize import HiEMSegmenterV3  # noqa: E402
from hi_em.sem_core_optimize_pe import HiEMSegmenterV32  # noqa: E402

DATA_DIR = REPO_ROOT / "benchmarks" / "tiage" / "data" / "personachat" / "anno"


def load_split(split: str) -> dict[str, list[tuple[str, str]]]:
    path = DATA_DIR / split / f"anno_{split}.json"
    raw = json.loads(path.read_text())
    return {cid: [(t[0], t[1]) for t in dialog] for cid, dialog in raw.items()}


def gt_shifts(dialog: list[tuple[str, str]]) -> list[bool]:
    labels = [t[1] for t in dialog]
    return [labels[i] == "1" for i in range(1, len(labels))]


def f1_score(gt: list[bool], pred: list[bool]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for g, p in zip(gt, pred):
        if g and p:
            tp += 1
        elif p and not g:
            fp += 1
        elif g and not p:
            fn += 1
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def _topic_stats(seg) -> tuple[int, float]:
    counts = seg.counts[seg.counts > 0]
    if len(counts) == 0:
        return 0, 0.0
    return int(len(counts)), float(counts.max() / counts.sum())


def run_v2(dialogs, embeddings, alpha, lmda, sigma0_sq):
    gt_all, pred_all = [], []
    n_topics_per_conv = []
    max_share_per_conv = []
    t0 = time.perf_counter()
    n_turns = 0
    for (cid, dialog), emb in zip(dialogs.items(), embeddings):
        seg = HiEMSegmenter(dim=emb.shape[1], alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq)
        ids = [seg.assign(s)[0] for s in emb]
        gt_all.extend(gt_shifts(dialog))
        pred_all.extend(ids[i] != ids[i - 1] for i in range(1, len(ids)))
        nt, ms = _topic_stats(seg)
        n_topics_per_conv.append(nt)
        max_share_per_conv.append(ms)
        n_turns += len(dialog)
    return f1_score(gt_all, pred_all), time.perf_counter() - t0, n_turns, \
        n_topics_per_conv, max_share_per_conv


def run_v3_1(dialogs, embeddings, alpha, lmda, tau, cos_threshold):
    gt_all, pred_all = [], []
    n_topics_per_conv = []
    max_share_per_conv = []
    t0 = time.perf_counter()
    n_turns = 0
    for (cid, dialog), emb in zip(dialogs.items(), embeddings):
        seg = HiEMSegmenterV3(
            dim=emb.shape[1], alpha=alpha, lmda=lmda,
            tau=tau, cos_threshold=cos_threshold,
        )
        ids = [seg.assign(s)[0] for s in emb]
        gt_all.extend(gt_shifts(dialog))
        pred_all.extend(ids[i] != ids[i - 1] for i in range(1, len(ids)))
        nt, ms = _topic_stats(seg)
        n_topics_per_conv.append(nt)
        max_share_per_conv.append(ms)
        n_turns += len(dialog)
    return f1_score(gt_all, pred_all), time.perf_counter() - t0, n_turns, \
        n_topics_per_conv, max_share_per_conv


def run_v3_2(dialogs, embeddings, alpha, lmda, tau, cos_threshold, rho):
    gt_all, pred_all = [], []
    n_topics_per_conv = []
    max_share_per_conv = []
    t0 = time.perf_counter()
    n_turns = 0
    for (cid, dialog), emb in zip(dialogs.items(), embeddings):
        seg = HiEMSegmenterV32(
            dim=emb.shape[1], alpha=alpha, lmda=lmda,
            tau=tau, cos_threshold=cos_threshold, rho=rho,
        )
        ids = [seg.assign(s)[0] for s in emb]
        gt_all.extend(gt_shifts(dialog))
        pred_all.extend(ids[i] != ids[i - 1] for i in range(1, len(ids)))
        nt, ms = _topic_stats(seg)
        n_topics_per_conv.append(nt)
        max_share_per_conv.append(ms)
        n_turns += len(dialog)
    return f1_score(gt_all, pred_all), time.perf_counter() - t0, n_turns, \
        n_topics_per_conv, max_share_per_conv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument(
        "--threshold-sweep", nargs="+", type=float,
        default=list(np.arange(0.3, 0.95, 0.025)),
    )
    parser.add_argument("--tau-sweep", nargs="+", type=float,
                        default=[10, 30, 50, 100])
    parser.add_argument("--cos-thr-sweep", nargs="+", type=float,
                        default=[0.5, 0.7, 0.85])
    parser.add_argument("--rho-sweep", nargs="+", type=float,
                        default=[0.3, 0.5, 0.7])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str,
                        default=str(REPO_ROOT / "outputs" / "tiage_v3_compare.md"))
    args = parser.parse_args()

    print(f"[1/3] loading TIAGE {args.split}...")
    dialogs = load_split(args.split)
    n_convs = len(dialogs)
    n_turns = sum(len(d) for d in dialogs.values())
    n_shifts = sum(sum(gt_shifts(d)) for d in dialogs.values())
    print(f"  {n_convs} conv / {n_turns} turns / {n_shifts} GT shifts")

    print("[2/3] encoding...")
    enc = QueryEncoder(device=args.device)
    all_utts = [t[0] for d in dialogs.values() for t in d]
    t0 = time.perf_counter()
    emb_flat = np.asarray(enc.encode(all_utts))
    enc_sec = time.perf_counter() - t0
    print(f"  encoded {n_turns} turns in {enc_sec:.1f}s on {enc.device}")

    embeddings: list[np.ndarray] = []
    idx = 0
    for d in dialogs.values():
        embeddings.append(emb_flat[idx : idx + len(d)])
        idx += len(d)

    print("[3/3] segmenting...")

    # (a) all-boundary
    gt, pred = [], []
    for d in dialogs.values():
        gs = gt_shifts(d)
        gt.extend(gs)
        pred.extend([True] * len(gs))
    ab_prf = f1_score(gt, pred)
    print(f"  (a) all-boundary       : P={ab_prf[0]:.3f} R={ab_prf[1]:.3f} F1={ab_prf[2]:.3f}")

    # (b) cosine threshold sweep
    best_cos = (0.0, 0.0, -1.0, None)
    for thr in args.threshold_sweep:
        gt, pred = [], []
        for d, emb in zip(dialogs.values(), embeddings):
            gt.extend(gt_shifts(d))
            for i in range(1, len(d)):
                pred.append(float(np.dot(emb[i], emb[i - 1])) < thr)
        prf = f1_score(gt, pred)
        if prf[2] > best_cos[2]:
            best_cos = (*prf, float(thr))
    print(f"  (b) cosine θ={best_cos[3]:.3f} : P={best_cos[0]:.3f} R={best_cos[1]:.3f} F1={best_cos[2]:.3f}")

    # (c) v2 persistence (default)
    (p_v2p, r_v2p, f_v2p), v2p_sec, _, ntopics_v2p, mshare_v2p = run_v2(
        dialogs, embeddings, alpha=1.0, lmda=10.0, sigma0_sq=0.01,
    )
    print(f"  (c) v2 persistence     : P={p_v2p:.3f} R={r_v2p:.3f} F1={f_v2p:.3f} "
          f"avg#topics={np.mean(ntopics_v2p):.1f} avg_max_share={np.mean(mshare_v2p):.2f}")

    # (d) v2 freq-shift
    (p_v2f, r_v2f, f_v2f), v2f_sec, _, ntopics_v2f, mshare_v2f = run_v2(
        dialogs, embeddings, alpha=10.0, lmda=1.0, sigma0_sq=0.1,
    )
    print(f"  (d) v2 freq-shift      : P={p_v2f:.3f} R={r_v2f:.3f} F1={f_v2f:.3f} "
          f"avg#topics={np.mean(ntopics_v2f):.1f} avg_max_share={np.mean(mshare_v2f):.2f}")

    # (e) v3.1 sweep
    best_v31 = None
    for tau in args.tau_sweep:
        for thr in args.cos_thr_sweep:
            (p31, r31, f31), v31_sec, _, ntopics31, mshare31 = run_v3_1(
                dialogs, embeddings, alpha=1.0, lmda=10.0,
                tau=tau, cos_threshold=thr,
            )
            print(f"  (e) v3.1 τ={tau:5.1f} thr={thr:.2f}: "
                  f"P={p31:.3f} R={r31:.3f} F1={f31:.3f} "
                  f"avg#topics={np.mean(ntopics31):.1f} "
                  f"avg_max_share={np.mean(mshare31):.2f}")
            entry = (f31, p31, r31, tau, thr, ntopics31, mshare31, v31_sec)
            if best_v31 is None or f31 > best_v31[0]:
                best_v31 = entry
    f_v31, p_v31, r_v31, tau31, thr31, ntopics31, mshare31, v31_sec = best_v31
    print(f"  v3.1 BEST τ={tau31} thr={thr31:.2f} F1={f_v31:.3f}")

    # (f) v3.2 sweep
    best_v32 = None
    for tau in args.tau_sweep:
        for thr in args.cos_thr_sweep:
            for rho in args.rho_sweep:
                (p32, r32, f32), v32_sec, _, ntopics32, mshare32 = run_v3_2(
                    dialogs, embeddings, alpha=1.0, lmda=10.0,
                    tau=tau, cos_threshold=thr, rho=rho,
                )
                print(f"  (f) v3.2 τ={tau:5.1f} thr={thr:.2f} ρ={rho:.2f}: "
                      f"P={p32:.3f} R={r32:.3f} F1={f32:.3f} "
                      f"avg#topics={np.mean(ntopics32):.1f} "
                      f"avg_max_share={np.mean(mshare32):.2f}")
                entry = (f32, p32, r32, tau, thr, rho,
                         ntopics32, mshare32, v32_sec)
                if best_v32 is None or f32 > best_v32[0]:
                    best_v32 = entry
    (f_v32, p_v32, r_v32, tau32, thr32, rho32,
     ntopics32, mshare32, v32_sec) = best_v32
    print(f"  v3.2 BEST τ={tau32} thr={thr32:.2f} ρ={rho32:.2f} F1={f_v32:.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    overhead_v2 = enc_sec / n_turns * 1000 + v2p_sec / n_turns * 1000
    overhead_v31 = enc_sec / n_turns * 1000 + v31_sec / n_turns * 1000
    overhead_v32 = enc_sec / n_turns * 1000 + v32_sec / n_turns * 1000
    md_lines = [
        f"# TIAGE {args.split} — Hi-EM 변형 모두 비교 (v2 / v3.1 / v3.2)",
        "",
        f"`split={args.split}` device={args.device or 'auto'}",
        "",
        f"Data: {n_convs} dialogs / {n_turns} turns / {n_shifts} GT shifts "
        f"(shift rate {n_shifts/(n_turns-n_convs):.3f} / transition)",
        "",
        "## Topic-shift F1 (turn-transition binary)",
        "",
        "| Method | Precision | Recall | F1 | avg #topics | avg max-share |",
        "|---|---|---|---|---|---|",
        f"| (a) all-boundary | {ab_prf[0]:.3f} | {ab_prf[1]:.3f} | {ab_prf[2]:.3f} | — | — |",
        f"| (b) cosine-threshold (θ={best_cos[3]:.3f}) | {best_cos[0]:.3f} | "
        f"{best_cos[1]:.3f} | {best_cos[2]:.3f} | — | — |",
        f"| (c) v2 hi-em-full-v1 persistence (α=1, λ=10, σ²=0.01) | {p_v2p:.3f} | "
        f"{r_v2p:.3f} | {f_v2p:.3f} | {np.mean(ntopics_v2p):.1f} | "
        f"{np.mean(mshare_v2p):.3f} |",
        f"| (d) v2 hi-em-full-v1 freq-shift (α=10, λ=1, σ²=0.1) | {p_v2f:.3f} | "
        f"{r_v2f:.3f} | {f_v2f:.3f} | {np.mean(ntopics_v2f):.1f} | "
        f"{np.mean(mshare_v2f):.3f} |",
        f"| (e) v3.1 hi-em-full-v3.1.1 (Bounded Cosine MAP, τ={tau31}, thr={thr31}) | "
        f"{p_v31:.3f} | {r_v31:.3f} | {f_v31:.3f} | {np.mean(ntopics31):.1f} | "
        f"{np.mean(mshare31):.3f} |",
        f"| (f) v3.2 hi-em-full-v3.2.1 (Cosine PE, τ={tau32}, thr={thr32}, ρ={rho32}) | "
        f"{p_v32:.3f} | {r_v32:.3f} | {f_v32:.3f} | {np.mean(ntopics32):.1f} | "
        f"{np.mean(mshare32):.3f} |",
        "",
        "## Latency",
        f"- embed: {enc_sec:.1f}s / {enc_sec/n_turns*1000:.2f} ms/turn",
        f"- v2 assign:   {v2p_sec*1000:.1f}ms / {v2p_sec/n_turns*1000:.3f} ms/turn",
        f"- v3.1 assign: {v31_sec*1000:.1f}ms / {v31_sec/n_turns*1000:.3f} ms/turn",
        f"- v3.2 assign: {v32_sec*1000:.1f}ms / {v32_sec/n_turns*1000:.3f} ms/turn",
        f"- 총 overhead (v2):   {overhead_v2:.2f} ms/turn",
        f"- 총 overhead (v3.1): {overhead_v31:.2f} ms/turn",
        f"- 총 overhead (v3.2): {overhead_v32:.2f} ms/turn",
        "",
    ]
    out.write_text("\n".join(md_lines) + "\n")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
