#!/usr/bin/env python3
"""Run Hi-EM + baselines on TopiOCQA dev, report topic shift F1 + latency.

Phase 1-3 (see ``plan.md``).

Ground truth
    shift at turn ``i`` iff ``Topic[i] != Topic[i-1]`` within a conversation.
    ``Topic_section`` 변화는 noise (Hi-EM이 해당 경계에서 분할 시 FP).

Baselines
    (a) all-boundary: predict every transition as a shift (recall ≡ 1).
    (b) cosine-threshold: predict shift if ``cos(s_i, s_{i-1}) < θ``;
        θ는 dev sweep에서 best-F1으로 선정 (optimistic baseline).
    (c) Hi-EM: sticky-CRP + 옵션 A (centroid + diag variance).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import v_measure_score, adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hi_em.embedding import QueryEncoder  # noqa: E402
from hi_em.sem_core import HiEMSegmenter  # noqa: E402


# --- Data loading --------------------------------------------------------

def load_topiocqa_dev() -> list[list[dict]]:
    """Load TopiOCQA dev.json and group turns by Conversation_no."""
    path = (
        REPO_ROOT
        / "benchmarks"
        / "topiocqa"
        / "downloads"
        / "data"
        / "topiocqa_dataset"
        / "dev.json"
    )
    raw = json.loads(path.read_text())
    buckets: dict[int, list[dict]] = defaultdict(list)
    for t in raw:
        buckets[t["Conversation_no"]].append(t)
    return [
        sorted(buckets[cno], key=lambda x: x["Turn_no"]) for cno in sorted(buckets)
    ]


def ground_truth_shifts(conv: list[dict]) -> list[bool]:
    """True iff `Topic` field changed at each transition (len == N-1)."""
    topics = [t["Topic"] for t in conv]
    return [topics[i] != topics[i - 1] for i in range(1, len(topics))]


def ground_truth_clusters(conv: list[dict]) -> list[int]:
    """Cluster id per turn (length == N) from ``Topic`` field equality.

    Each unique Topic string becomes a unique cluster id.
    """
    topics = [t["Topic"] for t in conv]
    seen: dict[str, int] = {}
    out: list[int] = []
    for topic in topics:
        if topic not in seen:
            seen[topic] = len(seen)
        out.append(seen[topic])
    return out


def boundaries_to_clusters(n: int, boundaries: list[bool]) -> list[int]:
    """Convert N-1 binary boundary flags to N cluster ids."""
    if n == 0:
        return []
    out = [0]
    cur = 0
    for is_b in boundaries:
        if is_b:
            cur += 1
        out.append(cur)
    return out


def clustering_metrics(
    gt_per_conv: list[list[int]], pred_per_conv: list[list[int]],
) -> tuple[float, float]:
    """Average V-measure and ARI across conversations (per-conv computation).

    Per-conv because cluster ids aren't comparable across conversations.
    Skips dialogs with fewer than 2 turns.
    """
    v_scores, ari_scores = [], []
    for gt, pred in zip(gt_per_conv, pred_per_conv):
        if len(gt) < 2:
            continue
        v_scores.append(v_measure_score(gt, pred))
        ari_scores.append(adjusted_rand_score(gt, pred))
    return float(np.mean(v_scores)), float(np.mean(ari_scores))


# --- Metrics -------------------------------------------------------------

def f1_score(gt: list[bool], pred: list[bool]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for g, p in zip(gt, pred):
        if g and p:
            tp += 1
        elif p and not g:
            fp += 1
        elif g and not p:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# --- Baselines + Hi-EM ---------------------------------------------------

def baseline_all_boundary(convs: list[list[dict]]):
    gt, pred = [], []
    for c in convs:
        gs = ground_truth_shifts(c)
        gt.extend(gs)
        pred.extend([True] * len(gs))
    return gt, pred


def baseline_cosine_threshold(
    convs: list[list[dict]], embeddings: list[np.ndarray], threshold: float
):
    gt, pred = [], []
    for conv, emb in zip(convs, embeddings):
        gt.extend(ground_truth_shifts(conv))
        for i in range(1, len(conv)):
            cos = float(np.dot(emb[i], emb[i - 1]))
            pred.append(cos < threshold)
    return gt, pred


def run_hi_em(
    convs: list[list[dict]],
    embeddings: list[np.ndarray],
    alpha: float,
    lmda: float,
    sigma0_sq: float,
):
    """Returns gt, pred, per_conv_cluster_ids, total_sec, n_assigns."""
    gt, pred = [], []
    per_conv_clusters: list[list[int]] = []
    total_sec = 0.0
    n_assigns = 0
    for conv, emb in zip(convs, embeddings):
        seg = HiEMSegmenter(
            dim=emb.shape[1], alpha=alpha, lmda=lmda, sigma0_sq=sigma0_sq
        )
        gt.extend(ground_truth_shifts(conv))
        t0 = time.perf_counter()
        assignments = [seg.assign(e) for e in emb]
        total_sec += time.perf_counter() - t0
        n_assigns += len(emb)
        pred.extend(a[1] for a in assignments[1:])
        per_conv_clusters.append([a[0] for a in assignments])
    return gt, pred, per_conv_clusters, total_sec, n_assigns


# --- Report --------------------------------------------------------------

def write_report(out_path: Path, convs, results, args) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_convs = len(convs)
    n_turns = sum(len(c) for c in convs)
    n_transitions = n_turns - n_convs
    n_shifts = sum(sum(ground_truth_shifts(c)) for c in convs)
    shift_rate = n_shifts / n_transitions if n_transitions else 0.0

    p_hi, r_hi, f1_hi = results["hi-em"]
    p_cos, r_cos, f1_cos = results["cosine-threshold"]
    p_ab, r_ab, f1_ab = results["all-boundary"]

    overhead_ms = results["embed-ms-per-turn"] + results["hi-em-assign-ms-per-turn"]

    lines = [
        "# Phase 1-3 — TopiOCQA dev segmentation 결과",
        "",
        f"실행 파라미터: α={args.alpha}, λ={args.lmda}, σ₀²={args.sigma0_sq}, "
        f"device={args.device or 'auto'}, limit_convs={args.limit_convs}",
        "",
        "## 데이터",
        f"- conversations: {n_convs}",
        f"- turns: {n_turns}",
        f"- transitions (turn 쌍): {n_transitions}",
        f"- ground-truth shifts (`Topic` 필드 변화): {n_shifts}",
        f"- shift rate per transition: {shift_rate:.3f}",
        "",
        "## Topic shift F1 + Clustering quality (V-measure / ARI, per-conv 평균)",
        "",
        "| Method | Precision | Recall | F1 | V-measure | ARI |",
        "|---|---|---|---|---|---|",
        f"| (a) all-boundary | {p_ab:.3f} | {r_ab:.3f} | {f1_ab:.3f} | "
        f"{results['all-boundary-v']:.3f} | {results['all-boundary-ari']:.3f} |",
        f"| (b) cosine-threshold (θ={results['cosine-threshold-best-thr']}) | "
        f"{p_cos:.3f} | {r_cos:.3f} | {f1_cos:.3f} | "
        f"{results['cosine-threshold-v']:.3f} | {results['cosine-threshold-ari']:.3f} |",
        f"| (c) Hi-EM (sCRP + option A) | {p_hi:.3f} | {r_hi:.3f} | {f1_hi:.3f} | "
        f"{results['hi-em-v']:.3f} | {results['hi-em-ari']:.3f} |",
        "",
        "**지표 의미**:",
        "- **F1**: turn-transition 단위 boundary 정확도 (binary classification)",
        "- **V-measure**: homogeneity × completeness 조화평균. 0~1, 1=완벽한 클러스터 일치",
        "- **ARI** (Adjusted Rand Index): chance-corrected pairwise agreement. 1=완벽, 0=random, 음수 가능",
        "",
        f"- cosine-threshold sweep 후보: {args.threshold_sweep}",
        "",
        "## Latency (Hi-EM overhead)",
        f"- embedding (bge-base-en-v1.5): {results['embed-sec']:.2f}s total, "
        f"{results['embed-ms-per-turn']:.2f} ms/turn",
        f"- HiEMSegmenter.assign(): {results['hi-em-assign-sec']:.3f}s total, "
        f"{results['hi-em-assign-ms-per-turn']:.3f} ms/turn",
        f"- Hi-EM 총 overhead ≈ {overhead_ms:.2f} ms/turn",
        "",
        "## 해석",
        "- **Ground truth**: `Topic` 필드 (Wikipedia doc) 변화만. `Topic_section` 변화는 noise → "
        "Hi-EM이 해당 경계에서 분할 시 False Positive.",
        "- **한계**: TopiOCQA 평균 12턴 → Hi-EM variance($\\sigma^2_k$)가 $n_e \\geq 3$ 이후에 "
        "학습되므로 본 Step은 **centroid 부분만 실측** 검증. variance 효과는 Phase 4 LongMemEval "
        "QA에서 간접 측정.",
        "- **cosine-threshold가 dev에서 sweep됨** — Hi-EM은 그 best θ를 이긴 것.",
        "",
        "## Gate 판정 (plan.md Step 1-4)",
    ]

    cond_baseline = f1_hi > f1_cos
    cond_abs = f1_hi > 0.4
    # +20% latency 제약: 전형 LLM latency 500~2000ms 대비
    cond_latency = overhead_ms <= 200.0  # 1000ms의 20% 기준
    passed = cond_baseline and cond_abs and cond_latency

    lines.extend(
        [
            f"- Hi-EM F1 > cosine baseline F1: **{cond_baseline}** "
            f"({f1_hi:.3f} vs {f1_cos:.3f})",
            f"- Hi-EM F1 > 0.4: **{cond_abs}** ({f1_hi:.3f})",
            f"- 턴당 overhead ≤ 200ms (LLM 1000ms의 20% 기준): **{cond_latency}** "
            f"({overhead_ms:.2f} ms)",
            "",
            f"**Gate 결과: {'PASS' if passed else 'FAIL'}**",
            "",
            (
                "→ **Phase 2 진입 가능**" if passed else
                "→ **옵션 A 번복 필요**. `context/06-decision-log.md`에 append 후 "
                "`context/01-hi-em-design.md §4`를 '번복됨' 마킹하고 옵션 D로 재설계."
            ),
        ]
    )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"report → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold-sweep",
        nargs="+",
        type=float,
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lmda", type=float, default=10.0)
    parser.add_argument("--sigma0-sq", type=float, default=0.01)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "outputs" / "phase-1-topiocqa.md"),
    )
    parser.add_argument(
        "--limit-convs",
        type=int,
        default=None,
        help="Only use first N conversations (quick iteration)",
    )
    args = parser.parse_args()

    print("[1/4] loading TopiOCQA dev...")
    convs = load_topiocqa_dev()
    if args.limit_convs:
        convs = convs[: args.limit_convs]
    n_convs = len(convs)
    n_turns = sum(len(c) for c in convs)
    n_shifts = sum(sum(ground_truth_shifts(c)) for c in convs)
    print(f"  {n_convs} conv / {n_turns} turns / {n_shifts} shift GT")

    print("[2/4] encoding queries with bge-base-en-v1.5...")
    encoder = QueryEncoder(device=args.device)
    print(f"  model on {encoder.device}")
    all_q = [t["Question"] for c in convs for t in c]
    t0 = time.perf_counter()
    emb_flat = np.asarray(encoder.encode(all_q))
    embed_sec = time.perf_counter() - t0
    print(
        f"  encoded {n_turns} turns in {embed_sec:.2f}s "
        f"({embed_sec/n_turns*1000:.2f} ms/turn)"
    )

    embeddings: list[np.ndarray] = []
    idx = 0
    for c in convs:
        embeddings.append(emb_flat[idx : idx + len(c)])
        idx += len(c)

    print("[3/4] baselines + Hi-EM...")
    results: dict = {}

    # GT cluster ids per conversation (Wikipedia Topic 단위)
    gt_clusters_per_conv = [ground_truth_clusters(c) for c in convs]

    # (a) all-boundary
    gt, pred = baseline_all_boundary(convs)
    results["all-boundary"] = f1_score(gt, pred)
    ab_pred_clusters = [list(range(len(c))) for c in convs]
    ab_v, ab_ari = clustering_metrics(gt_clusters_per_conv, ab_pred_clusters)
    results["all-boundary-v"] = ab_v
    results["all-boundary-ari"] = ab_ari
    print(
        f"  (a) all-boundary    : F1={results['all-boundary'][2]:.3f}  "
        f"V={ab_v:.3f} ARI={ab_ari:.3f}"
    )

    # (b) cosine threshold sweep — track best by F1, also save its clusters
    best = (0.0, 0.0, -1.0, None)
    best_pred_clusters: list[list[int]] | None = None
    for thr in args.threshold_sweep:
        gt, pred = baseline_cosine_threshold(convs, embeddings, thr)
        prf = f1_score(gt, pred)
        if prf[2] > best[2]:
            best = (*prf, thr)
            # rebuild per-conv clusters at this θ
            cur_clusters = []
            for conv, emb in zip(convs, embeddings):
                bnds = [
                    float(np.dot(emb[i], emb[i - 1])) < thr
                    for i in range(1, len(conv))
                ]
                cur_clusters.append(boundaries_to_clusters(len(conv), bnds))
            best_pred_clusters = cur_clusters
    results["cosine-threshold"] = best[:3]
    results["cosine-threshold-best-thr"] = best[3]
    cos_v, cos_ari = clustering_metrics(gt_clusters_per_conv, best_pred_clusters)
    results["cosine-threshold-v"] = cos_v
    results["cosine-threshold-ari"] = cos_ari
    print(
        f"  (b) cosine θ={best[3]:.2f}  : F1={best[2]:.3f} "
        f"(P={best[0]:.3f}, R={best[1]:.3f})  V={cos_v:.3f} ARI={cos_ari:.3f}"
    )

    # (c) Hi-EM
    gt, pred, hi_clusters, hi_sec, n_assigns = run_hi_em(
        convs, embeddings, args.alpha, args.lmda, args.sigma0_sq
    )
    results["hi-em"] = f1_score(gt, pred)
    results["hi-em-assign-sec"] = hi_sec
    results["hi-em-assign-ms-per-turn"] = hi_sec / n_assigns * 1000 if n_assigns else 0.0
    results["embed-sec"] = embed_sec
    results["embed-ms-per-turn"] = embed_sec / n_turns * 1000 if n_turns else 0.0
    hi_v, hi_ari = clustering_metrics(gt_clusters_per_conv, hi_clusters)
    results["hi-em-v"] = hi_v
    results["hi-em-ari"] = hi_ari
    p_hi, r_hi, f1_hi = results["hi-em"]
    print(
        f"  (c) Hi-EM           : F1={f1_hi:.3f} (P={p_hi:.3f}, R={r_hi:.3f}), "
        f"V={hi_v:.3f} ARI={hi_ari:.3f}, "
        f"assign={results['hi-em-assign-ms-per-turn']:.3f} ms/turn"
    )

    print("[4/4] writing report...")
    write_report(Path(args.output), convs, results, args)


if __name__ == "__main__":
    main()
