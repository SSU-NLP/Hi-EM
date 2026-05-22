"""Figure: per-turn graded score heatmap + boundary overlays for a single conversation.

Picks a conversation from MTB+ and visualizes:
1. (top row) v4.1.3 graded score as a horizontal heatmap (one cell per turn).
2. (next rows) boundary positions from baseline gpt-4o-mini and ours.

Highlights where the two methods agree / disagree and shows the calibrated
continuous signal underneath ours' binary decisions.

Output: ``plots/timeline_tape.{pdf,png}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXP = REPO_ROOT / "outputs/experiments/2026-05-21_v413_secom_swap"
OUT_DIR = EXP / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(p: Path):
    with p.open() as f:
        return [json.loads(line) for line in f]


def boundary_after_indices(segments: list[list[str]]) -> list[int]:
    """Returns the 0-based turn indices AFTER which a boundary occurs
    (i.e. the last turn of each segment except the final segment).

    For visualization purposes ALL utterances are concatenated across the
    conversation's sessions in order, so the indices are global.
    """
    bnd: list[int] = []
    cursor = -1
    for i, seg in enumerate(segments):
        cursor += len(seg)
        if i < len(segments) - 1 and len(seg) > 0:
            bnd.append(cursor)
    return bnd


def get_v413_graded_scores(conv: dict, delta_star: float):
    """Re-run v4.1.3 on the conversation to harvest per-turn graded_score.

    Returns: (flat_scores, session_break_indices, ours_boundaries_global).
    Each session is processed with a fresh segmenter (same as adapter).
    """
    from hi_em.sem_core_v413 import HiEMSegmenterV413

    encoder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    flat_scores: list[float] = []
    session_breaks: list[int] = []
    ours_bnd: list[int] = []
    offset = 0
    for sess in conv["sessions"]:
        if not sess:
            continue
        vecs = encoder.encode(sess, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        seg = HiEMSegmenterV413(dim=768, delta_star=delta_star)
        for t, (v, _) in enumerate(zip(vecs, sess)):
            _, is_bnd = seg.assign(v.astype(np.float64))
            flat_scores.append(seg.last_graded_score)
            if is_bnd:
                ours_bnd.append(offset + t)
        offset += len(sess)
        if offset < sum(len(s) for s in conv["sessions"]):
            session_breaks.append(offset - 1)
    return np.array(flat_scores), session_breaks, ours_bnd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtbp_path",
                    default=str(REPO_ROOT / "benchmarks/SeCom/experiment/data/mtbp/mtbp.jsonl"))
    ap.add_argument("--baseline_segs",
                    default=str(REPO_ROOT / "benchmarks/SeCom/experiment/result/mtbp/gpt4omini/segments_fixed.jsonl"))
    ap.add_argument("--ours_segs",
                    default=str(REPO_ROOT / "benchmarks/SeCom/experiment/result/mtbp/ours/segments.jsonl"))
    ap.add_argument("--delta_star", type=float, default=0.6194)
    ap.add_argument("--conv_index", type=int, default=3,
                    help="Which conversation from MTB+ to visualize (0-10).")
    args = ap.parse_args()

    mtbp = load_jsonl(Path(args.mtbp_path))
    bl_data = {s["conversation_id"]: s for s in load_jsonl(Path(args.baseline_segs))}
    ours_data = {s["conversation_id"]: s for s in load_jsonl(Path(args.ours_segs))}

    conv = mtbp[args.conv_index]
    cid = conv["conversation_id"]
    print(f"plotting conv {args.conv_index}: {cid}")

    flat_scores, session_breaks, ours_bnd_runtime = get_v413_graded_scores(conv, args.delta_star)
    n_turn = len(flat_scores)

    bl_bnd = boundary_after_indices(bl_data[cid]["segments"])
    ours_bnd_saved = boundary_after_indices(ours_data[cid]["segments"])

    fig, (ax_score, ax_bl, ax_ours) = plt.subplots(
        3, 1, figsize=(10.5, 2.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.12},
    )

    # Top: graded score heatmap
    cmap = plt.get_cmap("RdYlBu_r")
    norm = plt.Normalize(vmin=0, vmax=2.0)
    ax_score.imshow(flat_scores[None, :], cmap=cmap, norm=norm, aspect="auto",
                     extent=(-0.5, n_turn - 0.5, -0.5, 0.5))
    ax_score.set_yticks([])
    ax_score.set_ylabel("v4.1.3\n$\\delta_{\\mathrm{eff}}/\\delta^*$",
                         rotation=0, ha="right", va="center", fontsize=8.5)
    ax_score.axhline(0.5, color="black", linewidth=0.3)
    ax_score.axhline(-0.5, color="black", linewidth=0.3)
    # threshold annotations
    thresh_levels = [(0.7, "very_weak"), (1.0, "weak"), (1.3, "normal/strong")]

    # Mark ours boundaries with arrows on top row
    for b in ours_bnd_runtime:
        ax_score.annotate("", xy=(b, 0.55), xytext=(b, 1.1),
                          arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.3),
                          annotation_clip=False)

    # Session boundaries (dashed verticals)
    for sb in session_breaks:
        for ax_ in (ax_score, ax_bl, ax_ours):
            ax_.axvline(sb + 0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.65)

    # Middle: baseline boundary row
    ax_bl.set_yticks([])
    ax_bl.set_ylabel("baseline\n(gpt-4o-mini)", rotation=0, ha="right", va="center", fontsize=8.5)
    ax_bl.set_xlim(-0.5, n_turn - 0.5)
    ax_bl.set_ylim(0, 1)
    for b in bl_bnd:
        ax_bl.add_patch(plt.Rectangle((b + 0.1, 0.15), 0.8, 0.7,
                                       color="#C0504D", alpha=0.85, linewidth=0.4, edgecolor="black"))

    # Bottom: ours boundary row
    ax_ours.set_yticks([])
    ax_ours.set_ylabel("ours\n(v4.1.3)", rotation=0, ha="right", va="center", fontsize=8.5)
    ax_ours.set_xlim(-0.5, n_turn - 0.5)
    ax_ours.set_ylim(0, 1)
    for b in ours_bnd_saved:
        ax_ours.add_patch(plt.Rectangle((b + 0.1, 0.15), 0.8, 0.7,
                                          color="#4F81BD", alpha=0.85, linewidth=0.4, edgecolor="black"))

    # Highlight agreement turns
    agree = set(bl_bnd) & set(ours_bnd_saved)
    for b in agree:
        for ax_ in (ax_bl, ax_ours):
            ax_.add_patch(plt.Rectangle((b + 0.05, 0.05), 0.9, 0.9,
                                          fill=False, edgecolor="gold", linewidth=1.5))

    ax_ours.set_xlabel("Turn index (across all sessions of conv " + str(cid) + ")", fontsize=9)
    ax_ours.set_xticks(np.arange(0, n_turn, 5))

    # Colorbar
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.45, 0.012, 0.42])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 0.7, 1.0, 1.3, 2.0])
    cbar.set_ticklabels(["0", "0.7\nv.weak", "1.0\nweak/normal", "1.3\nstrong", "2.0"])
    cbar.ax.tick_params(labelsize=7)

    handles = [
        mpatches.Patch(color="#C0504D", alpha=0.85, label="baseline boundary"),
        mpatches.Patch(color="#4F81BD", alpha=0.85, label="ours boundary"),
        mpatches.Patch(facecolor="none", edgecolor="gold", linewidth=1.5, label="agreement"),
        mpatches.Patch(facecolor="none", edgecolor="gray", linestyle="--", label="session split"),
    ]
    ax_score.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.55),
                     ncol=4, fontsize=8, frameon=False)

    fig.suptitle(f"v4.1.3 graded boundary score with baseline ↔ ours boundary overlay  "
                 f"(conv {cid}, {n_turn} turns)", fontsize=10, y=1.02)
    plt.tight_layout(rect=(0, 0, 0.89, 0.98))

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"timeline_tape_conv{args.conv_index}.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
