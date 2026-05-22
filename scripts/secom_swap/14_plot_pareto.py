"""Figure G: Context length vs QA quality Pareto across 8 methods.

X-axis = avg # retrieved tokens (Context Length, paper Table 1).
Y-axis = GPT4Score (paper's headline QA metric).

Each method = one point. Pareto frontier drawn as connecting line of the
Pareto-optimal methods (max GPT4Score for each token budget).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP = REPO_ROOT / "outputs/experiments/2026-05-21_v413_secom_swap"
OUT_DIR = EXP / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


METHODS = {
    "zero":       ("Zero",            "#888888", "x",   None),
    "texttiling": ("TextTiling-style","#7B9F35", "v",   None),
    "graphseg":   ("GraphSeg-style",  "#9467BD", "P",   None),
    "greedyseg":  ("GreedySeg-style", "#E1812C", "s",   None),
    "csm":        ("CSM-style (ours-trained)","#1F77B4", "D", None),
    "ours":       ("Hi-Seg (Ours)",   "#D62728", "*",   220),
    "baseline":   ("GPT-4o-mini-Seg", "#2CA02C", "o",   None),
    "full":       ("Full History",    "#666666", "^",   None),
}


def main() -> None:
    pts = []
    for name, (label, color, marker, size) in METHODS.items():
        p = EXP / f"metrics_{name}.json"
        d = json.loads(p.read_text())
        n_tok = d.get("n_tokens") or 0
        g4 = d.get("gpt4_score_x10") or 0.0
        bs = d.get("bertscore_f1") or 0.0
        bleu = d.get("bleu") or 0.0
        pts.append({
            "name": name, "label": label, "color": color, "marker": marker,
            "size": size or 110,
            "n_tokens": n_tok, "gpt4": g4, "bertscore": bs, "bleu": bleu,
        })

    # Pareto frontier: sort by n_tokens ascending, keep points whose gpt4
    # exceeds all earlier ones (lower-token side).
    sorted_pts = sorted(pts, key=lambda p: p["n_tokens"])
    pareto = []
    best = -1
    for p in sorted_pts:
        if p["gpt4"] > best:
            pareto.append(p)
            best = p["gpt4"]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # Pareto frontier (drawn first, under scatter)
    if len(pareto) >= 2:
        px = [p["n_tokens"] for p in pareto]
        py = [p["gpt4"] for p in pareto]
        ax.plot(px, py, "--", color="#666666", linewidth=1.3, alpha=0.6,
                 label="Pareto frontier", zorder=1)

    # Scatter points
    for p in pts:
        ax.scatter(p["n_tokens"], p["gpt4"], s=p["size"],
                    marker=p["marker"], color=p["color"],
                    edgecolor="black", linewidth=0.7, zorder=3,
                    label=p["label"])
        dx, dy = 50, 0.8
        # offset annotation for legibility
        if p["name"] == "zero":
            dx, dy = 100, 1.2
        if p["name"] == "ours":
            dx, dy = 120, 1.3
        if p["name"] == "full":
            dx, dy = -2500, 1.2
        if p["name"] == "texttiling":
            dx, dy = -80, -2.5
        if p["name"] == "csm":
            dx, dy = -50, -2.5
        if p["name"] == "graphseg":
            dx, dy = 60, -2.0
        if p["name"] == "greedyseg":
            dx, dy = 60, 1.0
        ax.annotate(p["label"], xy=(p["n_tokens"], p["gpt4"]),
                     xytext=(p["n_tokens"] + dx, p["gpt4"] + dy),
                     fontsize=8.5, ha="left", zorder=4)

    # Hi-Seg star highlight
    ours = next(p for p in pts if p["name"] == "ours")
    ax.scatter(ours["n_tokens"], ours["gpt4"], s=400, marker="*",
                facecolor="#D62728", edgecolor="black", linewidth=1.0,
                zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(50, 50000)
    ax.set_xlabel("Average # retrieved tokens per query (log scale)", fontsize=10)
    ax.set_ylabel("GPT4Score (× 10)", fontsize=10)
    ax.set_ylim(35, 85)
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    ax.set_title(
        "Context-quality Pareto on Long-MT-Bench+\n"
        "(Hi-Seg achieves baseline-LLM-level quality at low token budget)",
        fontsize=10,
    )
    # legend disabled (labels are inline)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"pareto_qa_context.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
