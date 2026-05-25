#!/usr/bin/env python3
"""Paper figure — calib N convergence (gap-to-oracle).

inputs: outputs/experiments/2026-05-24_calib_n_option_a/results.json
outputs:
    main figure (bench-averaged, 3-panel): calib_n_convergence_main.pdf/.png
    appendix figure (9-cell small multiples): calib_n_convergence_appendix.pdf/.png

Design choices (from codex viz consult):
- y = gap to test-side oracle (Score) — lower=better, baseline at 0
- x = log N (calibration dialogs)
- threshold dashed line at gap=0.005 (= "near oracle")
- lines = p60 / p70 / p80 (label-free percentile), best=gray dashed (deemphasized)
- σ band (3-5 seed bootstrap) as shaded fill at α=0.18
- main = bench-averaged (3 encoders → 1 line per percentile per bench)
- appendix = per-cell (3×3 small multiples)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "outputs" / "experiments" / "2026-05-24_calib_n_option_a" / "results.json"
OUT_DIR = REPO / "outputs" / "experiments" / "2026-05-24_calib_n_option_a"

ENCS = ("mpnet", "minilm", "minilm-int8")
ENC_PRETTY = {"mpnet": "MPNet", "minilm": "MiniLM",
              "minilm-int8": "MiniLM-int8"}
BENCHES = ("tiage", "dialseg711", "superseg")
BENCH_PRETTY = {"tiage": "TIAGE", "dialseg711": "Dialseg711",
                "superseg": "SuperDialseg"}

# colorblind-friendly palette (tab10 subset)
PCOLORS = {
    "p60": "#1f77b4",   # blue
    "p70": "#ff7f0e",   # orange
    "p80": "#d62728",   # red
    "best": "#7f7f7f",  # gray
}
PSTYLES = {
    "p60": ("-", "o", 3),
    "p70": ("-", "s", 3),
    "p80": ("-", "^", 3),
    "best": ("--", "x", 3),
}

GAP_THRESHOLD = 0.005


def _setup_rc():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "lines.linewidth": 1.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _curve(ax, ns, mean, std, label, key, *, show_band=True):
    color = PCOLORS[key]
    style, marker, ms = PSTYLES[key]
    mean = np.asarray(mean)
    std = np.asarray(std)
    ax.plot(ns, mean, style, color=color, marker=marker,
            markersize=ms, label=label, zorder=3)
    if show_band:
        ax.fill_between(ns, mean - std, mean + std,
                        color=color, alpha=0.18, linewidth=0, zorder=1)


def _annotate_threshold(ax):
    ax.axhline(GAP_THRESHOLD, color="#444444", linestyle=":", linewidth=0.8,
               alpha=0.7, zorder=2)
    ax.text(3.2, GAP_THRESHOLD + 0.0015, f"gap = {GAP_THRESHOLD}",
            fontsize=7, color="#444444", alpha=0.85, ha="left", va="bottom")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.6, zorder=2)


def _bench_avg(results, bench, ref="oracle"):
    """encoder-averaged gap for a bench. ref = 'oracle' or 'sup' (best at same N)."""
    n_set = sorted({r["N"]
                    for enc in ENCS
                    for r in results[f"{enc}/{bench}"]["rows"]})
    avg = {p: {n: [] for n in n_set} for p in ("p60", "p70", "p80", "best")}
    std = {p: {n: [] for n in n_set} for p in ("p60", "p70", "p80", "best")}
    for enc in ENCS:
        oracle_val = results[f"{enc}/{bench}"]["oracle"]
        for row in results[f"{enc}/{bench}"]["rows"]:
            ref_val = oracle_val if ref == "oracle" else row["best_mean"]
            for p in ("p60", "p70", "p80", "best"):
                gap = ref_val - row[f"{p}_mean"]
                avg[p][row["N"]].append(gap)
                std[p][row["N"]].append(row[f"{p}_std"])
    out = {}
    for p in ("p60", "p70", "p80", "best"):
        ns_p = sorted(n for n in n_set if avg[p][n])
        out[p] = (np.array(ns_p),
                  np.array([np.mean(avg[p][n]) for n in ns_p]),
                  np.array([np.mean(std[p][n]) for n in ns_p]))
    return out


def plot_main(results):
    # EMNLP two-column friendly: 7in wide. 2-row × 3-bench:
    # row 0 = gap to oracle (test-side ceiling)
    # row 1 = gap to sup (train-side labeled tuning, at same N)
    # Linear x-axis with actual N proportion → user request 2026-05-24.
    # Wider figure (12.5 in) so tick labels {3, 10, 30, 100, 300} don't overlap
    # at the densely-packed left end on a proportional axis.
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 4.5), sharex=True)
    for col, bench in enumerate(BENCHES):
        for row, ref in enumerate(("oracle", "sup")):
            ax = axes[row, col]
            data = _bench_avg(results, bench, ref=ref)
            for key in ("p60", "p70", "p80"):
                ns, mean, std = data[key]
                _curve(ax, ns, mean, std, label=key, key=key)
            # gap=0 baseline + threshold (only on oracle row, sup can go negative)
            ax.axhline(0, color="black", linestyle="-", linewidth=0.6, zorder=2)
            if ref == "oracle":
                ax.axhline(GAP_THRESHOLD, color="#444", linestyle=":",
                           linewidth=0.7, alpha=0.7, zorder=2)
            # Linear scale → ticks at actual proportional positions (user 2026-05-24).
            ax.set_xticks([0, 100, 200, 300])
            ax.set_xlim(0, 310)
            if row == 1:
                ax.set_xlabel(r"Calibration dialogs $N$", fontsize=8.5)
            if row == 0:
                ax.set_title(BENCH_PRETTY[bench], fontsize=9, pad=2)
            if col == 0:
                ylab = "Gap to oracle" if ref == "oracle" else "Gap to sup"
                ax.set_ylabel(ylab, fontsize=8.5)
            ax.tick_params(labelsize=8)
    # legend only in top-left
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=7.5,
                      handlelength=1.2, borderaxespad=0.2, labelspacing=0.25)
    fig.tight_layout()
    main_pdf = OUT_DIR / "calib_n_convergence_main.pdf"
    main_png = OUT_DIR / "calib_n_convergence_main.png"
    fig.savefig(main_pdf)
    fig.savefig(main_png)
    plt.close(fig)
    print(f"main → {main_pdf}", flush=True)


def _plot_appendix_with_ref(results, ref: str):
    """3×3 small multiples. ref='oracle' or 'sup'."""
    fig, axes = plt.subplots(3, 3, figsize=(17.0, 7.4),
                             sharex=True, sharey="row")
    plt.rcParams.update({'font.size': 8})
    for r, enc in enumerate(ENCS):
        for c, bench in enumerate(BENCHES):
            ax = axes[r, c]
            cell = results[f"{enc}/{bench}"]
            oracle = cell["oracle"]
            ns = [row["N"] for row in cell["rows"]]
            for key in ("p60", "p70", "p80"):
                mean = np.array([(oracle if ref == "oracle" else row["best_mean"])
                                 - row[f"{key}_mean"]
                                 for row in cell["rows"]])
                std = np.array([row[f"{key}_std"] for row in cell["rows"]])
                _curve(ax, ns, mean, std, label=key, key=key)
            ax.axhline(0, color="black", linestyle="-", linewidth=0.6, zorder=2)
            if ref == "oracle":
                ax.axhline(GAP_THRESHOLD, color="#444", linestyle=":",
                           linewidth=0.7, alpha=0.7, zorder=2)
            ax.set_xticks([0, 100, 200, 300])
            ax.set_xlim(0, 310)
            ax.tick_params(labelsize=7.5)
            if r == 0:
                ax.set_title(BENCH_PRETTY[bench], fontsize=9, pad=2)
            if r == 2:
                ax.set_xlabel(r"Calibration dialogs $N$", fontsize=8)
            if c == 0:
                ylab_short = "oracle" if ref == "oracle" else "sup"
                ax.set_ylabel(f"{ENC_PRETTY[enc]}\nGap to {ylab_short}", fontsize=8)
            # (text box removed per user request — keep cells clean)
    axes[0, 2].legend(loc="upper right", frameon=False, fontsize=7.5,
                      handlelength=1.2, borderaxespad=0.2, labelspacing=0.25)
    fig.tight_layout()
    suffix = "appendix" if ref == "oracle" else "appendix_sup"
    apx_pdf = OUT_DIR / f"calib_n_convergence_{suffix}.pdf"
    apx_png = OUT_DIR / f"calib_n_convergence_{suffix}.png"
    fig.savefig(apx_pdf); fig.savefig(apx_png)
    plt.close(fig)
    print(f"appendix ({ref}) → {apx_pdf}", flush=True)


def plot_appendix(results):
    _plot_appendix_with_ref(results, ref="oracle")
    _plot_appendix_with_ref(results, ref="sup")


def main() -> None:
    _setup_rc()
    if not RESULTS.exists():
        raise SystemExit(f"results.json not yet — run diag_calib_n_option_a.py first ({RESULTS})")
    results = json.loads(RESULTS.read_text())
    plot_main(results)
    plot_appendix(results)


if __name__ == "__main__":
    main()
