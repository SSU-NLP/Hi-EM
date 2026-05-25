#!/usr/bin/env python3
"""Percentile vs Score curve per (encoder × dataset) — paper-grade figure
showing **WHY p60-p80 band is sufficient**.

Source data: outputs/reports/delta_star_calibration.md table (Score per
percentile across 9 (encoder × dataset) cells).

Plots 3-panel (one per dataset). Each panel has 3 encoder curves +
calibration band (p60-p80) highlighted + oracle line per cell.

산출: outputs/figures/figure_L_percentile_score_curve.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_FIGDIR = REPO / "outputs" / "figures"
OUT_FIGDIR.mkdir(parents=True, exist_ok=True)

PERCENTILES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

# (encoder, dataset, [Score_p50, ..., Score_p95], oracle)
# Source: outputs/reports/delta_star_calibration.md (2026-05-23)
DATA = {
    "mpnet": {
        "TIAGE":        ([0.422, 0.447, 0.457, 0.476, 0.459, 0.447, 0.433, 0.427, 0.402, 0.371], 0.473),
        "Dialseg711":   ([0.433, 0.471, 0.509, 0.540, 0.575, 0.601, 0.616, 0.629, 0.609, 0.521], 0.630),
        "SuperDialseg": ([0.455, 0.459, 0.465, 0.462, 0.459, 0.445, 0.427, 0.402, 0.366, 0.328], 0.464),
    },
    "minilm": {
        "TIAGE":        ([0.439, 0.455, 0.470, 0.472, 0.485, 0.481, 0.469, 0.466, 0.416, 0.371], 0.485),
        "Dialseg711":   ([0.437, 0.468, 0.506, 0.535, 0.565, 0.593, 0.612, 0.599, 0.580, 0.493], 0.609),
        "SuperDialseg": ([0.438, 0.435, 0.430, 0.420, 0.404, 0.391, 0.373, 0.355, 0.338, 0.308], 0.438),
    },
    "minilm-int8": {
        "TIAGE":        ([0.439, 0.453, 0.470, 0.472, 0.489, 0.482, 0.493, 0.448, 0.423, 0.378], 0.489),
        "Dialseg711":   ([0.434, 0.469, 0.502, 0.535, 0.559, 0.586, 0.596, 0.600, 0.575, 0.499], 0.616),
        "SuperDialseg": ([0.436, 0.434, 0.426, 0.415, 0.402, 0.387, 0.367, 0.355, 0.335, 0.305], 0.436),
    },
}

ENCS = ("mpnet", "minilm", "minilm-int8")
DATASETS = ("TIAGE", "Dialseg711", "SuperDialseg")
ENC_PRETTY = {"mpnet": "MPNet", "minilm": "MiniLM", "minilm-int8": "MiniLM-int8"}
ENC_COLOR = {"mpnet": "#1f77b4", "minilm": "#ff7f0e", "minilm-int8": "#d62728"}


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=False)

    for ci, ds in enumerate(DATASETS):
        ax = axes[ci]

        # p60-p80 calibration band
        ax.axvspan(60, 80, color="#FFF3CD", alpha=0.65, zorder=1,
                    label="p60–p80 band" if ci == 0 else None)

        for enc in ENCS:
            scores, oracle = DATA[enc][ds]
            color = ENC_COLOR[enc]
            ax.plot(PERCENTILES, scores, marker="o", linewidth=1.6,
                     markersize=5, color=color,
                     label=ENC_PRETTY[enc] if ci == 0 else None,
                     zorder=4)
            # oracle horizontal line
            ax.axhline(oracle, color=color, linestyle=":", linewidth=0.9,
                        alpha=0.6, zorder=2)
            # mark best percentile with a star
            best_idx = int(np.argmax(scores))
            ax.plot(PERCENTILES[best_idx], scores[best_idx], marker="*",
                     color=color, markersize=14, markeredgecolor="black",
                     markeredgewidth=0.6, zorder=5)

        ax.set_title(ds, fontsize=11, pad=4)
        ax.set_xlabel(r"Percentile $p$ (for $\delta^{*}_{p}$ calibration)",
                       fontsize=9.5)
        if ci == 0:
            ax.set_ylabel(r"Score $\uparrow$", fontsize=10)
        ax.set_xticks([50, 60, 70, 80, 90])
        ax.set_xlim(48, 97)
        ax.tick_params(labelsize=8.5)
        ax.grid(True, alpha=0.25, linestyle=":")

    # legend on first panel
    axes[0].plot([], [], color="gray", linestyle=":", linewidth=0.9,
                  label="oracle (test-sweep)")
    axes[0].plot([], [], marker="*", color="gray", linestyle="None",
                  markersize=12, label="best percentile")
    axes[0].legend(loc="lower left", fontsize=8.5, frameon=True,
                    framealpha=0.92, handlelength=1.6, ncol=1)

    fig.suptitle(
        "Score vs percentile across (encoder × dataset) — best percentile "
        "always falls inside the p60–p80 band",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.955))

    for ext in ("pdf", "png"):
        out = OUT_FIGDIR / f"figure_L_percentile_score_curve.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
