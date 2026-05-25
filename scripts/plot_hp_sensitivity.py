"""1D marginal sensitivity figure for (m, rho, a) at oracle delta*.

Reads: outputs/experiments/2026-05-25_hidots_mra_sweep/sweep.json
Outputs: outputs/figures/figure_I_hp_sensitivity.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
})

REPO = Path(__file__).resolve().parent.parent
SWEEP_JSON = REPO / "outputs" / "experiments" / "2026-05-25_hidots_mra_sweep" / "sweep.json"
FIG_DIR = REPO / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT = dict(m=2, rho=0.7, a=0.5)


def main() -> None:
    data = json.loads(SWEEP_JSON.read_text())
    grid = data["grid"]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)

    metrics_orac = "oracle_mean3"
    metrics_px   = "bestpx_mean3"
    color_orac = "#d62728"  # red
    color_px   = "#1f77b4"  # blue

    hps = [("m", "$m$  (context window size)", [2, 3, 4, 5, 6, 8]),
            ("rho", r"$\rho$  (context decay)", [0.5, 0.7, 0.9]),
            ("a", "$a$  (blend weight)", [0.0, 0.3, 0.5, 0.7, 1.0])]

    for ax, (hp, xlabel, values) in zip(axes, hps):
        # for each value of hp, aggregate over the other two dims
        max_orac = []
        max_px = []
        mean_orac = []
        mean_px = []
        for v in values:
            sub = [r for r in grid if r[hp] == v]
            max_orac.append(max(r[metrics_orac] for r in sub))
            max_px.append(max(r[metrics_px] for r in sub))
            mean_orac.append(np.mean([r[metrics_orac] for r in sub]))
            mean_px.append(np.mean([r[metrics_px] for r in sub]))

        # plot max (line) + mean (dotted)
        ax.plot(values, max_orac, "o-", color=color_orac, lw=1.3,
                  markersize=6, label="Oracle (max)")
        ax.plot(values, mean_orac, "o--", color=color_orac, lw=0.8,
                  markersize=4, alpha=0.55, label="Oracle (mean)")
        ax.plot(values, max_px, "s-", color=color_px, lw=1.3,
                  markersize=5.5, label="Best-$p_x$ (max)")
        ax.plot(values, mean_px, "s--", color=color_px, lw=0.8,
                  markersize=4, alpha=0.55, label="Best-$p_x$ (mean)")

        # mark default
        default_v = DEFAULT[hp]
        if default_v in values:
            ax.axvline(default_v, color="black", linestyle=":",
                          linewidth=0.7, alpha=0.5, zorder=1)

        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.22, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.set_xticks(values)
        ax.set_xticklabels([str(v) for v in values])

    axes[0].set_ylabel("Mean Score across 3 benchmarks")
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.95,
                    edgecolor="#bbb", borderpad=0.4, handletextpad=0.4,
                    labelspacing=0.3)

    fig.suptitle(
        "Hyperparameter sensitivity at per-bench best $\\delta^{*}$  "
        "(default $m=2,\\,\\rho=0.7,\\,a=0.5$ marked with dotted line)",
        fontsize=9.5, y=1.02,
    )

    plt.tight_layout()
    out = FIG_DIR / "figure_I_hp_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
