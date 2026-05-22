"""Figure: Per-band precision proves graded_score is a calibrated boundary signal.

Reuses v4.1.3's existing TIAGE/Dialseg711 measurements (recorded in
``context/methodology/v4.1.3.md``):

| dataset    | very_weak | weak  | normal | strong |
|------------|----------:|------:|-------:|-------:|
| TIAGE      | 0.000     | 0.238 | 0.345  | 0.520  |
| Dialseg711 | 0.000     | 0.129 | 0.383  | 0.800  |

Output: ``plots/band_precision.{pdf,png}``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs/experiments/2026-05-21_v413_secom_swap/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ["very_weak\n(<0.7)", "weak\n(0.7–1.0)", "normal\n(1.0–1.3)", "strong\n(≥1.3)"]
TIAGE = [0.000, 0.238, 0.345, 0.520]
DIALSEG = [0.000, 0.129, 0.383, 0.800]


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(BANDS))
    w = 0.38

    bars1 = ax.bar(x - w / 2, TIAGE, w, label="TIAGE", color="#4F81BD", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + w / 2, DIALSEG, w, label="Dialseg711", color="#C0504D", edgecolor="black", linewidth=0.6)

    for b in bars1 + bars2:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(BANDS, fontsize=9)
    ax.set_ylabel("Per-band precision\n(fraction of band turns that are gold boundaries)", fontsize=9)
    ax.set_xlabel("Graded score band  ($\\delta_{\\mathrm{eff}}/\\delta^*$)", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.set_title("Graded boundary score is monotonically calibrated\n"
                 "(precision grows from 0.00 → 0.52 (TIAGE) / 0.80 (Dialseg711))",
                 fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"band_precision.{ext}"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
