"""One-off: build learning-curve figure for slides. Run from repo root:
  uv run --with matplotlib python MPP_Code/charts/make_learning_curve_slide.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "MPP_Code" / "charts" / "learning_curve_train_frac_slide.png"

FRAC = [0.25, 0.5, 0.75, 1.0]
ACC = [0.628, 0.663, 0.689, 0.696]


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(
        FRAC,
        ACC,
        "o-",
        color="#2563eb",
        linewidth=2.75,
        markersize=12,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Accuracy & macro F1",
    )
    for x, y in zip(FRAC, ACC):
        ax.annotate(
            f"{y:.3f}",
            xy=(x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            color="#1e293b",
        )
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Accuracy / macro F1")
    ax.set_title(
        "Sarcasm detection vs. training size\n(fixed V–T–A features, adapted MUStARD++)",
        fontweight="semibold",
    )
    ax.set_xticks(FRAC)
    ax.set_xticklabels(["25%", "50%", "75%", "100%"])
    ax.set_ylim(0.58, 0.72)
    ax.set_xlim(0.2, 1.02)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.92)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
