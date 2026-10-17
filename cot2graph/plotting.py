"""Shared figure style."""

from __future__ import annotations

from pathlib import Path

COLORS = {
    "correct": "#1F9D55",
    "incorrect": "#E0483E",
    "neutral": "#AEB9C7",
    "amber": "#E8A33D",
    "trend": "#1D2B4A",
    "blue": "#215CAF",
}


def setup() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })


def save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
