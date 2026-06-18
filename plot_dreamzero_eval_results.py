#!/usr/bin/env python3
"""Plot DreamZero success-once evaluation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = {
    "LIBERO-Spatial": {
        "K=1": {"success_once": 0.9511719, "reward": 0.00947908, "n": 512},
        "K=4 Exec+IDM": {
            "success_once": 0.9765625,
            "reward": 0.00990457,
            "n": 512,
        },
    },
    "LIBERO-Object": {
        "K=1": {"success_once": 0.8991935, "reward": 0.00634655, "n": 496},
        "K=4 Exec+IDM": {
            "success_once": 0.9354839,
            "reward": 0.00656363,
            "n": 496,
        },
    },
    "LIBERO-10": {
        "K=1": {"success_once": 0.684, "reward": 0.00272868, "n": 500},
        "K=4 Exec+IDM": {
            "success_once": 0.754,
            "reward": 0.00299532,
            "n": 500,
        },
        "K=4 Random": {"success_once": 0.73, "reward": 0.00288717, "n": 500},
    },
}

METHOD_ORDER = ("K=1", "K=4 Exec+IDM", "K=4 Random")
COLORS = {
    "K=1": "#334155",
    "K=4 Exec+IDM": "#0F766E",
    "K=4 Random": "#D97706",
}


def style_axis(axis: plt.Axes) -> None:
    """Apply restrained publication-style axis formatting."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#94A3B8")
    axis.spines["bottom"].set_color("#94A3B8")
    axis.tick_params(colors="#334155", labelsize=9)
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    axis.set_axisbelow(True)


def add_success_panel(axis: plt.Axes, suite: str, values: dict) -> None:
    methods = [method for method in METHOD_ORDER if method in values]
    rates = [values[method]["success_once"] for method in methods]
    counts = [round(values[method]["success_once"] * values[method]["n"]) for method in methods]

    bars = axis.bar(
        range(len(methods)),
        rates,
        color=[COLORS[method] for method in methods],
        width=0.66,
        edgecolor="white",
        linewidth=0.8,
    )
    axis.set_ylim(0.0, 1.08)
    axis.set_title(suite, fontsize=13, fontweight="bold", color="#0F172A", pad=10)
    axis.set_ylabel("Success once" if suite == "LIBERO-Spatial" else "")
    axis.set_xticks(range(len(methods)), methods, rotation=0)
    style_axis(axis)

    for bar, rate, count, method in zip(bars, rates, counts, methods):
        n = values[method]["n"]
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.045,
            f"{rate * 100:.1f}%\n{count}/{n}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#0F172A",
        )

    baseline = values["K=1"]["success_once"]
    exec_rate = values["K=4 Exec+IDM"]["success_once"]
    axis.text(
        0.98,
        0.04,
        f"Exec+IDM delta: {(exec_rate - baseline) * 100:+.2f} pp",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=COLORS["K=4 Exec+IDM"],
        fontweight="bold",
        bbox={
            "boxstyle": "square,pad=0.2",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )


def create_figure(output: Path, dpi: int) -> None:
    """Create and save the comparison figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": "#334155",
            "text.color": "#0F172A",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    suites = list(RESULTS)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.7), constrained_layout=False)
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.72,
        bottom=0.16,
        wspace=0.20,
    )

    for column, suite in enumerate(suites):
        add_success_panel(axes[column], suite, RESULTS[suite])

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[method])
        for method in METHOD_ORDER
    ]
    figure.legend(
        legend_handles,
        METHOD_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    figure.suptitle(
        "DreamZero Best-of-K Evaluation",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    figure.text(
        0.5,
        0.92,
        "Observed success-once rates",
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
    )
    figure.text(
        0.5,
        0.018,
        "Spatial n=512; Object n=496; LIBERO-10 n=500. Random selection was evaluated only on LIBERO-10.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#64748B",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dreamzero_eval_success_once.png"),
        help="Output PNG path. A PDF with the same stem is also written.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    create_figure(args.output, args.dpi)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
