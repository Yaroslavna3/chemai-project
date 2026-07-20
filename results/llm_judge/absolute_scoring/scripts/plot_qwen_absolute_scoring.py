from __future__ import print_function

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PY2 = sys.version_info[0] == 2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.path.join(ROOT, "qwen_absolute_scoring_detailed.csv")
IMAGE_DIR = os.path.join(ROOT, "images")

STATUS_ORDER = ["bad", "ambiguous", "good"]
REPRESENTATION_ORDER = ["smiles", "iupac"]
COLORS = {
    "bad": "#D55E00",
    "ambiguous": "#E69F00",
    "good": "#009E73",
}


def read_csv_rows(path):
    if PY2:
        handle = open(path, "rb")
    else:
        handle = open(path, "r", newline="", encoding="utf-8-sig")
    try:
        return list(csv.DictReader(handle))
    finally:
        handle.close()


def median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def load_data():
    rows = []
    for row in read_csv_rows(DATA_PATH):
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
        status = row.get("status", "").strip().lower()
        representation = row.get("representation", "").strip().lower()
        if status not in STATUS_ORDER or representation not in REPRESENTATION_ORDER:
            continue
        rows.append(
            {
                "score": score,
                "status": status,
                "representation": representation,
            }
        )
    return rows


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "grid.color": "#DDDDDD",
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def scores_for(rows, representation, status):
    return [
        row["score"]
        for row in rows
        if row["representation"] == representation and row["status"] == status
    ]


def histogram_counts(values, bins):
    counts = [0] * (len(bins) - 1)
    if not values:
        return counts
    for value in values:
        if value == bins[-1]:
            counts[-1] += 1
            continue
        for idx in range(len(bins) - 1):
            if bins[idx] <= value < bins[idx + 1]:
                counts[idx] += 1
                break
    return counts


def plot_histograms(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    bins = [i / 20.0 for i in range(21)]

    for ax, representation in zip(axes, REPRESENTATION_ORDER):
        for status in STATUS_ORDER:
            values = scores_for(rows, representation, status)
            counts = histogram_counts(values, bins)
            ax.bar(
                bins[:-1],
                counts,
                width=0.05,
                align="edge",
                alpha=0.58,
                color=COLORS[status],
                label=status.capitalize(),
                edgecolor="white",
                linewidth=0.6,
            )
        ax.set_title(representation.upper())
        ax.set_xlabel("LLM drug-likeness score")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Molecules")
    axes[1].legend(loc="upper right")
    fig.suptitle("Qwen absolute scoring distributions by input representation", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, "qwen_absolute_scoring_histograms.png"), dpi=220)
    plt.close(fig)


def plot_boxplots(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    for ax, representation in zip(axes, REPRESENTATION_ORDER):
        values = [scores_for(rows, representation, status) for status in STATUS_ORDER]
        bp = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.55,
        )
        ax.set_xticklabels([status.capitalize() for status in STATUS_ORDER])
        for patch, status in zip(bp["boxes"], STATUS_ORDER):
            patch.set_facecolor(COLORS[status])
            patch.set_alpha(0.62)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(1.0)
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color("#333333")
                line.set_linewidth(1.0)
        for line in bp["medians"]:
            line.set_color("#222222")
            line.set_linewidth(1.8)
        for flier in bp.get("fliers", []):
            flier.set_marker("o")
            flier.set_markersize(3.5)
            flier.set_markerfacecolor("#555555")
            flier.set_markeredgecolor("none")
            flier.set_alpha(0.45)

        for idx, vals in enumerate(values, start=1):
            med = median(vals)
            if med is None:
                continue
            ax.text(
                idx,
                med + 0.035,
                "{:.2f}".format(med),
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
            )
        ax.set_title(representation.upper())
        ax.set_xlabel("Benchmark category")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("LLM drug-likeness score")
    fig.suptitle("Qwen absolute scoring by benchmark category", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, "qwen_absolute_scoring_boxplots.png"), dpi=220)
    plt.close(fig)


def main():
    if not os.path.isdir(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    apply_style()
    rows = load_data()
    plot_histograms(rows)
    plot_boxplots(rows)
    print("Wrote figures to {0}".format(IMAGE_DIR))


if __name__ == "__main__":
    main()
