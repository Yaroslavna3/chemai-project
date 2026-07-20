from __future__ import print_function

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PY2 = sys.version_info[0] == 2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TARGETS = ["1kzn", "3fqs"]
TRAJECTORIES = {
    "docking_and_metrics": {
        "title": "Docking + metrics",
        "metrics": ["DockingScoreProperty", "LogPProperty", "HeavyAtomCountProperty"],
    },
    "llm_and_docking": {
        "title": "LLM + docking",
        "metrics": ["DrugLikenessProperty", "DockingScoreProperty", "Reward"],
    },
    "llm_and_all_metrics": {
        "title": "LLM + docking + metrics",
        "metrics": ["DrugLikenessProperty", "DockingScoreProperty", "LogPProperty"],
    },
}


def open_csv(path):
    if PY2:
        return open(path, "rb")
    return open(path, "r", newline="", encoding="utf-8-sig")


def read_rows(path):
    handle = open_csv(path)
    try:
        return list(csv.DictReader(handle))
    finally:
        handle.close()


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.frameon": False,
        }
    )


def metric_label(metric):
    labels = {
        "DrugLikenessProperty": "LLM score",
        "DockingScoreProperty": "Docking score",
        "LogPProperty": "LogP",
        "HeavyAtomCountProperty": "Heavy atom count",
        "Reward": "Total reward",
    }
    return labels.get(metric, metric.replace("Property", ""))


def read_epoch_series(path, metric):
    points = []
    for row in read_rows(path):
        epoch = to_float(row.get("Epoch"))
        value = to_float(row.get(metric))
        if epoch is None or value is None:
            continue
        points.append((int(epoch), value))
    points.sort(key=lambda item: item[0])
    return points


def plot_training_dynamics():
    for trajectory, meta in sorted(TRAJECTORIES.items()):
        data_dir = os.path.join(ROOT, trajectory, "data")
        image_dir = os.path.join(ROOT, trajectory, "images")
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)

        metrics = meta["metrics"]
        fig, axes = plt.subplots(len(metrics), 1, figsize=(9.2, 2.55 * len(metrics)), sharex=True)
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            for target in TARGETS:
                path = os.path.join(data_dir, "{0}_training_epoch_means.csv".format(target))
                if not os.path.exists(path):
                    continue
                points = read_epoch_series(path, metric)
                if not points:
                    continue
                epochs = [point[0] for point in points]
                values = [point[1] for point in points]
                ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.8, label=target.upper())
                ax.annotate(
                    "{:.2f}".format(values[-1]),
                    xy=(epochs[-1], values[-1]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=9,
                    va="center",
                )
            ax.set_ylabel(metric_label(metric))
            ax.grid(axis="y", linewidth=0.7, alpha=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="best")

        axes[-1].set_xlabel("Epoch")
        fig.suptitle("{0}: epoch-level optimization dynamics".format(meta["title"]), y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(image_dir, "training_dynamics.png"), dpi=220)
        plt.close(fig)


def values_from_sample(path, metric):
    values = []
    for row in read_rows(path):
        value = to_float(row.get(metric))
        if value is not None:
            values.append(value)
    return values


def make_bins(values, count):
    if not values:
        return []
    low = min(values)
    high = max(values)
    if low == high:
        low -= 0.5
        high += 0.5
    step = (high - low) / float(count)
    return [low + step * idx for idx in range(count + 1)]


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


def plot_final_samples():
    metrics = [
        ("score", "LLM score / reward"),
        ("DockingScoreProperty", "Docking score"),
        ("LogPProperty", "LogP"),
        ("HeavyAtomCountProperty", "Heavy atom count"),
    ]

    for trajectory, meta in sorted(TRAJECTORIES.items()):
        data_dir = os.path.join(ROOT, trajectory, "data")
        image_dir = os.path.join(ROOT, trajectory, "images")
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        if not os.path.isdir(data_dir):
            continue

        sample_paths = [
            os.path.join(data_dir, name)
            for name in sorted(os.listdir(data_dir))
            if "_sample_" in name and name.endswith(".csv")
        ]
        if not sample_paths:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
        axes = axes.ravel()
        any_metric = False
        for ax, metric_info in zip(axes, metrics):
            metric, label = metric_info
            plotted = False
            series = []
            for path in sample_paths:
                target = os.path.basename(path).split("_sample_")[0].upper()
                values = values_from_sample(path, metric)
                if not values:
                    continue
                series.append((target, values))
            all_values = []
            for target, values in series:
                all_values.extend(values)
            bins = make_bins(all_values, 24)
            for target, values in series:
                counts = histogram_counts(values, bins)
                ax.bar(
                    bins[:-1],
                    counts,
                    width=(bins[1] - bins[0]) if len(bins) > 1 else 1,
                    align="edge",
                    alpha=0.55,
                    label=target,
                    edgecolor="white",
                )
                plotted = True
                any_metric = True
            ax.set_title(label)
            ax.grid(axis="y", linewidth=0.7, alpha=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if plotted:
                ax.legend(loc="best")
            else:
                ax.set_visible(False)

        if any_metric:
            fig.suptitle("{0}: final sample distributions".format(meta["title"]), y=1.02)
            fig.tight_layout()
            fig.savefig(os.path.join(image_dir, "final_sample_distributions.png"), dpi=220)
        plt.close(fig)


def plot_llm_score_summary():
    path = os.path.join(ROOT, "tables", "freedpp_llm_score_summary.csv")
    if not os.path.exists(path):
        return
    rows = read_rows(path)
    image_dir = os.path.join(ROOT, "images")
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    order = ["Docking + metrics", "LLM + docking", "LLM + docking + metrics"]
    by_target = {}
    for row in rows:
        by_target[(row.get("Target"), row.get("Trajectory"))] = row

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    width = 0.36
    colors = {"1KZN": "#0072B2", "3FQS": "#009E73"}
    offsets = {"1KZN": -width / 2.0, "3FQS": width / 2.0}
    x_positions = list(range(len(order)))

    for target in ["1KZN", "3FQS"]:
        positions = [x + offsets[target] for x in x_positions]
        means = [to_float(by_target[(target, trajectory)].get("Mean")) for trajectory in order]
        sds = [to_float(by_target[(target, trajectory)].get("SD")) for trajectory in order]
        ax.bar(
            positions,
            means,
            yerr=sds,
            width=width,
            capsize=4,
            label=target,
            color=colors[target],
            alpha=0.82,
        )
        for xpos, value in zip(positions, means):
            ax.text(xpos, value + 0.025, "{:.2f}".format(value), ha="center", fontsize=9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mean LLM drug-likeness score")
    ax.set_title("Final 1000-molecule LLM scores by target and trajectory")
    ax.grid(axis="y", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(image_dir, "freedpp_llm_score_summary.png"), dpi=220)
    plt.close(fig)


def main():
    apply_style()
    plot_training_dynamics()
    plot_final_samples()
    plot_llm_score_summary()
    print("Wrote FREED++ figures under {0}".format(ROOT))


if __name__ == "__main__":
    main()
