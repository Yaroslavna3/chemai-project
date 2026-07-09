import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import run_druglikeness_experiments as exp


ANALYSIS_DIR = exp.ANALYSIS_DIR
STATUS_ORDER = ["bad", "ambiguous", "good"]
STATUS_PALETTE = {
    "bad": "#c0392b",
    "ambiguous": "#d99a2b",
    "good": "#2e7d5b",
}


def find_latest_run_dir() -> Path:
    candidates = [
        path
        for path in ANALYSIS_DIR.iterdir()
        if path.is_dir() and (path / "results_detailed.csv").exists()
    ]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    legacy_path = ANALYSIS_DIR / "druglikeness_results_detailed.csv"
    if legacy_path.exists():
        return ANALYSIS_DIR
    raise FileNotFoundError(f"No experiment results found under {ANALYSIS_DIR}")


def details_path_for(run_dir: Path) -> Path:
    modern = run_dir / "results_detailed.csv"
    if modern.exists():
        return modern
    legacy = run_dir / "druglikeness_results_detailed.csv"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Detailed results not found in {run_dir}")


def sample_path_for(run_dir: Path) -> Path | None:
    for name in ["sample.csv", "druglikeness_sample.csv"]:
        path = run_dir / name
        if path.exists():
            return path
    return None


def add_condition_column(details: pd.DataFrame) -> pd.DataFrame:
    details = details.copy()
    details["condition"] = (
        details["model_family"]
        + " | "
        + details["representation"].str.replace("_", "+", regex=False)
        + " | "
        + details["scale"].astype(str)
    )
    return details


def make_group_tables(details: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        details.groupby(
            [
                "experiment_id",
                "model_family",
                "model",
                "representation",
                "scale",
                "condition",
                "status",
            ],
            as_index=False,
        )
        .agg(
            n=("score", "size"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            std_score=("score", "std"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .sort_values(["model_family", "representation", "scale", "status"])
    )

    pivot = grouped.pivot_table(
        index=[
            "experiment_id",
            "model_family",
            "model",
            "representation",
            "scale",
            "condition",
        ],
        columns="status",
        values="mean_score",
    ).reset_index()
    for status in STATUS_ORDER:
        if status not in pivot:
            pivot[status] = pd.NA
    pivot["good_minus_bad"] = pivot["good"] - pivot["bad"]
    pivot["good_minus_ambiguous"] = pivot["good"] - pivot["ambiguous"]
    pivot["ordered_separation"] = pivot["good"] - pivot[["bad", "ambiguous"]].max(axis=1)
    pivot = pivot.sort_values(["scale", "ordered_separation"], ascending=[True, False])
    return grouped, pivot


def save_grouped_barplot(run_dir: Path, grouped: pd.DataFrame):
    plot_df = grouped.copy()
    representations = list(plot_df["representation"].drop_duplicates())
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=plot_df,
        x="representation",
        y="mean_score",
        hue="status",
        col="model_family",
        row="scale",
        kind="bar",
        order=representations,
        hue_order=[status for status in STATUS_ORDER if status in set(plot_df["status"])],
        palette=STATUS_PALETTE,
        height=3.3,
        aspect=1.25,
        sharey=False,
    )
    g.set_axis_labels("Representation", "Mean score")
    g.set_titles(row_template="Scale {row_name}", col_template="{col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=20)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    plt.tight_layout()
    g.figure.savefig(run_dir / "grouped_mean_scores.png", dpi=180, bbox_inches="tight")
    plt.close(g.figure)


def save_separation_plot(run_dir: Path, pivot: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    scales = list(pivot["scale"].drop_duplicates())
    fig, axes = plt.subplots(1, len(scales), figsize=(7.5 * len(scales), 6), sharey=False)
    if len(scales) == 1:
        axes = [axes]
    for ax, scale in zip(axes, scales):
        plot_df = pivot[pivot["scale"].eq(scale)].copy().sort_values("ordered_separation", ascending=True)
        colors = ["#2e7d5b" if v >= 0 else "#c0392b" for v in plot_df["ordered_separation"]]
        labels = plot_df["model_family"] + " | " + plot_df["representation"].str.replace("_", "+", regex=False)
        bars = ax.barh(labels, plot_df["ordered_separation"], color=colors)
        ax.axvline(0, color="#374151", linewidth=1)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        ax.set_xlabel("Good mean - max(Bad mean, Ambiguous mean)")
        ax.set_ylabel("")
        ax.set_title(f"Scale {scale}")
    fig.suptitle("Which conditions separate good molecules best")
    plt.tight_layout()
    fig.savefig(run_dir / "condition_separation_ranking.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_boxplot(run_dir: Path, details: pd.DataFrame):
    sns.set_theme(style="whitegrid")
    present_statuses = [status for status in STATUS_ORDER if status in set(details["status"])]
    g = sns.catplot(
        data=details,
        x="model_family",
        y="score",
        hue="status",
        col="scale",
        kind="box",
        hue_order=present_statuses,
        palette=STATUS_PALETTE,
        height=4.2,
        aspect=1.25,
        sharey=False,
        legend_out=True,
    )
    g.set_axis_labels("Model family", "Score")
    g.set_titles(col_template="Scale {col_name}")
    g.figure.suptitle("Score distributions by model family and dataset group", y=1.03)
    if g._legend is not None:
        sns.move_legend(g, "center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    median_df = (
        details.groupby(["scale", "model_family", "status"], as_index=False)["score"]
        .median()
        .rename(columns={"score": "median_score"})
    )
    model_order = list(details["model_family"].drop_duplicates())
    offsets = {
        status: offset
        for status, offset in zip(present_statuses, [-0.27, 0.0, 0.27][: len(present_statuses)])
    }
    for ax, scale in zip(g.axes.flat, sorted(details["scale"].drop_duplicates())):
        sub = median_df[median_df["scale"].eq(scale)]
        for _, row in sub.iterrows():
            x = model_order.index(row["model_family"]) + offsets[row["status"]]
            y = row["median_score"]
            ax.text(
                x,
                y,
                f"{y:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="#111827",
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#cbd5e1", "alpha": 0.85},
            )
    plt.tight_layout()
    g.figure.savefig(run_dir / "status_score_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(g.figure)


def save_tables(run_dir: Path, details: pd.DataFrame, grouped: pd.DataFrame, ranking: pd.DataFrame):
    grouped.to_csv(run_dir / "grouped_by_status.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(run_dir / "condition_ranking.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(run_dir / "grouped_analysis.xlsx", engine="xlsxwriter") as writer:
        ranking.to_excel(writer, sheet_name="condition_ranking", index=False)
        grouped.to_excel(writer, sheet_name="grouped_by_status", index=False)
        details.to_excel(writer, sheet_name="detailed", index=False)
        sample_path = sample_path_for(run_dir)
        if sample_path is not None:
            pd.read_csv(sample_path).to_excel(writer, sheet_name="sample", index=False)


def save_html(run_dir: Path, grouped: pd.DataFrame, ranking: pd.DataFrame):
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Grouped drug-likeness analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2 {{ margin: 22px 0 10px; }}
    img {{ max-width: 100%; border: 1px solid #d7dde5; margin: 8px 0 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 22px; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 7px 8px; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    tr:nth-child(even) {{ background: #fafbfc; }}
    .note {{ color: #52606d; }}
  </style>
</head>
<body>
  <h1>Grouped drug-likeness analysis</h1>
  <p class="note">Score scales are kept as configured and compared separately. Higher separation means the condition assigns higher scores to good molecules than to both bad and ambiguous molecules within the same scale.</p>
  <h2>Mean score by group</h2>
  <img src="grouped_mean_scores.png" alt="Grouped mean scores">
  <h2>Condition ranking by group separation</h2>
  <img src="condition_separation_ranking.png" alt="Condition separation ranking">
  <h2>Score distributions</h2>
  <img src="status_score_distributions.png" alt="Status score distributions">
  <h2>Best conditions</h2>
  {ranking.head(18).to_html(index=False, escape=True, float_format=lambda x: f"{x:.3f}")}
  <h2>Grouped table</h2>
  {grouped.to_html(index=False, escape=True, float_format=lambda x: f"{x:.3f}")}
</body>
</html>"""
    (run_dir / "grouped_report.html").write_text(html, encoding="utf-8")


def run_analysis(run_dir: Path | None = None) -> Path:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    resolved_run_dir = run_dir or find_latest_run_dir()
    details = add_condition_column(pd.read_csv(details_path_for(resolved_run_dir)))
    grouped, ranking = make_group_tables(details)
    save_grouped_barplot(resolved_run_dir, grouped)
    save_separation_plot(resolved_run_dir, ranking)
    save_boxplot(resolved_run_dir, details)
    save_tables(resolved_run_dir, details, grouped, ranking)
    save_html(resolved_run_dir, grouped, ranking)
    print(f"Analysis written to: {resolved_run_dir}")
    print("Top conditions:")
    print(
        ranking[
            [
                "condition",
                "bad",
                "ambiguous",
                "good",
                "good_minus_bad",
                "good_minus_ambiguous",
                "ordered_separation",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )
    return resolved_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze grouped drug-likeness experiment results.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing results_detailed.csv.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(getattr(args, "run_dir"))
