import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


STATUS_ORDER = ["bad", "ambiguous", "good"]
STATUS_PALETTE = {
    "bad": "#b23a3a",
    "ambiguous": "#c88a2c",
    "good": "#2f7d59",
}
MODEL_ORDER = ["gpt", "claude", "qwen"]
REPRESENTATION_ORDER = ["smiles", "iupac"]


def normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(column) for column in df.columns]
    aliases = {
        "latin_name": "name",
        "status_reason": "status_reason",
        "smiles_type": "smiles_type",
        "status_pair": "status_pair",
        "drug_likeness_score": "score",
    }
    df = df.rename(columns={src: dst for src, dst in aliases.items() if src in df.columns})
    for column in ["model_family", "representation", "status", "status_reason", "category", "smiles_type", "status_pair"]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip().str.lower()
    if "score" not in df.columns:
        raise KeyError("Input table must contain a score column.")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def require_columns(df: pd.DataFrame, columns: list[str]):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")


def prepare_output_dir(input_path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        output_dir = input_path.with_name(f"{input_path.stem}_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ordered_values(values: pd.Series, preferred: list[str]) -> list[str]:
    present = [value for value in preferred if value in set(values.dropna())]
    extras = sorted(value for value in values.dropna().unique() if value not in present)
    return present + extras


def save_table(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def configure_plotting():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.family": "Arial",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def add_bar_labels(ax, fmt="{:.2f}", fontsize=8):
    for container in ax.containers:
        ax.bar_label(container, labels=[fmt.format(v) if pd.notna(v) else "" for v in container.datavalues], fontsize=fontsize, padding=2)


def is_problem_row(row: pd.Series) -> bool:
    if row["status"] == "bad":
        return row["score"] > 0.5
    if row["status"] == "good":
        return row["score"] < 0.5
    if row["status"] == "ambiguous":
        return row["score"] > 0.7
    return False


def analyze_absolute(input_path: Path, output_dir: Path | None):
    configure_plotting()
    output_dir = prepare_output_dir(input_path, output_dir)
    df = normalize_columns(pd.read_csv(input_path))
    require_columns(df, ["model_family", "representation", "name", "status", "status_reason", "score"])

    status_order = ordered_values(df["status"], STATUS_ORDER)
    model_order = ordered_values(df["model_family"], MODEL_ORDER)
    representation_order = ordered_values(df["representation"], REPRESENTATION_ORDER)

    summary = (
        df.groupby(["model_family", "representation", "status"], as_index=False)
        .agg(
            n=("score", "count"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            std_score=("score", "std"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .round(3)
    )
    save_table(summary, output_dir / "absolute_summary_by_status.csv")

    problem_df = df.copy()
    problem_df["is_problem"] = problem_df.apply(is_problem_row, axis=1)
    error_rates = (
        problem_df.groupby(["model_family", "representation", "status"], as_index=False)
        .agg(n=("score", "count"), problem_count=("is_problem", "sum"), problem_rate=("is_problem", "mean"))
        .round(3)
    )
    save_table(error_rates, output_dir / "absolute_error_rates.csv")

    reason_errors = (
        problem_df.groupby(["model_family", "representation", "status", "status_reason"], as_index=False)
        .agg(n=("score", "count"), problem_count=("is_problem", "sum"), problem_rate=("is_problem", "mean"))
        .sort_values(["model_family", "representation", "problem_rate"], ascending=[True, True, False])
        .round(3)
    )
    save_table(reason_errors, output_dir / "absolute_error_rates_by_status_reason.csv")

    paired_scores = (
        df.pivot_table(
            index=["model_family", "name", "status", "status_reason"],
            columns="representation",
            values="score",
            aggfunc="first",
        )
        .reset_index()
    )
    if {"smiles", "iupac"}.issubset(paired_scores.columns):
        paired_scores["delta_iupac_minus_smiles"] = paired_scores["iupac"] - paired_scores["smiles"]
        paired_scores["abs_delta"] = paired_scores["delta_iupac_minus_smiles"].abs()
        save_table(paired_scores.sort_values("abs_delta", ascending=False), output_dir / "absolute_representation_deltas.csv")

    g = sns.catplot(
        data=df,
        x="status",
        y="score",
        hue="representation",
        col="model_family",
        kind="box",
        order=status_order,
        hue_order=representation_order,
        col_order=model_order,
        palette="Set2",
        height=3.2,
        aspect=1.05,
        sharey=True,
    )
    g.set_axis_labels("Benchmark class", "LLM score")
    g.set_titles("{col_name}")
    g.figure.suptitle("Absolute scoring: score distributions by class", y=1.04)
    g.figure.savefig(output_dir / "absolute_score_distributions.png", bbox_inches="tight")
    plt.close(g.figure)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    plot_errors = error_rates.copy()
    plot_errors["condition"] = plot_errors["model_family"] + " / " + plot_errors["representation"]
    sns.barplot(
        data=plot_errors,
        x="condition",
        y="problem_rate",
        hue="status",
        hue_order=status_order,
        palette=STATUS_PALETTE,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Problem rate")
    ax.set_ylim(0, max(0.05, min(1.0, plot_errors["problem_rate"].max() * 1.25)))
    ax.set_title("Absolute scoring: benchmark-consistency errors")
    ax.tick_params(axis="x", rotation=25)
    add_bar_labels(ax, "{:.2f}", 7)
    fig.tight_layout()
    fig.savefig(output_dir / "absolute_error_rates.png", bbox_inches="tight")
    plt.close(fig)

    return output_dir


def analyze_pair_smiles(input_path: Path, output_dir: Path | None):
    configure_plotting()
    output_dir = prepare_output_dir(input_path, output_dir)
    df = normalize_columns(pd.read_csv(input_path))
    require_columns(df, ["model_family", "smiles_type", "name", "category", "score"])

    pivot = (
        df.pivot_table(
            index=["model_family", "name", "category"],
            columns="smiles_type",
            values="score",
            aggfunc="first",
        )
        .reset_index()
    )
    if {"canonical", "noncanonical"}.issubset(pivot.columns):
        pivot["delta_canonical_minus_noncanonical"] = pivot["canonical"] - pivot["noncanonical"]
        pivot["abs_delta"] = pivot["delta_canonical_minus_noncanonical"].abs()
    else:
        raise KeyError("Pair SMILES analysis requires canonical and noncanonical smiles_type values.")
    save_table(pivot.sort_values(["model_family", "abs_delta"], ascending=[True, False]), output_dir / "pair_smiles_deltas.csv")

    summary = (
        pivot.groupby("model_family", as_index=False)
        .agg(
            n_pairs=("name", "count"),
            changed_pairs=("abs_delta", lambda values: int((values.fillna(0) > 0).sum())),
            mean_abs_delta=("abs_delta", "mean"),
            median_abs_delta=("abs_delta", "median"),
            max_abs_delta=("abs_delta", "max"),
        )
        .round(3)
    )
    save_table(summary, output_dir / "pair_smiles_summary.csv")

    mean_scores = (
        df.groupby(["model_family", "smiles_type"], as_index=False)
        .agg(mean_score=("score", "mean"), median_score=("score", "median"), n=("score", "count"))
        .round(3)
    )
    save_table(mean_scores, output_dir / "pair_smiles_mean_scores.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    sns.boxplot(
        data=pivot,
        x="model_family",
        y="abs_delta",
        order=ordered_values(pivot["model_family"], MODEL_ORDER),
        color="#8fb7d9",
        ax=axes[0],
    )
    sns.stripplot(
        data=pivot,
        x="model_family",
        y="abs_delta",
        order=ordered_values(pivot["model_family"], MODEL_ORDER),
        color="#1f2933",
        alpha=0.35,
        size=3,
        ax=axes[0],
    )
    axes[0].set_title("Canonical vs noncanonical stability")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("|score delta|")

    sns.barplot(
        data=mean_scores,
        x="model_family",
        y="mean_score",
        hue="smiles_type",
        order=ordered_values(mean_scores["model_family"], MODEL_ORDER),
        ax=axes[1],
        palette="Set2",
    )
    axes[1].set_title("Mean score by SMILES type")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Mean LLM score")
    add_bar_labels(axes[1], "{:.2f}", 8)
    fig.tight_layout()
    fig.savefig(output_dir / "pair_smiles_stability.png", bbox_inches="tight")
    plt.close(fig)

    return output_dir


def analyze_pair_structure(input_path: Path, output_dir: Path | None):
    configure_plotting()
    output_dir = prepare_output_dir(input_path, output_dir)
    df = normalize_columns(pd.read_csv(input_path))
    require_columns(df, ["model_family", "representation", "status_pair", "score"])

    summary = (
        df.groupby(["model_family", "representation", "status_pair"], as_index=False)
        .agg(
            n=("score", "count"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            std_score=("score", "std"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .round(3)
    )
    save_table(summary, output_dir / "pair_structure_summary.csv")

    g = sns.catplot(
        data=summary,
        x="status_pair",
        y="mean_score",
        hue="model_family",
        col="representation",
        kind="bar",
        col_order=ordered_values(summary["representation"], REPRESENTATION_ORDER),
        hue_order=ordered_values(summary["model_family"], MODEL_ORDER),
        palette="Set2",
        height=4.0,
        aspect=1.25,
        sharey=True,
    )
    g.set_axis_labels("Pair status", "Mean LLM score")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=25)
        add_bar_labels(ax, "{:.2f}", 7)
    g.figure.suptitle("Pair structure benchmark: mean scores by structural pair status", y=1.04)
    g.figure.savefig(output_dir / "pair_structure_mean_scores.png", bbox_inches="tight")
    plt.close(g.figure)

    g = sns.catplot(
        data=df,
        x="status_pair",
        y="score",
        hue="model_family",
        col="representation",
        kind="box",
        col_order=ordered_values(df["representation"], REPRESENTATION_ORDER),
        hue_order=ordered_values(df["model_family"], MODEL_ORDER),
        palette="Set2",
        height=4.0,
        aspect=1.25,
        sharey=True,
    )
    g.set_axis_labels("Pair status", "LLM score")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=25)
    g.figure.suptitle("Pair structure benchmark: score distributions", y=1.04)
    g.figure.savefig(output_dir / "pair_structure_distributions.png", bbox_inches="tight")
    plt.close(g.figure)

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed LLM-judge benchmark result tables.")
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    for name in ["absolute", "pair-smiles", "pair-structure"]:
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--input", required=True, type=Path, help="CSV file with detailed LLM scores.")
        subparser.add_argument("--output-dir", type=Path, default=None, help="Directory for tables and plots.")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.benchmark == "absolute":
        output_dir = analyze_absolute(args.input, args.output_dir)
    elif args.benchmark == "pair-smiles":
        output_dir = analyze_pair_smiles(args.input, args.output_dir)
    elif args.benchmark == "pair-structure":
        output_dir = analyze_pair_structure(args.input, args.output_dir)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    print(f"Analysis written to: {output_dir}")


if __name__ == "__main__":
    main()
