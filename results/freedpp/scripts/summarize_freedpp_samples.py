from __future__ import print_function

import csv
import os
import statistics
import sys


PY2 = sys.version_info[0] == 2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TRAJECTORIES = [
    ("docking_and_metrics", "Docking + metrics"),
    ("llm_and_docking", "LLM + docking"),
    ("llm_and_all_metrics", "LLM + docking + metrics"),
]
TARGETS = ["1kzn", "3fqs"]


def open_csv(path):
    if PY2:
        return open(path, "rb")
    return open(path, "r", newline="", encoding="utf-8-sig")


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def quantile(sorted_values, q):
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = pos - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def score_stats(rows):
    values = [to_float(row.get("score")) for row in rows]
    values = [value for value in values if value is not None]
    ordered = sorted(values)
    return {
        "ValidN": len(values),
        "Mean": sum(values) / len(values),
        "SD": statistics.stdev(values) if len(values) > 1 else 0.0,
        "Median": statistics.median(values),
        "IQR_Q1": quantile(ordered, 0.25),
        "IQR_Q3": quantile(ordered, 0.75),
    }


def percentage(count, total):
    return 100.0 * count / total if total else 0.0


def prop(row, name):
    return to_float(row.get(name))


def alert_is_clear(row, name):
    value = prop(row, name)
    return value == 0.0 if value is not None else True


def build_scorecard(target):
    rows_out = []
    for folder, label in TRAJECTORIES:
        path = os.path.join(ROOT, folder, "data", "{0}_sample_050.csv".format(target))
        if not os.path.exists(path):
            path = os.path.join(ROOT, folder, "data", "{0}_sample_040.csv".format(target))
        if not os.path.exists(path):
            continue
        rows = read_rows(path)
        total = len(rows)
        smiles = [row.get("smiles") or row.get("Smiles") for row in rows]
        unique_rate = percentage(len(set(smiles)), total)

        high_llm = 0
        good_docking = 0
        logp_ok = 0
        lipinski_ok = 0
        no_alerts = 0
        balanced = 0
        pains = 0
        surechembl = 0
        glaxo = 0

        for row in rows:
            score = prop(row, "score")
            docking = prop(row, "DockingScoreProperty")
            logp = prop(row, "LogPProperty")
            heavy = prop(row, "HeavyAtomCountProperty")
            acceptors = prop(row, "NumHAcceptorsProperty")
            donors = prop(row, "NumHDonorsProperty")
            score_ok = score is not None and score >= 0.70
            docking_ok = docking is not None and docking <= -8.0
            logp_in_range = logp is not None and 0 <= logp <= 5
            lipinski_profile = (
                logp_in_range
                and heavy is not None
                and heavy <= 40
                and acceptors is not None
                and acceptors <= 10
                and donors is not None
                and donors <= 5
            )
            pains_alert = prop(row, "PAINSProperty") == 1.0
            sure_alert = prop(row, "SureChEMBLProperty") == 1.0
            glaxo_alert = prop(row, "GlaxoProperty") == 1.0
            clean_alerts = not (pains_alert or sure_alert or glaxo_alert)

            high_llm += int(score_ok)
            good_docking += int(docking_ok)
            logp_ok += int(logp_in_range)
            lipinski_ok += int(lipinski_profile)
            no_alerts += int(clean_alerts)
            balanced += int(score_ok and docking_ok and lipinski_profile and clean_alerts)
            pains += int(pains_alert)
            surechembl += int(sure_alert)
            glaxo += int(glaxo_alert)

        rows_out.append(
            {
                "Group": label,
                "UniqueRate": round(unique_rate, 1),
                "Drug-likeness >= 0.70": round(percentage(high_llm, total), 1),
                "Docking <= -8.0 kcal/mol": round(percentage(good_docking, total), 1),
                "LogP in 0-5": round(percentage(logp_ok, total), 1),
                "Lipinski-style profile": round(percentage(lipinski_ok, total), 1),
                "No structural alerts": round(percentage(no_alerts, total), 1),
                "Balanced profile": round(percentage(balanced, total), 1),
                "PAINSAlertRate": round(percentage(pains, total), 1),
                "SureChEMBLAlertRate": round(percentage(surechembl, total), 1),
                "GlaxoAlertRate": round(percentage(glaxo, total), 1),
            }
        )
    return rows_out


def main():
    table_dir = os.path.join(ROOT, "tables")
    if not os.path.isdir(table_dir):
        os.makedirs(table_dir)

    summary_rows = []
    for target in TARGETS:
        for folder, label in TRAJECTORIES:
            path = os.path.join(ROOT, folder, "data", "{0}_sample_050.csv".format(target))
            if not os.path.exists(path):
                path = os.path.join(ROOT, folder, "data", "{0}_sample_040.csv".format(target))
            if not os.path.exists(path):
                continue
            stats = score_stats(read_rows(path))
            summary_rows.append(
                {
                    "Target": target.upper(),
                    "Trajectory": label,
                    "ValidN": stats["ValidN"],
                    "Mean": "{:.3f}".format(stats["Mean"]),
                    "SD": "{:.3f}".format(stats["SD"]),
                    "Median": "{:.3f}".format(stats["Median"]),
                    "IQR_Q1": "{:.3f}".format(stats["IQR_Q1"]),
                    "IQR_Q3": "{:.3f}".format(stats["IQR_Q3"]),
                }
            )
    write_csv(
        os.path.join(table_dir, "freedpp_llm_score_summary.csv"),
        ["Target", "Trajectory", "ValidN", "Mean", "SD", "Median", "IQR_Q1", "IQR_Q3"],
        summary_rows,
    )

    scorecard_fields = [
        "Group",
        "UniqueRate",
        "Drug-likeness >= 0.70",
        "Docking <= -8.0 kcal/mol",
        "LogP in 0-5",
        "Lipinski-style profile",
        "No structural alerts",
        "Balanced profile",
        "PAINSAlertRate",
        "SureChEMBLAlertRate",
        "GlaxoAlertRate",
    ]
    for target in TARGETS:
        rows = build_scorecard(target)
        if rows:
            write_csv(
                os.path.join(table_dir, "{0}_trajectory_scorecard.csv".format(target)),
                scorecard_fields,
                rows,
            )
    print("Updated FREED++ summary tables in {0}".format(table_dir))


if __name__ == "__main__":
    main()
