import argparse
import json
import math
import os
import random
import re
import shutil
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "druglikeness_experiment.json"
PROMPT_TEMPLATE_PATH = REPO_ROOT / "configs" / "druglikeness_prompt_template.txt"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
ENV_PATH = REPO_ROOT / ".env"

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "druglikeness_v1",
    "dataset_path": "data/benchmark/absolute_scoring.csv",
    "output_root": "data/analysis",
    "api_base": "API_BASE",
    "api_key_env": "API_KEY",
    "sample_random_seed": 20260514,
    "max_retries": 3,
    "request_timeout_seconds": 180,
    "llm_parameters": {
        "temperature": 0.1,
        "max_tokens": 7000,
        "system_prompt": "You are precise, conservative, and return only valid JSON.",
        "extra_headers": {},
    },
    "sample": {
        "strategy": "balanced_per_status",
        "statuses": ["bad", "good", "ambiguous"],
        "per_status": 10,
        "balance_by": "STATUS_REASON",
    },
    "descriptor_columns": ["QED", "SA", "MW", "LOGP", "LIPINSKI", "BRENK", "PAINS", "GLAXO"],
    "models": [
        {"family": "gpt", "model": "openai/gpt-4.1-nano", "label": "GPT-4.1 Nano"},
        {"family": "claude", "model": "anthropic/claude-3-haiku", "label": "Claude 3 Haiku"},
        {"family": "qwen", "model": "qwen/qwen3-14b", "label": "Qwen3 14B"},
    ],
    "representations": ["smiles", "iupac"],
    "scales": [{"id": "0_1", "label": "0-1", "min": 0, "max": 1}],
    "prompt_template_path": "configs/druglikeness_prompt_template.txt",
}

COLUMN_CANDIDATES = {
    "name": ["NAME", "Name", "LATIN NAME", "Latin Name"],
    "smiles": ["SMILES", "smiles"],
    "iupac": ["IUPAC", "iupac"],
    "status": ["STATUS", "Status", "status"],
    "status_reason": ["STATUS_REASON", "Status_Reason", "status_reason"],
}


def load_env_file(path: Path = ENV_PATH):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> dict[str, Any]:
    load_env_file()
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            user_config = json.load(fh)
        config = deep_merge(DEFAULT_CONFIG, user_config)
    else:
        config = dict(DEFAULT_CONFIG)
    return config


def resolve_env_value(value: str, field_name: str) -> str:
    if value.startswith("http://") or value.startswith("https://"):
        return value
    env_value = os.getenv(value)
    if env_value:
        return env_value
    raise RuntimeError(f"Set {value} in .env or the environment for {field_name}.")


def extract_api_key(config: dict[str, Any]) -> str:
    env_name = config.get("api_key_env", "API_KEY")
    api_key = os.getenv(env_name)
    if api_key:
        return api_key
    raise RuntimeError(f"Set {env_name} before running the experiments.")


def request_json(
    api_base: str,
    path: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 120,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(api_base.rstrip("/") + path, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_decimal(value):
    if pd.isna(value):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"true", "false"}:
            return stripped.lower() == "true"
        if "," in stripped and re.fullmatch(r"-?\d+,\d+", stripped):
            return float(stripped.replace(",", "."))
    return value


def find_column(df: pd.DataFrame, logical_name: str, required: bool = True) -> str | None:
    candidates = COLUMN_CANDIDATES.get(logical_name, [logical_name])
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        raise KeyError(f"Required column not found for {logical_name}: tried {candidates}")
    return None


def row_value(row: pd.Series, logical_name: str, default: Any = "") -> Any:
    candidates = COLUMN_CANDIDATES.get(logical_name, [logical_name])
    for column in candidates:
        if column in row.index:
            value = row[column]
            if not pd.isna(value):
                return value
    return default


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    dataset_path = resolve_repo_path(config["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    df = pd.read_csv(dataset_path)
    for col in config["descriptor_columns"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_decimal)
    return df


def balanced_take(group: pd.DataFrame, n: int, rng: random.Random, balance_by: str) -> pd.DataFrame:
    reasons = list(group[balance_by].dropna().unique())
    rng.shuffle(reasons)
    buckets = {reason: group[group[balance_by] == reason].copy() for reason in reasons}
    for reason in reasons:
        buckets[reason] = buckets[reason].sample(frac=1, random_state=rng.randrange(1_000_000))

    chosen_indices = []
    positions = defaultdict(int)
    while len(chosen_indices) < n:
        progressed = False
        for reason in reasons:
            bucket = buckets[reason]
            pos = positions[reason]
            if pos < len(bucket) and len(chosen_indices) < n:
                chosen_indices.append(bucket.index[pos])
                positions[reason] += 1
                progressed = True
        if not progressed:
            break
    return group.loc[chosen_indices].sample(frac=1, random_state=rng.randrange(1_000_000))


def make_sample(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rng = random.Random(config.get("sample_random_seed", 20260514))
    sample_config = config["sample"]
    strategy = sample_config["strategy"]
    status_column = find_column(df, "status")

    if strategy == "all":
        sample = df.copy().sample(frac=1, random_state=config.get("sample_random_seed", 20260514)).reset_index(drop=True)
    elif strategy == "balanced_per_status":
        parts = []
        for status in sample_config["statuses"]:
            status_group = df[df[status_column].astype(str).str.lower() == str(status).lower()]
            parts.append(
                balanced_take(
                    status_group,
                    int(sample_config["per_status"]),
                    rng,
                    sample_config.get("balance_by", "STATUS_REASON"),
                )
            )
        sample = pd.concat(parts, ignore_index=True)
    else:
        raise ValueError(f"Unknown sample strategy: {strategy}")

    if "molecule_id" in sample.columns:
        sample = sample.drop(columns=["molecule_id"])
    sample.insert(0, "molecule_id", [f"M{i + 1:04d}" for i in range(len(sample))])
    return sample


def sample_export_columns(sample: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    preferred = [
        "molecule_id",
        "NAME",
        "Name",
        "SMILES",
        "STATUS",
        "Status",
        "STATUS_REASON",
        "Status_Reason",
        "IUPAC",
    ]
    preferred.extend(config["descriptor_columns"])
    preferred.extend(["BRENK STRUCTS", "PAINS STRUCTS", "GLAXO STRUCTS", "Glaxo STRUCTS"])
    return [column for column in preferred if column in sample.columns]


def molecule_payload(row: pd.Series, representation: str, descriptor_columns: list[str]) -> dict[str, Any]:
    base = {
        "molecule_id": row["molecule_id"],
        "name": row_value(row, "name"),
    }
    if representation == "smiles":
        base["smiles"] = row_value(row, "smiles")
    elif representation == "iupac":
        base["iupac"] = row_value(row, "iupac")
    elif representation == "smiles_descriptors":
        base["smiles"] = row["SMILES"]
        base["descriptors"] = {col: row[col] for col in descriptor_columns if col in row}
    else:
        raise ValueError(f"Unknown representation: {representation}")
    return base


def load_prompt_template(config: dict[str, Any]) -> str:
    template_path = resolve_repo_path(config["prompt_template_path"])
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def prompt_for_batch(
    sample: pd.DataFrame,
    representation: str,
    scale: dict[str, Any],
    config: dict[str, Any],
    prompt_template: str | None = None,
) -> str:
    template = prompt_template or load_prompt_template(config)
    molecules = [
        molecule_payload(row, representation, config["descriptor_columns"])
        for _, row in sample.iterrows()
    ]
    return template.format(
        min_score=scale["min"],
        max_score=scale["max"],
        representation=representation,
        molecule_batch=json.dumps(molecules, ensure_ascii=False),
    ).strip()


def parse_llm_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    block = re.search(r"\{.*\}", text, re.S)
    if block:
        return json.loads(block.group(0))
    raise ValueError("Failed to parse JSON")


def call_model(
    api_key: str,
    model: str,
    prompt: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    llm_params = config["llm_parameters"]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": llm_params["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        "temperature": llm_params["temperature"],
        "max_tokens": llm_params["max_tokens"],
    }
    last_error = None
    for attempt in range(1, int(config["max_retries"]) + 1):
        try:
            raw = request_json(
                config["api_base"],
                "/chat/completions",
                api_key,
                payload,
                timeout=int(config["request_timeout_seconds"]),
                extra_headers=llm_params.get("extra_headers", {}),
            )
            content = raw["choices"][0]["message"]["content"]
            return parse_llm_json(content), raw.get("usage", {})
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            last_error = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"Model call failed for {model}: {last_error}")


def make_run_dir(config: dict[str, Any], run_id: str | None) -> Path:
    output_root = resolve_repo_path(config["output_root"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config["experiment_name"]).strip("_")
    run_name = run_id or f"{timestamp}_{safe_name}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_metadata(run_dir: Path, config_path: Path, config: dict[str, Any]):
    (run_dir / "config.resolved.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if config_path.exists():
        shutil.copy2(config_path, run_dir / "config.input.json")


def scale_label(scale: dict[str, Any]) -> str:
    return scale.get("label") or str(scale["id"]).replace("_", "-")


def clamp_score(score: float, scale: dict[str, Any]) -> float:
    return max(float(scale["min"]), min(float(scale["max"]), score))


def run_experiments(config_path: Path = CONFIG_PATH, run_id: str | None = None, dry_run: bool = False) -> Path:
    config = load_config(config_path)
    run_dir = make_run_dir(config, run_id)
    save_run_metadata(run_dir, config_path, config)

    df = load_dataset(config)
    sample = make_sample(df, config)
    prompt_template = load_prompt_template(config)

    sample_to_save = sample[sample_export_columns(sample, config)]
    sample_to_save.to_csv(run_dir / "sample.csv", index=False, encoding="utf-8-sig")

    rendered_prompt_dir = run_dir / "rendered_prompts"
    rendered_prompt_dir.mkdir(exist_ok=True)

    if not dry_run:
        config["api_base"] = resolve_env_value(str(config["api_base"]), "api_base")
        api_key = extract_api_key(config)

    detail_rows = []
    experiment_rows = []

    for model_info in config["models"]:
        for representation in config["representations"]:
            for scale in config["scales"]:
                exp_id = f"{model_info['family']}_{representation}_{scale['id']}"
                print(f"Running {exp_id}", flush=True)
                prompt = prompt_for_batch(sample, representation, scale, config, prompt_template)
                (rendered_prompt_dir / f"{exp_id}.txt").write_text(prompt, encoding="utf-8")

                if dry_run:
                    continue

                parsed, usage = call_model(api_key, model_info["model"], prompt, config)
                results = parsed.get("results", [])
                by_id = {str(item.get("molecule_id")): item for item in results}

                scores = []
                for _, row in sample.iterrows():
                    item = by_id.get(row["molecule_id"], {})
                    raw_score = item.get("drug_likeness_score")
                    try:
                        score = clamp_score(float(raw_score), scale)
                    except (TypeError, ValueError):
                        score = math.nan
                    if not math.isnan(score):
                        scores.append(score)

                    detail_rows.append(
                        {
                            "experiment_id": exp_id,
                            "model_family": model_info["family"],
                            "model": model_info["model"],
                            "representation": representation,
                            "scale": scale_label(scale),
                            "scale_min": scale["min"],
                            "scale_max": scale["max"],
                            "molecule_id": row["molecule_id"],
                            "name": row_value(row, "name"),
                            "status": row_value(row, "status"),
                            "status_reason": row_value(row, "status_reason"),
                            "smiles": row_value(row, "smiles"),
                            "iupac": row_value(row, "iupac"),
                            "score": score,
                            "comment": item.get("comment", ""),
                        }
                    )

                experiment_rows.append(
                    {
                        "experiment_id": exp_id,
                        "model_family": model_info["family"],
                        "model": model_info["model"],
                        "representation": representation,
                        "scale": scale_label(scale),
                        "scale_min": scale["min"],
                        "scale_max": scale["max"],
                        "n_molecules": len(sample),
                        "valid_scores": len(scores),
                        "mean_score": statistics.mean(scores) if scores else math.nan,
                        "median_score": statistics.median(scores) if scores else math.nan,
                        "stdev_score": statistics.pstdev(scores) if len(scores) > 1 else 0,
                        "min_score": min(scores) if scores else math.nan,
                        "max_score": max(scores) if scores else math.nan,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    }
                )

    if dry_run:
        print(f"Dry run complete: {run_dir}", flush=True)
        return run_dir

    details = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(experiment_rows)
    details.to_csv(run_dir / "results_detailed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(run_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(run_dir / "experiment_results.xlsx", engine="xlsxwriter") as writer:
        sample_to_save.to_excel(writer, sheet_name="sample", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        details.to_excel(writer, sheet_name="detailed", index=False)

    write_html_report(run_dir, sample_to_save, summary, details)
    print(f"Done: {run_dir}", flush=True)
    return run_dir


def write_html_report(
    run_dir: Path,
    sample: pd.DataFrame,
    summary: pd.DataFrame,
    details: pd.DataFrame,
):
    style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }
      h1, h2 { margin: 22px 0 10px; }
      table { border-collapse: collapse; width: 100%; margin-bottom: 22px; font-size: 13px; }
      th, td { border: 1px solid #d7dde5; padding: 7px 8px; vertical-align: top; }
      th { background: #eef2f7; position: sticky; top: 0; }
      tr:nth-child(even) { background: #fafbfc; }
      .note { color: #52606d; }
    </style>
    """
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Drug-likeness LLM experiments</title>",
        style,
        "</head><body>",
        "<h1>Drug-likeness LLM experiments</h1>",
        "<p class='note'>Each run folder contains the resolved config, rendered prompts, sample, and results.</p>",
        "<h2>Sample</h2>",
        sample.to_html(index=False, escape=True),
        "<h2>Experiment Summary</h2>",
        summary.to_html(index=False, escape=True, float_format=lambda x: f"{x:.3f}"),
        "<h2>Detailed Results</h2>",
        details.to_html(index=False, escape=True, float_format=lambda x: f"{x:.3f}"),
        "</body></html>",
    ]
    (run_dir / "experiment_report.html").write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run config-driven drug-likeness LLM experiments.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to experiment JSON config.")
    parser.add_argument("--run-id", default=None, help="Optional output folder name under output_root.")
    parser.add_argument("--dry-run", action="store_true", help="Create run folder, sample, and rendered prompts without API calls.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(config_path=args.config, run_id=getattr(args, "run_id"), dry_run=getattr(args, "dry_run"))
