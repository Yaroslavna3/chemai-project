# LLM Result Analysis

This folder contains analysis scripts for completed LLM-judge result tables.
They are based on the earlier notebooks for the three benchmark subsets, but
are kept as command-line scripts so the analysis is easier to reproduce.

## Inputs

Each command expects a CSV file with LLM scores already computed.

- Absolute scoring: rows with `model_family`, `representation`, `name`,
  `status`, `status_reason`, `score`, and `comment`.
- Pair SMILES: rows with `model_family`, `smiles_type`, `name`, `category`,
  `score`, and `comment`.
- Pair structure: rows with `model_family`, `representation`, `status_pair`,
  `score`, and optional molecule metadata.

Column matching is case-insensitive and tolerant to spaces or hyphens.

## Usage

```bash
python scripts/analysis/analyze_llm_results.py absolute --input data/analysis/<run>/results_detailed.csv
python scripts/analysis/analyze_llm_results.py pair-smiles --input data/analysis/<run>/results_detailed.csv
python scripts/analysis/analyze_llm_results.py pair-structure --input data/analysis/<run>/results_detailed.csv
```

By default, outputs are written to a sibling folder named
`<input_stem>_analysis`. Use `--output-dir` to choose another location.

