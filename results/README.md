# Experiment Results

This directory contains curated result artifacts used for the paper and appendix.
It intentionally excludes raw FREED++ experiment directories, model checkpoints,
Slurm logs, and API responses with technical failures.

## LLM Judge Benchmark

- `llm_judge/absolute_scoring/qwen_absolute_scoring_detailed.csv` contains
  molecule-level Qwen3-14B scores and comments on the absolute-scoring subset.
- `llm_judge/absolute_scoring/scripts/plot_qwen_absolute_scoring.py` generates
  the histogram and boxplot figures in `llm_judge/absolute_scoring/images/`.

## FREED++ Experiments

The FREED++ results are grouped by optimization trajectory.

- `freedpp/docking_and_metrics/`: docking score plus classical molecular
  filters.
- `freedpp/llm_and_docking/`: LLM drug-likeness score plus docking score.
- `freedpp/llm_and_all_metrics/`: LLM drug-likeness score plus docking score
  and classical molecular filters.

Each trajectory folder contains:

- `data/`: epoch-level summaries and final-sample tables.
- `images/`: figures generated from the data tables.

Shared FREED++ scripts and cross-trajectory figures are stored in:

- `freedpp/scripts/plot_freedpp_results.py`
- `freedpp/scripts/summarize_freedpp_samples.py`
- `freedpp/images/`
- `freedpp/tables/`
