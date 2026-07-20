# Experiment Results

This directory contains curated result artifacts used for the paper and appendix.
It intentionally excludes raw FREED++ experiment directories, model checkpoints,
Slurm logs, and API responses with technical failures.

## LLM Judge Benchmark

- `llm_judge/absolute_scoring/qwen_absolute_scoring_detailed.csv` contains
  molecule-level Qwen3-14B scores and comments on the absolute-scoring subset.

## FREED++ Experiments

The FREED++ results are grouped by protein target.

- `freedpp/1kzn/training/` contains epoch-level summaries and plots for the
  1KZN optimization runs.
- `freedpp/1kzn/generation_1000/` contains final generated-molecule evaluation
  tables and comparison figures for three 1KZN training trajectories.
- `freedpp/3fqs/training/` contains epoch-level summaries and plots for the
  3FQS optimization runs.
- `freedpp/3fqs/generation_1000/` contains 3FQS post-training generation
  comparison figures and the classical docking-and-properties sample table.
- `freedpp/tables/` contains compact paper-ready summaries across targets.

The main trajectory names are:

- `docking_and_properties`: docking score plus classical molecular filters.
- `llm_and_docking`: LLM drug-likeness score plus docking score.
- `llm_and_properties`: LLM drug-likeness score plus docking score and
  classical molecular filters.
