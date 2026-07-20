# FreeDPP custom overlay

This folder contains the minimal FreeDPP files used for the LLM-reward
molecule-generation experiments. It is not a full FreeDPP fork. Copy these
files over a working FreeDPP checkout, or use them as a reference patch for the
changed files.

## Contents

- `freedpp/main.py`: FreeDPP entry point with the added `DrugLikeness`
  objective and CSV output columns `smiles`, `score`, and `comment`.
- `freedpp/env/reward.py`: LLM drug-likeness reward with a strict medicinal
  chemistry prompt, JSON parsing, retry logic, and comments.
- `configs/llm_config.example.json`: OpenAI-compatible API configuration
  template. Copy it to `configs/llm_config.json` and fill local credentials.
- `scripts/train.sh`: SLURM training script for the LLM + docking and
  LLM + docking + metrics trajectories.
- `scripts/sample_1000.sh`: sample 1000 molecules from a trained checkpoint.
- `scripts/evaluate.sh`: evaluate generated molecules with all metrics.
- `scripts/run_chain_example.sh`: example dependency chain for
  train -> sample -> evaluate.
- `receptors/`: receptor files used in the docking experiments.

## Local configuration

Create a private config file before running LLM-based objectives:

```bash
cp configs/llm_config.example.json configs/llm_config.json
```

Then fill:

```json
{
  "api_base": "https://your-provider.example/v1",
  "api_key": "YOUR_API_KEY",
  "model": "qwen/qwen3-14b",
  "temperature": 0.1,
  "max_tokens": 1500,
  "retries": 2,
  "extra_headers": {}
}
```

`configs/llm_config.json` is ignored by Git and must not be committed.

## Runtime environment

The scripts use paths relative to this folder. Optional environment variables:

```bash
export FREEDPP_PROJECT_ROOT=/path/to/freedpp_custom
export FREEDPP_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
export FREEDPP_CONDA_ENV=/path/to/conda/envs/freedpp310
export FREEDPP_RUNTIME_ROOT=/path/to/runtime
```

If `FREEDPP_CONDA_ENV` is not set, the scripts use the current Python
environment.

## Commands

Train on a target:

```bash
sbatch scripts/train.sh 1kzn llm_docking
sbatch scripts/train.sh 1kzn all_properties
sbatch scripts/train.sh 3fqs llm_docking
sbatch scripts/train.sh 3fqs all_properties
```

Sample 1000 molecules from a checkpoint:

```bash
sbatch scripts/sample_1000.sh 1kzn llm_docking experiments/1kzn_llm_docking_50_ep/ckpt/model_050.pth
```

Evaluate all metrics for a sampling experiment:

```bash
sbatch scripts/evaluate.sh 1kzn 1kzn_llm_docking_sample_1000 50
```

For an end-to-end SLURM dependency example, see `scripts/run_chain_example.sh`.
