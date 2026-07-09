# LLM-as-a-judge for Generated Molecule Evaluation

This project studies large language models as expert judges for evaluating the
quality of molecules produced by generative chemistry models.

Unlike binary validity checks, the framework can produce structured scores and
text feedback about medicinal-chemistry quality, feasibility, structural
liabilities, and potential risks.

## Experiments Framework

Drug-likeness LLM experiments are configured through:

- `configs/druglikeness_experiment.json` for models, score ranges, molecule
  representations, sampling strategy, and API environment variable names;
- `configs/druglikeness_prompt_template.txt` for the LLM prompt template.

Python entrypoints are stored in `scripts/`:

- `scripts/run_druglikeness_experiments.py` runs configured experiments;
- `scripts/analyze_grouped_results.py` aggregates results and builds plots.

### Dry Run Without API Calls

This creates a run folder, molecule sample, and rendered prompts without calling
an LLM:

```bash
python scripts/run_druglikeness_experiments.py --dry-run
```

### Full Experiment Run

Create a local `.env` file with the API base URL and API key before running:

```dotenv
API_BASE=https://example.com/v1
API_KEY=...
```

The `.env` file is ignored by Git and should not be committed. The config keeps
only the environment variable names:

```json
"api_base": "API_BASE",
"api_key_env": "API_KEY"
```

Run the experiment:

```bash
python scripts/run_druglikeness_experiments.py --config configs/druglikeness_experiment.json
```

### Analyze a Completed Run

```bash
python scripts/analyze_grouped_results.py --run-dir data/analysis/<run_folder>
```

### Evaluate All Molecules

To evaluate all molecules instead of a balanced subset, set the sampling
strategy in the config:

```json
"sample": {
  "strategy": "all"
}
```

Each run folder in `data/analysis/` stores the resolved config, rendered
prompts, sampled molecules, detailed results, summary tables, plots, and HTML
reports.
