# LLM-as-a-judge for Generated Molecule Evaluation

This project studies large language models as expert judges for evaluating the
quality of molecules produced by generative chemistry models.

Unlike binary validity checks, the framework can produce structured scores and
text feedback about medicinal-chemistry quality, feasibility, structural
liabilities, and potential risks.

## Experiments Framework

Drug-likeness LLM experiments are configured through:

- `configs/druglikeness_experiment.json` for models, score ranges, molecule
  representations, sampling strategy, and VseGPT parameters;
- `configs/druglikeness_prompt_template.txt` for the LLM prompt template.

Python entrypoints are stored in `scripts/`:

- `scripts/run_druglikeness_experiments.py` runs configured experiments;
- `scripts/analyze_grouped_results.py` aggregates results and builds plots.

### Dry Run Without VseGPT Calls

This creates a run folder, molecule sample, and rendered prompts without calling
an LLM:

```bash
python scripts/run_druglikeness_experiments.py --dry-run
```

### Full Experiment Run

Set the VseGPT API key through an environment variable before running:

```bash
export VSEGPT_API_KEY="..."
python scripts/run_druglikeness_experiments.py --config configs/druglikeness_experiment.json
```

PowerShell:

```powershell
$env:VSEGPT_API_KEY="..."
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
