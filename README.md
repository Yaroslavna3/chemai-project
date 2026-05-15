# LLM-as-a-judge в оценке сгенерированных молекул

Этот проект исследует применение больших языковых моделей в качестве экспертного
оценщика качества молекул, полученных генеративными химическими моделями.

В отличие от бинарных метрик вида «валидна / невалидна», система предоставляет
осмысленный текстовый фидбек, охватывающий химическую реализуемость, новизну,
синтезопригодность и потенциальные риски.

## Experiments framework

The drug-likeness LLM experiments are configured through
`configs/druglikeness_experiment.json` and
`configs/druglikeness_prompt_template.txt`.

Run a dry run without VseGPT calls:

```bash
python run_druglikeness_experiments.py --dry-run
```

Run the full experiment:

```bash
export VSEGPT_API_KEY="..."
python run_druglikeness_experiments.py --config configs/druglikeness_experiment.json
```

Analyze a completed run:

```bash
python analyze_grouped_results.py --run-dir data/analysis/<run_folder>
```

To evaluate all molecules instead of a balanced subset, set
`sample.strategy` to `all` in the config. Each run folder stores the resolved
config, prompt template, rendered prompts, sample, model metadata, raw results,
summary tables, and plots.
