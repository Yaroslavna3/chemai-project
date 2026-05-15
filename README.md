# LLM-as-a-judge в оценке сгенерированных молекул

Этот проект исследует применение больших языковых моделей в качестве экспертного
оценщика качества молекул, полученных генеративными химическими моделями.

В отличие от бинарных метрик вида «валидна / невалидна», система предоставляет
осмысленный текстовый фидбек, охватывающий химическую реализуемость, новизну,
синтезопригодность и потенциальные риски.

## Фреймворк экспериментов

Эксперименты по оценке лекарственности молекул через LLM настраиваются через:

- `configs/druglikeness_experiment.json` — модели, шкалы оценок, формат молекул,
  способ выборки и параметры VseGPT;
- `configs/druglikeness_prompt_template.txt` — шаблон промпта для LLM.

Основные скрипты лежат в папке `scripts/`:

- `scripts/run_druglikeness_experiments.py` — запуск экспериментов;
- `scripts/analyze_grouped_results.py` — агрегация результатов и построение
  графиков.

### Проверочный запуск без VseGPT

Такой запуск создает папку эксперимента, выборку молекул и отрендеренные
промпты, но не отправляет запросы в LLM:

```bash
python scripts/run_druglikeness_experiments.py --dry-run
```

### Полный запуск эксперимента

Перед запуском нужно передать ключ VseGPT через переменную окружения:

```bash
export VSEGPT_API_KEY="..."
python scripts/run_druglikeness_experiments.py --config configs/druglikeness_experiment.json
```

В PowerShell:

```powershell
$env:VSEGPT_API_KEY="..."
python scripts/run_druglikeness_experiments.py --config configs/druglikeness_experiment.json
```

### Анализ завершенного запуска

```bash
python scripts/analyze_grouped_results.py --run-dir data/analysis/<run_folder>
```

### Запуск на всех молекулах

Чтобы оценивать все молекулы, а не сбалансированную подвыборку, в конфиге нужно
поменять стратегию:

```json
"sample": {
  "strategy": "all"
}
```

Каждая папка запуска в `data/analysis/` хранит конфиг эксперимента, шаблон
промпта, отрендеренные промпты, выборку молекул, информацию о моделях,
подробные результаты, сводные таблицы, графики и HTML-отчеты.
