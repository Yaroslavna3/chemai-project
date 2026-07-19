# Data

This folder contains benchmark tables and supporting filter files used in the
LLM-as-a-judge experiments.

## Files

- `benchmark/absolute_scoring.csv`: 296 molecules labeled as `bad`,
  `ambiguous`, or `good` for absolute drug-likeness scoring.
- `benchmark/pair_smiles.csv`: 60 molecules with canonical and non-canonical
  SMILES for representation-robustness checks.
- `benchmark/pair_structure.csv`: 105 structurally related molecules with
  paired status labels and expert comments.
- `glaxo_filters.csv`: SMARTS patterns for Glaxo structural alerts.

## Common columns

- `NAME` / `LATIN NAME`: molecule name.
- `SMILES`, `CANONICAL SMILES`, `NON-CANONICAL SMILES`: molecular
  representations.
- `IUPAC`: IUPAC name.
- `STATUS`, `CATEGORY`, `STATUS_REASON`: expert-assigned benchmark labels.
- `CLASS`, `TOXICITY`, `COMMENT`, `EXPLANATION`: short expert annotations.
- `QED`: quantitative estimate of drug-likeness.
- `SA`: synthetic accessibility score; lower values are easier.
- `MW`: molecular weight.
- `LOGP`: calculated octanol-water partition coefficient.
- `RASCORE`: retrosynthetic accessibility score; higher values are easier.
- `LIPINSKI`: relaxed Lipinski Rule-of-Five pass flag, allowing up to one
  violation.
- `LIPINSKI 0`: strict Lipinski flag with zero violations.
- `BRENK`, `PAINS`, `GLAXO`: structural-alert flags.
- `BRENK STRUCTS`, `PAINS STRUCTS`, `GLAXO STRUCTS`: matched alert patterns.
